#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Unit tests for BaseOutputTransport.MediaSender.handle_audio_frame.

These tests cover the re-chunking path that splits incoming audio into
the transport's fixed audio_chunk_size before pushing onto
`_audio_queue`. The goal is to lock in the chunking semantics so the
allocation-reduction changes (memoryview slice + in-place buffer
trimming) stay behavior-preserving.
"""

import unittest
from unittest.mock import AsyncMock, MagicMock

from pipecat.frames.frames import OutputAudioRawFrame
from pipecat.transports.base_output import BaseOutputTransport
from pipecat.transports.base_transport import TransportParams


def _make_sender(
    chunk_size: int,
    sample_rate: int = 16000,
) -> BaseOutputTransport.MediaSender:
    """Construct a MediaSender wired with mocks for the audio path only.

    The full transport spins up audio/video/clock asyncio tasks in
    `start()`. For testing `handle_audio_frame` directly we only need a
    sender object with an `_audio_queue` and an `_resampler`; the audio
    task is never started.
    """
    transport = MagicMock(spec=BaseOutputTransport)
    params = TransportParams(audio_out_enabled=True)
    sender = BaseOutputTransport.MediaSender(
        transport,
        destination=None,
        sample_rate=sample_rate,
        audio_chunk_size=chunk_size,
        params=params,
    )
    # Stub the resampler to a passthrough so tests can assert on the
    # exact bytes that flow through chunking.
    sender._resampler = MagicMock()
    sender._resampler.resample = AsyncMock(side_effect=lambda audio, src, dst: audio)
    # Replace the queue with a plain awaitable mock - `_audio_queue` is
    # normally a FrameQueue created inside `_create_audio_task`.
    sender._audio_queue = AsyncMock()
    sender._audio_queue.put = AsyncMock()
    return sender


class TestMediaSenderChunking(unittest.IsolatedAsyncioTestCase):
    async def test_exact_multiple_produces_n_chunks(self):
        """Feeding exactly N * chunk_size bytes produces N chunks and
        leaves the buffer empty.
        """
        chunk_size = 320  # 10 ms at 16 kHz, mono, 16-bit
        sender = _make_sender(chunk_size=chunk_size)
        audio = bytes(range(256)) * 5  # 1280 bytes = 4 * chunk_size

        frame = OutputAudioRawFrame(
            audio=audio, sample_rate=16000, num_channels=1
        )
        await sender.handle_audio_frame(frame)

        self.assertEqual(sender._audio_queue.put.await_count, 4)
        self.assertEqual(len(sender._audio_buffer), 0)

        # Bytes must come out in order, untouched.
        for i, call in enumerate(sender._audio_queue.put.await_args_list):
            chunk = call.args[0]
            expected = audio[i * chunk_size : (i + 1) * chunk_size]
            self.assertIsInstance(chunk, OutputAudioRawFrame)
            self.assertEqual(chunk.audio, expected)
            self.assertEqual(chunk.sample_rate, 16000)

    async def test_partial_remainder_stays_buffered(self):
        """Bytes shorter than a full chunk remain in `_audio_buffer`.

        Sending 3.5 chunks worth of audio should dispatch 3 chunks and
        leave the trailing 0.5-chunk in the buffer.
        """
        chunk_size = 100
        sender = _make_sender(chunk_size=chunk_size)
        audio = bytes(i % 256 for i in range(350))

        frame = OutputAudioRawFrame(
            audio=audio, sample_rate=16000, num_channels=1
        )
        await sender.handle_audio_frame(frame)

        self.assertEqual(sender._audio_queue.put.await_count, 3)
        self.assertEqual(bytes(sender._audio_buffer), audio[300:])

    async def test_buffer_is_modified_in_place_not_rebound(self):
        """`del buf[:n]` must reuse the existing bytearray object.

        Regression guard for the perf change: the prior implementation
        did `self._audio_buffer = self._audio_buffer[n:]`, which
        allocated a fresh bytearray every iteration. The fix uses
        `del buf[:n]` so the same bytearray object survives across
        chunk dispatches.
        """
        chunk_size = 100
        sender = _make_sender(chunk_size=chunk_size)
        original_buffer = sender._audio_buffer
        audio = b"\x00" * 250  # two full chunks + 50 bytes remainder

        frame = OutputAudioRawFrame(
            audio=audio, sample_rate=16000, num_channels=1
        )
        await sender.handle_audio_frame(frame)

        # Same object, not a new bytearray.
        self.assertIs(sender._audio_buffer, original_buffer)
        self.assertEqual(len(sender._audio_buffer), 50)

    async def test_subsequent_calls_concatenate_buffer(self):
        """Two partial frames combine into a full chunk on the second call.

        Send half a chunk twice; the second call should produce one
        complete chunk whose bytes are the concatenation of the two
        halves in order.
        """
        chunk_size = 200
        sender = _make_sender(chunk_size=chunk_size)

        first = b"\xaa" * 100
        second = b"\xbb" * 100

        for audio in (first, second):
            frame = OutputAudioRawFrame(
                audio=audio, sample_rate=16000, num_channels=1
            )
            await sender.handle_audio_frame(frame)

        self.assertEqual(sender._audio_queue.put.await_count, 1)
        emitted = sender._audio_queue.put.await_args.args[0]
        self.assertEqual(emitted.audio, first + second)
        self.assertEqual(len(sender._audio_buffer), 0)

    async def test_chunk_emitted_is_bytes_not_bytearray(self):
        """Frames pushed to `_audio_queue` must carry `bytes`, not the
        live bytearray. Otherwise mutating `_audio_buffer` after
        emission would mutate the already-queued chunk's audio.
        """
        chunk_size = 100
        sender = _make_sender(chunk_size=chunk_size)
        audio = b"\x42" * 100

        frame = OutputAudioRawFrame(
            audio=audio, sample_rate=16000, num_channels=1
        )
        await sender.handle_audio_frame(frame)

        emitted = sender._audio_queue.put.await_args.args[0]
        self.assertIsInstance(emitted.audio, bytes)
        # Mutating the buffer after emission must not change the
        # already-dispatched chunk.
        sender._audio_buffer.extend(b"\xff" * 50)
        self.assertEqual(emitted.audio, audio)

    async def test_audio_out_disabled_short_circuits(self):
        """When `audio_out_enabled` is False the method returns
        immediately, without touching the buffer or the queue.
        """
        chunk_size = 100
        sender = _make_sender(chunk_size=chunk_size)
        sender._params.audio_out_enabled = False

        frame = OutputAudioRawFrame(
            audio=b"\x00" * 500, sample_rate=16000, num_channels=1
        )
        await sender.handle_audio_frame(frame)

        sender._audio_queue.put.assert_not_called()
        self.assertEqual(len(sender._audio_buffer), 0)


if __name__ == "__main__":
    unittest.main()
