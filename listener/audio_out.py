"""Persistent audio output: write PCM straight to the device.

This is the deliberate opposite of the old file. The previous version had a
WSOLA time-stretcher, a speed governor, a learned prebuffer, jitter margins,
a reactive "dry buffer" recovery net and half a dozen tuning knobs all trying
to smooth over gaps in network TTS streams. That machinery exists because Groq
delivers audio at roughly real-time over a socket and the player used to bleed
its buffer dry mid-reply.

Jarvis now talks with the Moonshine local voice first (listener/tts_moonshine),
which synthesises whole sentences locally before handing them over -- there is
nothing to smooth, so none of that should touch the samples. Whatever the TTS
produces is what the speakers play: one queue, one device stream, no
time-stretch, no speed control, no prebuffer tuning.

The persistent OutputStream is shared by cues (ready/done earcons speak through
play_blocking) and by TTS playback, so a cue never opens a competing device.
"""

import collections
import threading
import time
import traceback

import numpy as np
import sounddevice as sd

from listener import ui

DEFAULT_SAMPLE_RATE = 24000
CHANNELS = 1
BLOCKSIZE = 1024

# Device buffer to ask PortAudio for.
OUTPUT_LATENCY = 0.12

# Frames per stream.write().
WRITE_FRAMES = BLOCKSIZE * 2

# Waveform level granularity.
VIS_SLICE_SECONDS = 1.0 / 30.0

_stream = None
_stream_rate = None
_stream_lock = threading.Lock()


def get_stream(samplerate=DEFAULT_SAMPLE_RATE):
    global _stream, _stream_rate
    with _stream_lock:
        if _stream is not None and _stream_rate != samplerate:
            _stream.stop()
            _stream.close()
            _stream = None
            _stream_rate = None

        if _stream is None:
            try:
                _stream = sd.OutputStream(
                    samplerate=samplerate,
                    channels=CHANNELS,
                    dtype="int16",
                    blocksize=BLOCKSIZE,
                    latency=OUTPUT_LATENCY,
                )
            except sd.PortAudioError as e:
                print("audio_out: couldn't open output stream ({}); "
                      "trying without explicit latency".format(e))
                _stream = sd.OutputStream(
                    samplerate=samplerate,
                    channels=CHANNELS,
                    dtype="int16",
                    blocksize=BLOCKSIZE,
                )
            _stream.start()
            _stream_rate = samplerate

        return _stream


def close_stream():
    global _stream, _stream_rate
    with _stream_lock:
        if _stream is not None:
            try:
                _stream.stop()
                _stream.close()
            except Exception as e:
                print("audio_out: closing stream raised ({})".format(e))
            _stream = None
            _stream_rate = None


def play_blocking(samples, samplerate):
    """Writes a short int16 mono buffer straight to the shared output stream.

    listener/cues.py calls this for the two earcons (ready()/done()). It
    reuses the same persistent device stream (get_stream()) so a cue never
    fights TTS playback for the output device, and writes in WRITE_FRAMES-sized
    chunks so a longer cue can't hand PortAudio more than it wants in one call.

    Blocks until every sample has been handed to PortAudio.
    """
    if samples is None or len(samples) == 0:
        return
    stream = get_stream(samplerate)
    for start in range(0, len(samples), WRITE_FRAMES):
        stream.write(samples[start:start + WRITE_FRAMES])


def _rms(block):
    if len(block) == 0:
        return 0.0
    return float(np.sqrt(np.mean(block.astype(np.float32) ** 2)))


class StreamingPlayer:
    """Plays PCM as it is fed, as written -- nothing more.

    Thread A calls feed(pcm_bytes) with the TTS output, an arbitrary number of
    times and in arbitrary sizes. A single background thread drains the queue
    of complete int16 frames and writes them to the device immediately. There
    is deliberately no prebuffer, no time-stretch, no speed governor, and no
    dry-buffer recovery: whatever lands in feed() is what the speakers get.

    The only concession to byte boundaries is frame alignment -- the queue
    streams bytes, the device streams frames, so a partial frame is held back
    until its other half arrives.
    """

    def __init__(self, samplerate=DEFAULT_SAMPLE_RATE, on_first_audio=None):
        self.samplerate = samplerate
        self._on_first_audio = on_first_audio

        self._cv = threading.Condition()
        self._chunks = collections.deque()
        self._queued = 0
        self._done = False
        self._aborted = False

        self._leftover = b""
        self._thread = None
        self._first_audio_fired = False

    def start(self):
        get_stream(self.samplerate)
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def feed(self, pcm_bytes):
        if not pcm_bytes:
            return

        buf = self._leftover + pcm_bytes
        aligned = len(buf) - (len(buf) % 2)
        if aligned <= 0:
            self._leftover = buf
            return

        self._leftover = buf[aligned:]
        samples = np.frombuffer(buf[:aligned], dtype=np.int16)
        if len(samples) == 0:
            return

        with self._cv:
            if self._aborted:
                return
            self._chunks.append(samples)
            self._queued += len(samples)
            self._cv.notify_all()

    def finish(self, timeout=60.0):
        with self._cv:
            self._done = True
            self._cv.notify_all()
        if self._thread is not None:
            self._thread.join(timeout=timeout)

    def abort(self):
        with self._cv:
            self._aborted = True
            self._done = True
            self._chunks.clear()
            self._queued = 0
            self._cv.notify_all()

    def _publish_levels(self, block, lag):
        step = max(1, int(self.samplerate * VIS_SLICE_SECONDS))
        for index, start in enumerate(range(0, len(block), step)):
            piece = block[start:start + step]
            ui.push_rms(
                _rms(piece), delay=lag + index * VIS_SLICE_SECONDS
            )

    def _run(self):
        try:
            self._play()
        except Exception as e:
            print("playback failed: {}".format(e))
            traceback.print_exc()
            try:
                close_stream()
            except Exception:
                pass

    def _play(self):
        stream = get_stream(self.samplerate)

        try:
            lag = float(stream.latency) + BLOCKSIZE / float(self.samplerate)
        except Exception:
            lag = BLOCKSIZE / float(self.samplerate)

        # Drain and play. No pacing beyond "write as soon as it arrives";
        # a large feed() is split into WRITE_FRAMES writes so PortAudio is
        # never handed more than it likes in one call, which is the only
        # buffering shape left.
        while not self._aborted:
            with self._cv:
                while (self._queued == 0
                       and not self._done and not self._aborted):
                    self._cv.wait(timeout=0.05)
                if self._aborted or (self._queued == 0 and self._done):
                    break

                taken, out = 0, []
                while self._chunks and taken < WRITE_FRAMES:
                    chunk = self._chunks[0]
                    room = WRITE_FRAMES - taken
                    if len(chunk) <= room:
                        self._chunks.popleft()
                    else:
                        chunk, self._chunks[0] = chunk[:room], chunk[room:]
                    out.append(chunk)
                    taken += len(chunk)
                self._queued -= taken

            block = out[0] if len(out) == 1 else np.concatenate(out)

            if not self._first_audio_fired:
                self._first_audio_fired = True
                if self._on_first_audio:
                    try:
                        self._on_first_audio()
                    except Exception:
                        pass

            self._publish_levels(block, lag)
            stream.write(block)


if __name__ == "__main__":
    # python -m listener.audio_out -- play 1s of 440Hz through the pipeline.
    import math
    rate = 24000
    n = rate
    tone = (np.sin(2 * math.pi * 440 * np.arange(n) / rate) * 32767
            * 0.3).astype(np.int16)
    player = StreamingPlayer(samplerate=rate)
    player.start()
    for i in range(0, n, WRITE_FRAMES):
        player.feed(tone[i:i + WRITE_FRAMES].tobytes())
    player.finish()
    print("played 440Hz tone")