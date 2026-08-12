"""Persistent audio output, jitter buffer, and pitch-preserving time-stretch.

Goes in: listener/audio_out.py   (replaces existing)

Optimized for Kokoro-ONNX local stream synthesis.
"""

import collections
import threading
import time
import traceback

import numpy as np
import sounddevice as sd

from listener import ui

# Used to pre-warm the device at startup.
DEFAULT_SAMPLE_RATE = 24000
CHANNELS = 1
BLOCKSIZE = 1024

# Device buffer to ask PortAudio for.
OUTPUT_LATENCY = 0.12

# Frames per stream.write().
WRITE_FRAMES = BLOCKSIZE * 2

# Waveform level granularity.
VIS_SLICE_SECONDS = 1.0 / 30.0

# --- Prebuffer Sizing (Tweaked for Local ONNX Inference) ---
# Lower minimum prebuffer allows Kokoro to speak instantly without long opening lags.
PREBUFFER_MIN_SECONDS = 0.10
PREBUFFER_MAX_SECONDS = 0.80

RATE_SAMPLE_SECONDS = 0.20
RATE_MARGIN = 1.05
JITTER_MARGIN = 0.08
DEFICIT_SHARE = 0.30

# --- Speed Governor Settings ---
MIN_SPEED = 0.95
SPEED_EASE_LOW = 0.15
SPEED_EASE_HIGH = 0.40
SPEED_EASE_RATE = 0.18

# --- Reactive Net (Tuned for Punctuation Boundaries) ---
REBUFFER_FLOOR_SECONDS = 0.20
REBUFFER_GROWTH = 1.5
REBUFFER_DECAY = 0.95
REBUFFER_MIN_PAUSE = 0.08
CLEAN_EVIDENCE_SECONDS = 1.5

LOW_WATER_FRAMES = BLOCKSIZE

# Grace period increased to absorb natural punctuation pauses in Kokoro TTS
DRY_GRACE_SECONDS = 0.12

_stream = None
_stream_rate = None
_stream_lock = threading.Lock()

_learned = 0.0
_learned_lock = threading.Lock()


def learned_prebuffer():
    with _learned_lock:
        return _learned


def _grow_learned():
    global _learned
    with _learned_lock:
        _learned = min(
            PREBUFFER_MAX_SECONDS,
            max(REBUFFER_FLOOR_SECONDS, _learned * REBUFFER_GROWTH),
        )
        return _learned


def _relax_learned():
    global _learned
    with _learned_lock:
        _learned *= REBUFFER_DECAY
        if _learned < PREBUFFER_MIN_SECONDS:
            _learned = 0.0
        return _learned


def _rms(block):
    if len(block) == 0:
        return 0.0
    return float(np.sqrt(np.mean(block.astype(np.float32) ** 2)))


def get_stream(samplerate=DEFAULT_SAMPLE_RATE):
    global _stream, _stream_rate
    with _stream_lock:
        if _stream is not None and _stream_rate != samplerate:
            _stream.stop()
            _stream.close()
            _stream = None
            _stream_rate = None

        if _stream is None:
            _stream = sd.OutputStream(
                samplerate=samplerate,
                channels=CHANNELS,
                dtype="int16",
                blocksize=BLOCKSIZE,
                latency=OUTPUT_LATENCY,
            )
            _stream.start()
            _stream_rate = samplerate

        return _stream


def close_stream():
    global _stream, _stream_rate
    with _stream_lock:
        if _stream is not None:
            _stream.stop()
            _stream.close()
            _stream = None
            _stream_rate = None


class TimeStretcher:
    """WSOLA Time Stretcher. Bypassed during native 1.0x playback."""

    def __init__(self, speed, samplerate, frame=None, tolerance=None):
        self.speed = float(speed)
        self.frame = int(frame or (samplerate * 0.043)) // 2 * 2
        self.hop_s = self.frame // 2
        self.hop_a = max(1, int(round(self.hop_s * self.speed)))
        self.tolerance = int(tolerance or self.frame // 4)

        self.window = np.hanning(self.frame).astype(np.float32)
        self._buf = np.zeros(0, dtype=np.float32)
        self._base = 0
        self._nominal = 0
        self._template = None
        self._last = None
        self._olap = np.zeros(self.hop_s, dtype=np.float32)

    def set_speed(self, speed):
        speed = max(0.5, min(2.0, float(speed)))
        if abs(speed - self.speed) < 0.005:
            return
        self.speed = speed
        self.hop_a = max(1, int(round(self.hop_s * speed)))

    def _best_start(self, nominal):
        if self._template is None:
            return nominal

        lo = max(self._base, nominal - self.tolerance)
        hi = nominal + self.tolerance
        best, best_score = nominal, -np.inf

        for start in range(lo, hi + 1, 4):
            i = start - self._base
            seg = self._buf[i:i + self.frame]
            if len(seg) < self.frame:
                break
            score = float(np.dot(seg, self._template)) / (
                float(np.linalg.norm(seg)) + 1e-9
            )
            if score > best_score:
                best_score, best = score, start

        return best

    def feed(self, samples):
        if len(samples) == 0:
            return np.zeros(0, dtype=np.int16)

        self._buf = np.concatenate([self._buf, samples.astype(np.float32)])
        out = []

        while self._base + len(self._buf) >= (
            self._nominal + self.tolerance + self.frame + self.hop_s
        ):
            start = self._best_start(self._nominal)
            i = start - self._base
            windowed = self._buf[i:i + self.frame] * self.window

            out.append(windowed[:self.hop_s] + self._olap)
            self._olap = windowed[self.hop_s:].copy()

            follow = i + self.hop_s
            self._template = self._buf[follow:follow + self.frame].copy()
            self._last = start
            self._nominal = start + self.hop_a

            keep = min(
                (self._nominal - self.tolerance) - self._base,
                (self._last + self.hop_s) - self._base,
            )
            if keep > self.frame * 4:
                self._buf = self._buf[keep:]
                self._base += keep

        if not out:
            return np.zeros(0, dtype=np.int16)
        return np.clip(np.concatenate(out), -32768, 32767).astype(np.int16)

    def flush(self):
        if self._last is None:
            rest = self._buf
        else:
            rest = self._buf[(self._last + self.hop_s) - self._base:]

        self._olap = np.zeros(self.hop_s, dtype=np.float32)
        self._buf = np.zeros(0, dtype=np.float32)
        self._template = None
        self._last = None
        self._base = 0
        self._nominal = 0

        if len(rest) == 0:
            return np.zeros(0, dtype=np.int16)
        return np.clip(rest, -32768, 32767).astype(np.int16)


class StreamingPlayer:
    def __init__(self, samplerate=DEFAULT_SAMPLE_RATE, speed=1.0,
                 on_first_audio=None, expected_seconds=None):
        self.samplerate = samplerate
        self.speed = max(0.01, float(speed))
        self.expected_seconds = expected_seconds
        
        # Only instantiate TimeStretcher if speed is explicitly non-1.0
        self._stretcher = (
            TimeStretcher(speed, samplerate) if abs(speed - 1.0) > 0.01 else None
        )
        self._on_first_audio = on_first_audio

        self._cv = threading.Condition()
        self._chunks = collections.deque()
        self._queued = 0
        self._done = False
        self._aborted = False
        self._playing = False

        self._t0 = None
        self._produced = 0

        self._speed_now = self.speed
        self._speed_floor_seen = self.speed

        self._leftover = b""
        self._thread = None
        self._warned_dry = False

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

        if self._stretcher is not None:
            self._adapt_speed()
            samples = self._stretcher.feed(samples)

        self._enqueue(samples)

    def _adapt_speed(self):
        if self._stretcher is None:
            return

        with self._cv:
            queued = self._queued
            playing = self._playing
            aborted = self._aborted

        if aborted or not playing:
            return

        seconds = queued / float(self.samplerate)

        if seconds < SPEED_EASE_LOW:
            goal = MIN_SPEED
        elif seconds > SPEED_EASE_HIGH:
            goal = self.speed
        else:
            return

        current = self._speed_now
        updated = current + (goal - current) * SPEED_EASE_RATE

        if abs(goal - updated) < 0.01:
            updated = goal

        updated = max(MIN_SPEED, min(self.speed, updated))
        if abs(updated - current) < 0.005:
            return

        self._speed_now = updated
        if updated < self._speed_floor_seen:
            self._speed_floor_seen = updated
        self._stretcher.set_speed(updated)

    def _enqueue(self, samples):
        if len(samples) == 0:
            return
        with self._cv:
            if self._aborted:
                return
            if self._t0 is None:
                self._t0 = time.monotonic()
            self._chunks.append(samples)
            self._queued += len(samples)
            self._produced += len(samples)
            self._cv.notify_all()

    def finish(self, timeout=40.0):
        if self._stretcher is not None:
            self._enqueue(self._stretcher.flush())
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

    def _remaining_output_seconds(self):
        if not self.expected_seconds:
            return None
        with self._cv:
            produced = self._produced
        total = self.expected_seconds / self.speed
        return max(0.0, total - produced / float(self.samplerate))

    def _required_prebuffer(self, floor):
        if not self.expected_seconds:
            return floor

        with self._cv:
            t0, produced = self._t0, self._produced

        if t0 is None:
            return floor
        elapsed = time.monotonic() - t0
        if elapsed < RATE_SAMPLE_SECONDS:
            return floor

        rate = (produced / float(self.samplerate)) / elapsed
        rate /= RATE_MARGIN

        need = OUTPUT_LATENCY + JITTER_MARGIN
        if rate < 1.0:
            playtime = self.expected_seconds / self.speed
            need += playtime * (1.0 - rate) * DEFICIT_SHARE

        return max(floor, min(PREBUFFER_MAX_SECONDS, need))

    def _await_start(self):
        floor = max(PREBUFFER_MIN_SECONDS, learned_prebuffer())

        while True:
            with self._cv:
                queued, done, aborted = self._queued, self._done, self._aborted
            if aborted or done:
                return
            target = self._required_prebuffer(floor)
            if queued >= int(self.samplerate * target):
                return
            time.sleep(0.005)

    def _dry_locked(self):
        return (self._queued < LOW_WATER_FRAMES
                and not self._done and not self._aborted)

    def _take(self, want):
        warn = None

        with self._cv:
            if self._dry_locked():
                deadline = time.monotonic() + DRY_GRACE_SECONDS
                while self._dry_locked():
                    left = deadline - time.monotonic()
                    if left <= 0:
                        break
                    self._cv.wait(timeout=left)

            if self._dry_locked():
                refill = _grow_learned()
                left = self._remaining_output_seconds()
                if left is not None:
                    refill = min(refill, max(REBUFFER_MIN_PAUSE, left))

                if not self._warned_dry:
                    self._warned_dry = True
                    warn = refill

                need = int(self.samplerate * refill)
                while (self._queued < need
                        and not self._done and not self._aborted):
                    self._cv.wait(timeout=0.1)

            while (self._queued == 0
                    and not self._done and not self._aborted):
                self._cv.wait(timeout=0.1)

            if self._aborted or self._queued == 0:
                block = None
            else:
                out = []
                taken = 0
                while self._chunks and taken < want:
                    chunk = self._chunks[0]
                    room = want - taken
                    if len(chunk) <= room:
                        self._chunks.popleft()
                    else:
                        chunk, self._chunks[0] = chunk[:room], chunk[room:]
                    out.append(chunk)
                    taken += len(chunk)
                self._queued -= taken
                block = out[0] if len(out) == 1 else np.concatenate(out)

        if warn is not None:
            print("speech: buffer ran dry, holding {:.2f}s".format(warn))

        return block

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

        self._await_start()

        with self._cv:
            self._playing = True

        first = True
        while not self._aborted:
            block = self._take(WRITE_FRAMES)
            if block is None:
                break

            if first:
                first = False
                if self._on_first_audio:
                    try:
                        self._on_first_audio()
                    except Exception:
                        pass

            self._publish_levels(block, lag)
            stream.write(block)

        _relax_learned()