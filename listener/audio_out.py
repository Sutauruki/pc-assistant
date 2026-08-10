"""Persistent audio output, jitter buffer, and pitch-preserving time-stretch.

Goes in: listener/audio_out.py

One OutputStream stays open for the life of the process instead of being
opened and torn down per reply (~50-200ms of WASAPI setup each time on
Windows). Chunks are fed in as they arrive off the network, with a short
prebuffer to absorb jitter.
"""

import queue
import threading

import numpy as np
import sounddevice as sd

# Used to pre-warm the device at startup.
DEFAULT_SAMPLE_RATE = 24000
CHANNELS = 1
BLOCKSIZE = 1024

# Too low and a slow chunk causes an audible gap; too high and we hand back
# the latency we just saved.
PREBUFFER_SECONDS = 0.15

_stream = None
_stream_rate = None
_stream_lock = threading.Lock()


def get_stream(samplerate=DEFAULT_SAMPLE_RATE):
    """Opens the shared output stream, reusing it across calls. Reopens only
    if the requested sample rate differs from the open one."""
    global _stream, _stream_rate
    with _stream_lock:
        if _stream is not None and _stream_rate != samplerate:
            print("output stream: {}Hz -> {}Hz (reopening)".format(
                _stream_rate, samplerate
            ))
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
    """Speeds up speech without raising its pitch, incrementally.

    Orpheus ignores the API's `speed` parameter, and simply playing samples
    out faster would pitch-shift the voice into a chipmunk. WSOLA instead
    drops overlapping windows of audio and cross-fades the seams, choosing
    each seam where the waveforms line up best -- so periodicity survives and
    pitch stays put.

    Works chunk-by-chunk so it can sit in the streaming path: samples are
    stretched as they arrive rather than after the clip is complete.
    """

    def __init__(self, speed, samplerate, frame=None, tolerance=None):
        self.speed = float(speed)

        # ~43ms window: long enough to contain a pitch period at speaking
        # fundamentals, short enough that seams stay inaudible.
        self.frame = int(frame or (samplerate * 0.043)) // 2 * 2
        self.hop_s = self.frame // 2  # hanning at 50% overlap sums to unity
        self.hop_a = max(1, int(round(self.hop_s * self.speed)))
        self.tolerance = int(tolerance or self.frame // 4)

        self.window = np.hanning(self.frame).astype(np.float32)
        self._buf = np.zeros(0, dtype=np.float32)
        self._base = 0        # absolute sample index of _buf[0]
        self._nominal = 0     # absolute index where the next window ideally starts
        self._template = None # what should naturally follow the last window
        self._olap = np.zeros(self.hop_s, dtype=np.float32)

    def _best_start(self, nominal):
        """Searches near `nominal` for the window that continues the previous
        one most smoothly. This alignment is the whole trick -- a fixed hop
        would cut mid-period and buzz."""
        if self._template is None:
            return nominal

        lo = max(self._base, nominal - self.tolerance)
        hi = nominal + self.tolerance
        best, best_score = nominal, -np.inf

        # step 4 rather than 1: near-identical alignment at a quarter the cost
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
        """int16 in, int16 out. Returns an empty array while filling."""
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
            self._nominal = start + self.hop_a

            # discard what the search window can no longer reach
            keep = (self._nominal - self.tolerance) - self._base
            if keep > self.frame * 4:
                self._buf = self._buf[keep:]
                self._base += keep

        if not out:
            return np.zeros(0, dtype=np.int16)
        return np.clip(np.concatenate(out), -32768, 32767).astype(np.int16)

    def flush(self):
        """Emits the final overlap tail so the last word isn't clipped."""
        tail = self._olap
        self._olap = np.zeros(self.hop_s, dtype=np.float32)
        return np.clip(tail, -32768, 32767).astype(np.int16)


class StreamingPlayer:
    """Accepts raw PCM bytes as they arrive and plays them in order.

        player = StreamingPlayer(samplerate=24000, speed=1.25)
        player.start()
        for chunk in network_stream:
            player.feed(chunk)
        player.finish()   # blocks until queued audio has played out
    """

    def __init__(self, samplerate=DEFAULT_SAMPLE_RATE, speed=1.0,
                 on_first_audio=None):
        self.samplerate = samplerate
        self._prebuffer_samples = int(samplerate * PREBUFFER_SECONDS)
        self._stretcher = (
            TimeStretcher(speed, samplerate) if abs(speed - 1.0) > 0.01 else None
        )
        self._q = queue.Queue()
        self._thread = None
        self._leftover = b""
        self._buffered = []
        self._buffered_samples = 0
        self._playing = False
        self._on_first_audio = on_first_audio
        self._stop = threading.Event()

    def start(self):
        get_stream(self.samplerate)  # open up front, not on the first chunk
        self._thread = threading.Thread(target=self._drain, daemon=True)
        self._thread.start()

    def feed(self, pcm_bytes):
        """Push raw little-endian int16 PCM. Chunk boundaries don't have to
        land on sample boundaries -- the odd byte carries over."""
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
            samples = self._stretcher.feed(samples)

        self._enqueue(samples)

    def _enqueue(self, samples):
        if len(samples) == 0:
            return

        if self._playing:
            self._q.put(samples)
            return

        self._buffered.append(samples)
        self._buffered_samples += len(samples)
        if self._buffered_samples >= self._prebuffer_samples:
            self._flush_prebuffer()

    def _flush_prebuffer(self):
        self._playing = True
        for block in self._buffered:
            self._q.put(block)
        self._buffered = []
        self._buffered_samples = 0

    def finish(self, timeout=30.0):
        """Signals end of stream, waits for playback to drain."""
        if self._stretcher is not None:
            self._enqueue(self._stretcher.flush())
        if not self._playing:
            # clip was shorter than the prebuffer -- play what we have
            self._flush_prebuffer()
        self._q.put(None)
        if self._thread is not None:
            self._thread.join(timeout=timeout)

    def abort(self):
        """Drops anything still queued. Hook for barge-in."""
        self._stop.set()
        try:
            while True:
                self._q.get_nowait()
        except queue.Empty:
            pass
        self._q.put(None)

    def _drain(self):
        stream = get_stream(self.samplerate)
        first = True
        while not self._stop.is_set():
            block = self._q.get()
            if block is None:
                break
            if first:
                first = False
                if self._on_first_audio is not None:
                    try:
                        self._on_first_audio()
                    except Exception:
                        pass
            stream.write(block.reshape(-1, CHANNELS))


def play_blocking(samples, samplerate=DEFAULT_SAMPLE_RATE):
    """Plays an already-complete int16 array on the shared stream."""
    if samples is None or len(samples) == 0:
        return
    stream = get_stream(samplerate)
    stream.write(np.asarray(samples, dtype=np.int16).reshape(-1, CHANNELS))