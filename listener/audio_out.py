"""Persistent audio output, jitter buffer, and pitch-preserving time-stretch.

Goes in: listener/audio_out.py   (replaces existing)

One OutputStream stays open for the life of the process instead of being
opened and torn down per reply (~50-200ms of WASAPI setup each time on
Windows). Chunks are fed in as they arrive off the network, with a prebuffer
sized to the connection rather than a fixed guess.

WHY THE PREBUFFER IS SIZED AND NOT FIXED
----------------------------------------
Playback runs at SPEECH_SPEED (1.25x) because Orpheus ignores the API's speed
parameter, so the samples are stretched locally. That means the player drains
audio 25% faster than realtime while the network delivers it at whatever rate
Groq generates -- roughly realtime. A fixed 0.15s prebuffer against a 0.25
seconds-per-second deficit runs dry after 0.6s of speech, every single reply,
and you hear the sentence in pieces with words missing at the seams.

The shortfall over a whole reply is P >= T * (1 - r), where T is how long the
reply takes to play and r is the delivery rate in played-seconds per wall
second. Both are measurable: T from the character count the caller passes in,
r from watching the first few chunks land. On a connection that outruns
playback (r >= 1) this collapses to PREBUFFER_MIN_SECONDS and costs nothing.
When the estimate is wrong anyway, _take() catches the dry queue, holds a
longer buffer, and remembers it for the rest of the session.
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

# Device buffer to ask PortAudio for. A cushion at the hardware level covers
# what Python can't be relied on for: a GC pause, the GIL, a slow console
# repaint. Costs this much added time-to-first-sample, so it stays modest.
OUTPUT_LATENCY = 0.12

# Frames per stream.write(). Was one write per ~800-sample visual slice, which
# is *smaller* than BLOCKSIZE -- every write was a partial block, and the
# device couldn't be handed a full one until two writes had landed. Keep this a
# multiple of BLOCKSIZE.
#
# Two blocks (~85ms) rather than more: the queue drops by a full write the
# instant one is issued but only refills at the delivery rate, so a large
# write makes the buffer sawtooth hard enough to dip under LOW_WATER_FRAMES on
# its own and trip the growth path for no reason.
WRITE_FRAMES = BLOCKSIZE * 2

# Waveform level granularity. Independent of WRITE_FRAMES now: levels for the
# whole write are published up front with staggered delays, so the visual rate
# no longer dictates how the audio is chopped up.
VIS_SLICE_SECONDS = 1.0 / 30.0

# Prebuffer bounds. The floor is what a fast connection pays; the ceiling is
# the most latency we're willing to trade for smooth playback.
#
# The floor has to clear the opening transient with room to spare. The first
# writes go out while PortAudio's own buffer is still empty, so it absorbs
# OUTPUT_LATENCY worth of them instantly and returns long before that audio
# has played -- the queue takes the full hit of those writes against almost no
# wall time to refill in. It has to survive that dip without falling under
# LOW_WATER_FRAMES, or the growth path fires on the first word of every reply.
PREBUFFER_MIN_SECONDS = 0.35
PREBUFFER_MAX_SECONDS = 2.50

# How long to watch chunks arrive before trusting the rate estimate.
RATE_SAMPLE_SECONDS = 0.35

# Pessimism applied to the measured rate. Delivery is bursty over 0.35s and
# under-buffering is audible while over-buffering is not.
RATE_MARGIN = 1.15

# Reactive net, for when the estimate is wrong: where growth starts, how fast
# it grows. Remembered across replies so a session stutters at most once.
#
# It relaxes again on replies that play clean, or one bad moment on the
# network would cost every later reply the full ceiling in added latency.
# Growth is fast and decay is slow on purpose -- the cost of being slightly
# over is latency you can measure, the cost of being under is speech you can
# hear breaking up.
REBUFFER_FLOOR_SECONDS = 0.60
REBUFFER_GROWTH = 2.0
REBUFFER_DECAY = 0.9

# A reply has to be at least this long before playing cleanly counts as
# evidence the buffer is oversized. A one-second confirmation proves nothing
# about a two-second buffer.
CLEAN_EVIDENCE_SECONDS = 1.5

# Below this the queue counts as dry. Zero is too late and too narrow -- a
# queue hovering at 200 samples never hits zero but still can't fill a block.
LOW_WATER_FRAMES = BLOCKSIZE

# How long a dry queue has to stay dry before it counts. Without this, the
# ordinary end of a healthy reply -- last chunk delivered, finish() a
# millisecond behind it -- reads as an underrun, and the session's prebuffer
# creeps to the ceiling one false alarm at a time. The device is holding
# OUTPUT_LATENCY of audio, so waiting this out is inaudible.
DRY_GRACE_SECONDS = 0.05

_stream = None
_stream_rate = None
_stream_lock = threading.Lock()

_learned = 0.0
_learned_lock = threading.Lock()


def learned_prebuffer():
    with _learned_lock:
        return _learned


def _grow_learned():
    """Raises the session's prebuffer after a dry queue. Returns the new one."""
    global _learned
    with _learned_lock:
        _learned = min(
            PREBUFFER_MAX_SECONDS,
            max(REBUFFER_FLOOR_SECONDS, _learned * REBUFFER_GROWTH),
        )
        return _learned


def _relax_learned():
    """Walks it back down after a reply that played without a gap."""
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
        self._last = None     # absolute index the last window actually started at
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
            self._last = start
            self._nominal = start + self.hop_a

            # Discard what the search window can no longer reach -- but never
            # past the point flush() will need to resume from, or the tail
            # would replay audio that has already gone out. The two bounds
            # only cross above speed 1.5 (hop_a > hop_s + tolerance), which is
            # inside the range the docstring invites you to try.
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
        """Emits everything the loop couldn't reach, unstretched.

        The loop needs tolerance + frame + hop_s of lookahead before it will
        place a window, so when the stream ends there is always up to ~75ms
        of input left over that it never got to. This used to return only
        _olap -- the fading half of the last window, ~21ms -- and drop the
        rest on the floor, which clipped the final consonant off every reply.
        A clip shorter than one window fared worse still: feed() never
        produced anything, so the whole thing came out as silence.

        The remainder is emitted raw rather than windowed. It starts exactly
        where the last window's first half ended, so it's a continuous
        signal -- no seam, no click -- and it replaces _olap rather than
        following it, since _olap is that same audio already fading out with
        nothing left to cross-fade into. The last ~75ms plays at 1x instead
        of 1.25x, which is not something you can hear at the end of a
        sentence; a missing consonant is.
        """
        if self._last is None:
            rest = self._buf              # never placed a window at all
        else:
            rest = self._buf[(self._last + self.hop_s) - self._base:]

        # Full reset, not just the buffer: _base and _nominal are absolute
        # indices into a stream that has now ended, and leaving them set would
        # put anything fed afterwards at the wrong offset. Nothing reuses a
        # stretcher today -- one per reply -- but it shouldn't be a trap.
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
    """Accepts raw PCM bytes as they arrive and plays them in order.

        player = StreamingPlayer(samplerate=24000, speed=1.25,
                                 expected_seconds=len(text) / 14.5)
        player.start()
        for chunk in network_stream:
            player.feed(chunk)
        player.finish()   # blocks until queued audio has played out

    `expected_seconds` is a rough guess at how much *source* audio is coming,
    used only to size the prebuffer. Wrong is fine; absent is fine too, it
    just falls back to the reactive path.
    """

    def __init__(self, samplerate=DEFAULT_SAMPLE_RATE, speed=1.0,
                 on_first_audio=None, expected_seconds=None):
        self.samplerate = samplerate
        self.speed = max(0.01, float(speed))
        self.expected_seconds = expected_seconds
        self._stretcher = (
            TimeStretcher(speed, samplerate) if abs(speed - 1.0) > 0.01 else None
        )
        self._on_first_audio = on_first_audio

        # One condition guards the whole handoff: the chunk deque, how many
        # samples are in it, and the two flags that end the wait. A plain
        # Queue can't answer "how many seconds are buffered", which is the
        # question this class now exists to ask.
        self._cv = threading.Condition()
        self._chunks = collections.deque()
        self._queued = 0
        self._done = False
        self._aborted = False

        # Delivery rate bookkeeping, read under _cv.
        self._t0 = None
        self._produced = 0

        self._leftover = b""
        self._thread = None
        self._warned_dry = False

    # --- producer side -----------------------------------------------------

    def start(self):
        get_stream(self.samplerate)  # open up front, not on the first chunk
        self._thread = threading.Thread(target=self._run, daemon=True)
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
        """Signals end of stream, waits for playback to drain."""
        if self._stretcher is not None:
            self._enqueue(self._stretcher.flush())
        with self._cv:
            self._done = True
            self._cv.notify_all()
        if self._thread is not None:
            self._thread.join(timeout=timeout)

    def abort(self):
        """Drops anything still queued. Hook for barge-in."""
        with self._cv:
            self._aborted = True
            self._done = True
            self._chunks.clear()
            self._queued = 0
            self._cv.notify_all()

    # --- consumer side -----------------------------------------------------

    def _required_prebuffer(self, floor):
        """Seconds to hold before starting, from the measured delivery rate.

        Returns `floor` until there's enough evidence to do better.
        """
        if not self.expected_seconds:
            return floor

        with self._cv:
            t0, produced = self._t0, self._produced

        if t0 is None:
            return floor
        elapsed = time.monotonic() - t0
        if elapsed < RATE_SAMPLE_SECONDS:
            return floor

        # Played-seconds delivered per wall second. Above 1 and the network is
        # outrunning playback, so there's nothing to hold back for.
        rate = (produced / float(self.samplerate)) / elapsed
        rate /= RATE_MARGIN
        if rate >= 1.0:
            return floor

        playtime = self.expected_seconds / self.speed

        # The shortfall over the reply, plus the device buffer: PortAudio
        # takes OUTPUT_LATENCY out of the queue in the first few writes,
        # before any of it has played.
        need = playtime * (1.0 - rate) + OUTPUT_LATENCY
        return max(floor, min(PREBUFFER_MAX_SECONDS, need))

    def _await_start(self):
        """Holds the first sample back until there's enough buffered to get
        through the reply without running dry.

        Polled rather than event-driven on purpose: the target moves as the
        rate estimate firms up, and a 10ms poll for the fraction of a second
        this runs is not worth the extra condition-variable bookkeeping.
        """
        floor = max(PREBUFFER_MIN_SECONDS, learned_prebuffer())
        target = floor

        while True:
            with self._cv:
                queued, done, aborted = self._queued, self._done, self._aborted
            if aborted or done:
                return
            target = self._required_prebuffer(floor)
            if queued >= int(self.samplerate * target):
                return
            time.sleep(0.01)

    def _dry_locked(self):
        """Too little buffered to keep the device fed, with more still to
        come. Caller must hold _cv."""
        return (self._queued < LOW_WATER_FRAMES
                and not self._done and not self._aborted)

    def _take(self, want):
        """Pops up to `want` samples. None means end of stream.

        Running dry mid-reply is the bug this class was rewritten for, so it
        isn't treated as an ordinary wait. We stop writing, raise the
        prebuffer for the rest of the session, and refill to the new figure
        before carrying on. One clean pause beats a dozen stutters, and the
        next reply starts with a buffer that already knows better.
        """
        warn = None

        with self._cv:
            if self._dry_locked():
                # Give it a grace period first. The normal end of a healthy
                # reply looks identical from here -- last chunk delivered,
                # finish() a millisecond behind it -- and counting that as an
                # underrun would raise the prebuffer on every single reply.
                deadline = time.monotonic() + DRY_GRACE_SECONDS
                while self._dry_locked():
                    left = deadline - time.monotonic()
                    if left <= 0:
                        break
                    self._cv.wait(timeout=left)

            if self._dry_locked():
                refill = _grow_learned()
                if not self._warned_dry:
                    self._warned_dry = True
                    warn = refill
                need = int(self.samplerate * refill)
                while (self._queued < need
                        and not self._done and not self._aborted):
                    self._cv.wait(timeout=0.25)

            while (self._queued == 0
                    and not self._done and not self._aborted):
                self._cv.wait(timeout=0.25)

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

        # Outside the lock, deliberately. print() goes through the dashboard's
        # stdout proxy, which takes the render lock and repaints the panel --
        # a slow console write. Holding _cv across that would block the
        # network thread's next feed() at the exact moment the buffer is
        # already starving, which is the stall this class exists to avoid.
        if warn is not None:
            print("speech: buffer ran dry, now holding {:.2f}s".format(warn))

        return block

    def _publish_levels(self, block, lag):
        """Waveform levels for one write, staggered across its duration.

        Published before the write, not after: stream.write() blocks until the
        device has room, so a level pushed afterwards is already late. `lag`
        covers what the device is still holding; the stagger covers position
        within the block, since the tail of a 170ms write is heard 170ms after
        its head.
        """
        step = max(1, int(self.samplerate * VIS_SLICE_SECONDS))
        for index, start in enumerate(range(0, len(block), step)):
            piece = block[start:start + step]
            ui.push_rms(
                _rms(piece), delay=lag + index * VIS_SLICE_SECONDS
            )

    def _run(self):
        # A daemon thread that raises is silent by construction, and this one
        # holds the only path to the speakers.
        try:
            self._play()
        except Exception as e:
            print("playback failed: {}".format(e))
            traceback.print_exc()
            # Drop the shared stream. get_stream() only reopens on a sample
            # rate change, so a device that died mid-reply -- headset
            # unplugged, driver reset -- would otherwise stay dead for the
            # life of the process and take every later reply with it.
            try:
                close_stream()
            except Exception:
                pass

    def _play(self):
        stream = get_stream(self.samplerate)

        # How far ahead of the ear this thread is running. sounddevice reports
        # the device's own latency; the rest is whatever PortAudio has already
        # accepted but not played. Without it the bars visibly lead the voice.
        try:
            lag = float(stream.latency) + BLOCKSIZE / float(self.samplerate)
        except Exception:
            lag = BLOCKSIZE / float(self.samplerate)

        self._await_start()

        first = True
        written = 0
        while not self._aborted:
            block = self._take(WRITE_FRAMES)
            if block is None:
                break

            if first:
                first = False
                ui.set_state(ui.SPEAKING)
                if self._on_first_audio is not None:
                    try:
                        self._on_first_audio()
                    except Exception:
                        pass

            self._publish_levels(block, lag)
            stream.write(block.reshape(-1, CHANNELS))
            written += len(block)

        if self._aborted or first:
            return

        # PortAudio is still holding the last blocks. finish() claiming
        # "played out" while the closing word is unheard is how a deferred
        # set_volume ends up jumping the end of the sentence -- and how
        # the timing line reports a reply as spoken before it is.
        time.sleep(min(lag, 0.5))

        # A reply of real length that never ran dry is evidence the buffer is
        # bigger than it needs to be, so give some of it back. Without this,
        # one bad thirty seconds on the network would cost every reply for the
        # rest of the session up to 2.5s of added silence before the first
        # word -- and past the point where the whole reply fits inside the
        # buffer, streaming has quietly stopped happening at all.
        played = written / float(self.samplerate)
        if not self._warned_dry and played >= CLEAN_EVIDENCE_SECONDS:
            _relax_learned()


def play_blocking(samples, samplerate=DEFAULT_SAMPLE_RATE):
    """Plays an already-complete int16 array on the shared stream."""
    if samples is None or len(samples) == 0:
        return
    stream = get_stream(samplerate)
    stream.write(np.asarray(samples, dtype=np.int16).reshape(-1, CHANNELS))