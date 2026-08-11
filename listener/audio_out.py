"""Persistent audio output, jitter buffer, and pitch-preserving time-stretch.

Goes in: listener/audio_out.py   (replaces existing)

One OutputStream stays open for the life of the process instead of being
opened and torn down per reply (~50-200ms of WASAPI setup each time on
Windows). Chunks are fed in as they arrive off the network.

WHAT CHANGED IN THIS ROUND, AND WHY YOU COULD STILL HEAR A GAP
--------------------------------------------------------------
The previous version got the direction right -- size the prebuffer to the
connection instead of guessing -- but it had three weaknesses that together
produce exactly the symptom "it chops sometimes":

1. The delivery rate was measured once, over the first 0.35s, and never
   revised. That window is the least representative one in the whole reply:
   the server has usually buffered a head start and TCP hasn't settled, so
   the first burst *overstates* the sustained rate. If the estimate came out
   at >= 1.0 the player held only the floor, and then the real rate showed up
   halfway through the sentence.

2. When the estimate was wrong, recovery was a flat REBUFFER_FLOOR_SECONDS
   stop -- 0.6s of dead air mid-sentence. The old comment called it "one
   clean pause beats a dozen stutters". A 600ms hole in a 4-second sentence
   is not clean; it is the chop you are hearing.

3. It could only ever pay for the deficit with *latency* up front. Playing at
   1.25x while audio arrives at ~1x means the queue drains 0.25 played-seconds
   per wall second no matter how big the head start is -- a head start only
   buys time, it never stops the bleeding.

The fix for all three is the same idea: **the playback rate is a control
knob, not a constant.** WSOLA already resamples time without touching pitch,
and its hop size can be changed between windows with no seam. So when the
queue gets thin, the player eases the stretch factor down toward 1.0 -- which
*reduces the drain rate*, refills the buffer from the same incoming bytes, and
is heard as a slight change of pace rather than as a hole. When the queue is
healthy again it eases back up to the speed you asked for.

That turns the sustained deficit into a closed loop, so the prebuffer only has
to cover jitter rather than the whole reply. PREBUFFER_MAX_SECONDS came down
from 2.50 to 1.20 and DEFICIT_SHARE pre-funds only part of the shortfall,
which also means the first word arrives noticeably sooner than it used to.

The reactive stop is still here as a last resort, but it is now capped by how
much audio is actually still coming -- pausing 600ms to refill when only 200ms
of speech remains was never sensible.
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
# what Python cannot be relied on for: a GC pause, the GIL, a slow console
# repaint. Raised from 0.12 -- on an older machine the 24fps dashboard repaint
# is a synchronous console write holding the GIL, and 120ms was not enough
# headroom to ride one out. Costs this much added time-to-first-sample.
OUTPUT_LATENCY = 0.18

# Frames per stream.write(). Keep this a multiple of BLOCKSIZE so the device
# is never handed a partial block.
#
# Two blocks (~85ms) rather than more: the queue drops by a full write the
# instant one is issued but only refills at the delivery rate, so a large
# write makes the buffer sawtooth hard enough to dip under LOW_WATER_FRAMES on
# its own and trip the recovery path for no reason.
WRITE_FRAMES = BLOCKSIZE * 2

# Waveform level granularity. Independent of WRITE_FRAMES: levels for the
# whole write are published up front with staggered delays, so the visual rate
# does not dictate how the audio is chopped up.
VIS_SLICE_SECONDS = 1.0 / 30.0

# --- Prebuffer sizing ------------------------------------------------------
#
# The floor has to clear the opening transient with room to spare. The first
# writes go out while PortAudio's own buffer is still empty, so it absorbs
# OUTPUT_LATENCY worth of them instantly and returns long before that audio
# has played -- the queue takes the full hit of those writes against almost no
# wall time to refill in. It has to survive that dip without falling under
# LOW_WATER_FRAMES, or the recovery path fires on the first word of every
# reply.
PREBUFFER_MIN_SECONDS = 0.30

# Ceiling. Much lower than it used to be: the speed governor below now handles
# the sustained deficit, so holding a second and a half of speech back before
# the first word buys nothing except a slower-feeling assistant.
PREBUFFER_MAX_SECONDS = 1.20

# How long to watch chunks arrive before trusting the rate estimate.
RATE_SAMPLE_SECONDS = 0.35

# Pessimism applied to the measured rate. Delivery is bursty over 0.35s and
# under-buffering is audible while over-buffering is not.
RATE_MARGIN = 1.15

# Slack over and above the device buffer, to absorb one bad scheduling moment.
JITTER_MARGIN = 0.15

# How much of the projected shortfall to pre-fund with latency. The rest is
# absorbed by the speed governor as the reply plays. 1.0 would be the old
# behaviour (pay for all of it up front); 0.0 would lean entirely on the
# governor and start fast but audibly slow down on every long reply.
DEFICIT_SHARE = 0.40

# --- Speed governor --------------------------------------------------------
#
# The stretch factor tts asks for is a *target*, not a promise. When the queue
# runs thin the player eases toward MIN_SPEED, which cuts the drain rate and
# lets the same incoming bytes refill the buffer.
#
# MIN_SPEED sits a hair under 1.0 on purpose. If Groq is generating slightly
# slower than realtime, a floor of exactly 1.0 can only break even and never
# recover; 0.95 lets the buffer actually gain. A 5% rate change with no pitch
# shift is not something you can hear in isolation, and it is reached
# gradually.
MIN_SPEED = 0.95

# Queue depth, in played seconds, at which to start easing down / easing back
# up. The gap between them is a dead band -- without it the governor hunts
# around a single threshold and the pace wobbles audibly.
SPEED_EASE_LOW = 0.22
SPEED_EASE_HIGH = 0.55

# Fraction of the remaining distance to close per feed() call. Feeds land
# every ~40-80ms, so this reaches the goal in roughly a fifth of a second --
# fast enough to matter, slow enough to read as prosody.
SPEED_EASE_RATE = 0.18

# --- Reactive net ----------------------------------------------------------
#
# For when the estimate is wrong anyway. Remembered across replies so a
# session stutters at most once, and relaxed again on replies that play clean
# so one bad moment on the network does not tax every later reply.
#
# The floor is much lower than it was (0.60 -> 0.35) because it no longer has
# to carry the deficit on its own.
REBUFFER_FLOOR_SECONDS = 0.35
REBUFFER_GROWTH = 2.0
REBUFFER_DECAY = 0.9

# Hard floor on a mid-reply refill pause, and the cap that keeps it honest:
# never wait longer than there is audio still to come.
REBUFFER_MIN_PAUSE = 0.12

# A reply has to be at least this long before playing cleanly counts as
# evidence the buffer is oversized. A one-second confirmation proves nothing
# about a one-second buffer.
CLEAN_EVIDENCE_SECONDS = 1.5

# Below this the queue counts as dry. Zero is too late and too narrow -- a
# queue hovering at 200 samples never hits zero but still cannot fill a block.
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
    stretched as they arrive rather than after the clip is complete. The
    stretch factor can be changed while it runs -- see set_speed.
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

    def set_speed(self, speed):
        """Changes the stretch factor mid-stream.

        Only `hop_a` -- how far the *input* read head advances per output
        window -- depends on the speed. The window length, the overlap, the
        hanning envelope and the half-window currently held in `_olap` are all
        untouched, so the next window cross-fades into the last one exactly as
        it would have anyway. There is no seam and no discontinuity; the
        listener hears the pace change, nothing else.

        Only ever called with a value at or below the constructor's, which
        keeps the `keep` bound in feed() on the safe side of its one crossing
        condition (hop_a > hop_s + tolerance, i.e. speed above ~1.5).
        """
        speed = max(0.5, min(2.0, float(speed)))
        if abs(speed - self.speed) < 0.005:
            return
        self.speed = speed
        self.hop_a = max(1, int(round(self.hop_s * speed)))

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
            # would replay audio that has already gone out.
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
        of input left over that it never got to. Returning only _olap -- the
        fading half of the last window, ~21ms -- drops the rest on the floor
        and clips the final consonant off every reply. A clip shorter than one
        window fares worse still: feed() never produces anything, so the whole
        thing comes out as silence.

        The remainder is emitted raw rather than windowed. It starts exactly
        where the last window's first half ended, so it is a continuous
        signal -- no seam, no click -- and it replaces _olap rather than
        following it, since _olap is that same audio already fading out with
        nothing left to cross-fade into. The last ~75ms plays at 1x, which is
        not something you can hear at the end of a sentence; a missing
        consonant is.
        """
        if self._last is None:
            rest = self._buf              # never placed a window at all
        else:
            rest = self._buf[(self._last + self.hop_s) - self._base:]

        # Full reset, not just the buffer: _base and _nominal are absolute
        # indices into a stream that has now ended, and leaving them set would
        # put anything fed afterwards at the wrong offset.
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

        player = StreamingPlayer(samplerate=24000, speed=1.15,
                                 expected_seconds=len(text) / 13.0)
        player.start()
        for chunk in network_stream:
            player.feed(chunk)
        player.finish()   # blocks until queued audio has played out

    `expected_seconds` is a rough guess at how much *source* audio is coming,
    used only to size the prebuffer and to cap recovery pauses. Wrong is fine;
    absent is fine too, it just falls back to the reactive path.

    `speed` is a target. The player will ease below it -- never above -- when
    the buffer cannot sustain it. See the speed governor notes at the top.
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
        # samples are in it, and the flags that end the wait. A plain Queue
        # cannot answer "how many seconds are buffered", which is the question
        # this class exists to ask.
        self._cv = threading.Condition()
        self._chunks = collections.deque()
        self._queued = 0
        self._done = False
        self._aborted = False
        self._playing = False

        # Delivery rate bookkeeping, read under _cv.
        self._t0 = None
        self._produced = 0

        # Speed governor state. Only the producer thread writes these.
        self._speed_now = self.speed
        self._speed_floor_seen = self.speed

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
            # Adjust before stretching, so this chunk is already produced at
            # whatever rate the buffer can currently afford.
            self._adapt_speed()
            samples = self._stretcher.feed(samples)

        self._enqueue(samples)

    def _adapt_speed(self):
        """Eases the stretch factor toward whatever the buffer can sustain.

        Runs on the network thread, which is the only thread that stretches,
        so no lock is needed on the governor's own state. The queue depth is
        read under _cv and released immediately -- nothing slow happens while
        it is held.

        Deliberately inert until playback has actually started: while the
        prebuffer is filling the queue is legitimately small and nothing is
        draining it, so reacting to that depth would slow every reply down
        before its first word for no reason.
        """
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
            return  # dead band -- leave it alone rather than hunt

        current = self._speed_now
        updated = current + (goal - current) * SPEED_EASE_RATE

        # Snap the last sliver rather than approach it asymptotically forever.
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

    def _remaining_output_seconds(self):
        """How much audio is still expected to arrive, in played seconds.

        None when there is no estimate to work from. Approximate by
        construction -- if the governor has eased the speed down, the real
        total runs a little longer than this and the figure reads low. That
        bias is the safe one: it only ever shortens a recovery pause.
        """
        if not self.expected_seconds:
            return None
        with self._cv:
            produced = self._produced
        total = self.expected_seconds / self.speed
        return max(0.0, total - produced / float(self.samplerate))

    def _required_prebuffer(self, floor):
        """Seconds to hold before starting, from the measured delivery rate.

        Returns `floor` until there is enough evidence to do better.

        Only part of the projected shortfall is funded here (DEFICIT_SHARE);
        the speed governor absorbs the rest as the reply plays. Paying for all
        of it up front is what used to put a second or more of silence in
        front of every long reply.
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
        # outrunning playback, so there is nothing to hold back for beyond the
        # device buffer and a jitter margin.
        rate = (produced / float(self.samplerate)) / elapsed
        rate /= RATE_MARGIN

        need = OUTPUT_LATENCY + JITTER_MARGIN
        if rate < 1.0:
            playtime = self.expected_seconds / self.speed
            need += playtime * (1.0 - rate) * DEFICIT_SHARE

        return max(floor, min(PREBUFFER_MAX_SECONDS, need))

    def _await_start(self):
        """Holds the first sample back until there is enough buffered to get
        through the opening of the reply without running dry.

        Polled rather than event-driven on purpose: the target moves as the
        rate estimate firms up, and a 10ms poll for the fraction of a second
        this runs is not worth the extra condition-variable bookkeeping. The
        polling is also what makes the estimate self-correcting -- a slowdown
        during the wait drags the cumulative rate down and raises the target
        before the first word is ever played.
        """
        floor = max(PREBUFFER_MIN_SECONDS, learned_prebuffer())

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

        Running dry mid-reply should now be rare -- the governor gives ground
        on pace long before the queue empties -- but if the network stops
        outright there is nothing pace can do about it. In that case we stop
        writing, raise the prebuffer for the rest of the session, and refill
        before carrying on.

        The refill is capped by how much audio is actually still coming.
        Waiting 350ms to top up when 150ms of speech remains is how a recovery
        ends up longer than the gap it was recovering from.
        """
        warn = None

        with self._cv:
            if self._dry_locked():
                # Grace period first. The normal end of a healthy reply looks
                # identical from here -- last chunk delivered, finish() a
                # millisecond behind it -- and counting that as an underrun
                # would raise the prebuffer on every single reply.
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
            print("speech: buffer ran dry, holding {:.2f}s".format(warn))

        return block

    def _publish_levels(self, block, lag):
        """Waveform levels for one write, staggered across its duration.

        Published before the write, not after: stream.write() blocks until the
        device has room, so a level pushed afterwards is already late. `lag`
        covers what the device is still holding; the stagger covers position
        within the block, since the tail of an 85ms write is heard 85ms after
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

        # Arm the governor only now. Everything before this point had nothing
        # draining the queue, so a thin queue meant nothing.
        with self._cv:
            self._playing = True

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
        # set_volume ends up jumping the end of the sentence -- and how the
        # timing line reports a reply as spoken before it is.
        time.sleep(min(lag, 0.5))

        # Report *after* playback, never during: this print goes through the
        # dashboard proxy and repaints the panel, and the audio path must not
        # queue behind a console write.
        if self._speed_floor_seen < self.speed - 0.02:
            print("speech: eased to {:.2f}x to keep up (target {:.2f}x)".format(
                self._speed_floor_seen, self.speed
            ))

        # A reply of real length that never ran dry is evidence the buffer is
        # bigger than it needs to be, so give some of it back. Without this,
        # one bad thirty seconds on the network would cost every reply for the
        # rest of the session -- and past the point where the whole reply fits
        # inside the buffer, streaming has quietly stopped happening at all.
        played = written / float(self.samplerate)
        if not self._warned_dry and played >= CLEAN_EVIDENCE_SECONDS:
            _relax_learned()


def play_blocking(samples, samplerate=DEFAULT_SAMPLE_RATE):
    """Plays an already-complete int16 array on the shared stream."""
    if samples is None or len(samples) == 0:
        return
    stream = get_stream(samplerate)
    stream.write(np.asarray(samples, dtype=np.int16).reshape(-1, CHANNELS))