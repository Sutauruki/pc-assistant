"""TTS facade: Fish Audio first, Moonshine as the offline fallback.

Goes in: listener/tts.py

A stable public surface for the rest of the codebase (main.py calls
tts.speak / tts.SentenceFeeder / tts.speak_stream). Every reply is tried on
Fish Audio's free S2.1 Pro cloud voice (listener/tts_fish.py) first; if that
produces no audio at all -- no FISH_API_KEY, no network, rate limited, free
tier over -- the same text is spoken by the local Moonshine voice
(listener/tts_moonshine.py). last_engine() says which one actually spoke, for
the dev log and the UI.

speak_stream() is the streaming path for main.py's _Voice. On Fish it starts
talking on the first finished sentence while the rest is still being written
(tts_fish.speak_stream); anything Fish couldn't voice is handed to Moonshine.
Moonshine alone still drains the whole reply into one pass, since each of its
calls carries seconds of fixed overhead.
"""

import queue
import threading

from listener import tts_fish, tts_moonshine

_DONE = object()

_last_engine = "fish" if tts_fish.available() else "moonshine"


def primary_engine():
    """The engine the next reply will try first."""
    return "fish" if tts_fish.available() else "moonshine"


def last_engine():
    """The engine that spoke the most recent reply."""
    return _last_engine


def warm():
    """Pre-warms both voices: Fish's connection, and Moonshine's model so a
    fallback doesn't pay several seconds of load on top of the reply."""
    tts_fish.warm()
    tts_moonshine.warm()


def speak(text, on_first_audio=None):
    """Speak `text`: Fish Audio first, Moonshine if Fish produced nothing.
    Returns the player, or None."""
    global _last_engine
    # With a key configured, Fish is asked even after it switched itself off:
    # its phrase cache can still voice a repeated short reply without a
    # request, and it returns None straight away otherwise.
    if tts_fish.API_KEY:
        player = tts_fish.speak(text, on_first_audio=on_first_audio)
        if player is not None:
            _last_engine = "fish"
            return player
    _last_engine = "moonshine"
    return tts_moonshine.speak(text, on_first_audio=on_first_audio)


def speak_stream(pieces, on_first_audio=None, expected_chars=None):
    """Speak text that is still being generated.

    `pieces` is any iterable of strings -- typically a SentenceFeeder fed by
    commands/chat.py's on_sentence callback.
    """
    global _last_engine
    if pieces is None:
        return None
    if tts_fish.available():
        player, unspoken = tts_fish.speak_stream(
            pieces, on_first_audio=on_first_audio)
        if player is not None:
            _last_engine = "fish"
        if unspoken.strip():
            # Fish stopped partway (or never started): finish the reply in
            # Moonshine's voice rather than drop the rest of it.
            fallback = tts_moonshine.speak(
                unspoken,
                on_first_audio=on_first_audio if player is None else None)
            if player is None:
                _last_engine = "moonshine"
                return fallback
        return player
    text = " ".join(str(p) for p in pieces if p)
    if not text.strip():
        return None
    _last_engine = "moonshine"
    return tts_moonshine.speak(text, on_first_audio=on_first_audio)


def speak_async(text, on_first_audio=None):
    """Fire-and-forget speak() on its own thread. Returns the thread."""
    thread = threading.Thread(
        target=speak,
        args=(text,),
        kwargs={"on_first_audio": on_first_audio},
        daemon=True,
    )
    thread.start()
    return thread


class SentenceFeeder:
    """A hand-off between something producing sentences and speak_stream().

        feeder = tts.SentenceFeeder()
        threading.Thread(target=tts.speak_stream, args=(feeder,)).start()
        chat.answer(question, on_sentence=feeder.put)
        feeder.close()

    Iterating blocks until the next sentence arrives or close() is called.
    """

    def __init__(self):
        self._q = queue.Queue()
        self._closed = threading.Event()

    def put(self, sentence):
        if sentence and str(sentence).strip() and not self._closed.is_set():
            self._q.put(str(sentence).strip())

    def close(self):
        if not self._closed.is_set():
            self._closed.set()
            self._q.put(_DONE)

    def __iter__(self):
        while True:
            item = self._q.get()
            if item is _DONE:
                return
            yield item