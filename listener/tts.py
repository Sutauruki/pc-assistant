"""TTS facade: everything speaks through Moonshine.

Goes in: listener/tts.py   (replaces the multi-tier version)

This module used to chain cloud (Groq/Orpheus) and two local voices
(Moonshine, then Kokoro-ONNX) with a fallback order. The other voices are
gone -- Moonshine is the only voice now -- and this file is what remains: a
stable public surface for the rest of the codebase (main.py calls tts.speak /
tts.SentenceFeeder / tts.speak_stream) that hands everything to
listener/tts_moonshine.py.

speak_stream() keeps its name and shape for the streaming path in main.py's
_Voice, but Moonshine synthesises complete text rather than a live feed, so
the `pieces` iterator is drained into one reply and spoken as a single
coalesced call. That is the same behaviour the old local-first path had.
"""

import queue
import threading

from listener import tts_moonshine

_DONE = object()


def speak(text, on_first_audio=None):
    """Speak `text` through Moonshine. Returns the player, or None."""
    return tts_moonshine.speak(text, on_first_audio=on_first_audio)


def speak_stream(pieces, on_first_audio=None, expected_chars=None):
    """Speak text that is still being generated.

    `pieces` is any iterable of strings -- typically a SentenceFeeder fed by
    commands/chat.py's on_sentence callback. All of it is drained into one
    reply before Moonshine synthesises the whole thing.
    """
    if pieces is None:
        return None
    text = " ".join(str(p) for p in pieces if p)
    if not text.strip():
        return None
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