"""Who the assistant is: its name, and the phrase that wakes it.

Goes in: commands/identity.py   (new file)

    ASSISTANT_NAME=Novyx      (.env; default Novyx)

The name is used wherever the assistant talks about itself -- its prompts,
the window, the terminal panel, notifications -- and to recognise "minimize
Novyx" as being about its own window. Speech-to-text doesn't know a made-up
word, so ALIASES lists the ways it tends to come out ("Novix", "Nova X").

The wake phrase depends on which wake-word model is loaded
(listener/wake_word.py): "Hey Novyx" once models/hey_novyx.onnx exists,
"Hey Jarvis" (the built-in model) until then. wake_phrase() always says the
one that actually works, so the window never tells you to say something that
won't wake it.

Folder names (Documents\\Jarvis) and JARVIS_* settings keep their names --
renaming them would split your existing files and settings.
"""

import os

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:  # pragma: no cover
    pass

NAME = (os.getenv("ASSISTANT_NAME") or "Novyx").strip() or "Novyx"

# How speech-to-text tends to spell a name it has never seen.
_MISHEARINGS = {
    "novyx": ["novix", "novics", "novicks", "nova x", "novax", "novex", "noviks",
              "nov x", "nobics", "novis"],
}
ALIASES = sorted({NAME.lower()} | set(_MISHEARINGS.get(NAME.lower(), [])) | {"jarvis"})

_wake_phrase = "Hey Jarvis"


def set_wake_phrase(phrase):
    global _wake_phrase
    _wake_phrase = phrase


def wake_phrase():
    """The phrase that wakes the assistant right now."""
    return _wake_phrase


def is_me(text):
    """True if `text` names the assistant (or its old name)."""
    spoken = " ".join(str(text or "").lower().replace("'s", "").split())
    return spoken in ALIASES
