"""Plain-text session log -- who said what, and when.

Goes in: commands/logbook.py  (new file)

A running transcript of the conversation in log.txt at the repo root.
.gitignore already ignores *.txt (except requirements.txt), so the log is
never committed. Append-only, thread-safe, local timestamps, one line per
utterance:

    2026-08-13 14:32:05  USER     what's on my screen
    2026-08-13 14:32:09  JARVIS   Let me take a look.

Every path that speaks goes through _Voice in main.py, so hooking log() in
those two methods (say and stream) captures the whole conversation without
call-site sprawl. User input is logged in _run_turn, which both the spoken
and typed paths already funnel through -- including confirmation answers.
"""

import os
import threading
import time

LOG_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "log.txt"
)

_lock = threading.Lock()

# How much of one utterance is worth keeping. A router filler is short; a
# 200-word conversational answer should not be truncated, but a pathological
# model dump is still capped so the log can't balloon on one bad turn.
MAX_CHARS = 4000


def _stamp():
    return time.strftime("%Y-%m-%d %H:%M:%S")


def log(speaker, text):
    if not text or not str(text).strip():
        return
    line = "{}  {:<7}  {}".format(
        _stamp(), speaker, str(text).strip()[:MAX_CHARS]
    )
    with _lock:
        try:
            with open(LOG_PATH, "a", encoding="utf-8") as fh:
                fh.write(line + "\n")
        except Exception as e:
            # A log failure must never take the turn down with it.
            print("logbook: couldn't write to {} ({})".format(LOG_PATH, e))


def user(text):
    log("USER", text)


def jarvis(text):
    log("JARVIS", text)


def error(text):
    """Non-utterance diagnostics -- rate limits, failed tiers, bad models. The
    milestone is watching a rate-limit happen in the log instead of only on
    the console."""
    log("ERROR", text)


if __name__ == "__main__":
    # python -m commands.logbook  -- append a sample pair.
    user("what's on my screen")
    jarvis("Let me take a look.")
    error("sample error line")
    print("wrote a sample pair to {}".format(LOG_PATH))