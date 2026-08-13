"""Detailed developer log -- every service, every call, every millisecond.

Goes in: commands/devlog.py   (new file)

Gated entirely by DEV_LOG=1 in .env. When it is set, a second, much more
detailed transcript is appended to dev_log.txt at the repo root, alongside the
human-readable log.txt. When it is unset, every call in here is a no-op that
costs nothing.

WHAT'S IN IT, AND WHY IT'S SEPARATE FROM log.txt
------------------------------------------------
log.txt is the conversation -- who said what. This is the machinery: which STT
tier answered and in how long, which free model the chat brain picked and why
it fell through to the next, how the reply was split into synthesis segments,
which TTS provider voiced it, and how long each waited. When something is
slow, or a tier silently died, or a model misbehaved, this is the file that
shows it.

Output is one aligned row per event. Fixed column widths make the shape of a
turn visible at a glance:

    timestamp  category  service            event          status   duration  detail
    14:32:05.123  STT  moonshine          transcribe      OK          1.24s  "open chrome"
    14:32:06.000  PARSER  gpt-oss-20b     route           OK          0.62s  mode=command
    14:32:06.500  TTS  orpheus            speak           OK          2.10s  1 segment / 8 chars

Duration is right-aligned in its column so timings line up down the file; the
detail column is free-form and may be empty.
"""

import os
import threading
import time

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:  # pragma: no cover - .env is optional
    pass

LOG_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "dev_log.txt"
)

# The whole module is a no-op when this is false. Checked once, at import.
ENABLED = os.getenv("DEV_LOG", "") == "1"

_lock = threading.Lock()

_HEADER = (
    "timestamp                 category  service            event           "
    "status   duration  detail\n"
)

# Column geometry, kept in one place so the alignment can't drift apart.
_W = (23, 10, 20, 16, 8, 10)


def _stamp():
    now = time.time()
    base = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(now))
    return "{}.{:03d}".format(base, int((now % 1.0) * 1000))


def _dur(seconds):
    if seconds is None:
        return "-"
    if seconds >= 1.0:
        return "{:.2f}s".format(seconds)
    if seconds >= 0.001:
        return "{:.0f}ms".format(seconds * 1000)
    return "<1ms"


def _row(category, service, event, status, seconds, detail):
    stamp, _, _, _, _, _ = _W
    parts = (
        _stamp().rjust(stamp),
        category.ljust(10),
        service.ljust(20),
        event.ljust(16),
        _dur(seconds).rjust(7),
        " " + str(detail or "").strip(),
    )
    return "  ".join(p for p in parts if p) + "\n"


def log(category, service, event, status="", detail="", seconds=None):
    """Append one event row, or do nothing when DEV_LOG is off.

    `category` groups rows (STT, TTS, THINK, PARSER, EXEC, WAKE, TURN, SYS);
    `service` names the specific provider or tier; `event` is what happened;
    `status` is usually OK/FAIL/SKIP/RETRY/INFO; `seconds` right-aligns in the
    duration column; `detail` is the free-form tail.
    """
    if not ENABLED:
        return
    line = _row(category, service, event, status, seconds, detail)
    with _lock:
        try:
            with open(LOG_PATH, "a", encoding="utf-8") as fh:
                if os.path.getsize(LOG_PATH) == 0:
                    fh.write(_HEADER)
                fh.write(line)
        except Exception as e:
            # A dev log failure is not a reason to take the turn down with it.
            print("devlog: couldn't write to {} ({})".format(LOG_PATH, e))


# --- Subsystem helpers ------------------------------------------------------

def stt(tier, event, status="", detail="", seconds=None):
    log("STT", tier, event, status, detail, seconds)


def tts(service, event, status="", detail="", seconds=None):
    log("TTS", service, event, status, detail, seconds)


def think(service, event, status="", detail="", seconds=None):
    log("THINK", service, event, status, detail, seconds)


def route(event, status="", detail="", seconds=None):
    log("PARSER", "gpt-oss-20b", event, status, detail, seconds)


def exec_event(action, event, status="", detail="", seconds=None):
    log("EXEC", action, event, status, detail, seconds)


def wake(event, status="", detail="", seconds=None):
    log("WAKE", "openwakeword", event, status, detail, seconds)


def turn(event, status="", detail="", seconds=None):
    log("TURN", "-", event, status, detail, seconds)


def sys_event(event, status="", detail="", seconds=None):
    log("SYS", "-", event, status, detail, seconds)


if __name__ == "__main__":
    # python -m commands.devlog  -- append a full sample turn so the format is
    # visible without running the assistant.
    print("DEV_LOG={}".format("on" if ENABLED else "off"))
    if not ENABLED:
        raise SystemExit("set DEV_LOG=1 in .env to see the format")

    wake("detected", detail="hey_jarvis 0.93")
    stt("moonshine", "transcribe", "OK", '""', 0.31)
    turn("start", detail='"open chrome"')
    route("route", "OK", "mode=command", 0.42)
    exec_event("open_app", "run", "OK", '"Chrome"', 1.1)
    tts("orpheus", "speak", "OK", "1 segment / 12 chars", 0.8)
    turn("done", "OK", "1.9s total")
    print("wrote a sample turn to {}".format(LOG_PATH))