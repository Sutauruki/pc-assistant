"""Control Jarvis from your phone, through a Telegram bot.

Goes in: listener/telegram_bot.py   (new file)

SETUP
-----
1. In Telegram, message @BotFather, send /newbot, pick a name and a username
   ending in "bot". It gives you a token.
2. Put it in .env:  TELEGRAM_BOT_TOKEN=123456:ABC...
3. Restart Jarvis and send the bot any message. A 6-digit code appears on
   the PC (terminal, window and a notification). Send that code to the bot.
   That account is now the only one Jarvis answers.

WHAT IT DOES
------------
Text messages run exactly like typed commands; voice notes are transcribed
by Groq Whisper first. Replies come back as text and the PC stays quiet
(TELEGRAM_SPEAK=1, or /speak on, also says them out loud). Files Jarvis makes
during the turn are sent back. "send me a screenshot" and "send me <file>"
(from Documents\\Jarvis only) are handled here. A command started at the PC
that takes longer than TELEGRAM_NOTIFY_AFTER seconds (default 60) messages
you when it finishes.

SECURITY
--------
- No open ports. Jarvis asks Telegram for new messages over an outgoing
  HTTPS connection (long polling); nothing can connect in to the PC.
- One account. Only the paired account in a private chat is answered;
  everyone else is ignored without a reply. Pairing needs the code shown on
  the PC, so a leaked token alone controls nothing. Wrong codes lock pairing
  for 15 minutes after 5 tries.
- Remote is the lower-trust path. Confirmations that need a physical yes at
  the PC (banking apps, password managers...) can't be approved from here,
  and the computer agent and shell commands are refused outright
  (REMOTE_BLOCKED). Ordinary yes/no questions can be answered in the chat.
- Messages older than 2 minutes are dropped, so nothing queued while the PC
  was off runs hours later.
- Every remote command shows a notification on the PC and goes in the log.
  At most 20 commands a minute.
- Screenshots refuse private windows and black out password boxes, the same
  rules as screen vision. The token is never printed or logged.
- TELEGRAM_ENABLED=0 turns all of it off. "unpair telegram" at the PC, or
  /unpair in the chat, forgets the account.
"""

import json
import os
import queue
import random
import re
import threading
import time
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:  # pragma: no cover
    pass

TOKEN = (os.getenv("TELEGRAM_BOT_TOKEN") or "").strip()
ENABLED = (os.getenv("TELEGRAM_ENABLED", "1").strip().lower()
           not in ("0", "false", "no", "off"))
SPEAK = (os.getenv("TELEGRAM_SPEAK", "0").strip().lower()
         in ("1", "true", "yes", "on"))
try:
    NOTIFY_AFTER = float(os.getenv("TELEGRAM_NOTIFY_AFTER", "60").strip() or 60)
except ValueError:
    NOTIFY_AFTER = 60.0

# Actions a remote message may never start. The agent clicks and types with
# nobody watching the screen; a shell command can do anything at all.
REMOTE_BLOCKED = {
    "computer_task": "taking over the mouse and keyboard",
    "browser_run": "taking over the mouse and keyboard",
    "run_command": "running shell commands",
}

API = "https://api.telegram.org"
POLL_SECONDS = 30            # long-poll wait; the thread sleeps in the socket
MAX_AGE_SECONDS = 120        # older messages are dropped
CODE_SECONDS = 600           # a pairing code lives this long
CODE_TRIES = 5
LOCKOUT_SECONDS = 900
RATE_PER_MINUTE = 20
MAX_TEXT = 4000              # Telegram's limit is 4096
MAX_FILE_BYTES = 49 * 1024 * 1024
MAX_FILES_PER_TURN = 3
# Right after a restart the previous run's long poll can hold the bot for up
# to POLL_SECONDS, so only a conflict that outlasts that is reported.
CONFLICT_GRACE = POLL_SECONDS + 15

_STATE = Path(__file__).resolve().parent.parent / "data" / "telegram.json"

_inbox = queue.Queue()
_lock = threading.Lock()
_owner = None            # {"user_id", "chat_id", "name", "paired"}
_pairing = {"code": None, "until": 0.0, "tries": 0, "locked_until": 0.0}
_recent = []             # timestamps of accepted commands, for the rate limit
_thread = None
_status = "off"
_speak = SPEAK
_session = None
_stop = threading.Event()
_warned = set()


# ---------------------------------------------------------------------------
# Owner (the one paired account)
# ---------------------------------------------------------------------------

def _load_owner():
    global _owner
    try:
        data = json.loads(_STATE.read_text(encoding="utf-8"))
        if int(data.get("user_id")) and int(data.get("chat_id")):
            _owner = data
    except FileNotFoundError:
        _owner = None
    except Exception as e:
        print("telegram: couldn't read {} ({}); not paired".format(_STATE.name, e))
        _owner = None


def _save_owner(owner):
    _STATE.parent.mkdir(parents=True, exist_ok=True)
    tmp = _STATE.with_suffix(".tmp")
    tmp.write_text(json.dumps(owner, indent=2), encoding="utf-8")
    os.replace(tmp, _STATE)


def paired():
    return _owner is not None


def unpair():
    """Forgets the paired account. Returns what to say."""
    global _owner
    with _lock:
        was = _owner
        _owner = None
        _pairing.update(code=None, until=0.0, tries=0)
    try:
        _STATE.unlink()
    except FileNotFoundError:
        pass
    except Exception as e:
        print("telegram: couldn't delete {} ({})".format(_STATE.name, e))
    if not was:
        return "Telegram isn't paired with anyone."
    return "Telegram unpaired. Send the bot a message to pair again."


# ---------------------------------------------------------------------------
# Bot API
# ---------------------------------------------------------------------------

def _scrub(text):
    """Error text without the token -- request URLs carry it."""
    text = str(text)
    return text.replace(TOKEN, "<token>") if TOKEN else text


def _http():
    global _session
    if _session is None:
        import requests
        _session = requests.Session()
    return _session


def _call(method, http_timeout=15, files=None, **params):
    """One Bot API call. `params` go to Telegram as-is -- including its own
    "timeout" (the long-poll wait), which is why the HTTP timeout has a
    different name."""
    resp = _http().post("{}/bot{}/{}".format(API, TOKEN, method),
                        data=params if files else None,
                        json=None if files else params,
                        files=files, timeout=http_timeout)
    try:
        body = resp.json()
    except ValueError:
        body = {}
    if not body.get("ok"):
        raise RuntimeError("{} {}: {}".format(
            method, resp.status_code, body.get("description") or resp.text[:200]))
    return body.get("result")


def _send(chat_id, text):
    text = str(text or "").strip() or "Done."
    for start in range(0, len(text), MAX_TEXT):
        try:
            # Plain text, no parse_mode: nothing in a reply can turn into a
            # link or formatting the model didn't mean.
            _call("sendMessage", chat_id=chat_id, text=text[start:start + MAX_TEXT],
                  disable_web_page_preview=True)
        except Exception as e:
            print("telegram: couldn't send a message ({})".format(_scrub(e)))
            return


def _send_file(chat_id, path, photo=False, caption=None):
    path = Path(path)
    try:
        if path.stat().st_size > MAX_FILE_BYTES:
            _send(chat_id, "{} is too big to send over Telegram.".format(path.name))
            return False
        with open(path, "rb") as fh:
            _call("sendPhoto" if photo else "sendDocument", http_timeout=120,
                  files={"photo" if photo else "document": (path.name, fh)},
                  chat_id=chat_id, caption=caption or "")
        return True
    except Exception as e:
        print("telegram: couldn't send {} ({})".format(path.name, _scrub(e)))
        _send(chat_id, "I couldn't send {}.".format(path.name))
        return False


def send(text):
    """Messages the paired account. False when not paired or not running."""
    owner = _owner
    if not owner or not TOKEN or not ENABLED:
        return False
    _send(owner["chat_id"], text)
    return True


# ---------------------------------------------------------------------------
# Incoming messages
# ---------------------------------------------------------------------------

def _show_on_pc(title, text):
    """Pairing codes and remote commands are always visible at the PC."""
    if _stop.is_set():
        return
    print("telegram: {} -- {}".format(title, text))
    try:
        from commands import notify
        notify.notify(title, text)
    except Exception:
        pass


def _new_code(name):
    code = "{:06d}".format(random.SystemRandom().randrange(1000000))
    _pairing.update(code=code, until=time.time() + CODE_SECONDS, tries=0)
    _show_on_pc("Telegram pairing",
                "{} wants to control Jarvis. Code: {}. If that isn't you, "
                "ignore it.".format(name, code))
    try:
        from listener import ui, window_ui
        ui.set_reply("Telegram pairing code: {}".format(code))
        window_ui.peek()
    except Exception:
        pass


def _pair(msg, user, chat_id):
    """Handles a message from an unknown account while nobody is paired."""
    global _owner
    now = time.time()
    name = " ".join(filter(None, (user.get("first_name"), user.get("last_name")))) \
        or user.get("username") or "Someone"
    text = (msg.get("text") or "").strip()
    with _lock:
        if now < _pairing["locked_until"]:
            return  # locked out: say nothing
        if _pairing["code"] and now < _pairing["until"] and re.fullmatch(r"\d{6}", text):
            if text == _pairing["code"]:
                _owner = {"user_id": user["id"], "chat_id": chat_id, "name": name,
                          "paired": time.strftime("%Y-%m-%d %H:%M")}
                _pairing.update(code=None, until=0.0, tries=0)
                _save_owner(_owner)
                ok = True
            else:
                _pairing["tries"] += 1
                if _pairing["tries"] >= CODE_TRIES:
                    _pairing.update(code=None, until=0.0, tries=0,
                                    locked_until=now + LOCKOUT_SECONDS)
                    _show_on_pc("Telegram pairing", "Too many wrong codes -- pairing "
                                "locked for 15 minutes.")
                    return
                ok = False
        else:
            if not (_pairing["code"] and now < _pairing["until"]):
                _new_code(name)
            ok = None
    if ok:
        _show_on_pc("Telegram", "Paired with {}.".format(name))
        _send(chat_id, "Paired. I'm listening -- send a command, a voice note, or "
                       "/help.")
    elif ok is False:
        _send(chat_id, "That code isn't right.")
    else:
        _send(chat_id, "Send me the 6-digit code shown on the PC.")


def _rate_ok():
    now = time.time()
    _recent[:] = [t for t in _recent if now - t < 60]
    if len(_recent) >= RATE_PER_MINUTE:
        return False
    _recent.append(now)
    return True


def _handle_update(update):
    msg = update.get("message")
    if not msg:
        return  # edits, channel posts, button callbacks: ignored
    chat = msg.get("chat") or {}
    user = msg.get("from") or {}
    if chat.get("type") != "private" or not user.get("id") or user.get("is_bot"):
        return
    if time.time() - float(msg.get("date") or 0) > MAX_AGE_SECONDS:
        return

    owner = _owner
    if owner is None:
        _pair(msg, user, chat["id"])
        return
    if user["id"] != owner["user_id"] or chat["id"] != owner["chat_id"]:
        return  # not the paired account: no reply at all

    if not _rate_ok():
        _send(chat["id"], "Slow down a little -- try again in a minute.")
        return

    text = (msg.get("text") or msg.get("caption") or "").strip()
    voice = msg.get("voice") or msg.get("audio")
    if voice and not text:
        text = _transcribe_voice(chat["id"], voice)
        if not text:
            return
        _send(chat["id"], 'Heard: "{}"'.format(text))
    if not text:
        _send(chat["id"], "I can read text and voice notes.")
        return
    _inbox.put({"chat_id": chat["id"], "text": text[:2000]})


def _transcribe_voice(chat_id, voice):
    from listener import stt_groq
    if not stt_groq.available():
        _send(chat_id, "Voice notes need Groq speech-to-text (GROQ_TTS_API_KEY). "
                       "Send it as text instead.")
        return ""
    if int(voice.get("file_size") or 0) > 20 * 1024 * 1024:
        _send(chat_id, "That voice note is too long.")
        return ""
    try:
        info = _call("getFile", file_id=voice["file_id"])
        resp = _http().get("{}/file/bot{}/{}".format(API, TOKEN, info["file_path"]),
                           timeout=60)
        resp.raise_for_status()
        name = Path(info["file_path"]).name or "voice.ogg"
        if name.endswith(".oga"):
            name = name[:-4] + ".ogg"
        return stt_groq.transcribe_file(resp.content, name)
    except Exception as e:
        print("telegram: voice note failed ({})".format(_scrub(e)))
        _send(chat_id, "I couldn't make out that voice note.")
        return ""


def _get_updates(offset):
    params = {"timeout": POLL_SECONDS, "allowed_updates": ["message"]}
    if offset is not None:
        params["offset"] = offset
    return _call("getUpdates", http_timeout=POLL_SECONDS + 10, **params)


def _warn_once(key, text):
    """A problem that needs you, said once on the PC rather than silently
    retried forever."""
    if key in _warned:
        return
    _warned.add(key)
    _show_on_pc("Telegram", text)


_CONFLICT = (
    "Another program is reading this bot's messages, so Jarvis can't see them "
    "-- usually Dexter/OpenClaw, a bot-builder site, or a second copy of Jarvis "
    "still running. Close it, or in @BotFather send /revoke for this bot and "
    "put the new token in .env (that cuts everything else off).")


def _check_bot():
    """Says who the bot is, and whether a webhook (another service) has
    claimed it -- while one is set, Telegram hands every message to that
    service and polling here gets nothing. Returns False to stop."""
    global _status
    try:
        me = _call("getMe")
        if "me" not in _warned:
            _warned.add("me")
            print("telegram: connected as @{}".format(me.get("username")))
    except Exception as e:
        text = _scrub(e)
        if " 401" in text or "Unauthorized" in text or " 404" in text:
            _status = "off (token rejected)"
            _warn_once("token", "Telegram rejected TELEGRAM_BOT_TOKEN -- phone control "
                       "is off. Copy the token from @BotFather again.")
            return False
        return True  # offline: the poll loop retries
    try:
        hook = _call("getWebhookInfo") or {}
    except Exception:
        hook = {}
    if hook.get("url"):
        from urllib.parse import urlparse
        _status = "blocked (bot is connected to another service)"
        # Not removed automatically: taking a bot off a service you set up
        # should be your decision.
        _warn_once("webhook", "This bot is connected to another service ({}), which "
                   "gets your messages instead of Jarvis. In @BotFather send /revoke "
                   "for this bot and put the new token in .env.".format(
                       urlparse(hook["url"]).hostname or "a webhook"))
    return True


def _loop():
    global _status
    offset = None
    backoff = 2
    conflict_since = None
    if not _check_bot():
        return
    # Skip whatever piled up while Jarvis was off.
    try:
        old = _call("getUpdates", offset=-1)
        if old:
            offset = old[-1]["update_id"] + 1
    except Exception:
        pass
    while not _stop.is_set():
        try:
            updates = _get_updates(offset)
            _status = "on" + (" (paired)" if _owner else " (not paired)")
            backoff, conflict_since = 2, None
        except Exception as e:
            if _stop.is_set():
                return
            text = _scrub(e)
            if " 401" in text or "Unauthorized" in text:
                _status = "off (token rejected)"
                _warn_once("token", "Telegram rejected TELEGRAM_BOT_TOKEN -- phone "
                           "control is off. Copy the token from @BotFather again.")
                return
            if " 409" in text:
                conflict_since = conflict_since or time.time()
                _status = "blocked (another program is using this bot)"
                if "webhook" in text.lower():
                    _check_bot()
                elif time.time() - conflict_since >= CONFLICT_GRACE:
                    _warn_once("conflict", _CONFLICT)
                    backoff = 2  # keep trying: it frees up once the other one stops
            else:
                _status = "reconnecting"
                kind = text.split(":", 1)[0][:60]
                if kind not in _warned:  # print each kind of failure once
                    _warned.add(kind)
                    print("telegram: can't reach Telegram ({}) -- retrying".format(text[:200]))
            _stop.wait(backoff)
            backoff = min(backoff * 2, 60)
            continue
        for update in updates or []:
            offset = update["update_id"] + 1
            try:
                _handle_update(update)
            except Exception as e:
                print("telegram: couldn't handle a message ({})".format(_scrub(e)))


def stop():
    """Ends the poller on exit: wakes it out of its long poll so Ctrl+C
    doesn't leave a thread mid-request while Python shuts down."""
    _stop.set()
    session = _session
    if session is not None:
        try:
            session.close()
        except Exception:
            pass
    if _thread is not None:
        _thread.join(timeout=1.0)


def start():
    """Starts the background poller. Returns False when not configured."""
    global _thread, _status
    if not ENABLED:
        _status = "off (TELEGRAM_ENABLED=0)"
        return False
    if not TOKEN:
        _status = "off (no TELEGRAM_BOT_TOKEN)"
        return False
    if _thread is not None:
        return True
    _load_owner()
    _status = "connecting"
    _thread = threading.Thread(target=_loop, name="telegram", daemon=True)
    _thread.start()
    return True


def status():
    return _status


def poll():
    """The next remote command for the main loop, or None. Never blocks."""
    try:
        return _inbox.get_nowait()
    except queue.Empty:
        return None


# ---------------------------------------------------------------------------
# Remote turns (called by main.py on its own thread)
# ---------------------------------------------------------------------------

def speak_aloud():
    return _speak


_HELP = (
    "Send any command you'd say to Jarvis -- \"open Chrome\", \"how's my battery\", "
    "\"make a PDF about solar panels\" -- as text or a voice note.\n\n"
    "/screenshot -- see the screen (private windows are refused)\n"
    "show me <app> -- a picture of just that app's window\n"
    "is Claude done? / tell me when Claude is done -- check or watch an app\n"
    "send me <file> -- a file from Documents\\Jarvis\n"
    "/speak on|off -- also say replies out loud at the PC\n"
    "/status -- is the PC OK\n"
    "/unpair -- forget this account\n\n"
    "Not from here: taking over the mouse and keyboard, shell commands, and "
    "anything that needs approval at the PC."
)

_SCREENSHOT = re.compile(
    r"^/screenshot\b|^(please )?(send|show|take|get)( me)?( a| the)? (screenshot|screen ?shot|"
    r"picture of (my |the )?screen)\b", re.I)
_APP_PICTURE = re.compile(
    r"^(?:please )?(?:show me|send me (?:a )?(?:picture|screenshot|pic|photo) of|"
    r"(?:screenshot|picture|pic|photo) of) (?:the |my )?(.+?)(?: window| app)?\s*$", re.I)
_SEND_FILE = re.compile(r"^(?:please )?send me (?:the |my |that )?(.+?)\s*$", re.I)
_LATEST = re.compile(r"\b(latest|last|newest|recent|most recent)\b", re.I)


def local_command(item):
    """Handles what never needs the router: /help, /status, /unpair, /speak,
    screenshots and "send me <file>". Returns True when it handled `item`."""
    global _speak
    chat_id, text = item["chat_id"], item["text"].strip()
    low = text.lower()
    if low in ("/start", "/help", "help"):
        _send(chat_id, _HELP)
        return True
    if low == "/unpair":
        _send(chat_id, unpair())
        _show_on_pc("Telegram", "Unpaired from the phone.")
        return True
    if low.startswith("/speak"):
        arg = low.split(None, 1)[1] if " " in low else ""
        _speak = arg not in ("off", "0", "no")
        _send(chat_id, "I'll say replies out loud at the PC too." if _speak
              else "Replies stay on your phone; the PC stays quiet.")
        return True
    if low == "/status":
        _send(chat_id, _status_text())
        return True
    if _SCREENSHOT.search(text):
        _screenshot(chat_id)
        return True
    m = _APP_PICTURE.match(text)
    if m and _app_picture(chat_id, m.group(1)):
        return True
    m = _SEND_FILE.match(text)
    if m and "screenshot" not in low:
        path = find_file(m.group(1))
        if path:
            _send_file(chat_id, path)
            return True
    return False


def _status_text():
    lines = ["Online."]
    try:
        from commands import system_info
        lines += [system_info.battery(), system_info.ram(), system_info.uptime()]
    except Exception:
        pass
    return " ".join(lines)


def _screenshot(chat_id):
    import comtypes
    from commands import pc_policy, screen_actions
    comtypes.CoInitialize()
    try:
        window = screen_actions.active_window()
        private = pc_policy.sensitive_reason(window)
        if private:
            _send(chat_id, "Not sending the screen right now -- {}.".format(private))
            return
        img, bounds = screen_actions.capture(window)
        screen_actions.mask(img, screen_actions.password_rects(window, bounds), bounds)
        jpeg, _ = screen_actions.to_jpeg(img, max_size=1600, quality=75)
    except Exception as e:
        _send(chat_id, "I couldn't capture the screen ({}).".format(e))
        return
    finally:
        comtypes.CoUninitialize()
    from commands import paths
    folder = paths.jarvis_dir("Screenshots")
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / "telegram {}.jpg".format(time.strftime("%Y-%m-%d %H-%M-%S"))
    path.write_bytes(jpeg)
    _send_file(chat_id, path, photo=True,
               caption=(window or {}).get("title", "")[:200])


def _app_picture(chat_id, target):
    """"show me Claude": a picture of just that window. False when no window
    matches, so the words go on to the router ("show me my files")."""
    import comtypes
    from commands import app_watch
    comtypes.CoInitialize()
    try:
        if app_watch.find(target) is None:
            return False
        path, caption = app_watch.picture(target)
    except Exception as e:
        _send(chat_id, "I couldn't capture that ({}).".format(e))
        return True
    finally:
        comtypes.CoUninitialize()
    if path is None:
        _send(chat_id, caption)
    else:
        _send_file(chat_id, path, photo=True, caption=caption)
    return True


def _jarvis_files():
    from commands import paths
    base = paths.jarvis_dir()
    if not base.exists():
        return []
    return [p for p in base.rglob("*") if p.is_file() and not p.name.startswith(".")]


_FILLER = {"the", "a", "an", "my", "file", "from", "earlier", "today", "latest",
           "last", "newest", "recent", "most", "one", "i", "made", "you", "that",
           "please", "me", "of", "about", "on", "for"}
_KINDS = {"pdf": ".pdf", "doc": ".docx", "document": ".docx", "word": ".docx",
          "slides": ".pptx", "presentation": ".pptx", "deck": ".pptx",
          "spreadsheet": ".xlsx", "excel": ".xlsx", "sheet": ".xlsx"}


def find_file(spec):
    """Best match for `spec` among the files in Documents\\Jarvis, or None.
    "latest" (or just a kind, "the pdf") picks the newest; otherwise the
    name with the most words in common, newest first on a tie."""
    files = _jarvis_files()
    if not files:
        return None
    files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    words = [w for w in re.findall(r"[a-z0-9]+", spec.lower()) if w not in _FILLER]
    want = {_KINDS[w] for w in words if w in _KINDS}
    if want:
        files = [p for p in files if p.suffix.lower() in want]
        if not files:
            return None
    best, best_score = None, 0
    for p in files:
        name = p.stem.lower()
        score = sum(1 for w in words if w not in _KINDS and w in name)
        if score > best_score:
            best, best_score = p, score
    if best:
        return best
    if _LATEST.search(spec) or (want and all(w in _KINDS for w in words)):
        return files[0]
    return None


def snapshot_files():
    """{path: mtime} of Jarvis's output, taken before a remote turn."""
    try:
        return {p: p.stat().st_mtime for p in _jarvis_files()}
    except Exception:
        return {}


def reply(item, lines, before):
    """Sends a remote turn's result: what Jarvis said, then any files it made
    (screenshots excluded -- those only go out through /screenshot, which
    applies the privacy rules)."""
    chat_id = item["chat_id"]
    text = " ".join(s.strip() for s in lines if s and s.strip())
    try:
        from commands import confirm
        if confirm.is_pending() and confirm.needs_physical():
            text += " (That one needs approval at the PC itself.)"
    except Exception:
        pass
    _send(chat_id, text or "Done.")
    new = [p for p, m in snapshot_files().items()
           if (p not in before or m > before[p]) and "Screenshots" not in p.parts]
    for path in sorted(new, key=lambda p: p.stat().st_mtime)[:MAX_FILES_PER_TURN]:
        _send_file(chat_id, path)


def notify_long_task(asked, said, seconds):
    """A command started at the PC took a while: tell the phone it's done."""
    if NOTIFY_AFTER <= 0 or seconds < NOTIFY_AFTER or not paired():
        return
    asked = (asked or "").strip()
    said = (said or "").strip() or "Done."
    send('Finished "{}" ({:.0f}s): {}'.format(asked[:80], seconds, said))
