"""Streaming intent parse, now with routing.

Goes in: commands/parser.py   (replaces existing)

THE CHANGE THAT MAKES CONVERSATION POSSIBLE
-------------------------------------------
Until now an empty `actions` list meant exactly one thing to main.py: "I
didn't catch anything to do." That single hardcoded assumption is the whole
reason asking Jarvis a question got you a confession of failure -- there was no
way for the model to say "nothing to run, because this wasn't a command."

So the object gained a first key:

    {"mode": "command" | "chat" | "both", "reply": "...", "actions": [...]}

  command  do the thing.        "open chrome", "delete that screenshot"
  chat     answer the question. "what's a monad", "how's your day"
  both     do it AND say something real about the result.
           "what's my ip", "read my clipboard and is that a valid email"

`mode` is first in the key order deliberately. This response is streamed and
keys are acted on as they complete, so the route is known after a handful of
tokens -- which lets a chat turn abandon this stream about a second early and
hand straight to the conversation model rather than paying for a command parse
it will never use. That is what returning True from `on_mode` does.

WHY ROUTE HERE AND NOT IN THE BIG MODEL
---------------------------------------
The obvious design is to send everything to the conversational model and let it
decide. That spends a full round trip to a slow model to discover that "open
chrome" was a command. This model already runs on every turn, it is fast, and
deciding "command or question" is a far easier problem than answering the
question. So the cheap model routes, and only real questions pay for the
expensive one.

RETURN SIGNATURE CHANGED
------------------------
    (actions, reply)  ->  (mode, actions, reply)

main.py has to be updated in the same drop or every turn breaks.

The action schema itself is untouched. Still `"action"` as the discriminator,
still `"value"` on set_volume, still the same eight layout names -- the twelve
existing actions pass through commands/executor.py exactly as before.
"""

import json
import os
import re
import threading
import time

from dotenv import load_dotenv
from groq import Groq

load_dotenv()

MODEL_NAME = "openai/gpt-oss-20b"

# How much conversation the router sees. Enough to resolve "delete it" and
# "open the second one", and no more -- this is a router, not the thing holding
# the conversation. commands/chat.py keeps the real history.
ROUTER_HISTORY_TURNS = 4
HISTORY_CHARS = 160

VALID_MODES = ("command", "chat", "both")

_client = None
_client_lock = threading.Lock()

_history = []
_history_lock = threading.Lock()


# Key order is mandatory and the prompt says so twice. "mode" first so the
# route is known early; "reply" second so speech can start while "actions" is
# still generating.

def _named(prompt):
    """Puts the assistant's name (commands/identity.py) into a prompt."""
    from commands import identity
    aliases = ", ".join('"{}"'.format(a) for a in identity.ALIASES)
    return prompt.replace("{NAME}", identity.NAME).replace("{ALIASES}", aliases)

SYSTEM_PROMPT = """You are {NAME}, a voice-controlled PC assistant. Turn the user's words into ONE
JSON object with keys in THIS ORDER (it is read as it streams): "mode", "reply",
"actions".

"mode":
- "command": do something on this PC ("open chrome", "delete that file").
- "chat": talk or general knowledge, nothing on their machine ("capital of
  Norway", "how do I center a div", "tell me a joke").
- "both": do something AND answer from what came back ("what's my ip").
If unsure, prefer "chat" over an action that CHANGES something. But reading is
free: anything about THEIR machine (clipboard, windows, files, IP, what's
installed or running, where something is, what's on screen) must be read by an
action first -- you can't see their PC, so never answer those from memory and
never say "I don't have access".

"reply": one short spoken sentence (under 20 words), said BEFORE the actions
run, so acknowledge and assume success: "Opening Chrome." For chat/both, only a
lead-in ("One sec.") -- the real answer comes from elsewhere. No markdown,
emoji or symbols.

"actions": exact shapes below; [] for chat. Several in one sentence -> all of
them, in order. Never add an action just to acknowledge something.

APPS AND WINDOWS
{"action": "open_app", "target": "<app as said>"} | close_app (quits the whole app) | focus_window (switch to / bring up / show me X)
{"action": "minimize_window", "target": "<app or all>"} | maximize_window | restore_window (no target = undo the last minimize)
{"action": "arrange_windows", "layout": "left|right|top|bottom|center|side_by_side|grid|cascade", "targets": ["<app>", ...]} (omit targets = every window)
{"action": "list_windows"} | {"action": "set_volume", "value": <0-100>} | {"action": "take_screenshot"} | {"action": "refresh_apps"} (they installed something new)
{"action": "jarvis_window", "mode": "show|hide"}
Pass app names exactly as said, minus "the"/"my"; a local catalog resolves them -- never guess paths or exe names. "minimize everything" / "show desktop" -> minimize_window all.
YOUR OWN WINDOW: "maximize", "minimize", "show yourself", "hide", "go small" with NO app named, or naming you ({ALIASES}, "yourself") -> jarvis_window, never minimize/maximize_window. "show me that" after you said something -> jarvis_window show.

FILES (paths as spoken: "downloads/*.tmp", "the report on my desktop"; never invent drive letters; "here" / "this location" = where you run -> "this location/idea.txt")
{"action": "move_file"|"copy_file", "source": "<what>", "destination": "<where>"} | {"action": "rename_file", "target": "<what>", "name": "<new>"}
{"action": "delete_file", "target": "<what>"} (recycle bin; delete / bin / get rid of) | {"action": "organize_folder", "target": "<folder>", "by": "type|date"}
{"action": "make_folder", "target": "<where/name>"} | {"action": "list_files", "target": "<folder>"} | {"action": "find_file", "query": "<name>", "where": "<optional>"}
{"action": "open_path", "target": "<file or folder>"} | {"action": "where_is", "target": "<app or file>"} -- "where is X" is ALWAYS where_is, mode both.

CLIPBOARD: read_clipboard | write_clipboard {"text"} | append_clipboard {"text"} | clear_clipboard | open_link

PC STATUS: {"action": "system_info", "what": "ram|cpu|disk|battery|uptime|all"} -- RAM, CPU, disk, battery, uptime are ALWAYS this, never run_command. Mode command.

SCREEN AND APPS
{"action": "understand_screen", "question": "<their words>"} -- the screen in front: "what's on my screen", "what does that error say". Mode both.
{"action": "app_status", "target": "<claude|terminal|vs code|chrome...>", "question": "<their words>"} -- ONE named app, even behind other windows: "is Claude done", "what's my terminal doing". Mode command.
{"action": "watch_app", "target": "<app>"} ("tell me when Claude is done") | {"action": "stop_watching", "target": "<app or omit>"}
{"action": "notify", "title": "<short>", "message": "<one line>"}

SHELL
{"action": "run_command", "command": "<exact command>"} -- ONE quick command, no folder, only when no other action fits (ipconfig, ping, git status). Never to move/delete/copy/rename files or open/close apps. Don't add flags. Approval is asked for automatically.
{"action": "command_task", "folder": "<folder or empty>", "commands": ["<cmd>", "<next>"]} -- a named folder, several commands in a row, or anything that installs, builds, tests, clones or pulls. Runs in the background, stops at the first failure, reports when done.
task_status ("is the task done") | cancel_task | save_recipe {"name"} ("save that as X") | run_recipe {"name"} ("run X", a saved name) | list_recipes | delete_recipe {"name"}
{"action": "explain_command", "command": "..."} -- "would you run X", "is X safe".
ALREADY RAN: a question about a command they ALREADY ran ("how was the build I ran", "check the npm command I ran in X", "did my install finish") is app_status target "terminal" (covers VS Code's terminal too) with their exact words -- NEVER run it again, never doskey/history. Right after you started a command_task, "is it done" / "how's it going" -> task_status.

MAKING THINGS (mode command; full request in "prompt"; "target" only if they named a file or folder)
make_doc | make_pptx | make_xlsx | make_pdf {"prompt": "...", "target": ""} | build_website {"prompt", "target"} | stop_preview (mode both)

READING AND RESEARCH (mode both; pick these even without "search" or "read")
{"action": "analyze_file", "target": "<file>", "question": ""} -- a PDF, sheet, CSV, audio or text file.
{"action": "youtube_summary", "target": "<url or description>", "question": ""}
{"action": "web_search", "query": "...", "question": ""} -- anything to look up that ISN'T about their PC (weather, news, reviews).

COMPUTER CONTROL (mode command)
{"action": "computer_task", "task": "<their full request, every detail and name exactly as said>"} -- an agent that clicks and types like a person, for things in an app or site no other action covers: messaging on WhatsApp, filling forms, pressing keys/shortcuts ("press windows r"), clicking or typing on screen. Never chat for these (chat can't press anything). Not for what a dedicated action does.
If your last reply in the history was the agent's QUESTION ("which window has the search box?"), the user's words answer it: emit computer_task again with the original task plus their answer.
research_explain {"question"} ("explain what you found") | research_save {"format": "doc|sheet|slides|pdf", "target"} ("save it as a pdf") -- follow-ups to a research computer_task.

MEMORY
{"action": "set_preference", "text": "<a standing rule, imperative>"} -- how to BEHAVE from now on ("keep answers short", "always ask before deleting"). A CORRECTION is a preference too: "no, shorter", "too long", "stop doing that" -> also emit set_preference with the general rule behind it, alongside anything else. Not for one-offs ("keep this one short") or corrected facts.
{"action": "remember_this", "text": "<third-person fact>"} -- people, places, projects ("The user's sister is Tolu."). Emit it in passing too when they mention one. Never passwords, PINs, codes or numbers.
{"action": "set_profile", "field": "name|call_me|location|work|about", "value": "..."} -- "my name is", "call me", "I'm in Lagos", "I'm a developer"; in passing too.
get_to_know_me (only that; reply "Sure.") | list_memory ("what do you remember", mode both) | forget_preference {"text"} | forget_fact {"text"}
{"action": "my_style", "do": "show|forget|learn"} -- "how do you think I talk" / "forget my style" / "learn how I talk now".

If you truly can't tell what they mean, use chat and ask.

EXAMPLES
open chrome and mute the volume
{"mode": "command", "reply": "Opening Chrome and muting.", "actions": [{"action": "open_app", "target": "chrome"}, {"action": "set_volume", "value": 0}]}
delete idea.txt in this location
{"mode": "command", "reply": "Sending it to the recycle bin.", "actions": [{"action": "delete_file", "target": "this location/idea.txt"}]}
hey, how's it going
{"mode": "chat", "reply": "One sec.", "actions": []}
what's my ip address
{"mode": "both", "reply": "Checking.", "actions": [{"action": "run_command", "command": "ipconfig"}]}
what did I copy
{"mode": "both", "reply": "One sec.", "actions": [{"action": "read_clipboard"}]}
what does that error on screen say
{"mode": "both", "reply": "Reading it.", "actions": [{"action": "understand_screen", "question": "what does that error on screen say"}]}
where's vs code installed
{"mode": "both", "reply": "One sec.", "actions": [{"action": "where_is", "target": "vs code"}]}
minimize yourself
{"mode": "command", "reply": "Minimized.", "actions": [{"action": "jarvis_window", "mode": "hide"}]}
no, shorter
{"mode": "command", "reply": "Got it.", "actions": [{"action": "set_preference", "text": "Keep answers short."}]}
text my sister Tolu on whatsapp that I'll be late
{"mode": "command", "reply": "On it.", "actions": [{"action": "computer_task", "task": "open WhatsApp web and message my sister Tolu that I'll be late"}, {"action": "remember_this", "text": "The user's sister is Tolu."}]}
in pc-assistant run git pull then pip install -r requirements.txt and tell me when it's done
{"mode": "command", "reply": "Setting that up.", "actions": [{"action": "command_task", "folder": "pc-assistant", "commands": ["git pull", "pip install -r requirements.txt"]}]}
how was the npm run build command I ran in ExamSlingDesktop
{"mode": "command", "reply": "Let me look.", "actions": [{"action": "app_status", "target": "terminal", "question": "how was the npm run build command I ran in ExamSlingDesktop"}]}
is claude done
{"mode": "command", "reply": "Checking on Claude.", "actions": [{"action": "app_status", "target": "claude", "question": "is Claude done"}]}
how much ram am I using
{"mode": "command", "reply": "Checking.", "actions": [{"action": "system_info", "what": "ram"}]}
make a pdf report on this week's numbers
{"mode": "command", "reply": "Putting it together.", "actions": [{"action": "make_pdf", "prompt": "a one page report on this week's numbers", "target": ""}]}
what's the weather in tokyo
{"mode": "both", "reply": "Looking it up.", "actions": [{"action": "web_search", "query": "weather in tokyo"}]}

Respond with ONLY the JSON object, nothing else."""
SYSTEM_PROMPT = _named(SYSTEM_PROMPT)


_ESCAPES = {
    "n": "\n",
    "t": "\t",
    "r": "\r",
    "b": "\b",
    "f": "\f",
    '"': '"',
    "\\": "\\",
    "/": "/",
}


def get_client():
    """One Groq client, built on first use. commands/chat.py borrows this for
    its last-resort fallback, so it stays public."""
    global _client
    with _client_lock:
        if _client is None:
            api_key = os.getenv("GROQ_API_KEY")
            if not api_key:
                raise ValueError("GROQ_API_KEY not found in .env")
            # No silent retries: the SDK's default is to sleep and retry on a
            # 429, and Groq's free budget is 8K tokens a minute -- about one
            # parse of this prompt -- so a second command within the minute
            # sat there for 20-48s. Failing fast lets _open_stream hand the
            # turn to Gemini instead.
            _client = Groq(api_key=api_key, max_retries=0, timeout=20.0)
        return _client


# --- Router history --------------------------------------------------------
#
# Short, and trimmed hard. "delete it" needs the previous turn to mean
# anything; nothing here needs the turn from four minutes ago.

def remember(user_text, assistant_text):
    # Every turn also goes to disk (commands/recall.py), so a restart brings
    # the last few back instead of starting from nothing.
    try:
        from commands import recall
        recall.log_turn(user_text, assistant_text)
    except Exception as e:
        print("parser: couldn't log the turn ({})".format(e))
    with _history_lock:
        _history.append((
            str(user_text or "")[:HISTORY_CHARS],
            str(assistant_text or "")[:HISTORY_CHARS],
        ))
        del _history[:-ROUTER_HISTORY_TURNS]
    # The chat brain keeps its own history; without command turns in it,
    # "what did you do then?" gets answered from older, unrelated talk.
    try:
        from commands import chat
        chat.remember(user_text, assistant_text)
    except Exception as e:
        print("parser: couldn't share the turn with chat ({})".format(e))


def remember_event(text):
    """Something that happened in the background (a task finished, a watched
    app is done), so the next question about it has context."""
    remember("", "[while you were away] " + str(text or ""))


def forget():
    with _history_lock:
        _history.clear()


def _memory_block():
    """Remembered preferences and session facts as prompt text, or "".

    Wrapped in try/except and imported lazily on purpose. Routing is the one
    thing that must never break: if the preferences file is unreadable the
    right outcome is an assistant that forgets how you like to be spoken to,
    not one that can't take an instruction at all.
    """
    try:
        from commands import memory
        return memory.prompt_block()
    except Exception as e:
        print("parser: couldn't load memory ({})".format(e))
        return ""


def _system_prompt():
    """The router prompt plus what the user has told us and what's going on.

    Both halves matter here for different reasons. Preferences change how the
    reply field is written ("keep it brief" applies to the router's own
    talking). Session facts change what a target *means*: once "I'm working in
    the pc-assistant folder" is remembered, "delete idea.txt in this location"
    has a folder to point at even when the phrasing is loose.
    """
    block = _memory_block()
    if not block:
        return SYSTEM_PROMPT
    return SYSTEM_PROMPT + "\n\n" + block


def _history_messages():
    with _history_lock:
        pairs = list(_history)

    messages = []
    for user_text, assistant_text in pairs:
        if user_text:
            messages.append({"role": "user", "content": user_text})
        if assistant_text:
            messages.append({"role": "assistant", "content": assistant_text})
    return messages


# --- Streaming string extraction -------------------------------------------

def _extract_string_value(buf, key):
    """Pulls a complete string value for `key` out of a partial JSON buffer.

    Returns None while the value is still arriving, so the caller can just
    retry on the next token. This is what lets us start speaking before the
    model has finished writing the actions array -- and now, what lets us know
    the route before it has written anything else at all.

    json.loads can't do this: a partial object is a syntax error, not a partial
    result.
    """
    marker = '"' + key + '"'
    i = buf.find(marker)
    if i == -1:
        return None

    j = buf.find(":", i + len(marker))
    if j == -1:
        return None

    k = j + 1
    while k < len(buf) and buf[k] in " \t\r\n":
        k += 1
    if k >= len(buf) or buf[k] != '"':
        return None

    k += 1
    out = []
    while k < len(buf):
        c = buf[k]
        if c == "\\":
            if k + 1 >= len(buf):
                return None  # escape split across chunks
            out.append(_ESCAPES.get(buf[k + 1], buf[k + 1]))
            k += 2
            continue
        if c == '"':
            return "".join(out)
        out.append(c)
        k += 1
    return None  # closing quote hasn't arrived yet


def _normalise_mode(value):
    """Anything recognisable -> one of VALID_MODES, else None."""
    value = str(value or "").strip().lower()
    if not value:
        return None

    # Check "both" first: "command" is a substring of nothing here, but a
    # model that writes "command and chat" should land on "both", and testing
    # in VALID_MODES order would catch "command" first.
    if "both" in value:
        return "both"
    if "chat" in value or value in ("talk", "answer", "question",
                                    "conversation", "converse", "reply"):
        return "chat"
    if "command" in value or value in ("do", "act", "action", "run", "execute"):
        return "command"
    return None


# --- Salvage ---------------------------------------------------------------

_FENCE_RE = re.compile(r"^\s*```(?:json)?\s*|\s*```\s*$", re.IGNORECASE)


def _load_object(buf):
    """The JSON object out of whatever came back, or None.

    Even with response_format set, a small model occasionally wraps the object
    in a fence or bolts a sentence onto the end. Recovering from that is
    cheaper than a retry and very much cheaper than an apology.
    """
    text = _FENCE_RE.sub("", (buf or "").strip())
    try:
        return json.loads(text)
    except Exception:
        pass

    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        try:
            return json.loads(text[start:end + 1])
        except Exception:
            pass
    return None


def _clean_actions(raw):
    """Only well-formed dicts carrying an `action` survive.

    A malformed entry is dropped rather than passed along: executor.py would
    report it as an unrecognized action, main.py would see the word "failed"
    in the result, and the turn would end in a spoken apology for something the
    user never asked for.
    """
    actions = []
    if not isinstance(raw, list):
        return actions
    for item in raw:
        if isinstance(item, dict) and str(item.get("action", "")).strip():
            actions.append(item)
    return actions


# --- The parse -------------------------------------------------------------

# Room for the JSON answer plus gpt-oss's reasoning at "low" effort. Groq
# counts max_tokens toward the free tier's tokens-per-minute budget before the
# request runs; left unset it defaults to the model's maximum, which on top of
# this ~6k-token prompt is what produced "413 Request too large".
MAX_TOKENS = 1024
# Shorter reasoning: a router picks one of a few dozen shapes, it doesn't need
# to deliberate -- and gpt-oss's default effort is where 20-30s parses came from.
REASONING_EFFORT = os.getenv("PARSER_REASONING", "low")


# After Groq refuses for being over its per-minute budget, skip it for this
# long and go straight to the fallback -- asking again would just fail again.
GROQ_COOLDOWN_SECONDS = 60
_groq_cool_until = 0.0
# Groq's free tier refuses any single request bigger than its per-minute
# token budget (413), so a request that can't fit is trimmed first -- oldest
# history goes -- and skipped only if it still can't. Asking anyway costs a
# round trip and a 60-second cooldown for nothing.
GROQ_TPM = int(os.getenv("GROQ_TPM", "8000") or 8000)
GEMINI_PARSE_TIMEOUT = 15.0   # seconds; the fallback must not hang a turn
_gemini_parse_client = None


def _estimate_tokens(messages):
    # ~3.5 characters a token for English -- deliberately pessimistic.
    return int(sum(len(m["content"]) for m in messages) / 3.5) + MAX_TOKENS


def _fit_for_groq(messages):
    """messages trimmed to fit GROQ_TPM (oldest history first), or None
    when even the bare system prompt and question are too big."""
    trimmed = list(messages)
    while _estimate_tokens(trimmed) > GROQ_TPM and len(trimmed) > 2:
        del trimmed[1]
    return trimmed if _estimate_tokens(trimmed) <= GROQ_TPM else None


def _is_rate_limit(error):
    text = str(error).lower()
    return "429" in text or "rate limit" in text or "413" in text or "too large" in text


def _gemini_parse(system, convo):
    """The router on Gemini with minimal thinking: a fast JSON answer when
    Groq is over its limit. Raises when Gemini isn't available."""
    from commands import genbrain
    if not os.getenv("GEMINI_API_KEY"):
        raise RuntimeError("no GEMINI_API_KEY")
    global _gemini_parse_client
    if _gemini_parse_client is None:
        # Its own client so it can carry a timeout: the shared one also makes
        # long documents, which legitimately take longer than a route should.
        from google import genai
        _gemini_parse_client = genai.Client(
            api_key=os.getenv("GEMINI_API_KEY"),
            http_options={"timeout": int(GEMINI_PARSE_TIMEOUT * 1000)})
    client = _gemini_parse_client
    config = {"system_instruction": system, "response_mime_type": "application/json",
              "max_output_tokens": MAX_TOKENS,
              "thinking_config": {"thinking_level": "minimal"}}
    try:
        response = client.models.generate_content(
            model=genbrain.MODEL_NAME, contents=convo, config=config)
    except Exception as e:
        if "thinking" not in str(e).lower():
            raise
        config.pop("thinking_config")
        response = client.models.generate_content(
            model=genbrain.MODEL_NAME, contents=convo, config=config)
    text = (response.text or "").strip()
    if not text:
        raise RuntimeError("empty response")
    return text


def _open_stream(messages, t0):
    """(stream, None) from Groq, or (None, text) from the genbrain chain
    (Gemini, then OpenRouter) when Groq refuses -- a 413/429 on the free tier
    shouldn't turn a command into a chat turn that pretends to have done it.
    (None, None) when nothing answered at all."""
    global _groq_cool_until
    from commands import devlog
    client = get_client()
    attempts = [{"extra_body": {"reasoning_effort": REASONING_EFFORT}}, {}]
    last = None
    if time.time() < _groq_cool_until:
        attempts = []  # still over the minute's budget: straight to Gemini
        last = "Groq cooling down after a rate limit"
    fitted = _fit_for_groq(messages)
    if fitted is None:
        attempts = []
        last = "request too big for Groq's {} tokens a minute".format(GROQ_TPM)
    elif len(fitted) < len(messages):
        print("parser: trimmed {} history messages to fit Groq".format(len(messages) - len(fitted)))
    for extra in attempts:
        try:
            return client.chat.completions.create(
                model=MODEL_NAME,
                messages=fitted,
                response_format={"type": "json_object"},
                temperature=0.2,
                max_tokens=MAX_TOKENS,
                stream=True,
                **extra
            ), None
        except Exception as e:
            last = e
            if _is_rate_limit(e):
                _groq_cool_until = time.time() + GROQ_COOLDOWN_SECONDS
            # Only the reasoning option can be retried away; anything else
            # (rate limit, size, network) goes straight to the fallback.
            if "reasoning" not in str(e).lower():
                break
    print("parser: Groq failed ({}), trying the fallback chain".format(last))
    devlog.route("reach", "FAIL", str(last)[:120], seconds=time.time() - t0)
    convo = "\n".join("{}: {}".format(m["role"].upper(), m["content"])
                      for m in messages[1:])
    try:
        text = _gemini_parse(messages[0]["content"],
                             convo + "\nReturn the JSON object for the last USER message.")
        devlog.route("reach", "OK", "gemini fallback", seconds=time.time() - t0)
        return None, text
    except Exception as e:
        print("parser: Gemini fallback failed ({}), trying the rest".format(e))
    try:
        from commands import genbrain
        text = genbrain._chain_text(messages[0]["content"],
                                    convo + "\nReturn the JSON object for the last USER message.",
                                    MAX_TOKENS, json_mode=True)
        devlog.route("reach", "OK", "fallback chain", seconds=time.time() - t0)
        return None, text
    except Exception as e:
        print("parser: fallback chain failed too ({})".format(e))
        return None, None

def parse_command(text, on_reply=None, on_mode=None):
    """Takes transcribed text, returns (mode, actions, reply).

    on_reply(reply)  fires as soon as the spoken line is fully formed --
                     typically several hundred ms before the full response
                     lands. Start TTS from that callback.

    on_mode(mode)    fires as soon as the route is known, which is earlier
                     still. Returning True from it ABORTS the parse: the caller
                     has decided it doesn't need the actions, and every
                     remaining token is pure latency. That is how a chat turn
                     saves about a second.

    Never raises. A failure comes back as ("chat", [], <something to say>) so
    the turn still produces speech instead of silence.
    """
    from commands import devlog  # local import: avoids an import cycle
    t0 = time.time()

    if not text or not text.strip():
        fallback = "I didn't catch anything to do there."
        if on_reply is not None:
            on_reply(fallback)
        devlog.route("parse", "SKIP", "empty input", seconds=time.time() - t0)
        return "chat", [], fallback

    messages = [{"role": "system", "content": _system_prompt()}]
    messages.extend(_history_messages())
    messages.append({"role": "user", "content": text})

    stream, fallback_text = _open_stream(messages, t0)

    buf = ""
    mode = None
    reply = None
    aborted = False

    if stream is None:
        if fallback_text is None:
            return "chat", [], "I couldn't reach my language model just then."
        # Not streamed, so the early callbacks fire here instead.
        buf = fallback_text
        found = _extract_string_value(buf, "reply")
        if found and on_reply is not None:
            reply = found
            try:
                on_reply(reply)
            except Exception as e:
                print("parser: on_reply raised ({})".format(e))
        stream = ()

    try:
        for chunk in stream:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta.content or ""
            if not delta:
                continue
            buf += delta

            if mode is None:
                found = _extract_string_value(buf, "mode")
                if found is not None:
                    mode = _normalise_mode(found)
                    if mode is not None and on_mode is not None:
                        try:
                            if on_mode(mode):
                                aborted = True
                                break
                        except Exception as e:
                            # A callback that raises must not take the turn
                            # down with it -- worst case we finish the parse
                            # we were trying to skip.
                            print("parser: on_mode raised ({})".format(e))

            if reply is None:
                found = _extract_string_value(buf, "reply")
                if found:
                    reply = found
                    if on_reply is not None:
                        try:
                            on_reply(reply)
                        except Exception as e:
                            print("parser: on_reply raised ({})".format(e))
    except Exception as e:
        print("parser: stream broke ({})".format(e))
    finally:
        # Walking away from a stream without closing it leaves the connection
        # open until the SDK's socket times out. One of those per chat turn
        # adds up over a session.
        try:
            if hasattr(stream, "close"):
                stream.close()
        except Exception:
            pass

    if aborted:
        # Deliberate early exit. No actions were read, and none are wanted.
        devlog.route("parse", "OK", "mode={} (aborted early)".format(mode),
                     seconds=time.time() - t0)
        return mode, [], (reply or "")

    parsed = _load_object(buf)
    if parsed is None:
        devlog.route("parse", "FAIL", "unparseable response", seconds=time.time() - t0)
        print("parser: couldn't parse the response -- {!r}".format(buf[:300]))
        return "chat", [], (reply or "Something went wrong parsing that, sorry.")

    if mode is None:
        mode = _normalise_mode(parsed.get("mode"))
    if reply is None:
        reply = str(parsed.get("reply") or "").strip()

    actions = _clean_actions(parsed.get("actions"))

    # --- Reconciliation ----------------------------------------------------
    # Mode and action list disagree often enough to handle explicitly. When
    # they do, the list is the half to trust: it's concrete, and a wrong mode
    # costs one unnecessary round trip whereas a dropped action costs the thing
    # the user actually asked for.

    if mode is None:
        mode = "command" if actions else "chat"
    elif mode == "chat" and actions:
        mode = "both"
    elif mode == "both" and not actions:
        mode = "chat"

    if mode == "command" and not actions:
        # The old behaviour, now reached only when the model insisted this was
        # a command and then handed over nothing to do.
        devlog.route("parse", "OK", "mode={} no actions".format(mode),
                     seconds=time.time() - t0)
        return mode, [], (reply or "I didn't catch anything to do there.")

    devlog.route("parse", "OK",
                 detail="mode={} actions={}".format(
                     mode, [a.get("action") if isinstance(a, dict) else "?"
                            for a in actions]),
                 seconds=time.time() - t0)
    return mode, actions, reply


REPLY_SYSTEM_PROMPT = """You are {NAME}, a helpful voice assistant speaking out loud.
You just performed some actions on the user's PC, and at least one of them failed.
Given the raw result messages, respond with ONE short, natural, spoken sentence
that honestly reflects what happened -- including the failure. Sound casual and
human, not robotic. Keep it under 20 words when possible. No markdown, no
symbols, no lists -- this is read aloud."""
REPLY_SYSTEM_PROMPT = _named(REPLY_SYSTEM_PROMPT)


def _reply_system_prompt():
    """The correction prompt plus preferences only -- no session facts.

    Session context is deliberately left out. This call has one job: say what
    just went wrong in a sentence. Facts about which folder we're in don't help
    it do that and would only give it material to pad with. Preferences do
    apply, because "keep it short" is exactly the kind of thing that should
    govern an apology.
    """
    try:
        from commands import memory
        block = memory.prompt_block(include_session=False)
    except Exception:
        block = ""  # already reported by _memory_block on the routing path

    if not block:
        return REPLY_SYSTEM_PROMPT
    return REPLY_SYSTEM_PROMPT + "\n\n" + block


def generate_reply(results):
    """Fallback only -- used when something actually failed during execution
    and the pre-generated reply would be inaccurate.

    Never raises: it runs on the failure path, and an exception here would turn
    a reported failure into a silent one.
    """
    if not results:
        return "I didn't catch anything to do there."

    joined = "\n".join(str(r) for r in results if r)
    if not joined.strip():
        return "That didn't work."

    try:
        response = get_client().chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": _reply_system_prompt()},
                {"role": "user", "content": joined[:800]},
            ],
            temperature=0.3,
            max_tokens=80,
        )
        spoken = (response.choices[0].message.content or "").strip()
        return spoken or joined[:180]
    except Exception as e:
        print("parser: couldn't generate a correction ({})".format(e))
        return joined[:180]


if __name__ == "__main__":
    # python -m commands.parser  -- routing only, nothing is executed.
    probes = [
        "open chrome",
        "what's the capital of Norway",
        "delete the tmp files in downloads",
        "how do I center a div",
        "what's my ip address",
        "organize my downloads folder",
        "read my clipboard",
        "hey how are you",
        "minimize everything and tell me a joke",
        "rename that screenshot to receipt",
    ]
    width = max(len(p) for p in probes)
    for probe in probes:
        mode, actions, reply = parse_command(probe)
        print("{:<{w}}  {:<8} {:<30} {}".format(
            probe, mode, str([a.get("action") for a in actions]), reply, w=width
        ))