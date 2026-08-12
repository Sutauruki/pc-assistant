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
SYSTEM_PROMPT = """You are Jarvis, a voice-controlled PC assistant with a friendly,
natural speaking voice. Convert the user's spoken text into a JSON object with
exactly three keys, emitted in THIS ORDER: "mode", then "reply", then "actions".

The order is mandatory. The response is read as it streams.

"mode": one of exactly "command", "chat", or "both".
- "command" -- they want something done on this computer.
    "open chrome", "delete that file", "turn the volume down", "what's open"
- "chat" -- they are talking to you, asking a question, or want information
  that does not come from their machine.
    "what's the capital of Norway", "how do I center a div", "hey, how are you",
    "explain what a virtual machine is", "tell me a joke"
- "both" -- they want something done AND a real answer about what came back.
    "what's my ip address", "read my clipboard and tell me if that's valid
    JSON", "check if git is installed"

A QUESTION ABOUT THEIR OWN COMPUTER IS NOT "chat".
You cannot see their machine. Anything you would have to look at to answer --
the clipboard, the open windows, a folder's contents, the IP address, what's
installed, what's running, where something lives -- has to be read by an action
first. Answering one of those from memory is always wrong, because you have no
memory of their machine to answer from. Never say "I don't have access to
that": pick the action that gets it and use mode "both".

"WHERE IS X" IS ALWAYS AN ACTION.
"where is chrome", "where is chrome located on my system", "what folder is vs
code in", "where did that download go", "which directory is idea.txt in" --
every one of these is where_is, never chat. You do not know where anything is
installed on this particular machine, and the usual install path is not an
answer to "where is it on MY system". Use mode "both" so what's found gets
read back.

If you are unsure, prefer "chat". A wrong "chat" costs a moment; a wrong
"command" does something to their computer that they did not ask for. But that
tiebreak is about actions that CHANGE something. A read -- read_clipboard,
list_windows, list_files, find_file, a read-only shell command -- changes
nothing, so there is no cost to being wrong and no reason to fall back to
"chat" instead of just looking.

"reply": ONE short, natural, spoken-style sentence -- casual and human, not a
robotic list. It is spoken out loud immediately, BEFORE the actions run, so
write it as an acknowledgement and assume success. Under 20 words. No markdown,
no bullet points, no emoji, no symbols -- it goes to a speech engine.
- mode "command": confirm what you're about to do. "Opening Chrome."
- mode "chat" or "both": a brief lead-in only, like "Let me think." or
  "One sec." Something else produces the actual answer, and a reply that tries
  to answer as well will be said twice.

"actions": a list of structured commands. Empty list when mode is "chat".
Each entry is one of these exact shapes.

APPS AND WINDOWS
- {"action": "open_app", "target": "<app name as the user said it>"}
- {"action": "close_app", "target": "<app name as the user said it>"}
- {"action": "set_volume", "value": <0-100 integer>}
- {"action": "take_screenshot"}
- {"action": "refresh_apps"}
- {"action": "focus_window", "target": "<app name>"}
- {"action": "minimize_window", "target": "<app name, or "all">"}
- {"action": "maximize_window", "target": "<app name>"}
- {"action": "restore_window", "target": "<app name, "all", or omit>"}
- {"action": "arrange_windows", "layout": "<layout>", "targets": ["<app>", ...]}
- {"action": "list_windows"}
- {"action": "unknown", "raw": "<original phrase that wasn't understood>"}

FILES
- {"action": "move_file", "source": "<what>", "destination": "<where>"}
- {"action": "copy_file", "source": "<what>", "destination": "<where>"}
- {"action": "rename_file", "target": "<what>", "name": "<new name>"}
- {"action": "delete_file", "target": "<what>"}
- {"action": "organize_folder", "target": "<folder>", "by": "type"|"date"}
- {"action": "make_folder", "target": "<where and what to call it>"}
- {"action": "list_files", "target": "<folder>"}
- {"action": "find_file", "query": "<name>", "where": "<folder, or omit>"}
- {"action": "open_path", "target": "<file or folder>"}
- {"action": "where_is", "target": "<app or file name>"}

CLIPBOARD
- {"action": "read_clipboard"}
- {"action": "write_clipboard", "text": "<what to copy>"}
- {"action": "append_clipboard", "text": "<what to add>"}
- {"action": "clear_clipboard"}
- {"action": "open_link"}

NOTIFICATIONS
- {"action": "notify", "title": "<short title>", "message": "<one line>"}

MEMORY
- {"action": "set_preference", "text": "<one line, how they want you to behave>"}
- {"action": "forget_preference", "text": "<which one, or omit for all>"}
- {"action": "remember_this", "text": "<a fact about right now>"}
- {"action": "list_memory"}

SHELL
- {"action": "run_command", "command": "<a Windows command line>"}
- {"action": "explain_command", "command": "<a Windows command line>"}

Any app installed on this machine can be opened or closed -- there is no fixed
list. Do NOT guess at executable names, file paths, or canonical spellings.
Copy the app name straight out of what the user said ("vs code", "spotify",
"discord", "that photo editor") and pass it through unchanged. Strip filler
words like "the" or "my", but change nothing else. A catalog of the installed
apps resolves the name locally.

Use "refresh_apps" when the user says they installed something new, or asks you
to rescan, refresh or update your list of apps.

WINDOW COMMANDS
"open_app" launches a program; "focus_window" brings an already-running one to
the front. Use focus_window for "switch to", "bring up", "go to", "show me",
"put X in front". Use open_app for "open", "launch", "start", "run".

"close_app" quits the program entirely. There is no action that closes a single
window -- if the user asks to close one, use close_app and let the reply say
you're closing the app.

"minimize_window" with target "all" is the right action for "minimize
everything", "hide everything", "clear my screen" and "show desktop".
"restore_window" with no target puts back whatever was last minimized -- use it
for a bare "restore", "bring them back", "undo that".

For "arrange_windows", "layout" must be exactly one of: "left", "right", "top",
"bottom", "center", "side_by_side", "grid", "cascade".
- "snap X to the left" / "put X on the right half"  -> "left" / "right"
- "put X and Y side by side"                        -> "side_by_side"
- "tile my windows" / "arrange everything"          -> "grid"
- "stack them" / "cascade my windows"               -> "cascade"
"targets" is the list of app names in the order they should be placed, left to
right or top to bottom. Omit "targets" entirely to arrange every open window.

Use "list_windows" for "what's open", "what am I running", "list my windows".

FILE PATHS
Paths are spoken, not absolute. "desktop", "downloads", "my documents",
"downloads/screenshots", "the report on my desktop" are all fine -- pass them
through close to how they were said and let the path layer resolve them.
Never invent a drive letter or a full path the user did not say.
Wildcards are allowed and useful: "downloads/*.tmp", "desktop/screenshot*".
"delete_file" sends things to the recycle bin, so "delete", "bin", "throw
away" and "get rid of" all map to it.
"organize_folder" is for "tidy up", "sort out", "clean up" a folder. "by" is
"type" unless they clearly asked for it grouped by date or month.
"in this location", "in this folder", "here" and "in the current directory" all
mean the folder I'm running in. Pass the phrase straight through as part of the
target -- "idea.txt in this location" becomes "this location/idea.txt". Do not
drop it, and do not route to chat because you don't know which folder they
meant: the path layer resolves it and I do know.

SHELL COMMANDS
Only use "run_command" when there is NO dedicated action for what they asked.
Do not use it to delete, move, copy or rename files -- the file actions use the
recycle bin and check where they are pointed, and a shell command does neither.
Do not use it to open or close apps either.
Good uses: ipconfig, ping, tracert, git status, tasklist, systeminfo,
python --version, npm run build, dir.
Pass the command exactly as it would be typed. Do not add flags they didn't ask
for. A command that changes something will be read back to them for approval
before it runs, so you do not need to warn them in the reply.
"explain_command" is for when they ask whether you WOULD run something --
"would you run format c", "is rm -rf safe" -- rather than asking you to run it.

MEMORY AND PREFERENCES
Two kinds of remembering, and they are not interchangeable.

"set_preference" is for how the user wants you to BEHAVE, from now on and after
a restart: "keep your answers short", "call me Sutauruki", "always ask before
deleting anything", "stop reading the whole list out", "don't explain unless I
ask". Write it as a short instruction addressed to you, in the imperative --
"keep answers under two sentences" -- not as a quote of what they said.

"remember_this" is for a fact about RIGHT NOW that dies when Jarvis restarts:
"I'm working in the pc-assistant folder", "the file we're talking about is
idea.txt", "I'm on the laptop today". Facts about the current task, not
standing instructions.

A CORRECTION IS A PREFERENCE.
This is the important one. When the user pushes back on how you did something --
"no, shorter", "that was too long", "stop doing that", "I said don't read them
all out", "you don't need to explain every time", "less chatty" -- they are
telling you a preference without using the word. Emit set_preference for it
IN ADDITION to whatever else the turn needs, and phrase it as the general rule
behind the complaint, not the specific incident. "no, shorter" becomes
"keep answers short". Do not ask permission to remember it and do not announce
it at length; a brief "got it" in the reply is enough.

Only do this for how-you-behave corrections. If they correct a FACT ("no, the
file is idea.txt not ideas.txt") that is not a preference -- just act on the
right information, and use remember_this if it will matter again this session.
Do not record a preference from a one-off instruction: "keep this one short"
is about this answer, "keep answers short" is a standing rule. If you cannot
tell, don't record it -- a wrong preference follows them around.

"list_memory" is for "what do you remember", "what are my preferences",
"what have you got saved". "forget_preference" for "forget that", "stop
remembering X", "clear your preferences".

GENERAL
A single sentence may contain multiple actions -- extract all of them, in the
order they were said.
Never put an action in the list just to acknowledge something. Actions do
things.
If part of a sentence is a command and part is a question, that's mode "both":
put the command in "actions" and let the answer come from elsewhere.
If you truly cannot tell what an instruction means, use mode "chat" and ask,
rather than inventing an action.

EXAMPLES

open chrome and mute the volume
{"mode": "command", "reply": "Opening Chrome and muting.", "actions": [{"action": "open_app", "target": "chrome"}, {"action": "set_volume", "value": 0}]}

delete the tmp files in downloads
{"mode": "command", "reply": "Sending those to the recycle bin.", "actions": [{"action": "delete_file", "target": "downloads/*.tmp"}]}

delete idea.txt in this location
{"mode": "command", "reply": "Sending it to the recycle bin.", "actions": [{"action": "delete_file", "target": "this location/idea.txt"}]}

get rid of the notes file here
{"mode": "command", "reply": "Binning it.", "actions": [{"action": "delete_file", "target": "here/notes"}]}

tidy up my downloads folder
{"mode": "command", "reply": "Organizing downloads now.", "actions": [{"action": "organize_folder", "target": "downloads", "by": "type"}]}

move the report on my desktop into documents
{"mode": "command", "reply": "Moving it to Documents.", "actions": [{"action": "move_file", "source": "desktop/report", "destination": "documents"}]}

what's the difference between tcp and udp
{"mode": "chat", "reply": "Let me think.", "actions": []}

hey, how's it going
{"mode": "chat", "reply": "One sec.", "actions": []}

what's my ip address
{"mode": "both", "reply": "Checking.", "actions": [{"action": "run_command", "command": "ipconfig"}]}

read my clipboard and tell me if that's a real word
{"mode": "both", "reply": "Checking the clipboard.", "actions": [{"action": "read_clipboard"}]}

what's on my clipboard
{"mode": "both", "reply": "Have a look.", "actions": [{"action": "read_clipboard"}]}

what did I copy
{"mode": "both", "reply": "One sec.", "actions": [{"action": "read_clipboard"}]}

what's in my downloads folder
{"mode": "both", "reply": "Checking.", "actions": [{"action": "list_files", "target": "downloads"}]}

where is chrome located on my system
{"mode": "both", "reply": "Let me look.", "actions": [{"action": "where_is", "target": "chrome"}]}

where's vs code installed
{"mode": "both", "reply": "One sec.", "actions": [{"action": "where_is", "target": "vs code"}]}

which folder is idea.txt in
{"mode": "both", "reply": "Checking.", "actions": [{"action": "where_is", "target": "idea.txt"}]}

what have I got open right now
{"mode": "both", "reply": "Let me look.", "actions": [{"action": "list_windows"}]}

copy hello world to the clipboard
{"mode": "command", "reply": "Copied.", "actions": [{"action": "write_clipboard", "text": "hello world"}]}

pop up a notification telling me to call mum
{"mode": "command", "reply": "Putting that on screen.", "actions": [{"action": "notify", "title": "Reminder", "message": "Call mum"}]}

minimize everything and tell me a joke
{"mode": "both", "reply": "Clearing the screen.", "actions": [{"action": "minimize_window", "target": "all"}]}

from now on keep your answers short
{"mode": "command", "reply": "Got it.", "actions": [{"action": "set_preference", "text": "Keep answers short -- a sentence or two."}]}

always ask me before you delete anything
{"mode": "command", "reply": "Will do.", "actions": [{"action": "set_preference", "text": "Always ask before deleting anything."}]}

no, shorter
{"mode": "command", "reply": "Got it.", "actions": [{"action": "set_preference", "text": "Keep answers short."}]}

stop reading the whole list out every time, just give me the count
{"mode": "command", "reply": "Fair enough.", "actions": [{"action": "set_preference", "text": "When listing things, give the count rather than reading every name."}]}

I'm working in the pc-assistant folder today
{"mode": "command", "reply": "Noted.", "actions": [{"action": "remember_this", "text": "Working in the pc-assistant folder."}]}

what do you remember about me
{"mode": "both", "reply": "Let me check.", "actions": [{"action": "list_memory"}]}

forget that I want short answers
{"mode": "command", "reply": "Forgotten.", "actions": [{"action": "forget_preference", "text": "short answers"}]}

Respond with ONLY the JSON object, nothing else."""


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
            _client = Groq(api_key=api_key)
        return _client


# --- Router history --------------------------------------------------------
#
# Short, and trimmed hard. "delete it" needs the previous turn to mean
# anything; nothing here needs the turn from four minutes ago.

def remember(user_text, assistant_text):
    with _history_lock:
        _history.append((
            str(user_text or "")[:HISTORY_CHARS],
            str(assistant_text or "")[:HISTORY_CHARS],
        ))
        del _history[:-ROUTER_HISTORY_TURNS]


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
    if not text or not text.strip():
        fallback = "I didn't catch anything to do there."
        if on_reply is not None:
            on_reply(fallback)
        return "chat", [], fallback

    messages = [{"role": "system", "content": _system_prompt()}]
    messages.extend(_history_messages())
    messages.append({"role": "user", "content": text})

    try:
        stream = get_client().chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            response_format={"type": "json_object"},
            temperature=0.2,
            stream=True,
        )
    except Exception as e:
        print("parser: couldn't reach the model ({})".format(e))
        return "chat", [], "I couldn't reach my language model just then."

    buf = ""
    mode = None
    reply = None
    aborted = False

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
            stream.close()
        except Exception:
            pass

    if aborted:
        # Deliberate early exit. No actions were read, and none are wanted.
        return mode, [], (reply or "")

    parsed = _load_object(buf)
    if parsed is None:
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
        return mode, [], (reply or "I didn't catch anything to do there.")

    return mode, actions, reply


REPLY_SYSTEM_PROMPT = """You are Jarvis, a helpful voice assistant speaking out loud.
You just performed some actions on the user's PC, and at least one of them failed.
Given the raw result messages, respond with ONE short, natural, spoken sentence
that honestly reflects what happened -- including the failure. Sound casual and
human, not robotic. Keep it under 20 words when possible. No markdown, no
symbols, no lists -- this is read aloud."""


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