"""Streaming intent parse with reply-first extraction.

Goes in: commands/parser.py  (replaces existing)

Only the system prompt changed from the previous version: the fixed list of
four valid app names is gone. The model now passes the spoken app name through
verbatim and commands/app_index.py resolves it against what's really installed.
That keeps the prompt the same size no matter how much software is on the
machine, and the local resolver handles nicknames and mishearings better than
the model can guess at them.
"""

import json
import os
import threading

from dotenv import load_dotenv
from groq import Groq

load_dotenv()

MODEL_NAME = "openai/gpt-oss-20b"

_client = None
_client_lock = threading.Lock()

# Key order matters here. We stream this response and start speaking the
# moment "reply" is complete, without waiting for "actions" to finish
# generating -- so "reply" has to come out of the model first.
SYSTEM_PROMPT = """You are Jarvis, a voice-controlled PC assistant with a friendly,
natural speaking voice. Convert the user's spoken text into a JSON object with
exactly two keys, emitted in THIS ORDER: "reply" first, then "actions".

The order is mandatory. We begin speaking "reply" while the rest of your
response is still streaming, so it must be the first key in the object.

"reply": ONE short, natural, spoken-style sentence confirming what you're about
to do -- casual and human, like a helpful assistant talking out loud, not a
robotic list. Assume the actions will succeed. Keep it under 20 words.
If there is nothing actionable, say you didn't catch anything to do.

"actions": a list of structured commands, each one of these exact shapes:
- {"action": "open_app", "target": "<app name as the user said it>"}
- {"action": "close_app", "target": "<app name as the user said it>"}
- {"action": "set_volume", "value": <0-100 integer>}
- {"action": "take_screenshot"}
- {"action": "refresh_apps"}
- {"action": "unknown", "raw": "<original phrase that wasn't understood>"}

Any app installed on this machine can be opened or closed -- there is no fixed
list. Do NOT guess at executable names, file paths, or canonical spellings.
Copy the app name straight out of what the user said ("vs code", "spotify",
"discord", "that photo editor") and pass it through unchanged. Strip filler
words like "the" or "my", but change nothing else. A catalog of the installed
apps resolves the name locally.

Use "refresh_apps" when the user says they installed something new, or asks you
to rescan, refresh or update your list of apps.

A single sentence may contain multiple actions -- extract all of them in order.
If nothing actionable is said, "actions" should be an empty list.

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
    global _client
    with _client_lock:
        if _client is None:
            api_key = os.getenv("GROQ_API_KEY")
            if not api_key:
                raise ValueError("GROQ_API_KEY not found in .env")
            _client = Groq(api_key=api_key)
        return _client


def _extract_string_value(buf, key):
    """Pulls a complete string value for `key` out of a partial JSON buffer.

    Returns None while the value is still arriving, so the caller can just
    retry on the next token. This is what lets us start speaking before the
    model has finished writing the actions array.
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


def parse_command(text, on_reply=None):
    """Takes transcribed text, returns (actions, reply).

    on_reply is invoked as soon as the reply sentence is fully formed --
    typically several hundred ms before the full response lands. Start TTS
    from that callback.
    """
    if not text or not text.strip():
        fallback = "I didn't catch anything to do there."
        if on_reply is not None:
            on_reply(fallback)
        return [], fallback

    client = get_client()

    stream = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": text},
        ],
        response_format={"type": "json_object"},
        stream=True,
    )

    buf = ""
    reply = None

    for chunk in stream:
        if not chunk.choices:
            continue
        delta = chunk.choices[0].delta.content or ""
        if not delta:
            continue
        buf += delta

        if reply is None:
            found = _extract_string_value(buf, "reply")
            if found:
                reply = found
                if on_reply is not None:
                    on_reply(reply)

    try:
        parsed = json.loads(buf)
    except json.JSONDecodeError:
        print("Failed to parse Groq response as JSON: " + buf)
        return [], reply or "Something went wrong parsing that, sorry."

    return parsed.get("actions", []), reply or parsed.get("reply", "Done.")


REPLY_SYSTEM_PROMPT = """You are Jarvis, a helpful voice assistant speaking out loud.
You just performed some actions on the user's PC, and at least one of them failed.
Given the raw result messages, respond with ONE short, natural, spoken sentence
that honestly reflects what happened -- including the failure. Sound casual and
human, not robotic. Keep it under 20 words when possible."""


def generate_reply(results):
    """Fallback only -- used when something actually failed during execution
    and the pre-generated reply would be inaccurate."""
    if not results:
        return "I didn't catch anything to do there."

    client = get_client()
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": REPLY_SYSTEM_PROMPT},
            {"role": "user", "content": "\n".join(results)},
        ],
    )
    return response.choices[0].message.content.strip()