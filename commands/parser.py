import os
import json
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

MODEL_NAME = "openai/gpt-oss-20b"

_client = None

SYSTEM_PROMPT = """You are Jarvis, a voice-controlled PC assistant with a friendly,
natural speaking voice. Convert the user's spoken text into a JSON object with two keys:

"actions": a list of structured commands, each one of these exact shapes:
- {"action": "open_app", "target": "<app_name>"}
- {"action": "close_app", "target": "<app_name>"}
- {"action": "set_volume", "value": <0-100 integer>}
- {"action": "take_screenshot"}
- {"action": "unknown", "raw": "<original phrase that wasn't understood>"}

Valid app_name values: "chrome", "notepad", "calculator", "spotify"
If the user mentions an app not in this list, still use open_app/close_app
with their given name — we'll handle unsupported apps downstream.
A single sentence may contain multiple actions — extract all of them in order.
If nothing actionable is said, "actions" should be an empty list.

"reply": ONE short, natural, spoken-style sentence confirming what you're about
to do — casual and human, like a helpful assistant talking out loud, not a
robotic list. Assume the actions will succeed. Keep it under 20 words.
If "actions" is empty, "reply" should say you didn't catch anything to do.

Respond with ONLY the JSON object, nothing else."""


def get_client():
    global _client
    if _client is None:
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY not found in .env")
        _client = Groq(api_key=api_key)
    return _client


def parse_command(text):
    """Takes transcribed text, returns (actions, reply) — the structured
    action list plus a ready-to-speak natural confirmation, both generated
    in a single Groq call to minimize latency."""
    if not text or not text.strip():
        return [], "I didn't catch anything to do there."

    client = get_client()

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": text},
        ],
        response_format={"type": "json_object"},
    )

    raw = response.choices[0].message.content

    try:
        parsed = json.loads(raw)
        return parsed.get("actions", []), parsed.get("reply", "Done.")
    except json.JSONDecodeError:
        print(f"Failed to parse Groq response as JSON: {raw}")
        return [], "Something went wrong parsing that, sorry."


REPLY_SYSTEM_PROMPT = """You are Jarvis, a helpful voice assistant speaking out loud.
You just performed some actions on the user's PC, and at least one of them failed.
Given the raw result messages, respond with ONE short, natural, spoken sentence
that honestly reflects what happened — including the failure. Sound casual and
human, not robotic. Keep it under 20 words when possible."""


def generate_reply(results):
    """Fallback only — used when something actually failed during execution
    and the pre-generated reply from parse_command would be inaccurate."""
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