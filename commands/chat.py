"""The conversation brain -- free models on OpenRouter.

Goes in: commands/chat.py   (new file)

    .env
    OPENROUTER_API_KEY=sk-or-v1-...

Groq's gpt-oss-20b stays where it is, doing what it is good at: deciding in a
few hundred milliseconds whether you asked a question, gave an order, or both.
It is a router. When the answer is "this is a question", the question comes
here instead, and a larger free model answers it.

WHY THE MODEL LIST IS FETCHED, NOT HARDCODED
--------------------------------------------
OpenRouter's free roster turns over constantly -- models appear, get renamed,
lose their free tier. A hardcoded list is stale within a month and fails in the
least helpful way: 404 on every entry. So the list is pulled from
/api/v1/models, filtered to genuinely-free entries, ranked, and cached for a
few hours. PREFERRED below is only a ranking hint and a fallback for when the
listing call itself fails.

WHAT BRAHMA ECHO GOT WRONG HERE, AND WHY THIS IS DIFFERENT
----------------------------------------------------------
Brahma walks 22 free models with 2 retries and a 60-second timeout each, with
no overall deadline -- a bad afternoon on OpenRouter is a 44-minute wall of
silence. Three things prevent that here:

  - TOTAL_DEADLINE caps the entire chain. When it passes, it stops trying and
    says so, because a voice assistant that has said nothing for twenty seconds
    has already failed no matter what it eventually returns.
  - A model that rate-limits goes into a cooldown dict and is skipped until it
    expires, so the next question doesn't pay for the same 429 again.
  - Groq is the last resort. Slightly weaker for open conversation, but it is
    the key you know works, and something spoken beats nothing.

Brahma also has a bug worth not repeating: `_use_openrouter_first` is set True
on the first rate limit and never reset, so one hiccup degrades the whole
session until restart. Every bit of state here is either time-bounded or reset
on success.

STREAMING IS THE POINT
----------------------
`answer()` calls back on each complete sentence. listener/tts.py speaks
sentence by sentence, so you hear the first words while the model is still
writing the rest -- the same trick that makes command replies feel instant,
applied to answers that are five times longer.
"""

import json
import os
import re
import threading
import time
from collections import deque

from dotenv import load_dotenv

load_dotenv()

API_BASE = "https://openrouter.ai/api/v1"

# Whole-chain budget. Past this, stop and say so.
TOTAL_DEADLINE = 22.0

# Per-model budgets. First-token is the one that matters -- a model that hasn't
# started after 9 seconds is not going to finish in time to be worth waiting on.
FIRST_TOKEN_TIMEOUT = 9.0
STREAM_TIMEOUT = 30.0

RATE_LIMIT_COOLDOWN = 90.0
ERROR_COOLDOWN = 300.0

MODEL_CACHE_SECONDS = 6 * 3600
MAX_CHAIN = 4          # models tried before falling back to Groq
MAX_TOKENS = 400

# Rolling context. Six exchanges is enough for "what about the second one" and
# "do that again" without the prompt growing without bound.
HISTORY_TURNS = 6

# Ranking hints, best first. Substring match against the model id, so
# "deepseek-r1" matches whatever suffix the roster is using this month.
PREFERRED = (
    "deepseek-v3", "deepseek-chat", "deepseek-r1",
    "llama-4", "llama-3.3-70b", "qwen3", "qwen-2.5-72b", "qwen2.5-72b",
    "mistral-large", "mistral-small-3", "gemma-3-27b", "glm-4", "kimi",
    "phi-4", "gpt-oss-120b", "gpt-oss-20b",
)

# A free model id is a moving target: the roster turns over, and when a
# specialist model turns up with "content-safety" in its name it is ranked
# first and asked to hold a conversation. It cannot. Anything matching a
# disqualified keyword is excluded outright, so the safety classifiers that
# actually answered real questions with "User Safety: safe" can never be
# selected again even when OpenRouter labels them free.
_DISQUALIFYING = (
    "content-safety", "guard", "moderation", "moderator", "classifier",
    "safety", "shield", "reward", "judge", "embed", "rerank",
)

# Used only if the listing call fails outright.
FALLBACK_MODELS = (
    "deepseek/deepseek-chat-v3-0324:free",
    "meta-llama/llama-3.3-70b-instruct:free",
    "qwen/qwen-2.5-72b-instruct:free",
    "google/gemma-3-27b-it:free",
)

SYSTEM_PROMPT = """You are Jarvis, a voice assistant. Everything you say is read
aloud by a speech synthesiser, so write the way a person talks, not the way a
person types.

Rules that matter because this is spoken, not read:
- No markdown. No asterisks, no hashes, no bullet points, no numbered lists,
  no code blocks, no emoji. They get read out as literal symbols or dropped.
- Two or three sentences is the normal length. Only go longer when the user
  clearly asked for detail, and even then keep it flowing as prose.
- If you'd naturally write a list, say it as a sentence instead: "there are
  three -- x, y and z" rather than three bullets.
- Numbers, units and symbols spelled the way you'd say them. "About twenty
  percent", not "~20%".
- Contractions are good. You're talking, not writing a report.

You're running on the user's Windows PC and you can also open and close apps,
manage windows, move and delete files, read the clipboard, take screenshots and
run shell commands. If the user asks you to *do* one of those, it's already
being handled elsewhere -- your job is only the talking part. Don't claim you
can't do things you can do.

If you don't know something, say so briefly. Don't pad, don't hedge twice,
don't apologise more than once."""

_history = deque(maxlen=HISTORY_TURNS)
_history_lock = threading.Lock()


def _system_prompt():
    """The base prompt plus whatever the user has told us they prefer.

    Built per call rather than once at import, because a preference set two
    turns ago has to affect the next answer -- baking it in at startup would
    mean "keep it shorter" only took effect after a restart, which is the
    opposite of what saying it out loud implies.
    """
    try:
        from commands import memory
        block = memory.prompt_block()
    except Exception as e:  # a broken prefs file must not break conversation
        print("chat: couldn't load preferences ({})".format(e))
        block = ""

    if not block:
        return SYSTEM_PROMPT
    return SYSTEM_PROMPT + "\n\n" + block

_models_cache = None
_models_fetched = 0.0
_cooldown = {}
_state_lock = threading.Lock()


# --- HTTP ------------------------------------------------------------------
#
# requests if present, httpx otherwise. httpx is a hard dependency of the groq
# SDK, so one of the two is always there -- which means this feature needs no
# pip install of its own.

def _http():
    try:
        import requests
        return "requests", requests
    except ImportError:
        pass
    try:
        import httpx
        return "httpx", httpx
    except ImportError:
        raise RuntimeError(
            "chat needs the requests or httpx package (httpx ships with groq)"
        )


def api_key():
    key = os.getenv("OPENROUTER_API_KEY")
    if not key:
        raise RuntimeError("OPENROUTER_API_KEY not found in .env")
    return key


def _headers():
    return {
        "Authorization": "Bearer {}".format(api_key()),
        "Content-Type": "application/json",
        # OpenRouter uses these for its leaderboard; harmless, and it stops the
        # request being attributed to nobody.
        "HTTP-Referer": "https://localhost/jarvis",
        "X-Title": "Jarvis",
    }


def available():
    return bool(os.getenv("OPENROUTER_API_KEY"))


# --- Model discovery -------------------------------------------------------

def _is_free(entry):
    pricing = entry.get("pricing") or {}
    try:
        return (float(pricing.get("prompt", 1)) == 0.0
                and float(pricing.get("completion", 1)) == 0.0)
    except (TypeError, ValueError):
        return False


def _rank(model_id, entry):
    """Lower sorts first."""
    lowered = model_id.lower()
    for index, hint in enumerate(PREFERRED):
        if hint in lowered:
            return (0, index)

    # Unlisted models: prefer bigger context, which correlates well enough with
    # capability to be a better tiebreak than alphabetical.
    context = entry.get("context_length") or 0
    return (1, -int(context))


def _fetch_models():
    kind, lib = _http()
    url = API_BASE + "/models"

    if kind == "requests":
        response = lib.get(url, headers=_headers(), timeout=10)
        response.raise_for_status()
        payload = response.json()
    else:
        with lib.Client(timeout=10) as client:
            response = client.get(url, headers=_headers())
            response.raise_for_status()
            payload = response.json()

    entries = payload.get("data") or []
    free = [
        (entry.get("id"), entry) for entry in entries
        if entry.get("id") and _is_free(entry)
    ]

    # Drop the ones that can't hold a conversation: image/video/audio-only
    # endpoints, anything explicitly marked as a base (non-instruct) model,
    # and anything whose id smells like a classifier or safety gate rather
    # than a chat model.
    usable = []
    for model_id, entry in free:
        lowered = model_id.lower()
        if any(word in lowered for word in
               ("whisper", "tts", "embed", "rerank", "vision-only", "-base",
                "flux", "sdxl", "stable-diffusion")):
            continue
        if any(word in lowered for word in _DISQUALIFYING):
            print("chat: skipping {!r} -- not a conversation model".format(model_id))
            continue
        usable.append((model_id, entry))

    usable.sort(key=lambda pair: _rank(pair[0], pair[1]))
    return [model_id for model_id, _entry in usable]


def models(refresh=False):
    """Free model ids, best first. Cached; never raises."""
    global _models_cache, _models_fetched

    with _state_lock:
        fresh = (_models_cache is not None
                 and time.time() - _models_fetched < MODEL_CACHE_SECONDS)
        if fresh and not refresh:
            return list(_models_cache)

    try:
        found = _fetch_models()
    except Exception as e:
        print("chat: couldn't list free models ({}), using the built-in list".format(e))
        found = list(FALLBACK_MODELS)

    if not found:
        found = list(FALLBACK_MODELS)

    with _state_lock:
        _models_cache = found
        _models_fetched = time.time()
    return list(found)


def _cool(model, seconds):
    with _state_lock:
        _cooldown[model] = time.time() + seconds


def _is_cool(model):
    with _state_lock:
        until = _cooldown.get(model, 0.0)
        if until and until < time.time():
            del _cooldown[model]
            return False
        return bool(until)


def _chain():
    """The models to try this turn, cooled-down ones removed."""
    ordered = [m for m in models() if not _is_cool(m)]
    if not ordered:
        # Everything is cooling. Clear the oldest few rather than giving up --
        # a stale cooldown shouldn't outlive the problem that caused it.
        with _state_lock:
            _cooldown.clear()
        ordered = models()
    return ordered[:MAX_CHAIN]


# --- Sentence splitting ----------------------------------------------------
#
# Speech starts per sentence, so the split has to survive abbreviations. A naive
# split on "." cuts "Dr. Who" in half and the synthesiser reads the halves with
# two different intonations, which sounds broken.

_ABBREV = {"mr", "mrs", "ms", "dr", "prof", "sr", "jr", "st", "vs", "etc",
           "e.g", "i.e", "approx", "no", "fig", "inc", "ltd", "co", "u.s",
           "a.m", "p.m"}

_SENTENCE_END = re.compile(r"([.!?])(\s|$)")


def _split_sentences(buffer):
    """(complete_sentences, remainder)."""
    out = []
    start = 0
    for match in _SENTENCE_END.finditer(buffer):
        end = match.end(1)
        candidate = buffer[start:end].strip()
        if not candidate:
            continue

        # Don't cut after an abbreviation or a lone initial.
        word = re.split(r"[\s(]", candidate)[-1].rstrip(".").lower()
        if word in _ABBREV or (len(word) == 1 and word.isalpha()):
            continue
        # Or after a decimal point mid-number.
        if end < len(buffer) and buffer[end - 1] == "." and buffer[end:end + 1].isdigit():
            continue

        out.append(candidate)
        start = match.end()

    return out, buffer[start:]


# --- Cleanup ---------------------------------------------------------------

_MARKDOWN = (
    (re.compile(r"```.*?```", re.S), " "),
    (re.compile(r"`([^`]*)`"), r"\1"),
    (re.compile(r"\*\*([^*]+)\*\*"), r"\1"),
    (re.compile(r"(?<!\w)\*([^*\n]+)\*(?!\w)"), r"\1"),
    (re.compile(r"(?<!\w)_([^_\n]+)_(?!\w)"), r"\1"),
    (re.compile(r"^\s{0,3}#{1,6}\s*", re.M), ""),
    (re.compile(r"^\s{0,3}[-*+]\s+", re.M), ""),
    (re.compile(r"^\s{0,3}>\s?", re.M), ""),
    (re.compile(r"\[([^\]]+)\]\([^)]*\)"), r"\1"),
)


def despeak(text):
    """Strips the markdown the prompt asked it not to produce.

    Models emit bullets and bold anyway, especially when the question invites a
    list. Unstripped, the synthesiser reads "asterisk asterisk" out loud.
    """
    if not text:
        return ""
    for pattern, replacement in _MARKDOWN:
        text = pattern.sub(replacement, text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = re.sub(r"\n{2,}", "\n", text)
    return text.strip()


def _is_classifier_garbage(text):
    """True if a reply looks like a safety-classifier verdict instead of an
    answer.

    The model blacklist in _fetch_models is the real defence -- a classifier
    should never be selected. This is the backstop for a model that slips
    through anyway: a classifier verdict fits on one line and mentions
    "safety", and we would rather spend a second round trip than say it.
    """
    if not text:
        return False
    lowered = " ".join(text.lower().split())
    if len(lowered) > 200 or "\n" in lowered:
        return False
    return ("user safety" in lowered or "response safety" in lowered
            or lowered.startswith("safe"))


# --- History ---------------------------------------------------------------

def remember(user_text, assistant_text):
    if not user_text and not assistant_text:
        return
    with _history_lock:
        _history.append({
            "user": (user_text or "").strip(),
            "assistant": (assistant_text or "").strip(),
        })


def recent(limit=HISTORY_TURNS):
    with _history_lock:
        return list(_history)[-limit:]


def forget():
    with _history_lock:
        _history.clear()


def _messages(question, history):
    out = [{"role": "system", "content": _system_prompt()}]
    for turn in history or recent():
        if turn.get("user"):
            out.append({"role": "user", "content": turn["user"]})
        if turn.get("assistant"):
            out.append({"role": "assistant", "content": turn["assistant"]})
    out.append({"role": "user", "content": question})
    return out


# --- Streaming -------------------------------------------------------------

def _iter_sse(kind, lib, payload, deadline):
    """Yields content deltas from one model. Raises on failure."""
    url = API_BASE + "/chat/completions"
    headers = _headers()
    body = dict(payload, stream=True)

    remaining = max(1.0, deadline - time.time())
    read_timeout = min(STREAM_TIMEOUT, remaining)

    if kind == "requests":
        timeout = (6.0, min(FIRST_TOKEN_TIMEOUT, read_timeout))
        response = lib.post(url, headers=headers, json=body,
                            stream=True, timeout=timeout)
        if response.status_code >= 400:
            raise _http_error(response.status_code, response.text)
        lines = response.iter_lines(decode_unicode=True)
        try:
            for line in lines:
                for delta in _parse_sse_line(line):
                    yield delta
        finally:
            response.close()
        return

    timeout = lib.Timeout(read_timeout, connect=6.0,
                          read=min(FIRST_TOKEN_TIMEOUT, read_timeout))
    with lib.Client(timeout=timeout) as client:
        with client.stream("POST", url, headers=headers, json=body) as response:
            if response.status_code >= 400:
                raise _http_error(response.status_code, response.read().decode(
                    "utf-8", "replace"))
            for line in response.iter_lines():
                for delta in _parse_sse_line(line):
                    yield delta


def _parse_sse_line(line):
    if not line:
        return
    if isinstance(line, bytes):
        line = line.decode("utf-8", "replace")
    line = line.strip()
    if not line.startswith("data:"):
        return

    data = line[5:].strip()
    if not data or data == "[DONE]":
        return

    try:
        chunk = json.loads(data)
    except ValueError:
        return

    # An error can arrive mid-stream after a 200, which is why this is checked
    # here and not only on the status code.
    if chunk.get("error"):
        raise RuntimeError(str(chunk["error"])[:200])

    for choice in chunk.get("choices") or []:
        delta = (choice.get("delta") or {}).get("content")
        if delta:
            yield delta


class _Rate(RuntimeError):
    pass


class _NotAChatModel(RuntimeError):
    """A model that turned out to be a classifier or similar specialist, and
    announced it before producing an answer. Treated like any other model
    failure -- cooled down and skipped -- but worth its own name in the log."""


def _opens_like_a_verdict(buffer):
    """Early-abort probe: does an unspoken buffer look like a classifier
    verdict opening? Only ever called before the first sentence fires, so the
    cost is nothing on a normal stream and the benefit is never saying a
    verdict out loud."""
    if not buffer:
        return False
    probe = " ".join(buffer.strip().lower().split())
    return "user safety" in probe or "response safety" in probe


def _http_error(status, text):
    snippet = (text or "").strip()[:200]
    if status == 429:
        return _Rate("rate limited")
    if status in (401, 403):
        return RuntimeError("key rejected ({}): {}".format(status, snippet))
    return RuntimeError("HTTP {}: {}".format(status, snippet))


def _ask_one(model, messages, on_sentence, deadline):
    """One model, streamed. Returns the full answer or raises."""
    kind, lib = _http()
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": MAX_TOKENS,
        "temperature": 0.7,
    }

    buffer = ""
    spoken = []
    started = None

    for delta in _iter_sse(kind, lib, payload, deadline):
        if started is None:
            started = time.time()
        buffer += delta

        # Bail before the first word is spoken, not after. A classifier
        # announces itself in its opening tokens ("User Safety: ..."), and
        # once on_sentence has fired the synthesiser is already saying it --
        # checking the finished answer would be a check that arrives too late
        # to stop the sound.
        if not spoken and _opens_like_a_verdict(buffer):
            raise _NotAChatModel("returned a safety verdict, not an answer")

        if on_sentence is not None:
            sentences, buffer = _split_sentences(buffer)
            for sentence in sentences:
                cleaned = despeak(sentence)
                if cleaned:
                    spoken.append(cleaned)
                    on_sentence(cleaned)
            continue

        if time.time() > deadline + STREAM_TIMEOUT:
            break

    tail = despeak(buffer)
    if tail:
        # The verdict has no full stop, so it never splits into a sentence and
        # arrives here whole. Last chance to catch it before it is spoken.
        if not spoken and _is_classifier_garbage(tail):
            raise _NotAChatModel("returned a safety verdict, not an answer")
        spoken.append(tail)
        if on_sentence is not None:
            on_sentence(tail)

    answer = " ".join(part for part in spoken if part).strip()
    if not answer:
        raise RuntimeError("empty response")
    return answer


def _groq_fallback(messages):
    """Last resort. Not streamed -- by this point speed has already been lost,
    and the simpler path is the one least likely to fail too."""
    from commands import parser  # local import: avoids an import cycle

    client = parser.get_client()
    response = client.chat.completions.create(
        model=parser.MODEL_NAME,
        messages=messages,
        max_tokens=MAX_TOKENS,
        temperature=0.7,
    )
    return despeak(response.choices[0].message.content or "")


def answer(question, history=None, on_sentence=None, save=True):
    """Answers a conversational question. Never raises.

    on_sentence fires with each complete, markdown-stripped sentence as it
    arrives -- hand it to listener/tts.speak so the reply starts playing while
    the model is still writing.
    """
    question = (question or "").strip()
    if not question:
        return "I didn't catch that."

    messages = _messages(question, history)
    deadline = time.time() + TOTAL_DEADLINE
    tried = []

    if available():
        for model in _chain():
            if time.time() > deadline:
                print("chat: out of time after {}".format(", ".join(tried)))
                break

            try:
                started = time.time()
                reply = _ask_one(model, messages, on_sentence, deadline)
                print("chat: {} in {:.1f}s".format(model, time.time() - started))
                if save:
                    remember(question, reply)
                return reply
            except _NotAChatModel as e:
                # Cooled for the long interval, not the short one: this is not
                # a transient failure, it is the wrong kind of model, and it
                # will still be the wrong kind in ninety seconds.
                print("chat: {} {}".format(model, e))
                _cool(model, ERROR_COOLDOWN)
                tried.append(model)
            except _Rate:
                print("chat: {} is rate limited".format(model))
                _cool(model, RATE_LIMIT_COOLDOWN)
                tried.append(model)
            except Exception as e:
                print("chat: {} failed ({})".format(model, e))
                _cool(model, ERROR_COOLDOWN)
                tried.append(model)
    else:
        print("chat: OPENROUTER_API_KEY not set, using Groq")

    try:
        reply = _groq_fallback(messages)
        if reply:
            if save:
                remember(question, reply)
            if on_sentence is not None:
                # The fallback isn't streamed, so feed it through in sentences
                # anyway -- the caller's plumbing shouldn't care which path ran.
                sentences, tail = _split_sentences(reply + " ")
                for sentence in sentences + ([tail.strip()] if tail.strip() else []):
                    on_sentence(sentence)
            return reply
    except Exception as e:
        print("chat: Groq fallback failed too ({})".format(e))

    failure = "I couldn't reach a model to answer that just now."
    if on_sentence is not None:
        on_sentence(failure)
    return failure


def status():
    """Diagnostic line for the log."""
    if not available():
        return "chat: no OPENROUTER_API_KEY -- conversation falls back to Groq"
    ordered = models()
    cooling = sum(1 for m in ordered if _is_cool(m))
    return "chat: {} free models, {} cooling, first choice {}".format(
        len(ordered), cooling, ordered[0] if ordered else "(none)"
    )


def warm():
    """Fetches the model list on a background thread at startup, so the first
    question doesn't pay for it."""
    if not available():
        return

    def run():
        try:
            models()
            print(status())
        except Exception as e:
            print("chat: warm-up failed ({})".format(e))

    threading.Thread(target=run, daemon=True).start()


if __name__ == "__main__":
    # python -m commands.chat  -- one question, streamed to stdout.
    import sys

    print(status())
    prompt = " ".join(sys.argv[1:]) or "What's the difference between RAM and VRAM?"
    print("\nQ: {}\n".format(prompt))

    def show(sentence):
        print("  > {}".format(sentence))

    result = answer(prompt, on_sentence=show)
    print("\nFull answer:\n{}".format(result))