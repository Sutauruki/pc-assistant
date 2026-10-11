"""The content brain -- one shared entry to a generation model for the work
that is more than a status line: generating a document outline, a slide deck, a
spreadsheet, a website, or summarising a transcript / search results / a file.

Goes in: commands/genbrain.py   (new file)

WHY A SHARED MODULE
-------------------
Six Tier-6 features are "prompt in, rich content out" -- docs, slides, sheets,
PDF export, the website builder, and every summariser. Before this module the
only Gemini caller was screen_actions.py, and each new feature would have
re-implemented the same client, the same JSON salvage, the same spoken-answer
rules. That is how retry logic drifts out of sync. This module is the one place
that talks to a generation model for content; feature modules describe *what*
they want and get back either a parsed structure (for the generators) or a
spoken line (for the summarisers).

THE PROVIDER CHAIN
------------------
Content generation normally runs on Gemini (GEMINI_API_KEY). Gemini's free tier
is small and rate-limited, so nothing here assumes it will answer: the chain is
Gemini first, then Groq (GROQ_API_KEY, the same key the router already uses),
then the OpenRouter free roster (OPENROUTER_API_KEY). A failure or a 429 on one
provider just moves to the next, and the caller never sees the difference -- it
gets JSON or a spoken line, or a raised error only when every provider failed.
The OpenRouter leg reuses commands/chat.py's model discovery and cooldowns, so
a rate limit on one free model never costs the next.

WHY BRAHMA GOT THIS WRONG, AND WHY THE CHAIN IS SHAPED THIS WAY
--------------------------------------------------------------
Brahma treated each provider as a separate feature with its own hardcoded call
and no cross-provider fallback -- one quota exhausted meant "Couldn't draft
the report", full stop. The shaped failure here is the opposite: retry is a
property of the whole chain, capped by MAX_ATTEMPTS for output-shape problems
(truncation, prose instead of JSON) and unbounded across providers for
availability problems (429, key rejected, network).

TWO KINDS OF WORK
-----------------
`structured()` returns parsed JSON for the document/spreadsheet/slides/website
generators. `summarise()` returns a short spoken-style sentence for the
view-and-tell features (file analysis, YouTube transcripts, web search). Same
chain, different output shape, same failure contract.

NEVER RAISES PAST THE CALLER
----------------------------
These functions raise. Feature modules wrap them and return a spoken string
containing "couldn't"/"failed"/"didn't" (main.py scans results for those words
to decide whether to spend a second round trip on a correction). Raising here
keeps the feature's own message specific -- "Couldn't generate the outline" --
rather than a one-size-fits-all apology. The chain's own message says which
providers failed and why, so the reason is not hidden.

    .env
    GEMINI_API_KEY=sk-...
    GROQ_API_KEY=gsk_...
    OPENROUTER_API_KEY=sk-or-v1-...
"""

import json
import os
import re
import time

from dotenv import load_dotenv

from commands import devlog

load_dotenv()

MODEL_NAME = os.getenv("GENBRAIN_MODEL", "gemini-3.6-flash")

# Groq model for structured generation. Same family the router uses to keep a
# key we already know works; gpt-oss-20b is cheaper than the 120b and fine for
# outline JSON.
GROQ_MODEL = os.getenv("GENBRAIN_GROQ_MODEL", "openai/gpt-oss-20b")

# Retries before giving up on a structured generation. Each attempt runs
# through the whole provider chain again, so this bounds the retry even when
# one provider keeps producing prose.
MAX_ATTEMPTS = 2

# Summarisers work off a hard cap of transcript/result text. Enough to be
# meaningful, not so much that a request crawls or the answer becomes a wall.
SUMMARY_INPUT_CHARS = 16000
MAX_TOKENS = 4096

_client = None


def client():
    """One google.genai client, built on first use. Raises ValueError early."""
    global _client
    if _client is None:
        from google import genai

        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY not found in .env")
        _client = genai.Client(api_key=api_key)
    return _client


def _load(text):
    """JSON out of a model response, tolerating a fence or leading prose.

    Mirrors parser._load_object: strip a triple-back-tick fence, try a full
    load, then fall back to slicing between the first { and last }.
    """
    cleaned = re.sub(r"^\s*```(?:json)?\s*|\s*```\s*$", "", (text or ""), flags=re.I)
    try:
        return json.loads(cleaned)
    except Exception:
        pass

    start = cleaned.find("{")
    end = cleaned.rfind("}")
    array = cleaned.find("[") < 0
    if start >= 0 and end > start:
        try:
            return json.loads(cleaned[start:end + 1])
        except Exception:
            pass
    if not array:
        # The response may be a top-level array (sheets, slide decks).
        as_list = cleaned.find("[")
        ae = cleaned.rfind("]")
        if as_list >= 0 and ae > as_list:
            try:
                return json.loads(cleaned[as_list:ae + 1])
            except Exception:
                pass
    return None


# --- Providers --------------------------------------------------------------

def _gemini_text(system, contents, max_tokens):
    response = client().models.generate_content(
        model=MODEL_NAME,
        contents=contents,
        config={"system_instruction": system, "max_output_tokens": max_tokens},
    )
    text = (response.text or "").strip()
    if not text:
        raise RuntimeError("empty response")
    return text


def _groq_text(system, contents, max_tokens, json_mode=False):
    from groq import Groq

    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY not found in .env")

    kwargs = {}
    if json_mode:
        # Only for structured() -- forcing json_object on a summarise call
        # would return the prose answer wrapped in JSON, which then gets
        # spoken with the braces in it. The system prompt is already the
        # stricter shape enforcer for prose.
        kwargs["response_format"] = {"type": "json_object"}

    groq_client = Groq(api_key=api_key)
    response = groq_client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": contents},
        ],
        max_tokens=max_tokens,
        temperature=0.2,
        **kwargs,
    )
    text = (response.choices[0].message.content or "").strip()
    if not text:
        raise RuntimeError("empty response")
    return text


def _openrouter_text(system, contents, max_tokens):
    """One OpenRouter free model, reusing chat.py's discovery + cooldowns.

    Tries the free roster in ranking order, skipping anything currently in a
    cooldown and cooling a model that fails or rate-limits, so a bad model on
    the list never costs the whole request.
    """
    from commands import chat

    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY not found in .env")

    kind, lib = chat._http()
    headers = chat._headers()

    for model in chat.models():
        if chat._is_cool(model):
            continue
        body = {
            "model": model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": contents},
            ],
            "max_tokens": max_tokens,
            "temperature": 0.2,
        }
        try:
            if kind == "requests":
                response = lib.post(
                    chat.API_BASE + "/chat/completions",
                    headers=headers, json=body,
                    timeout=(6.0, 30.0),
                )
                status, raw = response.status_code, response.text
            else:
                with lib.Client(timeout=lib.Timeout(30.0, connect=6.0)) as client:
                    response = client.post(
                        chat.API_BASE + "/chat/completions",
                        headers=headers, json=body,
                    )
                    status, raw = response.status_code, response.text
        except Exception as e:
            chat._cool(model, chat.ERROR_COOLDOWN)
            print("genbrain: openrouter {} failed ({})".format(model, e))
            continue

        if status == 429:
            chat._cool(model, chat.RATE_LIMIT_COOLDOWN)
            print("genbrain: openrouter {} is rate limited".format(model))
            continue
        if status >= 400:
            chat._cool(model, chat.ERROR_COOLDOWN)
            print("genbrain: openrouter {} HTTP {}".format(model, status))
            continue

        try:
            payload = json.loads(raw or "{}")
            text = (payload.get("choices") or [{}])[0]\
                .get("message", {}).get("content") or ""
        except Exception:
            chat._cool(model, chat.ERROR_COOLDOWN)
            continue
        text = text.strip()
        if text:
            return text
        chat._cool(model, chat.ERROR_COOLDOWN)

    raise RuntimeError("no OpenRouter free model answered")


def _chain_text(system, contents, max_tokens, json_mode=False):
    """Gemini, then Groq, then OpenRouter. Raw text, or raises.

    The providers are intentionally coupled only by this chain, so adding one
    is a matter of writing a `_<name>_text` function and one more fallthrough
    here. Each leg logs OK/FAIL to the dev log; the caller cares only that it
    ends with text or an error.
    """
    errors = []

    if os.getenv("GEMINI_API_KEY"):
        t0 = time.time()
        try:
            text = _gemini_text(system, contents, max_tokens)
            devlog.think("gemini", "answer", "OK",
                         detail="{} {} chars".format(MODEL_NAME, len(text)),
                         seconds=time.time() - t0)
            return text
        except Exception as e:
            devlog.think("gemini", "answer", "FAIL",
                         detail=str(e)[:120], seconds=time.time() - t0)
            print("genbrain: gemini failed ({})".format(e))
            errors.append("gemini: {}".format(e))
    else:
        errors.append("gemini: no GEMINI_API_KEY")

    if os.getenv("GROQ_API_KEY"):
        t0 = time.time()
        try:
            text = _groq_text(system, contents, max_tokens, json_mode=json_mode)
            devlog.think("groq", "answer", "OK",
                         detail="{} {} chars".format(GROQ_MODEL, len(text)),
                         seconds=time.time() - t0)
            return text
        except Exception as e:
            devlog.think("groq", "answer", "FAIL",
                         detail=str(e)[:120], seconds=time.time() - t0)
            print("genbrain: groq failed ({})".format(e))
            errors.append("groq: {}".format(e))
    else:
        errors.append("groq: no GROQ_API_KEY")

    if os.getenv("OPENROUTER_API_KEY"):
        t0 = time.time()
        try:
            text = _openrouter_text(system, contents, max_tokens)
            devlog.think("openrouter", "answer", "OK",
                         detail="{} {} chars".format("openrouter", len(text)),
                         seconds=time.time() - t0)
            return text
        except Exception as e:
            devlog.think("openrouter", "answer", "FAIL",
                         detail=str(e)[:120], seconds=time.time() - t0)
            print("genbrain: openrouter chain failed ({})".format(e))
            errors.append("openrouter: {}".format(e))
    else:
        errors.append("openrouter: no OPENROUTER_API_KEY")

    raise RuntimeError("all providers failed: {}".format("; ".join(errors)))


# --- The two entry points ---------------------------------------------------

def structured(shape, user_prompt, max_tokens=MAX_TOKENS):
    """Ask the model for JSON matching a human-described `shape`.

    `shape` is a sentence describing the exact structure wanted, e.g.
    "a JSON object with keys title (string) and slides (array of objects each
    with keys title (string) and bullets (array of strings))". Returns the
    parsed JSON, or raises. Runs the whole provider chain per attempt; if the
    first try comes back as prose it is retried once with a sharper instruction.
    """
    system = (
        "You generate strictly valid JSON and nothing else -- no markdown code "
        "blocks, no surrounding prose, no trailing explanation. The user's "
        "request appears after REQUEST. Match exactly this shape:\n\n{}"
    ).format(shape)

    last_error = None
    for attempt in range(MAX_ATTEMPTS):
        if attempt == 0:
            push = "Return ONLY the JSON object -- no markdown, no prose."
        else:
            # The first pass usually fails here for one reason: the output got
            # long and was cut off mid-string. A retry that demands brevity
            # cures the truncation that a plain retry would reproduce.
            push = ("Return ONLY the JSON object now. Keep the whole response "
                    "within 6000 tokens, even if that means shorter values -- "
                    "a complete short answer beats a truncated long one.")
        try:
            text = _chain_text(
                system, "REQUEST: {}\n{}".format(user_prompt, push),
                max_tokens, json_mode=True,
            )
        except Exception as e:
            # The whole chain is down, or every provider failed -- a different
            # push cannot fix an availability failure, so stop retrying.
            raise e
        parsed = _load(text)
        if parsed is not None:
            return parsed
        last_error = RuntimeError("model returned prose I couldn't parse: {!r}".format(text[:200]))
    raise last_error


def summarise(text, question="", limit=SUMMARY_INPUT_CHARS):
    """One short, spoken-style answer about `text`.

    Used by file_analysis, youtube_video and web_search -- the "read this and
    tell me about it" features where the model's answer IS the result.
    """
    excerpt = (text or "").strip()
    if not excerpt:
        raise RuntimeError("there was nothing in the material to summarise")

    if len(excerpt) > limit:
        excerpt = excerpt[:limit]

    prompt = (
        "Here is the material a user asked about:\n\n{excerpt}\n\n"
        "QUESTION: {question}"
    ).format(excerpt=excerpt, question=question or "summarise it")

    system = (
        "You are Jarvis, a voice assistant. Answer in the natural, spoken style "
        "of a person talking -- two or three short sentences is the normal "
        "length, going longer only when the material genuinely needs it. No "
        "markdown, no bullet points, no symbols, no emoji: every word is read "
        "aloud. If the material does not contain enough to answer the question, "
        "say briefly what it does contain, do not invent anything."
    )

    text = _chain_text(system, prompt, 800)
    text = (text or "").strip()
    if not text:
        raise RuntimeError("model returned an empty answer")
    return text


def probe():
    """Diagnostic for `python -m commands.genbrain`. Never used at runtime."""
    t0 = time.time()
    try:
        out = summarise(
            "Jarvis is a voice assistant that lives on your PC and answers "
            "out loud. It can open apps, manage files and generate documents.",
            "what is this?",
        )
        return "genbrain OK in {:.1f}s\n  {}".format(time.time() - t0, out)
    except Exception as e:
        return "genbrain failed: {}".format(e)


if __name__ == "__main__":
    # python -m commands.genbrain  -- one call against the live chain.
    print(probe())