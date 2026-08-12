"""The middle ear: OpenRouter STT, between Gemini and Vosk.

Goes in: listener/stt_openrouter.py   (new file)

    .env
    OPENROUTER_STT_API_KEY=sk-or-v1-...

WHY A MIDDLE TIER AT ALL
------------------------
Gemini failing is almost never "this machine is broken". It's a rate limit, a
502, or three seconds of bad wifi. The old chain answered all of that by
dropping straight to Vosk -- and Vosk here runs a twenty-six word grammar
(listener/stt_local.py, COMMAND_VOCAB). So a transient cloud hiccup didn't
degrade transcription, it deleted it: "where is chrome located on my system"
has exactly one word in that vocabulary.

A second cloud tier means a Gemini blip costs you a couple of seconds and some
accuracy instead of your entire vocabulary. Vosk stays where it belongs, as the
thing that still works with the router unplugged.

A SEPARATE KEY, ON PURPOSE
--------------------------
The chat brain is already on OpenRouter's free tier. Sharing one key would put
conversation and hearing on the same rate-limit budget, so a chatty afternoon
would cost you your ears -- and it would fail in the confusing direction, where
asking too many questions stops the assistant understanding you at all. Two
keys, two budgets, independent failures.

If OPENROUTER_STT_API_KEY is missing this tier falls back to the chat key
rather than disappearing, because a shared budget still beats a 26-word
grammar. It says so out loud once, so a half-configured install is visible
instead of mysterious.

A REASONING MODEL ASKED TO TRANSCRIBE
-------------------------------------
nemotron-omni is a *reasoning* model that happens to accept audio. That buys
good robustness on accents and background noise, and it costs us two things:

  1. It thinks. Reasoning may arrive in a separate field, or leak into the
     content as "The audio contains a user asking..." -- so the transcript
     gets unwrapped (_clean_transcript) rather than trusted.
  2. It's chatty by nature. temperature 0 and a tight max_tokens keep it from
     narrating; _looks_like_commentary catches it when that isn't enough.

Same lesson as the content-safety model in commands/chat.py: a model that
returns something is not the same as a model that returned what you asked for.
Here the check is cheap because nothing is streamed -- a transcript is useless
until it's complete, so we wait for the whole thing and inspect it before
anyone downstream sees it.

A HARD DEADLINE, BECAUSE THE FALLBACK IS LOCAL
----------------------------------------------
Every second here lands between you finishing a sentence and Jarvis reacting.
Vosk is sitting right there and answers in well under a second. So this tier
gets TIMEOUT_SECONDS and not a moment more -- waiting 30s for a cloud retry
when a local answer is available is the wrong trade, even though the local
answer is worse.

On failure the model is cooled for COOLDOWN_SECONDS. A rate limit is not a
one-request event, and re-learning that on every single turn would add the
timeout to every turn.
"""

import base64
import io
import os
import re
import threading
import time
import wave

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:  # pragma: no cover - .env is optional
    pass

SAMPLE_RATE = 16000
MODEL_NAME = "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free"
API_URL = "https://openrouter.ai/api/v1/chat/completions"

# Between you speaking and Jarvis moving. Vosk is the alternative, and Vosk is
# fast, so this is deliberately impatient.
TIMEOUT_SECONDS = 12.0

# A failure is rarely isolated. Don't pay the timeout again on the next turn.
COOLDOWN_SECONDS = 120.0

# Transcripts are short. This is a ceiling on rambling, not on speech: 30s of
# talking is maybe 90 words, and reasoning models pad when given room.
MAX_TOKENS = 400

# Longer than any sane utterance -- if we get an essay back, it isn't a
# transcript, it's the model explaining itself.
MAX_TRANSCRIPT_CHARS = 1200

PROMPT = (
    "Transcribe the speech in this audio verbatim. Output only the words that"
    " were spoken, as a single line of plain text. Do not translate, do not"
    " summarise, do not describe the audio or the speaker, do not add"
    " quotation marks, and do not explain what you are doing. If the audio"
    " contains no intelligible speech, output exactly: (no speech)"
)

_cooled_until = 0.0
_lock = threading.Lock()
_warned_shared_key = False


# --- Key handling ----------------------------------------------------------

def api_key():
    """The STT key, or the chat key with a one-time warning."""
    global _warned_shared_key

    key = os.getenv("OPENROUTER_STT_API_KEY")
    if key:
        return key

    shared = os.getenv("OPENROUTER_API_KEY")
    if shared:
        if not _warned_shared_key:
            _warned_shared_key = True
            print("stt: OPENROUTER_STT_API_KEY not set -- sharing the chat key,"
                  " so chat and hearing share one rate limit")
        return shared

    raise RuntimeError("OPENROUTER_STT_API_KEY not found in .env")


def available():
    """True if this tier is configured and not currently cooling off."""
    if not (os.getenv("OPENROUTER_STT_API_KEY") or os.getenv("OPENROUTER_API_KEY")):
        return False
    with _lock:
        return time.time() >= _cooled_until


def _cool(seconds, why):
    global _cooled_until
    with _lock:
        _cooled_until = max(_cooled_until, time.time() + seconds)
    print("stt: openrouter cooling {:.0f}s ({})".format(seconds, why))


# --- Turning a model's answer back into a transcript -----------------------

class _NotATranscript(RuntimeError):
    """The model replied, but with something other than the spoken words."""


# Reasoning leaks. Some providers put chain-of-thought in a separate field,
# some wrap it in tags and send it inline; nemotron can do either.
_THINK_BLOCK = re.compile(
    r"<\s*(think|thinking|reasoning|scratchpad)\s*>.*?<\s*/\s*\1\s*>",
    re.IGNORECASE | re.DOTALL,
)

# An unclosed opener means the tokens ran out mid-thought and no transcript
# ever arrived.
_THINK_OPEN = re.compile(r"<\s*(think|thinking|reasoning|scratchpad)\s*>",
                         re.IGNORECASE)

# "Here is the transcription:" and friends. Anchored, so a sentence that
# happens to contain the word "transcript" isn't mangled.
_PREAMBLE = re.compile(
    r"^\s*(?:sure[,!.]?\s*)?"
    r"(?:here(?:'s| is| are)?\s+)?"
    r"(?:the\s+)?"
    r"(?:verbatim\s+|exact\s+|raw\s+)?"
    r"(?:transcription|transcript|transcribed\s+text|audio\s+says?|"
    r"speech|spoken\s+words?|text)"
    r"\s*(?:is|of\s+the\s+audio|from\s+the\s+audio)?\s*[:\-]\s*",
    re.IGNORECASE,
)

# Ways of saying "this is about the audio" rather than "this is the audio".
#
# Split into two groups because they need different strictness. A refusal
# opener like "I cannot" is also a perfectly normal thing to say to an
# assistant -- "i cannot open chrome" is a user complaining, not a model
# declining -- so those only count when the sentence also mentions the audio.
# The descriptive openers are already specific enough to stand alone.
_REFUSAL_OPENERS = (
    "i cannot", "i can't", "i am unable", "i'm unable", "i do not have",
    "i don't have", "i'm sorry", "i am sorry", "sorry, i", "as an ai",
    "unfortunately, i", "i am not able", "i'm not able",
)

# Job-language: ways of referring to *the request* rather than to sound in
# general. Deliberately narrow. "audio", "hear", "listen" and "speech" were in
# here and had to come out -- "i cannot hear the audio from my speakers" is a
# thing a person says to their assistant, and it was being thrown away.
_ABOUT_THIS_AUDIO = (
    "transcri", "this audio", "the audio file", "the audio clip",
    "audio recording", "provided audio", "audio provided", "audio input",
    "the given audio", "the recording", "this recording", "sound file",
    "voice clip", ".wav", ".mp3",
)

# First-person hardware talk. A model declining a job has no reason to mention
# your speakers; a person describing a problem does it constantly. This is the
# tie-breaker when a sentence trips both a refusal opener and the job-language
# above.
_USER_HARDWARE = (
    "my speaker", "my mic", "my microphone", "my headphone", "my headset",
    "my laptop", "my computer", "my pc", "my system", "on my", "from my",
    "in my", "my audio", "my sound",
)

_DESCRIPTION_OPENERS = (
    "the speaker says", "the speaker is", "the speaker appears",
    "this audio contains", "the audio contains", "the recording contains",
    "the audio file", "it sounds like the", "appears to be saying",
    "the user is asking", "the user asks", "the user says",
    "the person says", "the person is asking", "there is no audio",
    "no audio was", "the transcription of",
)


def _looks_like_commentary(lowered):
    """True if this is the model talking about the audio, not repeating it.

    Some sentences are unresolvable from words alone -- "i cannot listen to
    audio on this laptop" reads identically whether a person said it or a model
    wrote it. Those are let through, on purpose, because the two mistakes cost
    very different amounts:

      * Accepting a refusal as speech: it gets routed as chat, Jarvis answers
        something odd, the user repeats themselves. One wasted turn, and the
        tier stays warm.
      * Rejecting real speech: falls to Vosk's 26-word grammar, so the command
        is gone rather than degraded -- and _cool() then hobbles the next two
        minutes of turns as well.

    So the ambiguous middle is treated as speech.
    """
    if any(lowered.startswith(phrase) for phrase in _DESCRIPTION_OPENERS):
        return True

    # A refusal only counts as one if it's refusing to do *this* job. Checked
    # against the opening clause so "i can't hear you" from a user isn't
    # rescued by a stray "audio" thirty words later.
    if any(lowered.startswith(phrase) for phrase in _REFUSAL_OPENERS):
        if not any(word in lowered[:160] for word in _ABOUT_THIS_AUDIO):
            return False
        # Job-language plus a refusal opener -- but if they're also talking
        # about their own hardware, it's a person with a broken speaker.
        return not any(word in lowered[:160] for word in _USER_HARDWARE)

    return False

# The prompt's own "nothing here" token, plus the ways a model says it anyway.
_SILENCE = (
    "(no speech)", "no speech", "[no speech]", "(silence)", "silence",
    "[silence]", "(inaudible)", "inaudible", "[inaudible]", "(unintelligible)",
    "unintelligible", "n/a", "(empty)", "...", "[blank_audio]", "(blank)",
)


def _clean_transcript(text):
    """Model output -> the words that were spoken.

    Raises _NotATranscript when what came back is commentary, a refusal, or
    pure reasoning. That distinction matters: a bad transcript is worse than
    no transcript, because a bad one gets routed and acted on. Falling through
    to Vosk and hearing a clumsy version of the right words beats hearing a
    fluent description of them.
    """
    text = (text or "").strip()
    if not text:
        raise _NotATranscript("empty response")

    # Reasoning first, before anything measures length or looks for preambles.
    text = _THINK_BLOCK.sub(" ", text)
    if _THINK_OPEN.search(text):
        # Ran out of tokens mid-thought. Whatever follows isn't a transcript.
        raise _NotATranscript("response was all reasoning, no transcript")

    text = text.strip()

    # Code fences: some models wrap "plain text" in markdown regardless.
    if text.startswith("```"):
        parts = text.split("```")
        text = (parts[1] if len(parts) > 1 else text.lstrip("`")).strip()
        if "\n" in text and text.split("\n", 1)[0].strip().isalpha():
            first, rest = text.split("\n", 1)
            if len(first.strip()) < 12:  # a language tag, not speech
                text = rest.strip()

    text = _PREAMBLE.sub("", text).strip()

    # Speech is one utterance. Extra lines are the model adding notes.
    if "\n" in text:
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        if lines:
            text = lines[0]

    # Matched surrounding quotes -- the prompt says not to, models do anyway.
    for opener, closer in (('"', '"'), ("'", "'"), ("“", "”"),
                           ("‘", "’"), ("`", "`")):
        if len(text) > 1 and text.startswith(opener) and text.endswith(closer):
            text = text[1:-1].strip()
            break

    if len(text) > MAX_TRANSCRIPT_CHARS:
        raise _NotATranscript("response too long to be a transcript")

    lowered = text.lower().strip(" .!?")
    if lowered in _SILENCE:
        return ""  # heard nothing. Not a failure -- don't cascade to Vosk.

    if _looks_like_commentary(lowered):
        raise _NotATranscript("described the audio instead of transcribing it")

    return text.strip()


# --- Audio -----------------------------------------------------------------

def _wav_bytes(audio):
    """int16 mono numpy array -> WAV bytes. Same shape stt_gemini uses."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # int16
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(audio.tobytes())
    return buf.getvalue()


# --- HTTP ------------------------------------------------------------------

def _http():
    """requests if present, else httpx (which ships with groq)."""
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
            "stt needs the requests or httpx package (httpx ships with groq)"
        )


def _headers():
    return {
        "Authorization": "Bearer {}".format(api_key()),
        "Content-Type": "application/json",
        "HTTP-Referer": "https://localhost/jarvis",
        "X-Title": "Jarvis STT",
    }


def _payload(audio):
    """OpenRouter's audio shape: base64 in an input_audio content part.

    Not a file upload and not a URL -- the audio rides inside the chat
    completion request as a content part alongside the text prompt, which is
    why the recorder's array can go straight out without touching disk.
    """
    encoded = base64.b64encode(_wav_bytes(audio)).decode("ascii")
    return {
        "model": MODEL_NAME,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": PROMPT},
                    {
                        "type": "input_audio",
                        "input_audio": {"data": encoded, "format": "wav"},
                    },
                ],
            }
        ],
        # Zero temperature: transcription has a right answer. Creativity here
        # shows up as invented words.
        "temperature": 0.0,
        "max_tokens": MAX_TOKENS,
    }


def _post(payload):
    """One request, returns the parsed JSON body or raises."""
    kind, lib = _http()
    headers = _headers()

    if kind == "requests":
        response = lib.post(API_URL, headers=headers, json=payload,
                            timeout=TIMEOUT_SECONDS)
        status, text = response.status_code, response.text
        body = response.json() if status == 200 else None
    else:
        with lib.Client(timeout=TIMEOUT_SECONDS) as client:
            response = client.post(API_URL, headers=headers, json=payload)
        status, text = response.status_code, response.text
        body = response.json() if status == 200 else None

    if status == 429:
        raise RuntimeError("rate limited")
    if status in (401, 403):
        raise RuntimeError("key rejected ({})".format(status))
    if status != 200:
        raise RuntimeError("HTTP {}: {}".format(status, (text or "")[:160]))
    return body


def _content(body):
    """Pull the assistant text out, ignoring a separate reasoning field.

    Some providers return reasoning in message["reasoning"] and the answer in
    message["content"]; taking content only is what makes that case free.
    """
    try:
        message = body["choices"][0]["message"]
    except (KeyError, IndexError, TypeError):
        raise _NotATranscript("no choices in response")

    content = message.get("content")

    # Content can itself be a list of parts on some providers.
    if isinstance(content, list):
        pieces = []
        for part in content:
            if isinstance(part, dict) and part.get("type") in (None, "text"):
                pieces.append(part.get("text") or "")
            elif isinstance(part, str):
                pieces.append(part)
        content = " ".join(pieces)

    if not (content or "").strip():
        # Empty content plus a reasoning field means it thought and never
        # answered -- a real failure mode for reasoning models on a tight
        # token budget, and one worth naming precisely in the log.
        if (message.get("reasoning") or "").strip():
            raise _NotATranscript("returned reasoning but no transcript")
        raise _NotATranscript("empty content")

    return content


# --- The tier ---------------------------------------------------------------

def transcribe(audio):
    """audio: int16 numpy array, mono, 16kHz -- the recorder's own format.

    Returns the transcript, "" for silence, and raises on anything else so
    main.transcribe can drop to Vosk. Raising rather than returning "" for a
    failure is deliberate: "" is a real answer meaning nothing was said, and
    conflating the two would send an empty string down the router on every
    cloud hiccup.
    """
    if not available():
        raise RuntimeError("openrouter stt unavailable (no key or cooling off)")

    started = time.time()
    try:
        body = _post(_payload(audio))
        text = _clean_transcript(_content(body))
    except _NotATranscript as e:
        # The wrong kind of answer, not a transport failure. Cool it: a model
        # that narrates once will narrate again.
        _cool(COOLDOWN_SECONDS, str(e))
        raise
    except Exception as e:
        message = str(e).lower()
        if "rate limit" in message or "429" in message:
            _cool(COOLDOWN_SECONDS, "rate limited")
        elif "timeout" in message or "timed out" in message:
            # Short cooldown: a timeout is often the network, not the tier.
            _cool(30.0, "timed out")
        raise

    elapsed = time.time() - started
    used = body.get("model") or MODEL_NAME
    print("stt: openrouter ({}) in {:.1f}s".format(used.split("/")[-1], elapsed))
    return text


def status():
    with _lock:
        cooling = max(0.0, _cooled_until - time.time())
    if not (os.getenv("OPENROUTER_STT_API_KEY") or os.getenv("OPENROUTER_API_KEY")):
        return "stt openrouter: no key -- Gemini falls straight to Vosk"
    which = ("own key" if os.getenv("OPENROUTER_STT_API_KEY")
             else "sharing the chat key")
    if cooling:
        return "stt openrouter: cooling for {:.0f}s ({})".format(cooling, which)
    return "stt openrouter: ready ({}, {})".format(MODEL_NAME, which)


if __name__ == "__main__":
    # python -m listener.stt_openrouter [file.wav]
    # No argument: report configuration and exercise the cleaner offline.
    import sys

    print(status())

    if len(sys.argv) > 1:
        import numpy as np

        with wave.open(sys.argv[1], "rb") as wf:
            print("input: {}Hz {}ch {} frames".format(
                wf.getframerate(), wf.getnchannels(), wf.getnframes()
            ))
            raw = wf.readframes(wf.getnframes())
        audio = np.frombuffer(raw, dtype=np.int16)
        print("transcript: {!r}".format(transcribe(audio)))
        raise SystemExit

    # (text, expected) where expected is "pass", "silence" or "reject".
    # The third group is the one that matters: those are things a person
    # actually says, and every one of them has to come out untouched. Two of
    # them used to be thrown away as model commentary.
    probes = [
        # model output that needs unwrapping
        ('hello there', 'pass'),
        ('  "open chrome"  ', 'pass'),
        ('Here is the transcription: open chrome', 'pass'),
        ('Sure! The transcript is: "what is the weather"', 'pass'),
        ('<think>The user seems to be asking...</think>open notepad', 'pass'),
        ('```\nopen notepad\n```', 'pass'),
        ('open chrome\n\nNote: the audio was quiet.', 'pass'),
        ('(no speech)', 'silence'),
        ('[BLANK_AUDIO]', 'silence'),
        ('<think>Let me consider this carefully and at great length',
         'reject'),
        # commentary and refusals that must not reach the router
        ('I cannot transcribe this audio.', 'reject'),
        ('The speaker says they want to open Chrome.', 'reject'),
        ("I'm sorry, I cannot transcribe audio files.", 'reject'),
        ('I am unable to process the audio recording you sent.', 'reject'),
        ('As an AI, I cannot listen to the audio clip provided.', 'reject'),
        ('There is no audio in this file.', 'reject'),
        # real speech -- must survive verbatim
        ('i cannot open chrome', 'pass'),
        ('i cannot hear the audio from my speakers', 'pass'),
        ("i don't have vlc installed", 'pass'),
        ('sorry i meant close notepad', 'pass'),
        ('where is chrome located on my system', 'pass'),
        ('transcribe this meeting for me later', 'pass'),
        ('listen can you close task manager', 'pass'),
    ]

    bad = 0
    for probe, want in probes:
        try:
            out = _clean_transcript(probe)
            got, note = ("silence" if out == "" else "pass"), repr(out)
        except _NotATranscript as e:
            got, note = "reject", str(e)
        ok = got == want
        bad += not ok
        print("  {} {!r:54s} -> {}".format("ok  " if ok else "FAIL", probe, note))

    print("\n{}/{} as specified".format(len(probes) - bad, len(probes)))