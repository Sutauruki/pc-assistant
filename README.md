# Jarvis — PC Assistant

A voice-first Windows desktop assistant. Wake-word ("hey Jarvis") → speech-to-text
→ command router → executor → spoken reply, plus a live console UI, typed input,
history, notifications and an action-confirmation protocol.

## Features

- **Wake word loop** — openWakeWord (`hey_jarvis`), cooldown model reset, typed
  turns bypass the cooldown.
- **STT** — Moonshine, a single fully-offline, open-vocabulary tier (no cloud,
  no key, no rate limit). Nothing spoken has to leave the machine.
- **Command router** (`commands/parser.py`) — classifies a turn as `command`,
  `chat`, or `both`. Chat-only turns are answered without touching the action
  engine; `both` runs the action and feeds its real output back to the brain.
- **24 command modules** — see table below.
- **TTS** — Fish Audio's S2.1 Pro cloud voice (free `s2.1-pro-free` tier,
  streamed as raw PCM), falling back to Moonshine's offline voices
  (Kokoro/Piper, via its C core) when there's no key, no network, or Fish
  refuses. Chat replies start speaking after the first sentence while the
  rest is still being written, and short repeated replies ("Done.") play
  from an on-disk cache. Played straight through with no time-stretch, with overlap: the
  reply starts while actions are still running.
- **Screen understanding** — captures only the monitor you're working on,
  at 1024px, blacks out password boxes, and answers "what's on my screen" /
  "what does that error say" via Gemini with minimal thinking, speaking the
  answer as it streams (`commands/screen_actions.py`). Never captures private
  windows (password managers, banking, anything in `JARVIS_SENSITIVE`).
- **Computer control** — "message Tolu on WhatsApp that I'll be late", "help
  me gather info about X in Chrome": an agent that looks at the screen, clicks,
  scrolls and types like a person in any app (`commands/agent.py`). Research
  ends with "explain it, or save it as a doc, sheet, slides or PDF?". See
  **Computer control and security** below.
- **Detailed dev log** — `DEV_LOG=1` in `.env` appends a second, densely
  formatted log to `dev_log.txt`: every STT attempt, every TTS synthesis,
  the chat model chain and each model's latency, action executions, wake/typed
  events and per-turn timings, all in aligned columns (`commands/devlog.py`).
  Off by default.
- **Typed input** — full command parser available from the keyboard; typed turns
  skip the 0.4 s wake-word cooldown entirely.
- **Console UI** (`listener/ui.py`) — live ring, STT/TTS status, latency and
  turn counter, rerouted logs, and an overlay log viewer (`Ctrl+L`, arrows to
  scroll, `Esc` to close). Auto-disables when stdout isn't a terminal or
  `JARVIS_NO_UI` is set.
- **Confirmations** (`commands/confirm.py`) — destructive actions park a yes/no
  question; the *next* utterance resolves it (the loop re-records without the
  wake word while one is pending), bounded by `MAX_CONFIRM_FOLLOWUPS`.
- **Memory/preferences** (`commands/memory.py`) — persistent user preferences
  loaded at boot and shown as "loaded N preferences".
- **Notifications** (`commands/notify.py`) — Windows toasts (win11toast with
  fallbacks).

## Setup

Requires Python 3.10+ on Windows (the UI is a Windows console). Voice audio
works on any machine with a default input/output device.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
copy .env.example .env   # if you don't have one yet
python main.py
```

### Wake-word models

openWakeWord >= 0.5 no longer ships pretrained models in the wheel. `listener/wake_word.py`
auto-downloads the requested model (e.g. `hey_jarvis`) from Hugging Face on first
load and pins the ONNX inference framework on Windows (tflite-runtime has no
Windows wheels). Requires `onnxruntime`, which is in `requirements.txt`.
No manual download step.

## Configuration (.env)

| Variable | Used by | Required |
|---|---|---|
| `MOONSHINE_COMPRESSION` | Moonshine STT model size (`listener/stt_moonshine.py`); default `medium_streaming` | optional |
| `FISH_API_KEY` | primary TTS voice, Fish Audio S2.1 Pro (`listener/tts_fish.py`); without it Moonshine speaks | recommended |
| `FISH_VOICE_ID` | Fish Audio voice model id (`reference_id`); unset uses Fish's default voice | optional |
| `FISH_MODEL` | Fish Audio model header; default `s2.1-pro-free` (free tier), `s2.1-pro` for paid | optional |
| `FISH_LATENCY` | Fish Audio speed/quality trade-off: `low`, `normal` or `balanced` (default); `low` starts talking sooner with slightly lower voice quality | optional |
| `FISH_CACHE` | `0` turns off the on-disk cache of short Fish replies (`cache/tts_fish/`); on by default | optional |
| `MOONSHINE_VOICE` | fallback Moonshine TTS voice (`listener/tts_moonshine.py`); default `kokoro_af_bella` | optional |
| `GEMINI_API_KEY` | screen understanding (`commands/screen_actions.py`) + all Tier 6 content generation and summarising (`commands/genbrain.py`) | only if those features are used |
| `OPENROUTER_API_KEY` | chat brain (`commands/chat.py:184`) | for conversational chat |
| `GROQ_API_KEY` | parser brain (`commands/parser.py:388`) | for command routing |
| `JARVIS_SENSITIVE` | extra words (comma-separated) that mark a window as private — your bank's name, broker, health portal; never captured | recommended |
| `SCREEN_THINKING` | Gemini thinking level for screen questions: `minimal` (default), `low`, `medium`, `high` | optional |
| `SCREEN_MODEL` | Gemini model for screen questions and the agent; defaults to `GENBRAIN_MODEL` | optional |
| `AGENT_THINKING` | Gemini thinking level for the computer agent; default `low` | optional |
| `AGENT_MAX_STEPS` | step limit per computer task; default `30` | optional |
| `AGENT_TIMEOUT` | time limit per computer task in seconds; default `300` | optional |
| `JARVIS_NO_UI` | set non-empty to force the terminal UI off | optional |
| `JARVIS_NO_TYPING` | set non-empty to disable typed input (voice still works) | optional |
| `DEV_LOG` | `1` writes a detailed machine log to `dev_log.txt` (every STT/TTS/thinking/action event with timings, aligned columns) | optional |

## Tuning constants (`main.py`)

| Constant | Default | Meaning |
|---|---|---|
| `COOLDOWN_SECONDS` | 0.4 | sleep after a wake-word hit (model is reset explicitly) |
| `TYPED_COOLDOWN_SECONDS` | 0.0 | no cooldown for typed turns |
| `SAVE_RECORDINGS` | `True` | writes user recordings to `%LOCALAPPDATA%\\Jarvis\\recordings` |
| `OVERLAP_SPEECH_AND_ACTIONS` | `True` | reply starts while actions still run |
| `DEFERRED_ACTIONS` | `{"set_volume"}` | actions that wait for speech to finish before running |
| `SPOKEN_RESULTS` | set | actions whose result IS the spoken answer |
| `SPEECH_JOIN_TIMEOUT` | `45.0` | floor for waiting on speech to finish |
| `MAX_CONFIRM_FOLLOWUPS` | `2` | re-records allowed for a pending confirmation |
| `CONFIRM_SILENCE_SECONDS` | `6.0` | mic wait for a pending confirmation answer |

## Command modules (`commands/`)

| Module | Covers |
|---|---|
| `app_index.py` | app lookup / launch |
| `window_actions.py` | window listing, focus, minimize, arrange, close |
| `cmd_control.py` | raw command execution |
| `file_actions.py` | file ops (recycle-bin deletes via Send2Trash) |
| `locate.py` | file / window location lookups |
| `take_screenshot` | screen capture (in `executor.py:88`, via `mss`) |
| `clipboard_actions.py` | clipboard read/write |
| `parser.py` | command/chat routing |
| `executor.py` | action execution (worker threads) |
| `screen_actions.py` | screen understanding (active monitor → 1024px JPEG, passwords masked → Gemini, streamed) |
| `agent.py` | computer agent: look → decide → check → act loop |
| `agent_input.py` | real mouse/keyboard input (SendInput) + Esc/F8/corner brakes |
| `uia.py` | Windows UI Automation: every on-screen control and its position |
| `desktop.py` | foreground window, its process and monitor |
| `pc_policy.py` | security policy: private / protected windows, risky steps, banned keys, URLs, text |
| `research.py` | keeps the agent's research notes for "explain it" / "save it as a PDF" |
| `chat.py` | conversational answering |
| `confirm.py` | interactive yes/no confirmation protocol |
| `notify.py` | Windows toasts |
| `memory.py` | persistent preferences |
| `self_guard.py` | safety guardrails for system actions |
| `paths.py` | path resolution helpers |
| `genbrain.py` | shared Gemini content brain (structured JSON + summarising) |
| `docs.py` | prompt → Word document (python-docx) |
| `slides.py` | prompt → slide deck (python-pptx) |
| `sheets.py` | prompt → spreadsheet (openpyxl) |
| `pdf_export.py` | prompt → PDF report (reportlab) |
| `file_analysis.py` | read + answer about a PDF / CSV / Excel / audio / text file |
| `youtube_video.py` | fetch + summarise a YouTube transcript |
| `web_search.py` | query + summarise the web (ddgs + BeautifulSoup) |
| `website_builder.py` | prompt → small static site + local preview (single non-leaking server on 127.0.0.1) |

## Computer control and security

Say or type what you want done: "open WhatsApp web and tell mum I'm on my
way", "help me gather info about solar panel prices in Chrome", "fill in the
search box with cheap flights to Lagos". Jarvis uses your real mouse and
keyboard, in your normal browser with your existing logins, so **leave the
PC alone while it works** — it's quicker to watch than to fight it.

**Your controls**

| You do | It does |
|---|---|
| **Esc** (physical key) | stops the task immediately |
| Slam the mouse into the **top-left corner** | stops the task |
| Move the mouse yourself | stops the task — you've taken over |
| **F8** (physical key) | approves the one risky step it just described |

**What it will never do, whatever a web page or message says**

- **Look at private windows** — password managers, banking, crypto wallets,
  and anything matching a word in `JARVIS_SENSITIVE`. Nothing is captured.
- **Act in protected apps** — terminals/PowerShell (including Jarvis's own),
  the Run dialog, registry, Task Manager, Settings, Windows Security, control
  panel tools, installers, UAC and credential prompts. Matched by the
  process Windows reports, so a renamed window can't slip through.
- **Type passwords or your keys** — password boxes are blacked out in every
  screenshot and refused as targets; text containing any of your `.env`
  keys/tokens/passwords is refused.
- **Press Windows-key shortcuts** (Win+R, Win+X...) or anything outside a
  short list of safe keys and shortcuts.
- **Open non-web links** — `file:`, `javascript:`, `ms-settings:`,
  localhost and local-network addresses are refused.
- **Run code** — the AI picks from a fixed list of actions (click, type,
  press, scroll, open app/URL, note, done...). It never writes or runs code.

**What needs your F8 first** — sending, posting, submitting, buying, paying,
deleting, downloading, installing, sharing, calling, and pressing Enter in a
chat box (that's "send" on WhatsApp). Jarvis says exactly what it's about to
do — including the message it typed — then waits 30 seconds for F8. A spoken
"yes" doesn't count, so audio from a video can't approve anything.

**Prompt injection** — screen text goes to the AI marked as untrusted data,
and typing, opening apps or URLs, and every risky step are checked against
your original request by a second AI call that never sees the screen. Off-task
steps are blocked; if that check can't run, the step needs your F8 instead.

**Limits and records** — at most `AGENT_MAX_STEPS` steps and `AGENT_TIMEOUT`
seconds per task. Every step is appended to `agent_log.txt` (typed text
shortened, secrets masked). Screenshots are never saved to disk.

**Privacy trade-off** — screenshots go to Google's Gemini API. On the free
tier Google may use them to improve its products (including human review).
Private windows are never sent and password boxes are masked, but for
anything stricter use a paid Gemini key, whose data isn't used that way.

**If Chrome pages show no numbered boxes** — Chrome exposes page contents to
Windows accessibility on demand; if it doesn't on your machine, start Chrome
with `--force-renderer-accessibility` (add it to the shortcut's Target).

## How a turn flows

Wake or typed input → `record_until_silence` (16 kHz) → Moonshine STT (fully
offline) → `parse_command` routes `command` / `chat` / `both` →
`execute_actions` on a worker thread → reply via Fish Audio TTS (Moonshine fallback) → the
UI ring/status updates per state (`LISTENING` / `THINKING` / `SPEAKING`).

*One player per turn*: every path speaks through a single `_Voice` handle that
joins the previous thread first, so replies never interleave.

Design notes and known trade-offs live in `BRAHMA_ECHO_REFERENCE.md`.

When `DEV_LOG=1`, the same turn also writes aligned rows to `dev_log.txt` —
STT, TTS, the chat model chain with each model's latency, action executions,
and per-turn timings.