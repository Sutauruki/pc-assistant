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
- **TTS** — Moonshine's local voices (Kokoro/Piper, via its C core), played
  straight through with no time-stretch, with overlap: the reply starts while
  actions are still running.
- **Screen understanding** — grabs the screen, downscales to 640×360 JPEG
  quality 55, and answers "what's on my screen" / "what does that error say" via
  Gemini Vision (`commands/screen_actions.py`).
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
| `MOONSHINE_VOICE` | Moonshine TTS voice (`listener/tts_moonshine.py`); default `kokoro_af_sky` | optional |
| `GEMINI_API_KEY` | screen understanding (`commands/screen_actions.py`) + all Tier 6 content generation and summarising (`commands/genbrain.py`) | only if those features are used |
| `OPENROUTER_API_KEY` | chat brain (`commands/chat.py:184`) | for conversational chat |
| `GROQ_API_KEY` | parser brain (`commands/parser.py:388`) | for command routing |
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
| `screen_actions.py` | screen understanding (mss grab → 640×360 JPEG q55 → Gemini Vision) |
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
| `browser_automation.py` | bounded scripted browser workflows (Playwright, declarative steps only) |
| `website_builder.py` | prompt → small static site + local preview (single non-leaking server on 127.0.0.1) |

## How a turn flows

Wake or typed input → `record_until_silence` (16 kHz) → Moonshine STT (fully
offline) → `parse_command` routes `command` / `chat` / `both` →
`execute_actions` on a worker thread → reply via Moonshine TTS → the
UI ring/status updates per state (`LISTENING` / `THINKING` / `SPEAKING`).

*One player per turn*: every path speaks through a single `_Voice` handle that
joins the previous thread first, so replies never interleave.

Design notes and known trade-offs live in `BRAHMA_ECHO_REFERENCE.md`.

When `DEV_LOG=1`, the same turn also writes aligned rows to `dev_log.txt` —
STT, TTS, the chat model chain with each model's latency, action executions,
and per-turn timings.