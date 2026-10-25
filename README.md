# Jarvis — PC Assistant

A voice-first Windows desktop assistant. Wake-word ("hey Jarvis") → speech-to-text
→ command router → executor → spoken reply, plus a live console UI, typed input,
history, notifications and an action-confirmation protocol.

## Features

- **Wake word loop** — openWakeWord (`hey_jarvis`), cooldown model reset, typed
  turns bypass the cooldown.
- **STT** — Groq Whisper (`whisper-large-v3-turbo`, usually under a second)
  when `GROQ_TTS_API_KEY` is set, falling back to Moonshine: fully offline,
  open-vocabulary, no key, no rate limit. Without the key nothing spoken
  leaves the machine.
- **Command router** (`commands/parser.py`) — classifies a turn as `command`,
  `chat`, or `both`. Chat-only turns are answered without touching the action
  engine; `both` runs the action and feeds its real output back to the brain.
- **24 command modules** — see table below.
- **TTS** — Fish Audio's S2.1 Pro cloud voice (free `s2.1-pro-free` tier,
  streamed as raw PCM), falling back to Moonshine's offline voices
  (Kokoro/Piper, via its C core) when there's no key, no network, or Fish
  refuses. Chat replies start speaking while the rest is still being written:
  sentences are grouped into a few requests, the next one downloads while the
  current one plays, and the joins are trimmed to a natural pause, and short repeated replies ("Done.") play
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
- **Desktop window** (`listener/window_ui.py`) — rests as a **floating orb**
  on a transparent, click-through background (Windows): just the ball,
  changing colour (teal listening, rose hearing you, amber thinking with an
  orbiting comet, blue speaking and swelling with the voice). It opens into a
  small card only when there's a reason: **an approval** (Approve / Cancel,
  then it shrinks back ~1.5s after you decide), **something to read** (file
  lists, clipboard, command output, or "show me...") for ~25s, or **when you
  ask** — "maximize" / "show yourself" keeps it open until "minimize". A bare
  "maximize" or "minimize" means Jarvis's own window. Double-click the orb to
  open it, "–" to shrink; drag to move; right-click for Keep on top / Quit.
  Hidden while the computer agent works. `JARVIS_NO_WINDOW=1` turns it off.
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
| `GROQ_TTS_API_KEY` | Groq Whisper speech-to-text (`listener/stt_groq.py`), fast and accurate, with Moonshine as the fallback. A separate key from `GROQ_API_KEY` so hearing and parsing use different limits. Unset = offline Moonshine only | optional |
| `MIC_SENSITIVITY` | how quiet speech can be and still count: higher hears softer voices (`1.5`), lower ignores more room noise (`0.7`); default `1.0` | optional |
| `RECORD_SILENCE_SECONDS` | how long a pause ends a recording; default `1.2` (raise it if you get cut off) | optional |
| `STT_ENGINE` | `moonshine` keeps hearing offline even when a Groq key is set | optional |
| `MOONSHINE_COMPRESSION` | Moonshine STT model size (`listener/stt_moonshine.py`); default `medium_streaming` | optional |
| `FISH_API_KEY` | primary TTS voice, Fish Audio S2.1 Pro (`listener/tts_fish.py`); without it Moonshine speaks | recommended |
| `TELEGRAM_BOT_TOKEN` | Telegram bot token from @BotFather; turns on phone control (see "Control from your phone") | optional |
| `TELEGRAM_SPEAK` | `1` also says Telegram replies out loud at the PC (default `0`) | optional |
| `TELEGRAM_NOTIFY_AFTER` | seconds a PC command must run before the phone is told it's done (default `60`, `0` = never) | optional |
| `JARVIS_REMOTE_PIN` | lets Telegram approve commands, recipes and anything that needs a yes at the PC (6+ characters with a digit); unset = never from the phone | optional |
| `TELEGRAM_ENABLED` | `0` turns Telegram control off even with a token | optional |
| `FISH_VOICE_ID` | Fish Audio voice model id (`reference_id`); unset uses a calm British Jarvis voice, `default` uses Fish's stock voice (see "Name, voice and wake word") | optional |
| `FISH_MODEL` | Fish Audio model header; default `s2.1-pro-free` (free tier), `s2.1-pro` for paid | optional |
| `FISH_VOLUME_DB` | voice loudness boost in dB, applied by Fish (`prosody.volume`); default `3`, range -12 to 12 | optional |
| `FISH_SAMPLE_RATE` | Fish audio sample rate: `44100` (default, Fish's native rate, nothing downsampled), `32000`, `24000`, `16000`, `8000`; the speaker opens at this rate so it never switches mid-reply | optional |
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
| `JARVIS_NO_WINDOW` | set non-empty to skip the desktop window | optional |
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
| `agent_input.py` | real mouse/keyboard input (SendInput) + Esc, Ctrl+Enter approval, corner brakes |
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
| **Ctrl+Enter** (physical keys; **F8** also works) | approves the one risky step it just described |

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

**What needs your Ctrl+Enter first** — sending, posting, submitting, buying, paying,
deleting, downloading, installing, sharing, calling, and pressing Enter in a
chat box (that's "send" on WhatsApp). Jarvis says exactly what it's about to
do — including the message it typed — then waits 30 seconds for Ctrl+Enter. While it waits, those keys go only to
Jarvis -- they don't also reach the app in front. A spoken
"yes" doesn't count, so audio from a video can't approve anything.

**Prompt injection** — screen text goes to the AI marked as untrusted data,
and typing, opening apps or URLs, and every risky step are checked against
your original request by a second AI call that never sees the screen. Off-task
steps are blocked; if that check can't run, the step needs your Ctrl+Enter instead.

**Direct commands too** — outside the agent, opening a protected system app
("open PowerShell") or running a shell command that isn't on the read-only
safe list asks first, and that question only takes a **typed** "yes" in
Jarvis's window or the notification's Yes button, never a spoken one.
Read-only commands on the safe list (like `dir`) run straight away.

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

## Name, voice and wake word

The assistant is **Jarvis**, woken by **"Hey Jarvis"** (openWakeWord's
built-in model). Tune waking with `WAKE_THRESHOLD` (default 0.40; higher =
fewer false wakes, lower = wakes more easily).

**Voice** — by default Fish speaks with a calm British "Jarvis" voice
([preview](https://fish.audio/m/14129c3e320149449d6bada6862f7338)). To try
another, preview it on fish.audio and put its id in `.env` as
`FISH_VOICE_ID=<id>`:

| Voice | Id |
|---|---|
| JARVIS — sophisticated, subtly witty | `88b183bfbdda44a09ad4474712b967f4` |
| JARVIS — refined, measured | `41f0953d7a6b4c078445c7e65d620eeb` |
| Jarvis (MCU) | `05b36da8574341d0803391491850db20` |
| Brian — British, authoritative but gentle | `65c0b8155c464a648161af8877404f11` |

`FISH_VOICE_ID=default` goes back to Fish's own stock voice. Cached short
replies are kept per voice, so switching never plays the old one.

## Running commands in the background

*"In pc-assistant, git pull then pip install -r requirements.txt, and tell me
when it's done."* Jarvis finds the folder by name, reads the plan back once
("In pc-assistant: git pull, then pip install... Should I?"), and after your
yes runs it in the background with no time limit. Each command runs only if
the one before it worked; when it ends you get a notification and a Telegram
message saying what happened (or the error line). The full output is in
`Documents\Jarvis\Logs` -- "send me the task log" from Telegram.

- "Is the task done?" / "Cancel the task"
- **Recipes:** "save that as update jarvis" (at the PC), then "run update
  jarvis" any time. At the PC a recipe runs straight away -- you approved
  those exact commands when you saved it. From the phone it needs your PIN.

Every command is checked against the deny list first (format, diskpart,
mass deletes...), recipes included, every time they run.

**From your phone, with a PIN.** Set `JARVIS_REMOTE_PIN` in `.env` (6+
characters with a digit, not something like 123456). Then commands, recipes
and anything else that needs a yes at the PC can be approved from Telegram:
Jarvis asks, you send the PIN as its own message, and it deletes that
message from the chat so nobody can scroll back to it. The PIN never reaches
a model or a log. 3 wrong PINs lock phone approvals for 30 minutes; 3 more
unpair Telegram. Without a PIN, none of this works from the phone at all.
The computer agent (mouse and keyboard) is never available from the phone.

## Checking on your apps

Ask about one app and Jarvis looks at **that window alone** -- even behind
other windows, even minimized -- reads its text (terminals included), and
tells you what's going on:

- "Is Claude done?" / "What's Claude doing rn?"
- "What's happening in my terminal?" / "Is the build in VS Code finished?"
- "Tell me when Claude is done" -- it watches and pings you (notification,
  plus Telegram if paired) when it finishes, **needs your approval**, errors
  or the window closes. "Stop watching" ends it.
- From Telegram, "show me Claude" sends a picture of just that window.

The watcher is light: every 30 seconds it compares a tiny thumbnail of the
window on the PC (milliseconds, no model call), and asks Gemini only once the
window has stopped changing -- a long task costs a handful of calls. Up to 3
apps at once, for up to 4 hours.

Same privacy rules as screen vision: private windows are refused, password
boxes are blacked out, and text that looks like a key or token is hidden
before anything is sent. A status answer never sends a picture; only "show
me <app>" does. Needs `GEMINI_API_KEY`.

## Personality

Jarvis is calm and composed with a dry, understated wit: it answers first,
skips the filler, and keeps the jokes for relaxed moments (never when
something failed). It calls you by your name now and then -- whatever you
told it to call you ("call me Ezekiel") -- and never "sir".

**It learns how you talk.** After your first dozen messages, and every 30
after that, one small model call reads what you've said to it lately and
writes a short note on your style: message length, slang you actually use,
energy, how much detail you want. Jarvis drifts toward that while staying
itself. It's built only from what you say to Jarvis (anything that looks
like a password or card number is skipped) and kept in `data/style.json`.

- "How do you think I talk?" -- hear the note
- "Learn how I talk now" -- update it straight away
- "Forget my style" -- wipe it and start fresh

Direct instructions ("keep it shorter", "stop using my name") are saved as
preferences and always override the personality.

## Control from your phone (Telegram)

Message Jarvis from anywhere: text or voice notes, replies come back as text
and the PC stays quiet.

1. In Telegram, message **@BotFather**, send `/newbot`, and pick a name and a
   username ending in "bot".
2. Put the token it gives you in `.env`: `TELEGRAM_BOT_TOKEN=...`
3. Restart Jarvis and send your bot any message. A **6-digit code** appears
   on the PC; send it to the bot. That account is now the only one answered.

From the phone: any command ("open Chrome", "how's my battery", "make a PDF
about solar panels" -- the file comes back to you), voice notes (needs
`GROQ_TTS_API_KEY`), `/screenshot`, `send me <file>` (from
`Documents\Jarvis`), `/speak on|off`, `/status`, `/unpair`. A command you
start at the PC that runs over a minute messages you when it's done.

**Security.** No open ports: the PC fetches messages over an outgoing
connection, so nothing can connect in. Only the paired account in a private
chat gets a reply; pairing needs the code shown on the PC (5 wrong codes lock
it for 15 minutes), so a leaked token alone controls nothing. From the phone
you can answer ordinary yes/no questions; anything that needs approval at the
PC -- commands, recipes, protected apps -- needs your `JARVIS_REMOTE_PIN`
(see "Running commands in the background"), and the computer agent is
refused. Messages older than 2 minutes are dropped,
every remote command shows a notification on the PC, and screenshots refuse
private windows and black out password boxes. "Unpair telegram" at the PC
forgets the account; `TELEGRAM_ENABLED=0` turns it all off.

## Memory

| Kind | Example | Kept |
|---|---|---|
| Preferences | "keep answers short" | forever (`jarvis_prefs.json`) |
| Profile | your name, what to call you, where you live, what you do — in **every** prompt | forever (`data/profile.json`) |
| Facts | "my sister is Tolu, saved as Tolu Ade on WhatsApp" | forever (`data/facts.json`) |
| Conversation | the last few turns | across restarts within 6 hours (`data/conversation.jsonl`) |

Say **"get to know me"** for a quick interview (name, city, work, the people
you message, how you like answers, interests): one question at a time, no wake
word needed between them, "skip" or "stop" any time, and it picks up where you
left off if you go quiet. Or say "remember that..." -- or just mention a lasting fact ("text my sister
Tolu...") and Jarvis keeps it. "Forget that..." removes it; "what do you
remember" lists it. Each turn pulls in only the few facts relevant to what you
said (a local keyword match, well under a millisecond), so memory never slows
a reply or bloats the prompts. Passwords, PINs, codes, card/account numbers
and your API keys are refused -- the files are plain text on your disk.

## Where files go

Everything Jarvis makes lands in `Documents\Jarvis`, one folder per kind:

| Folder | What |
|---|---|
| `Documents` | Word documents |
| `Presentations` | slide decks |
| `Spreadsheets` | Excel files |
| `PDFs` | PDF reports |
| `Sites` | generated websites |
| `Screenshots` | screenshots |

Unnamed files are dated so they sort in order (`2026-10-23 Queue Theory.docx`);
a name you give is used as is ("called Q3 report" → `Q3 report.pdf`). Say a
folder to put it elsewhere ("...in downloads", "...on my desktop"). Nothing
is ever overwritten -- a clash gets `(2)`.

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