# Roadmap

Ideas taken from Dexter (an OpenClaw fork), rebuilt to stay light on a
low-end laptop (HP EliteBook 850 G1: 2-core Haswell, no GPU).

## Planned

1. **Telegram remote control.** *(Done -- README, "Control from your phone".)* Message Jarvis from your phone; it can
   message you back when a long task finishes. Only a paired account is
   answered, and protected actions still need Ctrl+Enter at the PC. Cost:
   one idle background thread.
2. **Reminders, timers and routines.** "Remind me in 20 minutes...", "every
   weekday at 8, tell me the weather and my battery". Saved to disk so they
   survive a restart; can reach your phone through Telegram. Cost: one
   sleeping thread.
3. **Session voice commands.** "Start fresh", "what model are you on",
   "think harder" / "be quick", "switch to Gemini". Free.
4. **Summarize any page.** The page open in Chrome, or any link. Runs in
   the cloud.
5. **Checkup.** "Run a checkup": which keys work, whether Fish and Groq are
   reachable, mic level, disk space, which speech engines are live. Runs
   only on request.

6. **Write as me.** Draft WhatsApp replies and emails in your own voice,
   from the learned style note, for you to approve before anything is sent.

## Done

- **Personality and learned style** (README, "Personality").
- **Checking on apps and the watcher** (README, "Checking on your apps").

## Not planned

Too heavy for this laptop, or not useful here: vector memory (LanceDB),
the Canvas UI, multi-agent routing, phone companion apps, bridges to many
chat apps.
