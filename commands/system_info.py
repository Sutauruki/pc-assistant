"""How the PC is doing -- RAM, CPU, disk, battery, uptime -- in plain English.

Goes in: commands/system_info.py   (new file)

"How much RAM am I using?" used to become a shell command (wmic), which
needed your approval and then read its raw output aloud:
"FreePhysicalMemory=2677600 TotalVisibleMemorySize=11986028". This reads the
same numbers through psutil (already a dependency) -- no shell, nothing to
approve -- and says them the way a person would: "You're using 8.9 of your
11.4 GB of RAM, about 78 percent. 2.6 GB is free."

PLAIN QUESTION OR SPECIFIC ONE
------------------------------
"How much RAM am I using" gets that sentence straight away. Anything more
specific -- "what's the smallest user of my RAM", "which app is eating my
CPU", "how many Chrome processes" -- can't be answered by a fixed sentence,
so main.py asks is_simple() and, when it's False, hands details() (every
app's memory, CPU per app, all drives...) to the chat model to answer the
question actually asked.
"""

import os
import re
import time

import psutil


def _gb(n):
    value = n / float(1024 ** 3)
    return "{:.1f}".format(value) if value < 100 else "{:.0f}".format(value)


def ram():
    m = psutil.virtual_memory()
    line = "You're using {} of your {} GB of RAM, about {} percent. {} GB is free.".format(
        _gb(m.total - m.available), _gb(m.total), int(round(m.percent)), _gb(m.available))
    top = _top_memory()
    if top:
        line += " The biggest user is {}, at {} GB.".format(top[0], _gb(top[1]))
    return line


def _memory_by_app():
    """{name: (bytes, process count)}, grouped by name -- Chrome is one
    entry, not forty."""
    totals = {}
    for proc in psutil.process_iter(["name", "memory_info"]):
        try:
            name = (proc.info["name"] or "").replace(".exe", "")
            mem = proc.info["memory_info"]
            if name and mem:
                used, count = totals.get(name, (0, 0))
                totals[name] = (used + mem.rss, count + 1)
        except (psutil.Error, KeyError):
            continue
    return totals


def _top_memory():
    """(name, bytes) of the app using the most memory."""
    totals = _memory_by_app()
    if not totals:
        return None
    name = max(totals, key=lambda n: totals[n][0])
    return name[:1].upper() + name[1:], totals[name][0]


def cpu():
    percent = psutil.cpu_percent(interval=0.5)
    cores = psutil.cpu_count(logical=True) or 0
    return "Your processor is at {} percent across {} threads.".format(int(round(percent)), cores)


def disk(path=None):
    path = path or (os.environ.get("SystemDrive", "C:") + "\\" if os.name == "nt" else "/")
    d = psutil.disk_usage(path)
    label = path.rstrip("\\/") or path
    # From free/total rather than psutil's percent, which leaves out
    # reserved space -- so the three numbers in the sentence agree.
    full = 100.0 * (d.total - d.free) / d.total if d.total else 0
    return "{} has {} GB free of {} GB, {} percent full.".format(
        label, _gb(d.free), _gb(d.total), int(round(full)))


def battery():
    b = psutil.sensors_battery()
    if b is None:
        return "This PC doesn't report a battery."
    state = "charging" if b.power_plugged else "on battery"
    line = "Battery is at {} percent, {}.".format(int(round(b.percent)), state)
    if not b.power_plugged and b.secsleft not in (psutil.POWER_TIME_UNKNOWN,
                                                   psutil.POWER_TIME_UNLIMITED):
        hours, minutes = divmod(int(b.secsleft) // 60, 60)
        line += " About {} left.".format(
            "{}h {}m".format(hours, minutes) if hours else "{} minutes".format(minutes))
    return line


def uptime():
    seconds = int(time.time() - psutil.boot_time())
    days, rest = divmod(seconds, 86400)
    hours = rest // 3600
    if days:
        return "The PC has been on for {} day{} and {} hour{}.".format(
            days, "" if days == 1 else "s", hours, "" if hours == 1 else "s")
    if hours:
        return "The PC has been on for {} hour{}.".format(hours, "" if hours == 1 else "s")
    minutes = max(1, seconds // 60)
    return "The PC has been on for {} minute{}.".format(minutes, "" if minutes == 1 else "s")


_TOPICS = {
    "ram": ram, "memory": ram, "mem": ram,
    "cpu": cpu, "processor": cpu,
    "disk": disk, "storage": disk, "space": disk, "drive": disk,
    "battery": battery, "power": battery,
    "uptime": uptime,
}


def system_info(what="all"):
    """Spoken answer for `what` (ram, cpu, disk, battery, uptime, or all)."""
    topic = str(what or "all").strip().lower()
    try:
        if topic in _TOPICS:
            return _TOPICS[topic]()
        return " ".join(f() for f in (ram, cpu, disk, battery))
    except Exception as e:
        return "I couldn't read that: {}".format(e)


# Words that ask for more than the one-line summary: a ranking, a specific
# app, a comparison, a reason. Any of them means "answer from details()".
_SPECIFIC = re.compile(
    r"\b(small(est)?|least|lowest|low|top|biggest|most|largest|heaviest|which|"
    r"what (app|apps|program|programs|process|processes)|who|list|each|every|"
    r"all (the|my) (apps|programs|processes)|why|slow|lag|compare|second|third|"
    r"top \d+|\d+ (apps|programs|processes|users)|how many (apps|programs|"
    r"processes|tabs|cores|threads|windows)|per|core|cores|speed|ghz|swap|"
    r"page ?file|drive|drives|enough|"
    r"partition|chrome|edge|firefox|claude|code|game|close|kill|should i|"
    r"health|temperature|hot|time left|how long will|history)\b", re.I)


def is_simple(text):
    """True when the one-line summary answers `text` ("how much RAM am I
    using", "battery?"); False when it asks for something more specific."""
    text = str(text or "")
    return len(text.split()) <= 14 and not _SPECIFIC.search(text)


def _mb(n):
    mb = n / float(1024 ** 2)
    return "{:.1f} GB".format(mb / 1024) if mb >= 1024 else "{:.0f} MB".format(mb)


def _ram_details():
    m = psutil.virtual_memory()
    lines = ["RAM: {} used of {} ({}%), {} available.".format(
        _mb(m.total - m.available), _mb(m.total), int(round(m.percent)), _mb(m.available))]
    try:
        s = psutil.swap_memory()
        if s.total:
            lines.append("Page file/swap: {} used of {}.".format(_mb(s.used), _mb(s.total)))
    except Exception:
        pass
    apps = sorted(_memory_by_app().items(), key=lambda kv: kv[1][0], reverse=True)
    lines.append("Apps by memory, largest first ({} apps, name: memory [processes]):".format(len(apps)))
    shown = apps[:40]
    for name, (used, count) in shown:
        lines.append("  {}: {}{}".format(name, _mb(used), " [{}]".format(count) if count > 1 else ""))
    if len(apps) > 40:
        lines.append("  ...{} more, each smaller. Smallest:".format(len(apps) - 40))
        for name, (used, count) in apps[-5:]:
            lines.append("  {}: {}".format(name, _mb(used)))
    return lines


def _cpu_details():
    procs = []
    for proc in psutil.process_iter(["name"]):
        try:
            proc.cpu_percent(None)  # first call primes the counter
            procs.append(proc)
        except psutil.Error:
            continue
    overall = psutil.cpu_percent(interval=0.5)
    by_app = {}
    for proc in procs:
        try:
            name = (proc.info["name"] or "").replace(".exe", "")
            pct = proc.cpu_percent(None)
            if name and name.lower() != "system idle process":
                by_app[name] = by_app.get(name, 0.0) + pct
        except psutil.Error:
            continue
    cores = psutil.cpu_count(logical=True) or 1
    lines = ["CPU: {}% overall, {} physical cores / {} threads.".format(
        int(round(overall)), psutil.cpu_count(logical=False) or "?", cores)]
    try:
        freq = psutil.cpu_freq()
        if freq and freq.current:
            lines.append("Clock: {:.1f} GHz (max {:.1f} GHz).".format(
                freq.current / 1000, (freq.max or freq.current) / 1000))
    except Exception:
        pass
    top = sorted(by_app.items(), key=lambda kv: kv[1], reverse=True)[:12]
    if top:
        # psutil counts per thread; divide so 100% means the whole processor.
        lines.append("Apps by CPU right now (% of the whole processor):")
        for name, pct in top:
            lines.append("  {}: {:.1f}%".format(name, pct / cores))
    return lines


def _disk_details():
    lines = ["Drives:"]
    for part in psutil.disk_partitions(all=False):
        if "cdrom" in part.opts or not part.fstype:
            continue
        try:
            d = psutil.disk_usage(part.mountpoint)
        except Exception:
            continue
        lines.append("  {} ({}): {} free of {}.".format(
            part.mountpoint.rstrip("\\/") or part.mountpoint, part.fstype,
            _mb(d.free), _mb(d.total)))
    return lines


def _battery_details():
    line = battery()
    return ["Battery: " + line]


def details(what="all"):
    """Everything worth knowing about `what`, as plain lines for the chat
    model to answer a specific question from. Not meant to be read aloud."""
    topic = str(what or "all").strip().lower()
    parts = {"ram": _ram_details, "cpu": _cpu_details, "disk": _disk_details,
             "battery": _battery_details, "uptime": lambda: [uptime()]}
    alias = {"memory": "ram", "mem": "ram", "processor": "cpu", "storage": "disk",
             "space": "disk", "drive": "disk", "power": "battery"}
    topic = alias.get(topic, topic)
    chosen = [topic] if topic in parts else ["ram", "cpu", "disk", "battery", "uptime"]
    lines = []
    for name in chosen:
        try:
            lines += parts[name]()
        except Exception as e:
            lines.append("{}: couldn't read ({})".format(name, e))
    return "\n".join(lines)


if __name__ == "__main__":
    # python -m commands.system_info [ram|cpu|disk|battery|uptime] [--details]
    import sys
    args = [a for a in sys.argv[1:] if a != "--details"]
    fn = details if "--details" in sys.argv else system_info
    print(fn(args[0] if args else "all"))
