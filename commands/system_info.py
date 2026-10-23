"""How the PC is doing -- RAM, CPU, disk, battery, uptime -- in plain English.

Goes in: commands/system_info.py   (new file)

"How much RAM am I using?" used to become a shell command (wmic), which
needed your approval and then read its raw output aloud:
"FreePhysicalMemory=2677600 TotalVisibleMemorySize=11986028". This reads the
same numbers through psutil (already a dependency) -- no shell, nothing to
approve -- and says them the way a person would: "You're using 8.9 of your
11.4 GB of RAM, about 78 percent. 2.6 GB is free."
"""

import os
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


def _top_memory():
    """(name, bytes) of the app using the most memory, grouped by name --
    Chrome is one entry, not forty."""
    totals = {}
    for proc in psutil.process_iter(["name", "memory_info"]):
        try:
            name = (proc.info["name"] or "").replace(".exe", "")
            mem = proc.info["memory_info"]
            if name and mem:
                totals[name] = totals.get(name, 0) + mem.rss
        except (psutil.Error, KeyError):
            continue
    if not totals:
        return None
    name = max(totals, key=totals.get)
    return name[:1].upper() + name[1:], totals[name]


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


if __name__ == "__main__":
    # python -m commands.system_info [ram|cpu|disk|battery|uptime]
    import sys
    print(system_info(sys.argv[1] if len(sys.argv) > 1 else "all"))
