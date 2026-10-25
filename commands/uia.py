"""What's on screen, as Windows itself describes it (UI Automation).

Goes in: commands/uia.py   (new file)

Every button, link, text box and menu item a window exposes -- the same tree
screen readers use -- with its exact on-screen rectangle. commands/agent.py
numbers these and draws the numbers on the screenshot, so the model answers
"click 14" instead of guessing pixel coordinates, which free vision models are
bad at. commands/screen_actions.py uses the password flag to black out
password boxes before a screenshot leaves the machine.

ONE CALL, NOT A TREE WALK
-------------------------
Walking the tree element by element is a cross-process COM call per property
per node -- seconds on a busy Chrome page. Instead this asks for every
on-screen descendant in one FindAllBuildCache with the properties we need
cached alongside, so the whole snapshot is one round trip.

Uses comtypes (already a dependency for pycaw); the UIAutomationCore type
library ships with Windows. No new packages.

PASSWORDS
---------
A box Windows flags as a password field (IsPassword) is reported with
"password": True and its value is never read. The agent refuses to type into
one and the screen reader blacks it out.
"""

import threading

_CONTROL_TYPES = {
    "Button": "button", "CheckBox": "checkbox", "ComboBox": "combobox",
    "Edit": "edit", "Hyperlink": "link", "ListItem": "listitem",
    "MenuItem": "menuitem", "RadioButton": "radio", "Slider": "slider",
    "SplitButton": "splitbutton", "TabItem": "tab", "TreeItem": "treeitem",
    "DataItem": "dataitem", "Spinner": "spinner", "Text": "text",
    "Document": "document", "Image": "image", "Custom": "custom",
    "Group": "group", "Pane": "pane", "Window": "window", "List": "list",
    "Menu": "menu", "MenuBar": "menubar", "ToolBar": "toolbar",
    "TitleBar": "titlebar", "Header": "header", "HeaderItem": "headeritem",
    "Table": "table", "Tree": "tree", "Tab": "tabs", "StatusBar": "statusbar",
    "ScrollBar": "scrollbar", "ProgressBar": "progressbar",
    "Calendar": "calendar", "Thumb": "thumb", "Separator": "separator",
    "ToolTip": "tooltip", "DataGrid": "datagrid", "AppBar": "appbar",
}

# What the agent can act on. Everything else is context at most.
INTERACTIVE = {
    "button", "checkbox", "combobox", "edit", "link", "listitem", "menuitem",
    "radio", "slider", "splitbutton", "tab", "treeitem", "dataitem",
    "spinner", "headeritem",
}

MAX_NAME = 80

_lock = threading.Lock()
_module = None


def _uia():
    """A fresh IUIAutomation for the calling thread, plus the type library.

    Fresh per call on purpose: COM objects belong to the thread (apartment)
    that made them, and this is called from the executor's worker threads,
    which come and go. Creating the object costs a few milliseconds.
    """
    global _module
    import comtypes
    import comtypes.client

    comtypes.CoInitialize()  # harmless when the thread already has COM
    with _lock:
        if _module is None:
            comtypes.client.GetModule("UIAutomationCore.dll")
            from comtypes.gen import UIAutomationClient
            _module = UIAutomationClient
    api = comtypes.client.CreateObject(
        _module.CUIAutomation, interface=_module.IUIAutomation)
    return api, _module


def _type_names(mod):
    out = {}
    for key, short in _CONTROL_TYPES.items():
        value = getattr(mod, "UIA_{}ControlTypeId".format(key), None)
        if value is not None:
            out[value] = short
    return out


def _clip(text):
    text = " ".join(str(text or "").split())
    return text if len(text) <= MAX_NAME else text[:MAX_NAME - 3] + "..."


def _inside(rect, bounds):
    left, top, right, bottom = rect
    if right - left < 4 or bottom - top < 4:
        return False
    if bounds is None:
        return True
    bl, bt, br, bb = bounds
    cx, cy = (left + right) // 2, (top + bottom) // 2
    return bl <= cx < br and bt <= cy < bb


def snapshot(hwnd, bounds=None, max_items=160, max_text_chars=5000):
    """Everything on screen in window `hwnd`, as plain Python.

    Returns {"elements": [...], "text": "..."}:
      elements: interactive things, in reading order, each
        {"type", "name", "value", "rect": (l, t, r, b), "password", "focused"}
      text:     visible static text (labels, paragraphs), joined and capped --
                what the agent reads when it's researching a page.
    `bounds` (l, t, r, b), usually the monitor, drops anything whose centre is
    outside it. Never raises: a window that won't answer gives empty results.
    """
    try:
        api, mod = _uia()
        root = api.ElementFromHandle(hwnd)
        if root is None:
            return {"elements": [], "text": ""}

        cache = api.CreateCacheRequest()
        for prop in (
            mod.UIA_NamePropertyId,
            mod.UIA_ControlTypePropertyId,
            mod.UIA_BoundingRectanglePropertyId,
            mod.UIA_IsEnabledPropertyId,
            mod.UIA_IsPasswordPropertyId,
            mod.UIA_HasKeyboardFocusPropertyId,
            mod.UIA_IsKeyboardFocusablePropertyId,
            mod.UIA_ValueValuePropertyId,
        ):
            cache.AddProperty(prop)
        onscreen = api.CreatePropertyCondition(mod.UIA_IsOffscreenPropertyId, False)
        found = root.FindAllBuildCache(mod.TreeScope_Descendants, onscreen, cache)
        names = _type_names(mod)

        elements, texts, seen = [], [], set()
        text_len = 0
        for i in range(found.Length if found is not None else 0):
            el = found.GetElement(i)
            r = el.CachedBoundingRectangle
            rect = (int(r.left), int(r.top), int(r.right), int(r.bottom))
            if not _inside(rect, bounds):
                continue
            kind = names.get(el.CachedControlType, "other")
            name = _clip(el.CachedName)
            password = bool(el.CachedIsPassword)
            focusable = bool(el.CachedIsKeyboardFocusable)

            if kind == "text":
                if name and text_len < max_text_chars:
                    texts.append(name)
                    text_len += len(name) + 1
                continue
            if password:
                # Odd custom controls can still be password boxes; keep every
                # one so it gets masked and refused, whatever its type.
                kind = "edit"
            elif kind not in INTERACTIVE and not (kind == "custom" and focusable):
                continue
            if not password and not bool(el.CachedIsEnabled):
                continue

            key = (kind, name, rect)
            if key in seen:
                continue
            seen.add(key)

            value = ""
            if not password:
                try:
                    value = _clip(el.GetCachedPropertyValue(
                        mod.UIA_ValueValuePropertyId) or "")
                except Exception:
                    value = ""
            elements.append({
                "type": kind,
                "name": name,
                "value": value,
                "rect": rect,
                "password": password,
                "focused": bool(el.CachedHasKeyboardFocus),
            })

        elements.sort(key=lambda e: (e["rect"][1] // 12, e["rect"][0]))
        # Password boxes always survive the cap: they're what gets masked.
        kept = elements[:max_items]
        kept += [e for e in elements[max_items:] if e["password"]]
        return {"elements": kept, "text": "\n".join(texts)}
    except Exception as e:
        print("uia: couldn't read that window ({})".format(e))
        return {"elements": [], "text": ""}


def bounded(fn, timeout, default):
    """fn() on its own thread, giving up after `timeout` seconds.

    UI Automation asks the app itself, and a big or busy one -- VS Code or
    Chrome with many tabs on a slow laptop -- can take minutes to answer.
    Without a bound that freezes whoever asked (the main loop, so Telegram
    stops answering). An abandoned call finishes in the background."""
    box = {}

    def run():
        try:
            box["value"] = fn()
        except Exception as e:
            box["error"] = e

    thread = threading.Thread(target=run, name="uia-bounded", daemon=True)
    thread.start()
    thread.join(timeout)
    if thread.is_alive():
        print("uia: the window took longer than {}s to answer -- skipped".format(timeout))
        return default
    return box.get("value", default)


def focused():
    """{"name", "type", "password"} for whatever has keyboard focus, or None."""
    try:
        api, mod = _uia()
        el = api.GetFocusedElement()
        if el is None:
            return None
        return {
            "name": _clip(el.CurrentName),
            "type": _type_names(mod).get(el.CurrentControlType, "other"),
            "password": bool(el.CurrentIsPassword),
        }
    except Exception:
        return None
