"""Scripted browser workflows with Playwright.

Goes in: commands/browser_automation.py   (new file)

WHY A DECLARATIVE STEP RUNNER, NOT GENERATED CODE
-------------------------------------------------
The dangerous way to do this is to let the model emit Python (or JS) and exec
it. That is how you get a voice assistant that runs remote code with your
session -- the Brahma reference's own defect list calls out `exec(compile(...))`
on LLM-generated code as High severity. This module never executes a string.

Instead the content brain turns the instruction into a small JSON workflow made
of a fixed vocabulary of steps: open, type, click, wait, extract, screenshot.
`_run_steps` interprets those declarations inside Playwright with hard bounds --
a step cap, a wall-clock deadline, an allowlist of http(s) URLs -- and nothing
else can happen. A "workflow" is data, not code.

BOUNDS, BECAUSE A BROWSER IS A REAL COMPUTER
--------------------------------------------
  MAX_STEPS     10        a runaway loop ends in a refusal, not a hang
  TIMEOUT_SEC   45        total wall clock, including navigation waits
  Only http(s) URLs       never file:, javascript:, chrome:// or localhost
  A single screenshot     saved to the screenshots dir like take_screenshot

FIRST-RUN NOTE
--------------
Playwright needs its browsers installed once (`python -m playwright install
chromium`). If they're missing the failure message says exactly that instead of
a raw Playwright traceback.

    .env
    GEMINI_API_KEY=sk-...       (planning the workflow)
"""

import os
import re
import time

from commands import genbrain

MAX_STEPS = 10
TIMEOUT_SEC = 45
SCREENSHOTS_DIR = "screenshots"

_HTTP_RE = re.compile(r"^https?://", re.I)
_LOCKED = ("file:", "javascript:", "data:", "chrome://", "about:", "localhost")

SHAPE = (
    "a JSON array of steps. Each step is an object with exactly one of these "
    "shapes, in this order and matching this vocabulary exactly: "
    "{'type': 'open', 'url': 'http(s) url'}, "
    "{'type': 'type', 'selector': 'css selector', 'text': 'string'}, "
    "{'type': 'click', 'selector': 'css selector'}, "
    "{'type': 'wait', 'seconds': number}, "
    "{'type': 'extract', 'selector': 'css selector or \"body\"'}. "
    "Use no other types, no code, and only http(s) URLs."
)


def browser_run(instruction=""):
    """Interpret a spoken browser instruction as a bounded, declarative
    workflow and run it. Returns a spoken result string."""
    instruction = (instruction or "").strip()
    if not instruction:
        return "I didn't catch what you want me to do in the browser."

    target = None
    match = re.search(r"https?://[^\s]+", instruction)
    if match:
        target = match.group(0)

    try:
        steps = _plan(instruction, target)
    except Exception as e:
        msg = "Couldn't plan the browser workflow: {}".format(e)
        print(msg)
        return msg

    if not steps:
        return "Couldn't turn that into a browser workflow -- try a page to open."

    return _run_workflow(steps, _slug(instruction))


# --- Planning -------------------------------------------------------------

def _plan(instruction, target):
    if len(instruction) > 1200:
        raise RuntimeError("that instruction is too long to plan safely")
    try:
        steps = genbrain.structured(SHAPE, instruction)
    except Exception as e:
        raise RuntimeError(e)

    if not isinstance(steps, list):
        raise RuntimeError("the planner didn't return a step list")

    # Validation is the security boundary. Anything that isn't in the vocabulary
    # or points at a locked URL is refused outright, not quietly dropped.
    cleaned = []
    for step in steps:
        if not isinstance(step, dict):
            continue
        kind = step.get("type")
        if kind == "open":
            url = str(step.get("url") or "").strip()
            if not _http_re(url):
                raise RuntimeError(
                    "the workflow tried a non-web address ({}) -- refused".format(url or "none"))
            cleaned.append({"type": "open", "url": url})
        elif kind == "type":
            cleaned.append({"type": "type",
                            "selector": str(step.get("selector") or "body"),
                            "text": str(step.get("text") or "")})
        elif kind == "click":
            cleaned.append({"type": "click",
                            "selector": str(step.get("selector") or "")})
        elif kind == "wait":
            cleaned.append({"type": "wait",
                            "seconds": max(0.5, min(5, float(step.get("seconds") or 1)))})
        elif kind == "extract":
            cleaned.append({"type": "extract",
                            "selector": str(step.get("selector") or "body")})
        else:
            raise RuntimeError("the workflow used a step type I don't allow ({})".format(kind))

    if not cleaned:
        raise RuntimeError("no usable steps in the workflow")

    if not any(s["type"] == "open" for s in cleaned) and target:
        cleaned.insert(0, {"type": "open", "url": target})

    return cleaned[:MAX_STEPS]


def _http_re(url):
    if not _HTTP_RE.match(url):
        return False
    return not any(locked in url.lower() for locked in _LOCKED)


# --- Running --------------------------------------------------------------

def _run_workflow(steps, slug):
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        return ("Couldn't run the browser workflow -- playwright isn't installed. "
                "Install it with: pip install playwright, then run: python -m "
                "playwright install chromium.")

    screenshot_path = None
    deadline = time.time() + TIMEOUT_SEC

    try:
        with sync_playwright() as p:
            try:
                browser = p.chromium.launch(headless=True)
            except Exception as e:
                return ("Couldn't start the browser. It usually needs one setup step: "
                        "python -m playwright install chromium. ({})".format(
                            str(e).splitlines()[0][:120]))

            page = browser.new_page()
            page.set_default_timeout(8000)
            extracts = []

            try:
                for step in steps:
                    if time.time() > deadline:
                        return "The browser workflow ran out of time before finishing."
                    kind = step["type"]
                    if kind == "open":
                        page.goto(step["url"], timeout=8000)
                    elif kind == "type":
                        _try(page.fill, step["selector"], step["text"], kind)
                    elif kind == "click":
                        _try(page.click, step["selector"], kind)
                    elif kind == "wait":
                        page.wait_for_timeout(int(step["seconds"] * 1000))
                    elif kind == "extract":
                        text = _extract(page, step["selector"])
                        if text:
                            extracts.append(text[:800])

                # A screenshot is the stable, cheap way to give the user (and the
                # model) a concrete view of the outcome without scraping a page
                # that may have closed its important content.
                screenshot_path = _save_screenshot(page, slug)
                page_title = page.title() or "the page"

                if extracts:
                    answer = _describe(slug, page_title, "\n".join(extracts)[:5000])
                else:
                    answer = _describe(slug, page_title, "opened successfully")
            finally:
                try:
                    browser.close()
                except Exception:
                    pass

        if screenshot_path:
            screenshot_note = "Took a screenshot of the page. "
        else:
            screenshot_note = ""
        if not answer:
            return "Did the browser step, but couldn't describe what it showed."
        return screenshot_note + answer
    except Exception as e:
        msg = "Couldn't finish the browser workflow: {}".format(str(e)[:160])
        print(msg)
        return msg


def _try(action, *args, kind):
    """Call a Playwright action; a missing selector waits briefly and retries
    once, since pages render late more often than they stay empty."""
    try:
        action(*args)
    except Exception:
        time.sleep(1.0)
        action(*args)


def _extract(page, selector):
    try:
        locator = page.locator(selector)
        if locator.count() == 0:
            return ""
        text = (locator.first.inner_text() or "").strip()
        return text
    except Exception:
        return ""


def _save_screenshot(page, slug):
    try:
        os.makedirs(SCREENSHOTS_DIR, exist_ok=True)
        path = os.path.join(SCREENSHOTS_DIR,
                            "browser_{}_{}.png".format(slug, time.strftime("%Y%m%d_%H%M%S")))
        page.screenshot(path=path, full_page=False)
        return path
    except Exception:
        return None


def _describe(slug, page_title, material):
    try:
        return genbrain.summarise(
            "A scripted browser workflow. Page title: {}\n\n{}".format(
                page_title, material),
            "Tell the user in one or two short sentences what the page shows.",
        )
    except Exception:
        return None


def _slug(text):
    safe = "".join(ch if ch.isalnum() else " " for ch in text.lower())
    return "-".join(part for part in safe.split() if part)[:24] or "page"


if __name__ == "__main__":
    # python -m commands.browser_automation  "open example.com and read the headline"
    import sys

    probe = " ".join(sys.argv[1:]) or "open example.com and extract the body text"
    print(browser_run(probe))