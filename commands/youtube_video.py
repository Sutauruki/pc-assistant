"""Fetch and summarise a YouTube transcript.

Goes in: commands/youtube_video.py   (new file)

     .env
     GEMINI_API_KEY=sk-...       (summarising answer)

TWO WAYS IN
-----------
The user gives a URL ("youtube.com/watch?v=..."), or a search term and we find
the top match ("the new iPhone review"). Parsing a video id out of a URL works
for watch links and short links, and the youtube-transcript-api handles the
fetch -- including auto-generated captions when the uploader never added real
ones.

WHAT COMES BACK
---------------
A spoken summary of the transcript, joined to the video title if we could
fetch it. The transcript stays on screen in the log; only its meaning is
spoken, because nobody wants a three-minute video read back word for word.
"""

import re

from commands import genbrain

ID_PATTERN = re.compile(
    r"(?:v=|youtu\.be/|shorts/|youtube\.com/embed/|youtube\.com/watch[?&]?v=)?([A-Za-z0-9_-]{11})"
)
_META_CHARS = 500


def youtube_summary(url_or_query="", question=""):
    """Return a spoken summary of a video's transcript."""
    query = (url_or_query or "").strip()
    if not query:
        return "I didn't catch which video you want the transcript of."

    try:
        from youtube_transcript_api import YouTubeTranscriptApi
    except ImportError:
        return "Couldn't fetch the transcript -- youtube-transcript-api isn't installed."

    video_id = _video_id(query)
    if not video_id:
        video_id = _search_first(query)
        if not video_id:
            return "Couldn't find a video matching {}.".format(query)

    try:
        transcript = YouTubeTranscriptApi().fetch(video_id)
        text = " ".join(part.get("text", "") for part in transcript)
    except Exception as e:
        msg = "Couldn't fetch the transcript for that video: {}".format(e)
        print(msg)
        return msg

    if not text.strip():
        return "That video has no captions I could pull down."

    title = _title(video_id)
    material = "Video title: {}\n\nTranscript: {}".format(title, text)

    try:
        answer = genbrain.summarise(material, question)
    except Exception as e:
        return "Couldn't summarise the transcript: {}".format(e)

    prefix = ("{}\n".format(title)) if title else ""
    out = prefix + "".join(answer)
    print(out)
    return out


def _video_id(query):
    """The 11-char id out of a YouTube URL, or the query itself when it is one."""
    match = ID_PATTERN.search(query)
    if not match:
        return None
    return match.group(1)


def _search_first(query):
    """Best-effort: find the top video for a phrase via DuckDuckGo."""
    try:
        import requests
        from bs4 import BeautifulSoup
    except ImportError:
        return None

    try:
        response = requests.get(
            "https://duckduckgo.com/html/",
            params={"q": query + " youtube"},
            headers={"User-Agent": "Mozilla/5.0"},
            timeout=10,
        )
        response.raise_for_status()
    except Exception:
        return None

    links = BeautifulSoup(response.text, "html.parser").select("a.result__a")
    for link in links[:6]:
        href = link.get("href", "")
        match = ID_PATTERN.search(href)
        if match:
            return match.group(1)
    return None


def _title(video_id):
    """The <title> tag without the site boilerplate."""
    try:
        import requests
        from bs4 import BeautifulSoup
    except ImportError:
        return ""

    try:
        response = requests.get(
            "https://www.youtube.com/watch?v={}".format(video_id),
            headers={"User-Agent": "Mozilla/5.0"},
            timeout=10,
        )
        soup = BeautifulSoup(response.text, "html.parser")
        title = (soup.title.string or "").strip()
        if " - YouTube" in title:
            return title.replace(" - YouTube", "").strip()
        return title[:_META_CHARS]
    except Exception:
        return ""


if __name__ == "__main__":
    # python -m commands.youtube_video  <url or query>
    import sys

    probe = " ".join(sys.argv[1:]) or "an intro to machine learning"
    print(youtube_summary(probe))