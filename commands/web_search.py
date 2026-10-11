"""Search the web and summarise the top results.

Goes in: commands/web_search.py   (new file)

     .env
     GEMINI_API_KEY=sk-...       (summarising answer)

`ddgs` does the lookup (no API key, no cost) and returns titles, snippets and
URLs. Those top results are handed to the content brain to be turned into one
spoken answer -- searching is only half the job, and a wall of ten snippets is
not what someone asking out loud wants to hear.

When the user asked a pointed question ("is it raining in Tokyo"), the top
result snippet is fetched and its body text appended to the mix, so the answer
can draw on a page rather than on its search-engine blurb. Everything stays a
read -- nothing is written to disk, which is why this joins SPOKEN_RESULTS.
"""

from commands import genbrain

MAX_RESULTS = 6
FRAGMENT_CHARS = 4000


def web_search(query="", question=""):
    """Query the web, summarise the top results, return a spoken answer."""
    query = (query or "").strip()
    if not query:
        return "I didn't catch what you want me to look up."

    try:
        from ddgs import DDGS
    except ImportError:
        return "Couldn't search the web -- the ddgs package isn't installed."

    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=MAX_RESULTS))
    except Exception as e:
        msg = "Couldn't search the web: {}".format(e)
        print(msg)
        return msg

    if not results:
        return "The search for {} came back empty -- try rewording it.".format(query)

    material = "Search: {}\n\n".format(query)
    for result in results:
        title = result.get("title") or ""
        body = result.get("body") or ""
        href = result.get("href") or ""
        material += "- {}\n{}\n{}\n\n".format(title, body, href)

    # A pointed question gets a look at the freshest single page so the answer
    # isn't built only from cached snippets. Best-effort: a failed fetch changes
    # nothing, the snippets are already in hand.
    if question:
        material += _page_fragment(_first_url(results))

    try:
        answer = genbrain.summarise(material, question or "summarise the top results")
    except Exception as e:
        return "Couldn't summarise the search: {}".format(e)

    out = answer
    print(out)
    return out


def _first_url(results):
    for result in results:
        if result.get("href"):
            return result["href"]
    return None

def _page_fragment(url):
    if not url:
        return ""
    try:
        import requests
        from bs4 import BeautifulSoup
    except ImportError:
        return ""

    try:
        response = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=8)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        for tag in soup(["script", "style", "nav", "header", "footer"]):
            tag.decompose()
        body = " ".join(soup.get_text(separator=" ", strip=True).split())
        if not body:
            return ""
        return "From the top page:\n{}".format(body[:FRAGMENT_CHARS])
    except Exception:
        return ""


if __name__ == "__main__":
    # python -m commands.web_search  "what's the capital of Tasmania"
    import sys

    probe = " ".join(sys.argv[1:]) or "current weather in Tokyo"
    print(web_search(probe))