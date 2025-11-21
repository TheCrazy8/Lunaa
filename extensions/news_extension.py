import requests
import datetime
import re
from xml.etree import ElementTree

DEFAULT_SOURCES = {
    "hn": "https://hnrss.org/frontpage",
    "r-ai": "https://www.reddit.com/r/MachineLearning/.rss",
    "r-python": "https://www.reddit.com/r/Python/.rss",
    "bbc-tech": "http://feeds.bbci.co.uk/news/technology/rss.xml",
}

def initialize():
    """
    Return dict mapping command name -> handler.
    The handler signature here is: handler(raw: str, append, ask_llm=None)
    - raw: full user input after `/news`
    - append: function(text: str) -> None
    - ask_llm: optional callback(question: str, system_prompt: str|None) -> str
    """
    return {"news": handle_news_command}

def handle_news_command(raw: str, append, ask_llm=None):
    """
    /news help
    /news latest [topic]
    /news source <name> [topic]
    """
    args = raw.strip().split()
    if not args or args[0] == "help":
        _print_help(append)
        return

    sub = args[0].lower()
    if sub == "latest":
        topic = " ".join(args[1:]) if len(args) > 1 else ""
        _handle_latest(topic, append, ask_llm)
    elif sub == "source":
        if len(args) < 2:
            append("[/news] Usage: /news source <name> [topic]")
            return
        name = args[1]
        topic = " ".join(args[2:]) if len(args) > 2 else ""
        _handle_source(name, topic, append, ask_llm)
    else:
        append(f"[/news] Unknown subcommand '{sub}'. Use '/news help'.")

def _print_help(append):
    append("=== /news commands ===")
    append("/news latest [topic]         - summarize latest tech/AI news")
    append("/news source <name> [topic]  - fetch from specific source")
    append("Available sources: " + ", ".join(sorted(DEFAULT_SOURCES.keys())))

def _handle_latest(topic, append, ask_llm):
    # For now just aggregate all DEFAULT_SOURCES and optionally filter in LLM
    append("[/news] Fetching latest headlines from default sources...")
    items = []
    for name, url in DEFAULT_SOURCES.items():
        items.extend(_fetch_rss(name, url, append))
    if not items:
        append("[/news] No headlines fetched.")
        return
    _summarize_items(items, topic, append, ask_llm)

def _handle_source(name, topic, append, ask_llm):
    url = DEFAULT_SOURCES.get(name)
    if not url:
        append(f"[/news] Unknown source '{name}'. Available: {', '.join(DEFAULT_SOURCES.keys())}")
        return
    append(f"[/news] Fetching headlines from {name} ...")
    items = _fetch_rss(name, url, append)
    if not items:
        append(f"[/news] No headlines from {name}.")
        return
    _summarize_items(items, topic, append, ask_llm)

def _fetch_rss(source_name, url, append, max_items=10):
    try:
        r = requests.get(url, timeout=10, headers={"User-Agent": "LunaaNews/1.0"})
    except Exception as e:
        append(f"[/news] Error fetching {source_name}: {e}")
        return []
    if r.status_code != 200:
        append(f"[/news] HTTP {r.status_code} from {source_name}")
        return []

    try:
        root = ElementTree.fromstring(r.content)
    except Exception as e:
        append(f"[/news] Failed to parse RSS from {source_name}: {e}")
        return []

    items = []
    for item in root.findall(".//item")[:max_items]:
        title_el = item.find("title")
        link_el = item.find("link")
        desc_el = item.find("description")
        title = (title_el.text or "").strip() if title_el is not None else ""
        link = (link_el.text or "").strip() if link_el is not None else ""
        desc = (desc_el.text or "").strip() if desc_el is not None else ""
        if not title:
            continue
        items.append({"source": source_name, "title": title, "link": link, "desc": _clean_html(desc)})
    return items

def _clean_html(s):
    # cheap HTML tag stripper
    s = re.sub(r"<[^>]+>", " ", s)
    s = re.sub(r"\s+", " ", s)
    return s.strip()

def _summarize_items(items, topic, append, ask_llm):
    # Build plain text from items
    lines = []
    for it in items:
        lines.append(f"- [{it['source']}] {it['title']}\n  {it['link']}\n  {it['desc']}")
    raw_text = "\n\n".join(lines)

    if not ask_llm:
        append("[/news] Headlines:")
        append(raw_text[:4000] + ("..." if len(raw_text) > 4000 else ""))
        return

    topic_hint = f" focusing on {topic}" if topic else ""
    system_prompt = (
        "You are a news summarization assistant. "
        "Given a list of headlines and brief descriptions, "
        f"produce a concise summary of key stories{topic_hint}. "
        "Group related stories together. Keep it under 15 bullet points."
    )
    append("[/news] Summarizing with model...")
    try:
        summary = ask_llm(raw_text, system_prompt)
    except Exception as e:
        append(f"[/news] Error during summarization: {e}")
        append("[/news] Raw headlines:\n" + raw_text[:4000])
        return

    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    append(f"=== News Summary ({now}) ===")
    append(summary)