import os
import re
import json

try:
    import PyPDF2
    _PDF_OK = True
except ImportError:
    _PDF_OK = False

INDEX_FILE = "lunaa_doc_index.json"

def register(extension_manager):
    extension_manager.register_command("doc", handle_doc_command,
                                       help_text="/doc index <path> | /doc ask <name> <question>")

def load_index():
    if not os.path.exists(INDEX_FILE):
        return {}
    try:
        with open(INDEX_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}

def save_index(idx):
    with open(INDEX_FILE, "w", encoding="utf-8") as f:
        json.dump(idx, f, ensure_ascii=False, indent=2)

def extract_text_from_pdf(path):
    if not _PDF_OK:
        raise RuntimeError("PyPDF2 not installed")
    text_chunks = []
    with open(path, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        for i, page in enumerate(reader.pages):
            try:
                t = page.extract_text() or ""
            except Exception:
                t = ""
            if t.strip():
                text_chunks.append({"page": i + 1, "text": t})
    return text_chunks

def index_path(path, append):
    idx = load_index()
    path = os.path.abspath(path)

    if os.path.isdir(path):
        for root, _, files in os.walk(path):
            for name in files:
                full = os.path.join(root, name)
                index_file(full, idx, append)
    else:
        index_file(path, idx, append)

    save_index(idx)
    append(f"[/doc] Index updated. Total docs: {len(idx)}")

def index_file(path, idx, append):
    base = os.path.basename(path)
    ext = os.path.splitext(base)[1].lower()
    append(f"[/doc] Indexing {path} ...")

    text_entries = []
    if ext == ".pdf":
        try:
            pages = extract_text_from_pdf(path)
            for p in pages:
                text_entries.append(f"[page {p['page']}] {p['text']}")
        except Exception as e:
            append(f"[/doc] Failed to read PDF {path}: {e}")
            return
    else:
        try:
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                text_entries.append(f.read())
        except Exception as e:
            append(f"[/doc] Failed to read {path}: {e}")
            return

    idx[base] = {
        "path": path,
        "content": "\n\n".join(text_entries),
    }
    append(f"[/doc] Indexed {base}")

def simple_search(text, query, max_snippets=5):
    # Dumb keyword search
    query = query.lower()
    lines = text.splitlines()
    hits = []
    for i, line in enumerate(lines):
        if query in line.lower():
            start = max(0, i - 2)
            end = min(len(lines), i + 3)
            snippet = "\n".join(lines[start:end])
            hits.append(snippet)
            if len(hits) >= max_snippets:
                break
    return hits

def handle_doc_command(args: str, append, ask_llm=None):
    """
    args: text after /doc
    append: output function
    ask_llm: optional callback to call Lunaa's text model; if ExtensionManager
             can pass it, we can do real QA. If None, we just show snippets.
    """
    parts = args.strip().split(maxsplit=2)
    if not parts:
        append("[/doc] Usage: /doc index <path> | /doc ask <name> <question>")
        return

    subcmd = parts[0].lower()
    if subcmd == "index":
        if len(parts) < 2:
            append("[/doc] Usage: /doc index <file_or_directory>")
            return
        index_path(parts[1], append)
    elif subcmd == "ask":
        if len(parts) < 3:
            append("[/doc] Usage: /doc ask <name> <question>")
            return
        name, question = parts[1], parts[2]
        idx = load_index()
        doc = idx.get(name)
        if not doc:
            append(f"[/doc] No indexed document named '{name}'.")
            return
        snippets = simple_search(doc["content"], question)
        if not snippets:
            append("[/doc] No relevant snippets found.")
            return
        if ask_llm is None:
            append("[/doc] Matching snippets:\n" + "\n\n---\n\n".join(snippets))
            return
        # If you wire ask_llm, you can send snippets + question to Ollama here.
    else:
        append(f"[/doc] Unknown subcommand: {subcmd}")