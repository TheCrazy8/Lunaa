import os
import subprocess
import shlex

def initialize():
    """
    /review file <path>
    /review diff [path]
    /review folder <path>
    """
    return {"review": handle_review_command}

def handle_review_command(raw: str, append, ask_llm=None):
    args = shlex.split(raw)
    if not args or args[0] == "help":
        _print_help(append)
        return

    if not ask_llm:
        append("[/review] Model integration not available. This command requires ask_llm.")
        return

    sub = args[0].lower()
    if sub == "file":
        if len(args) < 2:
            append("[/review] Usage: /review file <path>")
            return
        _review_file(args[1], append, ask_llm)
    elif sub == "diff":
        path = args[1] if len(args) > 1 else None
        _review_diff(path, append, ask_llm)
    elif sub == "folder":
        if len(args) < 2:
            append("[/review] Usage: /review folder <path>")
            return
        _review_folder(args[1], append, ask_llm)
    else:
        append(f"[/review] Unknown subcommand '{sub}'. Use '/review help'.")

def _print_help(append):
    append("=== /review commands ===")
    append("/review file <path>    - Review a single file")
    append("/review diff [path]    - Review current git diff (optionally filtered by path)")
    append("/review folder <path>  - Review small folders (may truncate)")

def _read_file(path, append, max_bytes=20000):
    if not os.path.exists(path):
        append(f"[/review] File not found: {path}")
        return None
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            data = f.read(max_bytes + 1)
    except Exception as e:
        append(f"[/review] Error reading file: {e}")
        return None
    if len(data) > max_bytes:
        append("[/review] File truncated for review (too large).")
        data = data[:max_bytes]
    return data

def _review_file(path, append, ask_llm):
    content = _read_file(path, append)
    if content is None:
        return
    prompt = f"File path: {path}\n\n{content}"
    system_prompt = (
        "You are a strict but helpful code reviewer. "
        "Review this single file and provide:\n"
        "1) Summary of what it does\n"
        "2) Potential bugs or edge cases\n"
        "3) Style/readability suggestions\n"
        "4) Any security or performance concerns."
    )
    append(f"[/review] Sending {path} to model for review...")
    try:
        review = ask_llm(prompt, system_prompt)
    except Exception as e:
        append(f"[/review] Error from model: {e}")
        return
    append(f"=== Code Review: {path} ===")
    append(review)

def _review_diff(path, append, ask_llm, max_bytes=20000):
    cmd = ["git", "diff"]
    if path:
        cmd.append(path)
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=20)
    except Exception as e:
        append(f"[/review] Error running git diff: {e}")
        return
    diff_text = proc.stdout
    if not diff_text.strip():
        append("[/review] No diff to review.")
        return
    if len(diff_text) > max_bytes:
        append("[/review] Diff truncated for review (too large).")
        diff_text = diff_text[:max_bytes]

    system_prompt = (
        "You are an experienced engineer performing a code review on this git diff. "
        "Provide:\n"
        "1) High-level summary\n"
        "2) Possible bugs/logic issues\n"
        "3) Tests you would add or update\n"
        "4) Any style, security, or performance concerns."
    )
    append("[/review] Sending diff to model for review...")
    try:
        review = ask_llm(diff_text, system_prompt)
    except Exception as e:
        append(f"[/review] Error from model: {e}")
        return
    append("=== Code Review: Diff ===")
    append(review)

def _review_folder(folder, append, ask_llm, max_bytes=20000):
    if not os.path.isdir(folder):
        append(f"[/review] Not a folder: {folder}")
        return
    collected = []
    total = 0
    for root, _, files in os.walk(folder):
        for name in files:
            if name.endswith((".py", ".js", ".ts", ".tsx", ".html", ".css")):
                path = os.path.join(root, name)
                try:
                    with open(path, "r", encoding="utf-8", errors="ignore") as f:
                        content = f.read(4000)
                except Exception:
                    continue
                collected.append(f"=== {path} ===\n{content}\n")
                total += len(content)
                if total >= max_bytes:
                    break
        if total >= max_bytes:
            break
    if not collected:
        append("[/review] No code files found in folder.")
        return

    bundle = "\n\n".join(collected)
    system_prompt = (
        "You are reviewing a small codebase snapshot (multiple files). "
        "Give a high-level overview, and then list key issues and suggestions."
    )
    append(f"[/review] Sending folder snapshot of {folder} to model...")
    try:
        review = ask_llm(bundle, system_prompt)
    except Exception as e:
        append(f"[/review] Error from model: {e}")
        return
    append(f"=== Code Review: Folder {folder} ===")
    append(review)