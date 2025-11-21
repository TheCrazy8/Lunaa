import subprocess
import shlex

def initialize():
    """
    /git status
    /git diff [path]
    /git log [n]
    /git explain <commit>
    """
    return {"git": handle_git_command}

def handle_git_command(raw: str, append, ask_llm=None):
    args = shlex.split(raw)
    if not args or args[0] == "help":
        _print_help(append)
        return

    sub = args[0].lower()
    if sub == "status":
        _run_git(["status", "--short"], append)
    elif sub == "diff":
        path = args[1] if len(args) > 1 else None
        cmd = ["diff"]
        if path:
            cmd.append(path)
        _run_git(cmd, append)
    elif sub == "log":
        n = args[1] if len(args) > 1 else "5"
        _run_git(["log", f"-{n}", "--oneline", "--decorate"], append)
    elif sub == "explain":
        if len(args) < 2:
            append("[/git] Usage: /git explain <commit-ish>")
            return
        commit = args[1]
        _explain_commit(commit, append, ask_llm)
    else:
        append(f"[/git] Unknown subcommand '{sub}'. Use '/git help'.")

def _print_help(append):
    append("=== /git commands ===")
    append("/git status            - git status --short")
    append("/git diff [path]       - git diff")
    append("/git log [n]           - last n commits (default 5)")
    append("/git explain <commit>  - summarize the diff using the model")

def _run_git(args, append):
    cmd = ["git"] + args
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
    except Exception as e:
        append(f"[/git] Error running git: {e}")
        return
    if proc.stdout:
        append(proc.stdout.rstrip())
    if proc.stderr:
        append("[stderr]\n" + proc.stderr.rstrip())

def _explain_commit(commit, append, ask_llm):
    if not ask_llm:
        append("[/git] LLM integration not available; showing raw diff instead.")
        _run_git(["show", commit], append)
        return

    try:
        proc = subprocess.run(
            ["git", "show", commit, "--stat", "--patch"],
            capture_output=True, text=True, timeout=20
        )
    except Exception as e:
        append(f"[/git] Error running git show: {e}")
        return

    if proc.returncode != 0:
        append(f"[/git] git show failed: {proc.stderr.strip()}")
        return

    diff_text = proc.stdout
    if not diff_text.strip():
        append("[/git] Empty diff.")
        return

    system_prompt = (
        "You are a senior engineer reviewing a git commit. "
        "Given the commit diff and metadata, write a concise explanation "
        "of what changed, why it might have changed, and any potential risks."
    )
    append(f"[/git] Explaining commit {commit} with model...")
    try:
        explanation = ask_llm(diff_text[:12000], system_prompt)
    except Exception as e:
        append(f"[/git] Error from model: {e}")
        append("[/git] Raw diff:\n" + diff_text[:4000])
        return

    append(f"=== Commit Explanation: {commit} ===")
    append(explanation)