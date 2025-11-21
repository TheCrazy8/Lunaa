import io
import contextlib
import math
import numpy as np

SAFE_BUILTINS = {
    "abs": abs,
    "min": min,
    "max": max,
    "sum": sum,
    "len": len,
    "range": range,
    "print": print,
}

SAFE_GLOBALS = {
    "__builtins__": SAFE_BUILTINS,
    "math": math,
    "np": np,
}

def register(extension_manager):
    """
    Called by ExtensionManager at load time.
    Should register the /code command.
    """
    extension_manager.register_command("code", handle_code_command, help_text="/code py <code> - run Python code in a sandbox")

def handle_code_command(args: str, append):
    """
    args: raw string after /code
    append: function to write output back to UI
    """
    parts = args.strip().split(maxsplit=1)
    if not parts:
        append("[/code] Usage: /code py <code>")
        return

    lang = parts[0].lower()
    if lang != "py":
        append(f"[/code] Unsupported language: {lang}. Only 'py' is supported.")
        return

    if len(parts) == 1:
        append("[/code] Please provide code after 'py'.")
        return

    code = parts[1]

    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            exec(code, SAFE_GLOBALS, {})
    except Exception as e:
        output = buf.getvalue()
        append(f"[/code] Error while executing code:\n{output}\n{type(e).__name__}: {e}")
        return

    output = buf.getvalue().strip()
    if not output:
        append("[/code] (no output)")
    else:
        append(f"[/code] Output:\n{output}")