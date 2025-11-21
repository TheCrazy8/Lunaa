import json
import os
import shlex

MACRO_FILE = "lunaa_macros.json"

def register(extension_manager):
    # We need access to a "send_input" callback; ExtensionManager may expose it
    extension_manager.register_command("macro", handle_macro_command,
                                       help_text="/macro create/run/list/delete")

def load_macros():
    if not os.path.exists(MACRO_FILE):
        return {}
    try:
        with open(MACRO_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}

def save_macros(macros):
    with open(MACRO_FILE, "w", encoding="utf-8") as f:
        json.dump(macros, f, ensure_ascii=False, indent=2)

def handle_macro_command(args: str, append, send_input=None):
    """
    args: text after /macro
    append: output function
    send_input: callback that takes a string and feeds it into Lunaa as if user typed it
    """
    parts = shlex.split(args)
    if not parts:
        append("[/macro] Usage: /macro create/run/list/delete ...")
        return

    subcmd = parts[0].lower()
    macros = load_macros()

    if subcmd == "list":
        if not macros:
            append("[/macro] No macros defined.")
            return
        append("[/macro] Macros:")
        for name, steps in macros.items():
            append(f" - {name}: {len(steps)} step(s)")
        return

    if subcmd == "delete":
        if len(parts) < 2:
            append("[/macro] Usage: /macro delete <name>")
            return
        name = parts[1]
        if name in macros:
            del macros[name]
            save_macros(macros)
            append(f"[/macro] Deleted '{name}'.")
        else:
            append(f"[/macro] No macro named '{name}'.")
        return

    if subcmd == "create":
        if len(parts) < 3:
            append("[/macro] Usage: /macro create <name> \"step1\" \"step2\" ...")
            return
        name = parts[1]
        steps = parts[2:]
        macros[name] = steps
        save_macros(macros)
        append(f"[/macro] Macro '{name}' saved with {len(steps)} step(s).")
        return

    if subcmd == "run":
        if len(parts) < 2:
            append("[/macro] Usage: /macro run <name>")
            return
        if send_input is None:
            append("[/macro] Macro execution not supported in this context.")
            return
        name = parts[1]
        steps = macros.get(name)
        if not steps:
            append(f"[/macro] No macro named '{name}'.")
            return
        append(f"[/macro] Running macro '{name}' ({len(steps)} step(s))...")
        for step in steps:
            send_input(step)
        return

    append(f"[/macro] Unknown subcommand: {subcmd}")