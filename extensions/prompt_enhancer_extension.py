import shlex

def initialize():
    """
    /prompt enhance <short description>
    /imgplus <short description>
    """
    return {
        "prompt": handle_prompt_command,
        "imgplus": handle_imgplus_command,
    }

def handle_prompt_command(raw: str, append, ask_llm=None):
    args = shlex.split(raw)
    if not args or args[0] == "help":
        _print_help(append)
        return

    sub = args[0].lower()
    if sub != "enhance":
        append(f"[/prompt] Unknown subcommand '{sub}'. Use '/prompt enhance <idea>'.")
        return

    if not ask_llm:
        append("[/prompt] Model integration not available.")
        return

    idea = " ".join(args[1:]) if len(args) > 1 else ""
    if not idea:
        append("[/prompt] Usage: /prompt enhance <short description>")
        return

    system_prompt = (
        "You are a prompt engineer for Stable Diffusion / Automatic1111. "
        "Given a short idea, produce a SINGLE detailed prompt suitable for "
        "text-to-image generation. Include:\n"
        "- subject\n"
        "- style (e.g. digital art, photo, painting)\n"
        "- lighting, mood, composition\n"
        "Return only the prompt, no extra commentary."
    )
    append("[/prompt] Enhancing your prompt with the model...")
    try:
        improved = ask_llm(idea, system_prompt)
    except Exception as e:
        append(f"[/prompt] Error from model: {e}")
        return

    append("=== Enhanced Prompt ===")
    append(improved.strip())

def handle_imgplus_command(raw: str, append, ask_llm=None, send_input=None):
    """
    /imgplus <short description>
    This will enhance the prompt, then internally call `/img <enhanced>`.
    """
    if not ask_llm:
        append("[/imgplus] Model integration not available.")
        return
    if not send_input:
        append("[/imgplus] Cannot trigger /img; send_input callback not provided.")
        return

    idea = raw.strip()
    if not idea:
        append("[/imgplus] Usage: /imgplus <short description>")
        return

    system_prompt = (
        "You are a prompt engineer for Stable Diffusion / Automatic1111. "
        "Given a short idea, produce a SINGLE detailed prompt suitable for "
        "text-to-image generation. Return only the prompt text."
    )
    append("[/imgplus] Enhancing image prompt and generating image...")
    try:
        enhanced = ask_llm(idea, system_prompt)
    except Exception as e:
        append(f"[/imgplus] Error from model: {e}")
        return

    final_prompt = enhanced.strip().replace("\n", " ")
    append(f"[/imgplus] Using enhanced prompt:\n{final_prompt}")
    # Simulate user typing `/img <prompt>`
    send_input(f"/img {final_prompt}")