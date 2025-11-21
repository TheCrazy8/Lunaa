"""
Example extension for Lunaa AI
This demonstrates how to create an extension
"""

def initialize():
    """
    Initialize the extension and return available commands
    Returns: dict of command_name -> function
    """
    return {
        'hello': hello_command,
        'calculate': calculate_command
    }

def hello_command(name: str = "World"):
    """Example greeting command"""
    return f"Hello, {name}! This is an example extension."

def calculate_command(expression: str):
    """Example calculation command - basic arithmetic only"""
    try:
        # Only allow basic arithmetic - no eval for security
        # For real calculations, use the /math command
        return f"For calculations, please use the /math command: /math {expression}"
    except Exception as e:
        return f"Error: {e}"

def cleanup():
    """Called when extension is unloaded"""
    print("Example extension unloaded")
