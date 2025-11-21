"""Extension system for Lunaa AI"""
import os
import importlib.util
from typing import Dict, Callable

class ExtensionManager:
    def __init__(self, extensions_dir: str = 'extensions'):
        self.extensions_dir = extensions_dir
        self.loaded_extensions = {}
        self.extension_commands = {}
        
        # Create extensions directory if it doesn't exist
        os.makedirs(extensions_dir, exist_ok=True)
    
    def load_extension(self, extension_name: str):
        """Load an extension from the extensions directory"""
        extension_path = os.path.join(self.extensions_dir, f"{extension_name}.py")
        
        if not os.path.exists(extension_path):
            return f"Extension not found: {extension_name}"
        
        try:
            spec = importlib.util.spec_from_file_location(extension_name, extension_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            # Look for an 'initialize' function in the extension
            if hasattr(module, 'initialize'):
                commands = module.initialize()
                if commands:
                    self.extension_commands[extension_name] = commands
            
            self.loaded_extensions[extension_name] = module
            return f"Extension loaded: {extension_name}"
        except Exception as e:
            return f"Error loading extension: {e}"
    
    def unload_extension(self, extension_name: str):
        """Unload an extension"""
        if extension_name in self.loaded_extensions:
            # Call cleanup if available
            module = self.loaded_extensions[extension_name]
            if hasattr(module, 'cleanup'):
                module.cleanup()
            
            del self.loaded_extensions[extension_name]
            if extension_name in self.extension_commands:
                del self.extension_commands[extension_name]
            
            return f"Extension unloaded: {extension_name}"
        return f"Extension not loaded: {extension_name}"
    
    def list_extensions(self):
        """List available extensions"""
        available = []
        if os.path.exists(self.extensions_dir):
            for file in os.listdir(self.extensions_dir):
                if file.endswith('.py') and not file.startswith('__'):
                    extension_name = file[:-3]
                    status = "loaded" if extension_name in self.loaded_extensions else "available"
                    available.append(f"{extension_name} ({status})")
        
        return '\n'.join(available) if available else "No extensions found"
    
    def execute_extension_command(self, extension_name: str, command: str, *args, **kwargs):
        """Execute a command from an extension"""
        if extension_name not in self.loaded_extensions:
            return f"Extension not loaded: {extension_name}"
        
        if extension_name not in self.extension_commands:
            return f"No commands registered for extension: {extension_name}"
        
        commands = self.extension_commands[extension_name]
        if command not in commands:
            return f"Command not found: {command}"
        
        try:
            return commands[command](*args, **kwargs)
        except Exception as e:
            return f"Error executing command: {e}"
