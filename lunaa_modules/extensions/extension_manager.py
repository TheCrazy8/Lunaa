"""Extension system for Lunaa AI"""
import os
import importlib.util
from typing import Dict, Callable
import sys

class ExtensionManager:
    def __init__(self, extensions_dir: str = 'extensions'):
        self.extensions_dir = extensions_dir
        self.loaded_extensions = {}
        self.extension_commands = {}
        
        # Create extensions directory if it doesn't exist
        os.makedirs(extensions_dir, exist_ok=True)
    
    def load_extension(self, extension_name: str):
        """Load an extension from the extensions directory"""
        # Validate extension name to prevent path traversal and other attacks
        # Check for path separators, parent directory references, null bytes
        if any(char in extension_name for char in ['..', '/', '\\', '\0', '\n', '\r']):
            return "Invalid extension name: contains unsafe characters"
        
        # Check for absolute paths
        if os.path.isabs(extension_name):
            return "Invalid extension name: absolute paths not allowed"
        
        # Normalize the name
        extension_name = os.path.normpath(extension_name)
        
        # Build path
        extension_path = os.path.join(self.extensions_dir, f"{extension_name}.py")
        
        if not os.path.exists(extension_path):
            return f"Extension not found: {extension_name}"
        
        # Verify file is within extensions directory (resolve symlinks)
        real_path = os.path.realpath(extension_path)
        real_ext_dir = os.path.realpath(self.extensions_dir)
        
        # Ensure the real path starts with the extensions directory
        if not real_path.startswith(real_ext_dir + os.sep) and real_path != real_ext_dir:
            return "Security error: Extension path outside extensions directory"
        
        # Verify it's actually a file
        if not os.path.isfile(real_path):
            return "Security error: Path is not a file"
        
        try:
            spec = importlib.util.spec_from_file_location(extension_name, extension_path)
            module = importlib.util.module_from_spec(spec)
            
            # Add to sys.modules to allow proper importing
            sys.modules[extension_name] = module
            spec.loader.exec_module(module)
            
            # Look for an 'initialize' function in the extension
            if hasattr(module, 'initialize'):
                commands = module.initialize()
                if commands:
                    # Validate commands are callables
                    for cmd_name, cmd_func in commands.items():
                        if not callable(cmd_func):
                            return f"Extension error: {cmd_name} is not callable"
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
                try:
                    module.cleanup()
                except Exception as e:
                    print(f"Extension cleanup error: {e}")
            
            # Remove from sys.modules
            if extension_name in sys.modules:
                del sys.modules[extension_name]
            
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
