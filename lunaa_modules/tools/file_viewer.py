"""File viewing and management capabilities"""
import os
import mimetypes
from pathlib import Path

class FileViewer:
    def __init__(self):
        self.supported_text_extensions = ['.txt', '.py', '.js', '.json', '.md', '.csv', '.log', '.xml', '.html', '.css']
        
    def view_file(self, filepath: str, max_lines: int = 100):
        """View contents of a file"""
        try:
            if not os.path.exists(filepath):
                return f"File not found: {filepath}"
            
            file_ext = os.path.splitext(filepath)[1].lower()
            
            if file_ext in self.supported_text_extensions:
                return self._view_text_file(filepath, max_lines)
            else:
                return self._file_info(filepath)
        except Exception as e:
            return f"Error viewing file: {e}"
    
    def _view_text_file(self, filepath: str, max_lines: int):
        """View text file contents"""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                
            if len(lines) <= max_lines:
                return ''.join(lines)
            else:
                preview = ''.join(lines[:max_lines])
                return f"{preview}\n... (showing first {max_lines} of {len(lines)} lines)"
        except Exception as e:
            return f"Error reading file: {e}"
    
    def _file_info(self, filepath: str):
        """Get file information"""
        stat = os.stat(filepath)
        mime_type = mimetypes.guess_type(filepath)[0] or 'unknown'
        
        return f"""File: {filepath}
Size: {stat.st_size} bytes
Type: {mime_type}
Modified: {stat.st_mtime}
(Binary or unsupported file type - cannot display contents)"""
    
    def list_directory(self, dirpath: str = '.'):
        """List directory contents"""
        try:
            if not os.path.isdir(dirpath):
                return f"Not a directory: {dirpath}"
            
            entries = []
            for item in sorted(os.listdir(dirpath)):
                full_path = os.path.join(dirpath, item)
                if os.path.isdir(full_path):
                    entries.append(f"[DIR]  {item}")
                else:
                    size = os.path.getsize(full_path)
                    entries.append(f"[FILE] {item} ({size} bytes)")
            
            return '\n'.join(entries)
        except Exception as e:
            return f"Error listing directory: {e}"
    
    def search_files(self, directory: str, pattern: str):
        """Search for files matching pattern"""
        try:
            matches = []
            for root, dirs, files in os.walk(directory):
                for file in files:
                    if pattern.lower() in file.lower():
                        matches.append(os.path.join(root, file))
            
            if matches:
                return '\n'.join(matches)
            return "No files found matching pattern"
        except Exception as e:
            return f"Error searching files: {e}"
