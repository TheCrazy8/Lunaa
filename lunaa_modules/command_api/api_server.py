"""Command API for external programs to interact with Lunaa"""
import json
import socket
import threading
from typing import Callable, Dict

class CommandAPI:
    def __init__(self, port: int = 8765):
        self.port = port
        self.commands = {}
        self.running = False
        self.server_thread = None
        self.max_payload_size = 1024 * 1024  # 1MB max payload
        
    def register_command(self, command_name: str, handler: Callable):
        """Register a command handler"""
        self.commands[command_name] = handler
    
    def start_server(self):
        """Start the command API server"""
        if self.running:
            return "Server already running"
        
        self.running = True
        self.server_thread = threading.Thread(target=self._run_server, daemon=True)
        self.server_thread.start()
        return f"Command API server started on port {self.port}"
    
    def stop_server(self):
        """Stop the command API server"""
        self.running = False
        return "Command API server stopped"
    
    def _run_server(self):
        """Internal server loop"""
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                s.bind(('localhost', self.port))
                s.listen(5)
                s.settimeout(1.0)
                
                while self.running:
                    try:
                        conn, addr = s.accept()
                        threading.Thread(target=self._handle_client, args=(conn,), daemon=True).start()
                    except socket.timeout:
                        continue
        except Exception as e:
            print(f"Command API server error: {e}")
    
    def _handle_client(self, conn):
        """Handle client connection"""
        try:
            # Set receive buffer limit
            conn.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, self.max_payload_size)
            
            # Receive data with size limit
            chunks = []
            bytes_received = 0
            while True:
                # Check size before receiving more data
                if bytes_received >= self.max_payload_size:
                    raise ValueError("Payload too large")
                
                chunk = conn.recv(4096)
                if not chunk:
                    break
                
                # Check if adding this chunk would exceed limit
                if bytes_received + len(chunk) > self.max_payload_size:
                    raise ValueError("Payload too large")
                
                chunks.append(chunk)
                bytes_received += len(chunk)
            
            data = b''.join(chunks).decode('utf-8')
            
            # Validate and parse JSON
            try:
                request = json.loads(data)
            except json.JSONDecodeError:
                response = {'status': 'error', 'message': 'Invalid JSON'}
                conn.sendall(json.dumps(response).encode('utf-8'))
                return
            
            # Validate request structure
            if not isinstance(request, dict):
                response = {'status': 'error', 'message': 'Request must be a JSON object'}
                conn.sendall(json.dumps(response).encode('utf-8'))
                return
            
            command = request.get('command')
            args = request.get('args', {})
            
            if not command or not isinstance(command, str):
                response = {'status': 'error', 'message': 'Invalid command'}
                conn.sendall(json.dumps(response).encode('utf-8'))
                return
            
            # Validate args is a dictionary
            if not isinstance(args, dict):
                response = {'status': 'error', 'message': 'Args must be a dictionary'}
                conn.sendall(json.dumps(response).encode('utf-8'))
                return
            
            if command in self.commands:
                result = self.commands[command](**args)
                response = {'status': 'success', 'result': result}
            else:
                response = {'status': 'error', 'message': f'Unknown command: {command}'}
            
            conn.sendall(json.dumps(response).encode('utf-8'))
        except ValueError as e:
            error_response = {'status': 'error', 'message': str(e)}
            try:
                conn.sendall(json.dumps(error_response).encode('utf-8'))
            except:
                pass
        except Exception as e:
            error_response = {'status': 'error', 'message': 'Internal server error'}
            try:
                conn.sendall(json.dumps(error_response).encode('utf-8'))
            except:
                pass
        finally:
            conn.close()
