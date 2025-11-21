"""
Memory Engine for Lunaa AI
Stores and retrieves conversation history, facts, and context
"""
import json
import os
from datetime import datetime
from typing import List, Dict, Optional

class MemoryEngine:
    def __init__(self, memory_file='lunaa_memory.json'):
        self.memory_file = memory_file
        self.memory = self._load_memory()
        
    def _load_memory(self) -> Dict:
        """Load memory from disk"""
        if os.path.exists(self.memory_file):
            try:
                with open(self.memory_file, 'r') as f:
                    return json.load(f)
            except Exception:
                return {'conversations': [], 'facts': [], 'context': {}}
        return {'conversations': [], 'facts': [], 'context': {}}
    
    def _save_memory(self):
        """Save memory to disk"""
        try:
            with open(self.memory_file, 'w') as f:
                json.dump(self.memory, f, indent=2)
        except Exception as e:
            print(f"Error saving memory: {e}")
    
    def add_conversation(self, role: str, content: str):
        """Add a conversation entry"""
        entry = {
            'role': role,
            'content': content,
            'timestamp': datetime.now().isoformat()
        }
        self.memory['conversations'].append(entry)
        self._save_memory()
    
    def add_fact(self, fact: str, source: str = 'user'):
        """Add a fact to memory"""
        entry = {
            'fact': fact,
            'source': source,
            'timestamp': datetime.now().isoformat()
        }
        self.memory['facts'].append(entry)
        self._save_memory()
    
    def get_recent_conversations(self, count: int = 10) -> List[Dict]:
        """Get recent conversations"""
        return self.memory['conversations'][-count:]
    
    def search_facts(self, query: str) -> List[Dict]:
        """Search facts by keyword"""
        query_lower = query.lower()
        return [f for f in self.memory['facts'] if query_lower in f['fact'].lower()]
    
    def get_context(self, key: str) -> Optional[str]:
        """Get context value"""
        return self.memory['context'].get(key)
    
    def set_context(self, key: str, value: str):
        """Set context value"""
        self.memory['context'][key] = value
        self._save_memory()
    
    def clear_memory(self):
        """Clear all memory"""
        self.memory = {'conversations': [], 'facts': [], 'context': {}}
        self._save_memory()
