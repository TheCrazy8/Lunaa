"""Context engine for maintaining conversation context"""
from typing import List, Dict, Optional
from collections import deque

class ContextEngine:
    def __init__(self, max_context_size: int = 20):
        self.max_context_size = max_context_size
        self.context_buffer = deque(maxlen=max_context_size)
        self.entities = {}
        self.topics = []
        
    def add_to_context(self, message: Dict):
        """Add message to context buffer"""
        self.context_buffer.append(message)
        self._extract_entities(message.get('content', ''))
    
    def _extract_entities(self, text: str):
        """Extract entities from text (simple implementation)"""
        # Placeholder for more sophisticated entity extraction
        words = text.split()
        for word in words:
            if word.istitle() and len(word) > 2:
                self.entities[word] = self.entities.get(word, 0) + 1
    
    def get_context_summary(self) -> str:
        """Get summary of current context"""
        if not self.context_buffer:
            return "No context available"
        
        summary_parts = []
        summary_parts.append(f"Context buffer size: {len(self.context_buffer)}")
        
        if self.entities:
            top_entities = sorted(self.entities.items(), key=lambda x: x[1], reverse=True)[:5]
            summary_parts.append(f"Key entities: {', '.join([e[0] for e in top_entities])}")
        
        return " | ".join(summary_parts)
    
    def get_recent_context(self, count: int = 5) -> List[Dict]:
        """Get recent context messages"""
        return list(self.context_buffer)[-count:]
    
    def clear_context(self):
        """Clear context buffer"""
        self.context_buffer.clear()
        self.entities.clear()
        self.topics.clear()
