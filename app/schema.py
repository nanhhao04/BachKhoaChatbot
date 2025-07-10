from dataclasses import dataclass
from typing import Optional, List, Dict, Any
from datetime import datetime


@dataclass
class Citation:
    """Citation data structure"""
    text: str
    source: str
    page: Optional[int] = None
    url: Optional[str] = None
    confidence: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'text': self.text,
            'source': self.source,
            'page': self.page,
            'url': self.url,
            'confidence': self.confidence
        }


@dataclass
class ChatMessage:
    """Chat message data structure"""
    content: str
    is_user: bool
    timestamp: datetime
    citations: Optional[List[Citation]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'content': self.content,
            'is_user': self.is_user,
            'timestamp': self.timestamp.isoformat(),
            'citations': [c.to_dict() for c in self.citations] if self.citations else []
        }


@dataclass
class ChatSession:
    """Chat session data structure"""
    id: str
    title: str
    messages: List[ChatMessage]
    created_at: datetime

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'id': self.id,
            'title': self.title,
            'messages': [m.to_dict() for m in self.messages],
            'created_at': self.created_at.isoformat()
        }