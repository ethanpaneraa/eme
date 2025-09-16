from dataclasses import dataclass, field
from typing import Dict

@dataclass
class MessageChunk:
    """
    Represents one *message-centric* chunk, possibly with +/- context lines composed in text.
    """
    text: str
    metadata: Dict = field(default_factory=lambda: {
        "msg_id": None,
        "group_id": None,
        "sender_id": None,
        "sender_name": None,
        "created_at_iso": None,   # e.g., "2025-09-15T21:07:00Z"
        "window": 0,              # context window used to build this chunk
        "raw_len": 0              # original message length (before composing context)
    })

    def to_id(self) -> str:
        # Stable id for Chroma: message id plus window version
        w = self.metadata.get("window", 0)
        return f"{self.metadata.get('msg_id','unknown')}__w{w}"
