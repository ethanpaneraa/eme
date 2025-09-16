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
        "created_at_iso": None,
        "window": 0,
        "raw_len": 0
    })

    def to_id(self) -> str:
        w = self.metadata.get("window", 0)
        return f"{self.metadata.get('msg_id','unknown')}__w{w}"
