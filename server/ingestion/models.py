from dataclasses import dataclass, field
from typing import Dict, Literal

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
    
AllowedSubjects = Literal["COMP_SCI", "COMP_ENG"]

@dataclass 
class ShortCourseRecord:
    

@dataclass
class FullCourseRecord:
    """
    Represents each unique course's information, including the historical times they were offered. 
    """

    name: AllowedSubjects
    description: str # Description of the class from the latest offering of it
    prereqs: str
    found_disc: str
    schools: list[str]

    def check_types(self):
        try:
            int(self.found_disc)
        except (TypeError, ValueError):
            raise AssertionError("found_disc must be an int or a string representing an int")
        

    

    def to_summary(self):
        self.check_types()


