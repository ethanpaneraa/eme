from dataclasses import dataclass, field
from typing import Dict, Literal
import re

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
    
AllowedSubject = Literal["COMP_SCI", "COMP_ENG"]
    
class FullCourseRecord:
    """
    Represents each unique course's information, including the historical times they were offered. 
    """
    subject: AllowedSubject = "COMP_SCI"
    catalog_number: str = "" # the '330-0' part in 'COMP_SCI 330-0'
    name: str = "" # Name, like 'Human Computer Interaction'
    description: str = "" # Description of the class from the latest offering of it
    prereqs: str = ""
    llm_message: str = "" 


    def __init__(self, subject="COMP_SCI", catalog_number="", name="", description="", prereqs=""):
        self.subject = subject
        self.catalog_number = catalog_number
        self.name = name
        self.description = description
        self.prereqs = prereqs

    def check_types(self):
        try:
            int(self.found_disc)
        except (TypeError, ValueError):
            raise AssertionError("found_disc must be an int or a string representing an int")

        assert self.subject in ("COMP_SCI", "COMP_ENG"), "subject must be COMP_SCI or COMP_ENG"
        assert isinstance(self.catalog_number, str), "catalog_number must be a string"
        assert re.match(r'^[\d\-]+$', self.catalog_number), "catalog_number must only contain numbers and dashes"
        assert isinstance(self.name, str), "name must be a string"
        assert isinstance(self.description, str), "description must be a string"
        assert isinstance(self.prereqs, str), "prereqs must be a string"

    def get_message(self) -> str:
        self.check_types()

        self.llm_message: str = f"Course information for {'COMP_SCI/CS' if self.subject == 'COMP_SCI' else 'COMP_ENG/CE'}{self.catalog_number} {self.name}:\nDescription: {self.description}\nPrerequisites: {self.prereqs}\n"
        return self.llm_message