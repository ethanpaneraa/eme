from dataclasses import dataclass, field
from typing import Dict, Literal
import re, math
from pydantic import BaseModel, field_validator

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
    
class FullCourseRecord(BaseModel):
    """
    Represents each unique course's information, including the historical times they were offered. 
    """
    subject: AllowedSubject
    catalog_number: str = "" # the '330-0' part in 'COMP_SCI 330-0'
    name: str = "" # Name, like 'Human Computer Interaction'
    description: str = "" # Description of the class from the latest offering of it
    prereqs: str = ""
    llm_message: str = "" 

    @field_validator("*", mode="before")
    @classmethod
    def clean_text(cls, v):
        if v is None or (isinstance(v, float) and math.isnan(v)):
            return "."
        if not isinstance(v, str):
            v = str(v)
        
        v = re.sub(r"[\u200B-\u200D\uFEFF]", "", v)
        v = re.sub(r"[\x00-\x1F\x7F]", "", v)  # remove control chars
        v = re.sub(r"\s+", " ", v).strip()
        if not v:
            v = "."

        # Optional: truncate very long fields to avoid token overflow
        MAX_CHARS = 30000
        if len(v) > MAX_CHARS:
            v = v[:MAX_CHARS]

        return v

    @field_validator("catalog_number", mode="after")
    @classmethod
    def catalog_number_checker(cls, v):
        if not re.match(r'^[\d\-]+$', v):
            raise ValueError("Catalog number must be only number and dashes")
        if v == "":
            raise ValueError("Catalog number cannot be empy")
        return v
    
    @field_validator("name", mode="after")
    @classmethod
    def name_checker(cls, v):
        if v == "":
            raise ValueError("Name of course cannot be empty")
        return v

    def get_message(self) -> str:
        self.llm_message = f"Course information for {'COMP_SCI/CS' if self.subject == 'COMP_SCI' else 'COMP_ENG/CE'} {self.catalog_number}: {self.name}\nDescription: {self.description}\nPrerequisites: {self.prereqs}\n"
        return self.llm_message