from groupme_loader import _clean_text
from ingestion.models import MessageChunk
from ingestion.models import FullCourseRecord
from types import Any
import json

def _load_data() -> Any:
    with open("../data/raw/plan.json", "r", encoding="utf-8") as f:
        data = json.load(f)
    return data

def _split_subject_catalog_number(course_code: str) -> tuple[str, str]:
    parts = course_code.split(',')
    if len(parts) != 2:
        raise ValueError(f"Invalid course code format: {course_code}")
    
    subject, catalog_number = parts
    return subject, catalog_number

def _make_course_records_from_data() -> list[FullCourseRecord]:
    data = _load_data()
    assert "courses" in data, "Data must contain 'courses' key"


    course_records = []
    for item in data['courses']:
        if 
        record = FullCourseRecord(
            subject=item.get("subject", "COMP_SCI"),
            catalog_number=item.get("catalog_number", ""),
            name=item.get("name", ""),
            description=item.get("description", ""),
            prereqs=item.get("prereqs", "")
        )
        course_records.append(record)
    return course_records

def make_messages_from_papernu() -> list[str]:
    all_records: list[FullCourseRecord] =  _make_course_records_from_data()

    return [record.get_message for record in all_records]