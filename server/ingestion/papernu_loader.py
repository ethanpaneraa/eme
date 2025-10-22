from ingestion.models import FullCourseRecord, AllowedSubject
import sys
from pathlib import Path
import json

def _load_data() -> dict:
    data_path = Path(__file__).parent.parent / "data" / "raw" / "papernu.json"
    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data

def _split_subject_catalog_number(course_code: str) -> tuple[str, str]:
    parts = course_code.split(' ')
    subject, catalog_number = parts
    if len(parts) != 2:
        return ("", "")
        
    return subject, catalog_number

def _make_course_records_from_data(data: dict) -> list[FullCourseRecord]:
    assert "courses" in data, "Data must contain 'courses' key"

    course_records: list[FullCourseRecord] = []
    for item in data['courses']:
        if "i" in item:
            subject, catalog_number = _split_subject_catalog_number(item["i"])
            name = item.get("n") or ""
            description = item.get("d") or ""
            prereqs = item.get("p") or ""

            try:                
                record = FullCourseRecord(
                    subject=subject,
                    catalog_number=catalog_number,
                    name=name,
                    description=description,
                    prereqs=prereqs
                )
                course_records.append(record)
            except ValueError as err:
                continue
    return course_records

def make_context_from_papernu_data() -> list[FullCourseRecord]:
    data = _load_data()
    all_records: list[FullCourseRecord] =  _make_course_records_from_data(data)
    return all_records