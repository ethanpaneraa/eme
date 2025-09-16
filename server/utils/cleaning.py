import json
from typing import Any, Dict

def sanitize_metadata_value(v: Any) -> Any:
    if v is None:
        return ""
    if isinstance(v, (str, int, float, bool)):
        return v
    return json.dumps(v, ensure_ascii=False)

def sanitize_metadata(d: Dict[str, Any]) -> Dict[str, Any]:
    return {k: sanitize_metadata_value(v) for k, v in d.items()}
