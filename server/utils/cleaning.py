import json
from typing import Any, Dict
from logging_config import get_logger

logger = get_logger(__name__)

def sanitize_metadata_value(v: Any) -> Any:
    """Sanitize a metadata value for storage in ChromaDB."""
    if v is None:
        return ""
    if isinstance(v, (str, int, float, bool)):
        return v
    try:
        return json.dumps(v, ensure_ascii=False)
    except Exception as e:
        logger.warning(f"Failed to serialize metadata value {v}: {str(e)}")
        return str(v)

def sanitize_metadata(d: Dict[str, Any]) -> Dict[str, Any]:
    """Sanitize all metadata values in a dictionary."""
    logger.debug(f"Sanitizing metadata with {len(d)} keys")
    return {k: sanitize_metadata_value(v) for k, v in d.items()}
