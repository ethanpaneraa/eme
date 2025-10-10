import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from ingestion.papernu_loader import make_messages_from_papernu
from pprint import pprint

def test_make_messages_from_papernu():
    llm_msgs: list[str] = make_messages_from_papernu()

    assert isinstance(llm_msgs, list)
    assert len(llm_msgs) > 100
    assert all(len(msg) > 10 for msg in llm_msgs)

    for msg in llm_msgs[:5]:
        print(msg)
    