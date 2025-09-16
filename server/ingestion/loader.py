import orjson, pathlib, datetime as dt, re
from typing import List, Dict, Tuple, Optional
from ingestion.models import MessageChunk

##############
# Utilities
##############

def _clean_text(s: Optional[str]) -> str:
    if not s:
        return ""
    s = re.sub(r"[\u200B-\u200D\uFEFF]", "", s)  # zero-width
    s = re.sub(r"\s+", " ", s).strip()
    return s

def _to_iso(ts: int) -> str:
    return dt.datetime.utcfromtimestamp(int(ts)).replace(tzinfo=dt.timezone.utc).isoformat()

def _looks_like_question(t: str) -> bool:
    t_l = t.lower()
    return (
        "?" in t_l
        or t_l.startswith(("does anyone", "anyone", "hi,", "question", "got a question"))
        or t_l.endswith("?")
    )

ANNOUNCE_PAT = re.compile(r"\b(hackathon|apply|intern(ship|ships)?|mentorship|deadline|poll|join(ed)?|has left|created new poll)\b", re.I)
SYSTEM_TYPES_TO_SUMMARIZE = {"membership.announce.joined","membership.notifications.exited","poll.created","poll.finished","poll.reminder","message.update","message.deleted"}

def _classify_type(text: str, is_system: bool) -> str:
    if is_system:
        return "system"
    t = text.lower()
    if _looks_like_question(t):
        return "question"
    if ANNOUNCE_PAT.search(t):
        return "announcement"
    return "message"

def _summarize_system(obj: Dict) -> Optional[str]:
    # Turn system messages into short summaries so they can still answer “who joined/when” queries.
    ev = (obj.get("event") or {}).get("type")
    if not ev:
        return None
    if ev == "membership.announce.joined":
        nick = ((obj["event"]["data"].get("user") or {}).get("nickname")) or "Someone"
        return f"{nick} joined the group."
    if ev == "membership.notifications.exited":
        nick = ((obj["event"]["data"].get("removed_user") or {}).get("nickname")) or "Someone"
        return f"{nick} left the group."
    if ev == "poll.created":
        pol = (obj["event"]["data"] or {}).get("poll") or {}
        subj = pol.get("subject", "A poll")
        return f"Poll created: “{subj}”."
    if ev == "poll.finished":
        pol = (obj["event"]["data"] or {}).get("poll") or {}
        subj = pol.get("subject", "A poll")
        opts = (obj["event"]["data"] or {}).get("options") or []
        top = sorted(opts, key=lambda o: o.get("votes",0), reverse=True)[:2]
        winners = ", ".join([f"{o['title']} ({o.get('votes',0)})" for o in top]) if top else ""
        return f"Poll finished: “{subj}”. Top: {winners}".strip()
    if ev == "message.deleted":
        return "A message was deleted by an admin."
    if ev == "message.update":
        return None
    return None

def load_jsonl(path: pathlib.Path) -> List[Dict]:
    recs = []
    with path.open("rb") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = orjson.loads(line)
            except orjson.JSONDecodeError:
                continue
            recs.append(obj)
    return recs

##############
# Core
##############

def normalize_records(records: List[Dict]) -> List[Dict]:
    """
    Convert raw GroupMe JSON objects into normalized dicts we can thread.
    - Keep system messages as short summaries (only some types).
    - Drop empty/attachments-only posts unless they’re replies.
    """
    out = []
    for r in records:
        is_system = bool(r.get("system"))
        raw_text = r.get("text")
        sys_summary = _summarize_system(r) if is_system else None

        # If not system: skip messages with no text and no reply context
        if not is_system and not _clean_text(raw_text):
            # keep ONLY if it's a reply container with no text? GroupMe sometimes has images+reply; we skip for now.
            continue

        text = _clean_text(sys_summary if is_system else raw_text or "")
        if not text and not is_system:
            continue

        created_iso = _to_iso(r.get("created_at", 0))
        msg = {
            "id": str(r.get("id")),
            "group_id": str(r.get("group_id")),
            "sender_id": str(r.get("sender_id") or ""),
            "sender_name": _clean_text(r.get("name") or ("System" if is_system else "")),
            "created_at_iso": created_iso,
            "text": text,
            "is_system": is_system,
            "attachments": r.get("attachments") or [],
        }

        reply_to = None
        for att in msg["attachments"]:
            if att.get("type") == "reply":
                reply_to = att.get("base_reply_id") or att.get("reply_id")
                break
        msg["reply_to"] = str(reply_to) if reply_to else None

        msg["has_link"] = bool(re.search(r"https?://", raw_text or ""))
        msg["has_image"] = any(a.get("type") == "image" for a in msg["attachments"])

        msg["msg_type"] = _classify_type(text, is_system=is_system)

        out.append(msg)

    out.sort(key=lambda m: m["created_at_iso"])
    return out

def build_threads(msgs: List[Dict]) -> Tuple[Dict[str, Dict], Dict[str, List[str]]]:
    """
    Return:
      id_to_msg: id -> message dict
      children:  parent_id -> [child_ids ...]
    """
    id_to_msg = {m["id"]: m for m in msgs}
    children: Dict[str, List[str]] = {}
    for m in msgs:
        p = m.get("reply_to")
        if p and p in id_to_msg:
            children.setdefault(p, []).append(m["id"])
    return id_to_msg, children

def harvest_qna_chunks(
    msgs: List[Dict],
    id_to_msg: Dict[str, Dict],
    children: Dict[str, List[str]],
    answer_horizon_minutes: int = 240,
    max_answers: int = 8,
) -> List[MessageChunk]:
    """
    For every message that looks like a question, build one chunk:
      [question] + replies thread + nearby answers within time horizon.
    """

    by_time = msgs

    def within_horizon(q_idx: int, a_idx: int) -> bool:
        t_q = by_time[q_idx]["created_at_iso"]
        t_a = by_time[a_idx]["created_at_iso"]
        t1 = dt.datetime.fromisoformat(t_q)
        t2 = dt.datetime.fromisoformat(t_a)
        return (t2 - t1).total_seconds() <= answer_horizon_minutes * 60

    chunks: List[MessageChunk] = []
    msg_index = {m["id"]: i for i, m in enumerate(by_time)}

    for i, q in enumerate(by_time):
        if q["msg_type"] != "question":
            continue

        lines = [f"→ [{q['created_at_iso']}] {q['sender_name']}: {q['text']}"]
        for rid in children.get(q["id"], [])[:max_answers]:
            r = id_to_msg[rid]
            lines.append(f"· [{r['created_at_iso']}] {r['sender_name']}: {r['text']}")

        added = 0
        j = i + 1
        while j < len(by_time) and added < max_answers:
            cand = by_time[j]
            if within_horizon(i, j) and not cand["is_system"]:
                if len(cand["text"].split()) >= 3 or cand["reply_to"] == q["id"]:
                    lines.append(f"· [{cand['created_at_iso']}] {cand['sender_name']}: {cand['text']}")
                    added += 1
            else:
                break
            j += 1

        composed = "\n".join(lines)
        chunks.append(
            MessageChunk(
                text=composed,
                metadata={
                    "msg_id": q["id"],
                    "group_id": q["group_id"],
                    "sender_id": q["sender_id"],
                    "sender_name": q["sender_name"],
                    "created_at_iso": q["created_at_iso"],
                    "window": 0,
                    "raw_len": len(q["text"]),
                    "msg_type": "qna",
                },
            )
        )
    return chunks

def harvest_announcement_chunks(msgs: List[Dict]) -> List[MessageChunk]:
    """
    One chunk per announcement-like message (jobs/hackathons/polls).
    """
    chunks: List[MessageChunk] = []
    for m in msgs:
        if m["msg_type"] not in {"announcement","system"}:
            continue
        text = m["text"]
        if not text:
            continue
        prefix = "→"
        lines = [f"{prefix} [{m['created_at_iso']}] {m['sender_name']}: {text}"]
        if m["has_link"]:
            lines.append("(contains link)")
        if m["has_image"]:
            lines.append("(contains images)")
        composed = "\n".join(lines)
        chunks.append(
            MessageChunk(
                text=composed,
                metadata={
                    "msg_id": m["id"],
                    "group_id": m["group_id"],
                    "sender_id": m["sender_id"],
                    "sender_name": m["sender_name"],
                    "created_at_iso": m["created_at_iso"],
                    "window": 0,
                    "raw_len": len(text),
                    "msg_type": m["msg_type"],
                },
            )
        )
    return chunks

def harvest_context_windows(msgs: List[Dict], window: int = 1) -> List[MessageChunk]:
    """
    Lightweight fallback like before: every non-system message becomes a +/-window chunk.
    We filter out very short “banter” unless it’s a reply.
    """
    chunks: List[MessageChunk] = []
    for i, m in enumerate(msgs):
        if m["is_system"]:
            continue
        if not m["text"]:
            continue
        if len(m["text"].split()) < 3 and not m["reply_to"]:
            continue

        left = max(0, i - window)
        right = min(len(msgs), i + window + 1)
        lines = []
        for j in range(left, right):
            tag = "→" if j == i else "·"
            n = msgs[j]
            lines.append(f"{tag} [{n['created_at_iso']}] {n['sender_name']}: {n['text']}")
        composed = "\n".join(lines)
        chunks.append(
            MessageChunk(
                text=composed,
                metadata={
                    "msg_id": m["id"],
                    "group_id": m["group_id"],
                    "sender_id": m["sender_id"],
                    "sender_name": m["sender_name"],
                    "created_at_iso": m["created_at_iso"],
                    "window": window,
                    "raw_len": len(m["text"]),
                    "msg_type": "context",
                },
            )
        )
    return chunks

def make_chunks_from_records(records: List[Dict], window: int = 1) -> List[MessageChunk]:
    msgs = normalize_records(records)
    id_to_msg, children = build_threads(msgs)

    chunks: List[MessageChunk] = []
    chunks += harvest_qna_chunks(msgs, id_to_msg, children, answer_horizon_minutes=240, max_answers=8)
    chunks += harvest_announcement_chunks(msgs)
    chunks += harvest_context_windows(msgs, window=window)

    seen = set()
    deduped = []
    for c in chunks:
        key = (c.metadata["msg_id"], c.metadata.get("msg_type"), c.text)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(c)
    return deduped
