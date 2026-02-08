"""Lightweight local memory utilities (index + daily journal)."""
from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List


def _storage_dir() -> Path:
    root = Path.home() / ".paper2ppt"
    root.mkdir(parents=True, exist_ok=True)
    return root


def index_path() -> Path:
    return _storage_dir() / "paper_index.json"


def journal_path() -> Path:
    return _storage_dir() / "journal.jsonl"


def summary_cache_path() -> Path:
    return _storage_dir() / "summary_cache.json"


def load_index() -> Dict[str, Any]:
    path = index_path()
    if not path.exists():
        return {"papers": []}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {"papers": []}


def save_index(data: Dict[str, Any]) -> None:
    index_path().write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def load_summary_cache() -> Dict[str, Any]:
    path = summary_cache_path()
    if not path.exists():
        return {"items": []}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {"items": []}


def save_summary_cache(data: Dict[str, Any]) -> None:
    summary_cache_path().write_text(
        json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def purge_summary_cache(max_age_seconds: int) -> int:
    data = load_summary_cache()
    items = data.get("items", [])
    if not items:
        return 0
    now = datetime.now().timestamp()
    kept = []
    removed = 0
    for it in items:
        ts = it.get("created_at_ts")
        if isinstance(ts, (int, float)) and now - ts > max_age_seconds:
            removed += 1
            continue
        kept.append(it)
    if removed:
        data["items"] = kept
        save_summary_cache(data)
    return removed


def get_cached_summary(cache_key: str, max_age_seconds: int) -> str | None:
    data = load_summary_cache()
    now = datetime.now().timestamp()
    for it in data.get("items", []):
        if it.get("key") != cache_key:
            continue
        ts = it.get("created_at_ts")
        if isinstance(ts, (int, float)) and now - ts <= max_age_seconds:
            return it.get("summary", "") or None
    return None


def put_cached_summary(cache_key: str, summary: str) -> None:
    data = load_summary_cache()
    items = data.get("items", [])
    now_ts = datetime.now().timestamp()
    now_iso = datetime.now().isoformat(timespec="seconds")
    for it in items:
        if it.get("key") == cache_key:
            it["summary"] = summary
            it["created_at"] = now_iso
            it["created_at_ts"] = now_ts
            save_summary_cache(data)
            return
    items.append(
        {
            "key": cache_key,
            "summary": summary,
            "created_at": now_iso,
            "created_at_ts": now_ts,
        }
    )
    data["items"] = items
    save_summary_cache(data)


def upsert_paper(entry: Dict[str, Any]) -> bool:
    data = load_index()
    papers = data.get("papers", [])
    key = entry.get("paper_id") or entry.get("title")
    updated = False
    for i, p in enumerate(papers):
        if p.get("paper_id") == key or (not entry.get("paper_id") and p.get("title") == entry.get("title")):
            papers[i] = entry
            updated = True
            break
    if not updated:
        papers.append(entry)
    data["papers"] = papers
    save_index(data)
    return updated


def search_index(query: str, limit: int = 10) -> List[Dict[str, Any]]:
    data = load_index()
    q = (query or "").strip().lower()
    if not q:
        return []
    terms = [t for t in re.findall(r"[A-Za-z0-9]+", q) if t]
    scored = []
    for p in data.get("papers", []):
        hay = " ".join(
            [
                str(p.get("title", "")),
                str(p.get("summary", "")),
                " ".join(p.get("key_claims", []) or []),
                " ".join(p.get("methods", []) or []),
                " ".join(p.get("datasets", []) or []),
                " ".join(p.get("keywords", []) or []),
            ]
        ).lower()
        score = 0
        if q in hay:
            score += 2
        for t in terms:
            if t in hay:
                score += 1
        if score > 0:
            scored.append((score, p))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [p for _score, p in scored[:limit]]


def append_journal(entry: Dict[str, Any]) -> None:
    path = journal_path()
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def load_journal_for_date(date_str: str) -> List[Dict[str, Any]]:
    path = journal_path()
    if not path.exists():
        return []
    entries = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            obj = json.loads(line)
        except Exception:
            continue
        if obj.get("date") == date_str:
            entries.append(obj)
    return entries


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def today_str() -> str:
    return datetime.now().strftime("%Y-%m-%d")
