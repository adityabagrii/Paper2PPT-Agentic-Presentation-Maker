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
