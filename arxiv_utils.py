"""arXiv helpers for metadata lookup and source download/extraction."""
from __future__ import annotations

import re
import shutil
import tarfile
from pathlib import Path
from typing import Any, Dict, Optional

import arxiv
import requests

ARXIV_SRC_URL = "https://arxiv.org/src/{arxiv_id}"
ARXIV_EPRINT_URL = "https://arxiv.org/e-print/{arxiv_id}"


def extract_arxiv_id(arxiv_link_or_id: str) -> str:
    """Normalize arXiv link or ID into a plain arXiv identifier."""
    s = arxiv_link_or_id.strip()
    m = re.search(r"arxiv\.org/(abs|pdf)/([^/?#]+)", s)
    if m:
        raw = m.group(2)
        return raw.replace(".pdf", "")
    return s


def get_arxiv_metadata(arxiv_id: str) -> Dict[str, Any]:
    """Fetch arXiv metadata (title, authors, abstract, url, published)."""
    r = next(arxiv.Search(id_list=[arxiv_id]).results(), None)
    if r is None:
        raise RuntimeError("arXiv metadata not found. Check the ID/link.")
    return {
        "title": r.title,
        "authors": [a.name for a in r.authors],
        "abstract": r.summary,
        "url": r.entry_id,
        "published": str(r.published),
    }


def download_and_extract_arxiv_source(arxiv_id: str, out_dir: Path) -> Path:
    """Download arXiv source tarball and extract into work directory."""
    out_dir = Path(out_dir)
    src_dir = out_dir / "arxiv_source"
    if src_dir.exists():
        shutil.rmtree(src_dir)
    src_dir.mkdir(parents=True, exist_ok=True)

    def _try(url: str) -> bytes:
        r = requests.get(url, timeout=60)
        r.raise_for_status()
        return r.content

    content = None
    last_err = None
    for url in [ARXIV_SRC_URL.format(arxiv_id=arxiv_id), ARXIV_EPRINT_URL.format(arxiv_id=arxiv_id)]:
        try:
            content = _try(url)
            break
        except Exception as e:
            last_err = e

    if content is None:
        raise RuntimeError(f"Failed to download arXiv source for {arxiv_id}: {last_err}")

    tmp = src_dir / "source.bin"
    tmp.write_bytes(content)

    extracted_any = False
    try:
        with tarfile.open(tmp, "r:*") as tf:
            tf.extractall(src_dir)
            extracted_any = True
    except tarfile.ReadError:
        pass

    if extracted_any:
        tmp.unlink(missing_ok=True)

    tex_files = list(src_dir.rglob("*.tex"))
    if not tex_files:
        raise RuntimeError(f"No .tex files found after extraction for {arxiv_id}.")
    return src_dir
