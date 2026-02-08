from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

try:
    from .arxiv_utils import download_and_extract_arxiv_source, get_arxiv_metadata
except Exception:
    from arxiv_utils import download_and_extract_arxiv_source, get_arxiv_metadata


class ArxivClient:
    def get_metadata(self, arxiv_id: str) -> Dict[str, Any]:
        """Get metadata.

        Args:
            arxiv_id (str):

        Returns:
            Dict[str, Any]:
        """
        return get_arxiv_metadata(arxiv_id)

    def download_source(self, arxiv_id: str, out_dir: Path) -> Path:
        """Download source.

        Args:
            arxiv_id (str):
            out_dir (Path):

        Returns:
            Path:
        """
        return download_and_extract_arxiv_source(arxiv_id, out_dir)
