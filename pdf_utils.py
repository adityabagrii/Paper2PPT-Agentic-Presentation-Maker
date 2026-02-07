"""Utilities for extracting text and images from local PDF files."""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List

import fitz  # PyMuPDF


def _clean_text(s: str) -> str:
    s = (s or "").replace("\x00", " ")
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()


def extract_pdf_content(pdf_path: Path, out_dir: Path, max_pages: int | None = None) -> Dict[str, Any]:
    """Extract text and images from a PDF.

    Returns:
        dict with keys: title, text, images (list of dicts with page, path)
    """
    pdf_path = Path(pdf_path).expanduser().resolve()
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    out_dir = Path(out_dir)
    images_dir = out_dir / "pdf_images"
    images_dir.mkdir(parents=True, exist_ok=True)

    doc = fitz.open(pdf_path)
    pages = list(range(len(doc)))
    if max_pages is not None:
        pages = pages[:max_pages]

    all_text: List[str] = []
    images: List[Dict[str, Any]] = []

    for pno in pages:
        page = doc.load_page(pno)
        text = page.get_text("text")
        if text:
            all_text.append(f"\n\n[PAGE {pno + 1}]\n{text}")

        for idx, img in enumerate(page.get_images(full=True), 1):
            xref = img[0]
            pix = fitz.Pixmap(doc, xref)
            if pix.n >= 5:
                pix = fitz.Pixmap(fitz.csRGB, pix)
            img_path = images_dir / f"page_{pno + 1}_img_{idx}.png"
            pix.save(img_path.as_posix())
            pix = None
            images.append({"page": pno + 1, "path": str(img_path)})

    doc.close()

    return {
        "title": pdf_path.stem,
        "text": _clean_text("\n".join(all_text)),
        "images": images,
    }
