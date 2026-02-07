"""Standalone image generation sanity check.

Generates a few images from an article text to validate image pipeline and API keys.
"""
from __future__ import annotations

import argparse
import os
import re
from pathlib import Path
from typing import List

from arxiv_utils import download_and_extract_arxiv_source, extract_arxiv_id
from image_gen_utils import generate_images_openai, generate_images_nvidia
from tex_utils import build_paper_text, find_main_tex_file, flatten_tex


def _load_article(path: str | None, text: str | None, arxiv_id_or_url: str | None) -> str:
    if arxiv_id_or_url:
        arxiv_id = extract_arxiv_id(arxiv_id_or_url)
        out_dir = Path("./image_sanity_outputs").expanduser().resolve()
        src_dir = download_and_extract_arxiv_source(arxiv_id, out_dir)
        main_tex = find_main_tex_file(src_dir)
        flat = flatten_tex(main_tex)
        return build_paper_text(flat, max_chars=200000)
    if path:
        p = Path(path).expanduser().resolve()
        if not p.exists():
            raise FileNotFoundError(f"Article file not found: {p}")
        return p.read_text(encoding="utf-8", errors="replace")
    if text:
        return text
    raise ValueError("Provide --arxiv, --article-file, or --article-text")


def _extract_sentences(text: str) -> List[str]:
    cleaned = re.sub(r"\s+", " ", text).strip()
    if not cleaned:
        return []
    sentences = re.findall(r"[^.!?]+[.!?]", cleaned)
    if not sentences:
        # Fallback: split by newlines
        sentences = [s.strip() for s in text.splitlines() if s.strip()]
    return [s.strip() for s in sentences if s.strip()]


def build_prompts_from_article(text: str, max_prompts: int = 4) -> List[str]:
    sentences = _extract_sentences(text)
    prompts: List[str] = []

    for s in sentences:
        if len(prompts) >= max_prompts:
            break
        if len(s) < 40:
            continue
        excerpt = s
        if len(excerpt) > 220:
            excerpt = excerpt[:220].rsplit(" ", 1)[0] + "..."
        prompt = (
            "Create a clean, minimal diagram suitable for a presentation slide. "
            "Use a neutral background, simple shapes, and minimal text labels. "
            f"Base it on this excerpt: {excerpt}"
        )
        prompts.append(prompt)

    if not prompts:
        # Fallback: use chunks of words
        words = re.findall(r"\w+", text)
        chunk = " ".join(words[:120])
        if chunk:
            prompts.append(
                "Create a clean, minimal diagram suitable for a presentation slide. "
                "Use a neutral background, simple shapes, and minimal text labels. "
                f"Base it on this excerpt: {chunk}"
            )
    return prompts


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Sanity-check image generation from an article.")
    p.add_argument("--arxiv", help="arXiv ID or URL (downloads LaTeX source)")
    p.add_argument("--article-file", help="Path to article text file")
    p.add_argument("--article-text", help="Article text inline (use quotes)")
    p.add_argument("--out-dir", default="./image_sanity_outputs", help="Output directory")
    p.add_argument("--provider", default="nvidia", choices=["nvidia", "openai"], help="Image provider")
    p.add_argument("--model", default="", help="Model name override")
    p.add_argument("--size", default="1:1", help="Aspect ratio (nvidia) or size (openai)")
    p.add_argument("--quality", default="medium", help="Quality (openai only)")
    p.add_argument("--max-images", type=int, default=4, help="Max images to generate")
    p.add_argument("--api-key", default="", help="API key override")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    article = _load_article(args.article_file, args.article_text, args.arxiv)
    prompts = build_prompts_from_article(article, max_prompts=max(1, args.max_images))
    if not prompts:
        raise RuntimeError("No prompts could be created from the article.")

    out_dir = Path(args.out_dir).expanduser().resolve()
    provider = args.provider.lower()

    if provider == "openai":
        api_key = args.api_key or os.getenv("OPENAI_API_KEY", "")
        model = args.model or "gpt-image-1.5"
        created = generate_images_openai(
            prompts,
            out_dir,
            api_key=api_key,
            model=model,
            size=args.size,
            quality=args.quality,
            max_images=args.max_images,
        )
    else:
        api_key = args.api_key or os.getenv("NVIDIA_API_KEY", "")
        model = args.model or "black-forest-labs/flux.1-kontext-dev"
        created = generate_images_nvidia(
            prompts,
            out_dir,
            api_key=api_key,
            model=model,
            aspect_ratio=args.size,
            max_images=args.max_images,
        )

    print("Generated images:")
    for p in created:
        print(f"- {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
