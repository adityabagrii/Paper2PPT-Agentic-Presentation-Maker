"""Image generation utilities."""
from __future__ import annotations

import base64
from pathlib import Path
from typing import List, Tuple

import requests


def generate_images_openai(
    prompts: List[str],
    out_dir: Path,
    api_key: str,
    model: str = "gpt-image-1.5",
    size: str = "1024x1024",
    quality: str = "medium",
    max_images: int = 6,
) -> List[Path]:
    """Generate images with OpenAI Images API."""
    out_dir.mkdir(parents=True, exist_ok=True)
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is required for OpenAI image generation.")

    created: List[Path] = []
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    url = "https://api.openai.com/v1/images/generations"

    for i, prompt in enumerate(prompts, 1):
        if len(created) >= max_images:
            break
        payload = {
            "model": model,
            "prompt": prompt,
            "size": size,
            "quality": quality,
            "response_format": "b64_json",
        }
        r = requests.post(url, headers=headers, json=payload, timeout=60)
        r.raise_for_status()
        data = r.json()
        b64 = data["data"][0]["b64_json"]
        img_bytes = base64.b64decode(b64)
        path = out_dir / f"gen_{i:03d}.png"
        path.write_bytes(img_bytes)
        created.append(path)

    return created


def generate_images_nvidia(
    prompts: List[str],
    out_dir: Path,
    api_key: str,
    model: str = "black-forest-labs/flux.1-kontext-dev",
    aspect_ratio: str = "1:1",
    steps: int = 30,
    cfg_scale: float = 3.5,
    seed: int = 0,
    max_images: int = 6,
) -> List[Path]:
    """Generate images with NVIDIA GenAI endpoint (Flux Kontext)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    if not api_key:
        raise RuntimeError("NVIDIA_API_KEY is required for NVIDIA image generation.")

    created: List[Path] = []
    headers = {"Authorization": f"Bearer {api_key}", "Accept": "application/json"}
    url = f"https://ai.api.nvidia.com/v1/genai/{model}"

    for i, prompt in enumerate(prompts, 1):
        if len(created) >= max_images:
            break
        payload = {
            "prompt": prompt,
            "aspect_ratio": aspect_ratio,
            "steps": steps,
            "cfg_scale": cfg_scale,
            "seed": seed,
        }
        r = requests.post(url, headers=headers, json=payload, timeout=90)
        r.raise_for_status()
        data = r.json()
        b64 = data.get("image") or data.get("data") or data.get("b64_json")
        if isinstance(b64, list):
            b64 = b64[0]
        if not b64:
            raise RuntimeError("NVIDIA image response missing image payload.")
        img_bytes = base64.b64decode(b64.split(",")[-1])
        path = out_dir / f"gen_{i:03d}.png"
        path.write_bytes(img_bytes)
        created.append(path)

    return created


def build_prompts_from_slides(slides) -> List[Tuple[int, str]]:
    """Return list of (slide_index, prompt)."""
    prompts: List[Tuple[int, str]] = []
    for idx, sl in enumerate(slides, 1):
        ideas = list(sl.figure_suggestions or [])
        if not ideas:
            continue
        for idea in ideas:
            prompt = (
                "Create a clean, minimal diagram suitable for a presentation slide. "
                f"Slide title: {sl.title}. Figure idea: {idea}. "
                "Use a neutral background, simple shapes, and minimal text labels."
            )
            prompts.append((idx, prompt))
    return prompts
