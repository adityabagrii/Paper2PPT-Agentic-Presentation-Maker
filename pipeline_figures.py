from __future__ import annotations

import json
import re
import shutil
from pathlib import Path
from typing import Dict, List, Optional

try:
    from .llm import safe_invoke
    from .models import DeckOutline
except Exception:
    from llm import safe_invoke
    from models import DeckOutline

try:
    from .pipeline_common import logger
    from .pipeline_outline import OutlineBuilder
except Exception:
    from pipeline_common import logger
    from pipeline_outline import OutlineBuilder


class FigureAsset:
    def __init__(self, tex_path: str, resolved_path: str, caption: str, label: Optional[str]) -> None:
        """Initialize.
        
        Args:
            tex_path (str):
            resolved_path (str):
            caption (str):
            label (Optional[str]):
        
        Returns:
            None:
        """
        self.tex_path = tex_path
        self.resolved_path = resolved_path
        self.caption = caption
        self.label = label


class FigurePlanner:
    FIG_ENV_RE = re.compile(r"\\begin\{figure\}[\s\S]*?\\end\{figure\}", re.MULTILINE)
    CAP_RE = re.compile(r"\\caption\*?\{([\s\S]*?)\}")
    LAB_RE = re.compile(r"\\label\{([\s\S]*?)\}")
    INC_RE = re.compile(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}")

    @staticmethod
    def _strip_tex(s: str) -> str:
        """Strip tex.
        
        Args:
            s (str):
        
        Returns:
            str:
        """
        s = re.sub(r"(?m)(?<!\\\\)%.*$", "", s)
        s = re.sub(r"\\[a-zA-Z]+\\*?(?:\\[[^]]*\\])?(?:\\{[^}]*\\})?", " ", s)
        s = s.replace("{", " ").replace("}", " ").replace("\\\\", " ")
        s = re.sub(r"\\s+", " ", s).strip()
        return s

    @staticmethod
    def resolve_graphic_path(src_dir: Path, tex_ref: str) -> Optional[Path]:
        """Resolve graphic path.
        
        Args:
            src_dir (Path):
            tex_ref (str):
        
        Returns:
            Optional[Path]:
        """
        tex_ref = tex_ref.strip()
        candidates = [
            src_dir / tex_ref,
            src_dir / (tex_ref + ".pdf"),
            src_dir / (tex_ref + ".png"),
            src_dir / (tex_ref + ".jpg"),
            src_dir / (tex_ref + ".jpeg"),
        ]
        for c in candidates:
            if c.exists() and c.is_file():
                return c

        base = Path(tex_ref).name
        for ext in [".pdf", ".png", ".jpg", ".jpeg"]:
            hits = list(src_dir.rglob(base if base.endswith(ext) else base + ext))
            if hits:
                return hits[0]
        return None

    def extract_figures(self, flat_tex: str, src_dir: Path) -> List[FigureAsset]:
        """Extract figures.
        
        Args:
            flat_tex (str):
            src_dir (Path):
        
        Returns:
            List[FigureAsset]:
        """
        figs: List[FigureAsset] = []
        for env in self.FIG_ENV_RE.findall(flat_tex):
            cap_m = self.CAP_RE.search(env)
            caption = self._strip_tex(cap_m.group(1)) if cap_m else ""
            lab_m = self.LAB_RE.search(env)
            label = lab_m.group(1).strip() if lab_m else None

            for inc_m in self.INC_RE.finditer(env):
                tex_ref = inc_m.group(1).strip()
                p = self.resolve_graphic_path(src_dir, tex_ref)
                if p is None:
                    continue
                figs.append(FigureAsset(tex_ref, str(p), caption, label))

        uniq: Dict[str, FigureAsset] = {}
        for f in figs:
            uniq[f.resolved_path] = f
        return list(uniq.values())

    def plan_with_llm(self, llm, outline: DeckOutline, fig_assets: List[FigureAsset], max_figs: int = 12) -> dict:
        """Plan with llm.
        
        Args:
            llm (Any):
            outline (DeckOutline):
            fig_assets (List[FigureAsset]):
            max_figs (int):
        
        Returns:
            dict:
        """
        if not fig_assets:
            return {"slides": []}

        figs = fig_assets[:max_figs]
        catalog = "\n".join([f"- {Path(f.resolved_path).name}: {f.caption[:120]}" for f in figs])
        slide_titles = "\n".join(
            [
                f"{i+1}. {getattr(s, 'title', None) or (s.get('title') if isinstance(s, dict) else 'Slide')}"
                for i, s in enumerate(outline.slides)
            ]
        )

        prompt = f"""
Return ONLY JSON.

Schema:
{{
  "slides": [
    {{
      "slide_index": 1,
      "figures": [{{"file": "filename.ext", "why": "short", "caption": "short"}}]
    }}
  ]
}}

Rules:
- Only choose from the filenames listed below.
- At most 1 figure per slide.
- Skip slides without a strong matching figure.
- Keep explanations short.
- Generate a short, descriptive caption for the selected figure.

Slides:
{slide_titles}

Available figures (filename: caption):
{catalog}
""".strip()

        raw = safe_invoke(logger, llm, prompt, retries=6)
        js = OutlineBuilder.try_extract_json(raw)
        if js is None:
            logger.warning("Figure plan JSON parse failed. Skipping figures.")
            return {"slides": []}

        try:
            obj = json.loads(js)
        except Exception:
            return {"slides": []}

        allowed = {Path(f.resolved_path).name for f in figs}
        cleaned = {"slides": []}
        for s in obj.get("slides", []):
            if not isinstance(s, dict):
                continue
            idx = s.get("slide_index")
            figs_out = []
            for g in s.get("figures", []):
                name = g.get("file")
                if name in allowed:
                    figs_out.append({
                        "file": name,
                        "why": g.get("why", ""),
                        "caption": g.get("caption", ""),
                    })
            if idx and figs_out:
                cleaned["slides"].append({"slide_index": idx, "figures": figs_out})

        return cleaned

    def materialize(self, fig_plan: dict, fig_assets: List[FigureAsset], out_dir: Path) -> dict:
        """Materialize.
        
        Args:
            fig_plan (dict):
            fig_assets (List[FigureAsset]):
            out_dir (Path):
        
        Returns:
            dict:
        """
        out_dir = Path(out_dir)
        fig_dir = out_dir / "figures"
        fig_dir.mkdir(parents=True, exist_ok=True)

        by_name = {Path(f.resolved_path).name: f for f in fig_assets}

        resolved = {"slides": []}
        for s in fig_plan.get("slides", []):
            new_item = {"slide_index": s["slide_index"], "figures": []}
            for g in s.get("figures", []):
                name = g.get("file")
                if not name or name not in by_name:
                    continue
                src_path = Path(by_name[name].resolved_path)
                dst_path = fig_dir / name
                shutil.copy2(src_path, dst_path)
                if not dst_path.exists():
                    continue
                new_item["figures"].append({
                    "file": str(Path("figures") / name),
                    "why": g.get("why", ""),
                    "caption": g.get("caption", ""),
                })
            if new_item["figures"]:
                resolved["slides"].append(new_item)

        return resolved
