"""Graphviz flowchart utilities."""
from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path
from typing import List


def _sanitize_label(s: str, max_chars: int = 60) -> str:
    s = re.sub(r"[\[\]{}()<>]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    if len(s) > max_chars:
        s = s[:max_chars].rsplit(" ", 1)[0] + "..."
    return s


def build_graphviz(steps: List[str], structure: str = "linear") -> str:
    lines = [
        "digraph Flowchart {",
        "  rankdir=LR;",
        "  node [shape=box, style=rounded, fontsize=12];",
    ]
    for i, s in enumerate(steps, 1):
        lines.append(f'  N{i} [label="{_sanitize_label(s)}"];')

    if structure == "branch":
        for i in range(1, min(3, len(steps))):
            lines.append(f"  N{i} -> N{i+1};")
        if len(steps) >= 4:
            lines.append("  N3 -> N4;")
        if len(steps) >= 5:
            lines.append("  N3 -> N5;")
        if len(steps) >= 6:
            lines.append("  N4 -> N6;")
            lines.append("  N5 -> N6;")
        for i in range(6, len(steps)):
            lines.append(f"  N{i} -> N{i+1};")
    elif structure == "cycle":
        for i in range(1, len(steps)):
            lines.append(f"  N{i} -> N{i+1};")
        if len(steps) >= 3:
            lines.append("  N3 -> N2;")
    else:
        for i in range(1, len(steps)):
            lines.append(f"  N{i} -> N{i+1};")

    lines.append("}")
    return "\n".join(lines)


def render_graphviz(dot_path: Path, out_path: Path) -> None:
    if shutil.which("dot"):
        subprocess.run(["dot", "-Tpng", str(dot_path), "-o", str(out_path)], check=True)
        return
    try:
        import graphviz  # type: ignore

        src = graphviz.Source(dot_path.read_text(encoding="utf-8"))
        src.format = "png"
        src.render(out_path.with_suffix("").as_posix(), cleanup=True)
        return
    except Exception:
        pass
    raise RuntimeError(
        "Graphviz renderer not found. Install with: `brew install graphviz` "
        "or `pip install graphviz` and ensure `dot` is on PATH."
    )
