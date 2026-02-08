"""TeX parsing, text extraction, and Beamer rendering utilities."""
from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

try:
    from .models import DeckOutline
except Exception:
    from models import DeckOutline

INPUT_RE = re.compile(r"\\(?:input|include)\{([^}]+)\}")


def read_text(path: Path) -> str:
    """Function read text.
    
    Args:
        path (Path):
    
    Returns:
        str:
    """
    return path.read_text(encoding="utf-8", errors="ignore")


def find_main_tex_file(src_dir: Path) -> Path:
    """Find main tex file.
    
    Args:
        src_dir (Path):
    
    Returns:
        Path:
    """
    candidates = []
    for p in src_dir.rglob("*.tex"):
        try:
            t = read_text(p)
        except Exception:
            continue
        score = 0
        if "\\documentclass" in t:
            score += 10
        if "\\begin{document}" in t:
            score += 5
        if "\\title" in t or "\\title*" in t:
            score += 3
        score += min(len(t), 200000) / 20000
        if score > 0:
            candidates.append((score, p))
    if not candidates:
        raise RuntimeError("Could not find a plausible main TeX file.")
    candidates.sort(reverse=True, key=lambda x: x[0])
    return candidates[0][1]


def resolve_tex_path(base: Path, ref: str) -> Optional[Path]:
    """Resolve tex path.
    
    Args:
        base (Path):
        ref (str):
    
    Returns:
        Optional[Path]:
    """
    ref = ref.strip()
    candidates = [base / ref, base / (ref + ".tex")]
    for c in candidates:
        if c.exists() and c.is_file():
            return c
    return None


def flatten_tex(main_tex_path: Path, max_files: int = 120) -> str:
    """Flatten tex.
    
    Args:
        main_tex_path (Path):
        max_files (int):
    
    Returns:
        str:
    """
    base = main_tex_path.parent
    seen = set()

    def _read(p: Path) -> str:
        """Function read.
        
        Args:
            p (Path):
        
        Returns:
            str:
        """
        if len(seen) >= max_files:
            return "\n% [flatten stopped: max_files reached]\n"
        rp = p.resolve()
        if rp in seen:
            return f"\n% [flatten skipped duplicate: {p.name}]\n"
        seen.add(rp)

        txt = read_text(p)
        txt = re.sub(r"(?m)(?<!\\)%.*$", "", txt)

        def repl(m):
            """Function repl.
            
            Args:
                m (Any):
            
            Returns:
                Any:
            """
            ref = m.group(1)
            child = resolve_tex_path(base, ref)
            if not child:
                return f"\n% [missing input: {ref}]\n"
            return "\n% ---- BEGIN " + child.name + " ----\n" + _read(child) + "\n% ---- END " + child.name + " ----\n"

        return INPUT_RE.sub(repl, txt)

    return _read(main_tex_path)


def strip_latex_commands(s: str) -> str:
    """Strip latex commands.
    
    Args:
        s (str):
    
    Returns:
        str:
    """
    s = re.sub(r"\\[a-zA-Z]+\*?(?:\[[^]]*\])?(?:\{[^}]*\})?", " ", s)
    s = s.replace("\\", " ").replace("{", " ").replace("}", " ")
    s = re.sub(r"\s+", " ", s).strip()
    return s


def build_paper_text(flat_tex: str, max_chars: Optional[int] = None) -> str:
    """Build paper text.
    
    Args:
        flat_tex (str):
        max_chars (Optional[int]):
    
    Returns:
        str:
    """
    if "\\begin{document}" in flat_tex and "\\end{document}" in flat_tex:
        flat_tex = flat_tex.split("\\begin{document}", 1)[1].rsplit("\\end{document}", 1)[0]
    txt = strip_latex_commands(flat_tex)
    if max_chars is None:
        return txt
    return txt[:max_chars]


def _esc(s: str) -> str:
    """Function esc.
    
    Args:
        s (str):
    
    Returns:
        str:
    """
    return (
        s.replace("\\", "\\textbackslash ")
        .replace("&", "\\&")
        .replace("%", "\\%")
        .replace("#", "\\#")
        .replace("_", "\\_")
        .replace("$", "\\$")
        .replace("{", "\\{")
        .replace("}", "\\}")
    )


def _slide_get(sl, key: str, default=None):
    if hasattr(sl, key):
        return getattr(sl, key)
    if isinstance(sl, dict):
        return sl.get(key, default)
    return default


def _slide_list(sl, key: str) -> list:
    val = _slide_get(sl, key, [])
    return val if isinstance(val, list) else []


def _image_frames(sl, title: str = "") -> list[str]:
    frames: list[str] = []
    captions = list(_slide_list(sl, "image_captions") or [])
    cap_idx = 0

    def _frame(img_path: str, caption: str, prefix: str) -> str:
        if caption:
            frame_title = f"{prefix}: {caption}"
        elif title:
            frame_title = f"{prefix}: {title}"
        else:
            frame_title = f"{prefix}"
        cap_line = f"\\vspace{{0.2em}}\n{{\\footnotesize\\textit{{{prefix}:}} {_esc(caption)}}}" if caption else ""
        return f"""
\\begin{{frame}}[t]{{{_esc(frame_title)}}}
\\begin{{center}}
\\includegraphics[width=0.9\\linewidth,height=0.8\\textheight,keepaspectratio]{{{_esc(img_path)}}}
\\end{{center}}
{cap_line}
\\end{{frame}}
""".strip()

    flow_caption = ""
    flowchart = _slide_get(sl, "flowchart", None)
    if flowchart:
        if isinstance(flowchart, dict):
            flow_caption = flowchart.get("caption", "") or ""
        else:
            flow_caption = getattr(flowchart, "caption", "") or ""

    for p in _slide_list(sl, "flowchart_images"):
        if p:
            frames.append(_frame(p, flow_caption, "Diagram"))

    for p in _slide_list(sl, "generated_images"):
        if p:
            cap = ""
            if cap_idx < len(captions):
                cap = captions[cap_idx]
                cap_idx += 1
            frames.append(_frame(p, cap, "Figure"))

    return frames


def beamer_from_outline(outline: DeckOutline) -> str:
    """Function beamer from outline.
    
    Args:
        outline (DeckOutline):
    
    Returns:
        str:
    """
    slides_tex = []
    for sl in outline.slides:
        bullets_list = _slide_list(sl, "bullets")
        bullets = "\n".join([f"\\item {_esc(b)}" for b in bullets_list])

        figs = ""
        fig_sugs = _slide_list(sl, "figure_suggestions")
        if fig_sugs:
            figs = "\\vspace{0.4em}\n{\\footnotesize\\textit{Figure ideas:} " + _esc("; ".join(fig_sugs)) + "}"

        tables_tex = ""
        tables = _slide_list(sl, "tables")
        if tables:
            for t in tables:
                if isinstance(t, dict):
                    cols = [str(c) for c in t.get("columns", [])]
                    rows = [r for r in t.get("rows", [])]
                    title = t.get("title", "") or "Results"
                else:
                    cols = [str(c) for c in getattr(t, "columns", [])]
                    rows = [r for r in getattr(t, "rows", [])]
                    title = getattr(t, "title", "") or "Results"
                if not cols or not rows:
                    continue
                col_spec = " | ".join(["l"] * len(cols))
                header = " & ".join([_esc(c) for c in cols]) + " \\\\ \\hline\n"
                body_lines = []
                for r in rows:
                    body_lines.append(" & ".join([_esc(str(x)) for x in r]) + " \\\\")
                body = "\n".join(body_lines)
                title = _esc(str(title))
                tables_tex += (
                    "\n\\vspace{0.4em}\n"
                    f"{{\\footnotesize\\textit{{Table:}} {title}}}\n"
                    "\\vspace{0.2em}\n"
                    "\\begin{tabular}{"
                    + col_spec
                    + "}\n\\hline\n"
                    + header
                    + body
                    + "\n\\hline\n\\end{tabular}\n"
                )

        notes = ""
        speaker_notes = _slide_get(sl, "speaker_notes", "") or ""
        if str(speaker_notes).strip():
            notes = (
                "\\vspace{0.4em}\n"
                f"{{\\footnotesize\\textit{{Notes:}} {_esc(str(speaker_notes))}}}"
            )

        slides_tex.append(
            f"""
\\begin{{frame}}[t,allowframebreaks]{{{_esc(str(_slide_get(sl, 'title', 'Slide')))}}}
\\begin{{itemize}}
{bullets}
\\end{{itemize}}
{figs}
{tables_tex}
{notes}
\\end{{frame}}
""".strip()
        )
        slides_tex.extend(_image_frames(sl, title=str(_slide_get(sl, "title", ""))))

    refs = ""
    if outline.citations:
        items = "\n".join([f"\\item {_esc(c)}" for c in outline.citations if c.strip()])
        if items.strip():
            refs = (
                f"""
\\begin{{frame}}[t,allowframebreaks]{{References}}
\\begin{{itemize}}
{items}
\\end{{itemize}}
\\end{{frame}}
""".strip()
            )

    credit = (
        f"""
\\begin{{frame}}[t]{{}}
\\centering
{{\\LARGE Presentation generated using Paper2PPT}}\\\\[0.6em]
{{\\large by Aditya Bagri}}
\\end{{frame}}
""".strip()
    )

    return (
        f"""
\\documentclass[aspectratio=169]{{beamer}}
\\usepackage[utf8]{{inputenc}}
\\usepackage{{hyperref}}
\\usepackage{{graphicx}}
\\usepackage{{amsmath}}
\\setbeamertemplate{{navigation symbols}}{{}}
\\title{{{_esc(outline.deck_title)}}}
\\subtitle{{Sources: {_esc(outline.arxiv_id)}}}
\\author{{Auto-generated}}
\\date{{}}

\\begin{{document}}

\\begin{{frame}}
\\centering
{{\\LARGE \\textbf{{{_esc(outline.deck_title)}}}}}
\\vspace{{0.6em}}

{{\\normalsize Sources: {_esc(outline.arxiv_id)}}}
\\end{{frame}}

{chr(10).join(slides_tex)}

{refs}

{credit}

\\end{{document}}
""".strip()
    )


def write_beamer(tex: str, out_dir: Path, filename_base: str = "presentation") -> Path:
    """Write beamer.
    
    Args:
        tex (str):
        out_dir (Path):
        filename_base (str):
    
    Returns:
        Path:
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tex_path = out_dir / f"{filename_base}.tex"
    tex_path.write_text(tex, encoding="utf-8")
    return tex_path


def beamer_from_outline_with_figs(outline: DeckOutline, fig_plan: dict) -> str:
    """Function beamer from outline with figs.
    
    Args:
        outline (DeckOutline):
        fig_plan (dict):
    
    Returns:
        str:
    """
    fig_map = {x["slide_index"]: x["figures"] for x in fig_plan.get("slides", [])}

    slides_tex = []
    for idx, sl in enumerate(outline.slides, 1):
        bullets_list = _slide_list(sl, "bullets")
        bullets = "\n".join([f"\\item {_esc(b)}" for b in bullets_list])
        slides_tex.append(
            f"""
\\begin{{frame}}[t,allowframebreaks]{{{_esc(str(_slide_get(sl, 'title', 'Slide')))}}}
\\begin{{itemize}}
{bullets}
\\end{{itemize}}
{("\\vspace{0.3em}\n{\\footnotesize\\textit{Notes:} " + _esc(str(_slide_get(sl, 'speaker_notes', ''))) + "}") if str(_slide_get(sl, 'speaker_notes', '')).strip() else ""}
\\end{{frame}}
""".strip()
        )
        slides_tex.extend(_image_frames(sl, title=str(_slide_get(sl, "title", ""))))
        figs = fig_map.get(idx, [])
        if figs:
            f0 = figs[0]["file"]
            cap = figs[0].get("caption", "")
            fig_frame = f"""
\\begin{{frame}}[t]{{Figure - {cap}}}
\\begin{{center}}
\\includegraphics[width=0.85\\linewidth,height=0.7\\textheight,keepaspectratio]{{{_esc(f0)}}}
\\end{{center}}
\\vspace{{0.2em}}
\\end{{frame}}
""".strip()
            slides_tex.append(fig_frame)

    refs = ""
    if outline.citations:
        items = "\n".join([f"\\item {_esc(c)}" for c in outline.citations if c.strip()])
        if items.strip():
            refs = (
                f"""
\\begin{{frame}}[t,allowframebreaks]{{References}}
\\begin{{itemize}}
{items}
\\end{{itemize}}
\\end{{frame}}
""".strip()
            )

    return (
        f"""
\\documentclass[aspectratio=169]{{beamer}}
\\usepackage[utf8]{{inputenc}}
\\usepackage{{hyperref}}
\\usepackage{{graphicx}}
\\usepackage{{amsmath}}
\\setbeamertemplate{{navigation symbols}}{{}}
\\title{{{_esc(outline.deck_title)}}}
\\subtitle{{Sources: {_esc(outline.arxiv_id)}}}
\\author{{Auto-generated}}
\\date{{}}

\\begin{{document}}

\\begin{{frame}}
\\centering
{{\\LARGE \\textbf{{{_esc(outline.deck_title)}}}}}
\\vspace{{0.6em}}

{{\\normalsize Sources: {_esc(outline.arxiv_id)}}}
\\end{{frame}}

{chr(10).join(slides_tex)}

{refs}

\\end{{document}}
""".strip()
    )
