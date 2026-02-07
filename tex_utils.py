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
    """Read file content with UTF-8 fallback."""
    return path.read_text(encoding="utf-8", errors="ignore")


def find_main_tex_file(src_dir: Path) -> Path:
    """Heuristically find the main TeX file in the arXiv source tree."""
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
    """Resolve a \\input/\\include reference to a concrete file path."""
    ref = ref.strip()
    candidates = [base / ref, base / (ref + ".tex")]
    for c in candidates:
        if c.exists() and c.is_file():
            return c
    return None


def flatten_tex(main_tex_path: Path, max_files: int = 120) -> str:
    """Inline \\input/\\include files into a single TeX string."""
    base = main_tex_path.parent
    seen = set()

    def _read(p: Path) -> str:
        if len(seen) >= max_files:
            return "\n% [flatten stopped: max_files reached]\n"
        rp = p.resolve()
        if rp in seen:
            return f"\n% [flatten skipped duplicate: {p.name}]\n"
        seen.add(rp)

        txt = read_text(p)
        txt = re.sub(r"(?m)(?<!\\)%.*$", "", txt)

        def repl(m):
            ref = m.group(1)
            child = resolve_tex_path(base, ref)
            if not child:
                return f"\n% [missing input: {ref}]\n"
            return "\n% ---- BEGIN " + child.name + " ----\n" + _read(child) + "\n% ---- END " + child.name + " ----\n"

        return INPUT_RE.sub(repl, txt)

    return _read(main_tex_path)


def strip_latex_commands(s: str) -> str:
    """Remove LaTeX commands to produce a readable text stream."""
    s = re.sub(r"\\[a-zA-Z]+\*?(?:\[[^\]]*\])?(?:\{[^}]*\})?", " ", s)
    s = s.replace("\\", " ").replace("{", " ").replace("}", " ")
    s = re.sub(r"\s+", " ", s).strip()
    return s


def build_paper_text(flat_tex: str, max_chars: Optional[int] = None) -> str:
    """Build a plain-text representation of the paper body."""
    if "\\begin{document}" in flat_tex and "\\end{document}" in flat_tex:
        flat_tex = flat_tex.split("\\begin{document}", 1)[1].rsplit("\\end{document}", 1)[0]
    txt = strip_latex_commands(flat_tex)
    if max_chars is None:
        return txt
    return txt[:max_chars]


def _esc(s: str) -> str:
    """Escape LaTeX special characters for safe rendering."""
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


def beamer_from_outline(outline: DeckOutline) -> str:
    """Render a Beamer LaTeX deck from the outline."""
    slides_tex = []
    for sl in outline.slides:
        bullets = "\n".join([f"\\item {_esc(b)}" for b in sl.bullets])

        figs = ""
        if sl.figure_suggestions:
            figs = "\\vspace{0.4em}\n{\\footnotesize\\textit{Figure ideas:} " + _esc("; ".join(sl.figure_suggestions)) + "}"

        notes = ""
        if sl.speaker_notes.strip():
            notes = f"\\vspace{{0.4em}}\\n\\\\footnotesize\\\\textit{{Notes:}} {_esc(sl.speaker_notes)}"

        slides_tex.append(
            f"""
\\begin{{frame}}[t,allowframebreaks]{{{_esc(sl.title)}}}
\\begin{{itemize}}
{bullets}
\\end{{itemize}}
{figs}
{notes}
\\end{{frame}}
""".strip()
        )

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

    maintenance = (
        f"""
\\begin{{frame}}[t]{{Maintenance Note}}
\\begin{{itemize}}
\\item After any version upgrade, run: \\texttt{{pip install -r requirements.txt}} from the codebase directory.
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

{maintenance}

\\end{{document}}
""".strip()
    )


def write_beamer(tex: str, out_dir: Path, filename_base: str = "presentation") -> Path:
    """Write Beamer LaTeX to the output directory."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tex_path = out_dir / f"{filename_base}.tex"
    tex_path.write_text(tex, encoding="utf-8")
    return tex_path


def beamer_from_outline_with_figs(outline: DeckOutline, fig_plan: dict) -> str:
    """Render a Beamer deck with separate figure slides."""
    fig_map = {x["slide_index"]: x["figures"] for x in fig_plan.get("slides", [])}

    slides_tex = []
    for idx, sl in enumerate(outline.slides, 1):
        bullets = "\n".join([f"\\item {_esc(b)}" for b in sl.bullets])

        slides_tex.append(
            f"""
\\begin{{frame}}[t,allowframebreaks]{{{_esc(sl.title)}}}
\\begin{{itemize}}
{bullets}
\\end{{itemize}}
{("\\vspace{0.3em}\\n\\\\footnotesize\\\\textit{Notes:} " + _esc(sl.speaker_notes)) if sl.speaker_notes.strip() else ""}
\\end{{frame}}
""".strip()
        )
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
