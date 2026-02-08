# Reading Mode (Non-Presentation)

Use this when you want understanding, not slides. It generates 1–2 pages of structured notes.

**Command**
```bash
paper2ppt -a 2401.12345 --read
```

**With diagrams embedded**
```bash
paper2ppt -a 2401.12345 --read --generate-flowcharts
```

**Intent-aware DAG diagrams**
```bash
paper2ppt -a 2401.12345 --read --generate-flowcharts --diagram-intent-aware --diagram-style dag
```
Note: `--diagram-intent-aware` and `--diagram-style` only affect diagram generation. To embed images in `reading_notes.md`, you must include `--generate-flowcharts`.

**Output**
- `outputs/reading_notes.md`
- Diagrams (if enabled): `outputs/flowcharts/reading_diagram_*.png` embedded in `reading_notes.md`
- Context cache for resume/chat: `outputs/paper_context.json` (summary + chunks)
- Progress cache: `outputs/progress.json` (summary + extraction state)

**Sections**
- Problem
- Key Idea
- Method
- Results
- Limitations
- What I Learned

**Notes**
- Works with arXiv IDs, local PDFs, or PDF URLs.
- You can combine with `--index-paper` to store a persistent memory entry.
- Reading notes are generated section-by-section with minimum length targets and a second-pass expansion if needed.
- Diagrams are embedded under the matching section header when `--generate-flowcharts` is enabled.

**Resume Read Mode**
```bash
# Reuse extracted text + summary from a prior run
paper2ppt --read --resume /path/to/run
```
- Resume prefers `outputs/paper_context.json` and falls back to `outputs/progress.json`.
- If the cached summary is missing, rerun without `--resume` to rebuild it.
