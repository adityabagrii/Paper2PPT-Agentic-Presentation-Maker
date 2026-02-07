# Changelog

## 0.5.1 - 2026-02-07
- Added `--pdf-url` for direct PDF links (repeatable or comma-separated).

## 0.5.0 - 2026-02-07
- Added multi-source support: multiple arXiv IDs, multiple PDFs, and PDF directories.
- Pretty-printed outline display for approval.
- Deck title is generated from the user query and source titles.

## 0.4.5 - 2026-02-07
- Added `--retry-slides` to control slide generation retries.

## 0.4.4 - 2026-02-07
- Retry slide generation on malformed JSON and fall back instead of crashing.

## 0.4.3 - 2026-02-07
- Print top web search results for user reference during query-guided runs.

## 0.4.2 - 2026-02-07
- Made file logging resilient to filesystem timeouts by falling back to a temp log or console-only.

## 0.4.1 - 2026-02-07
- Added auto-install of updated dependencies when `requirements.txt` changes.

## 0.4.0 - 2026-02-07
- Added `--query` to steer the presentation goal beyond summarization.
- Added automatic web search when `--query` is set (disable with `--no-web-search`).
- Saved user queries to `outputs/query.txt`.
- Added source citations from web search to outline and final references.

## 0.3.0 - 2026-02-07
- Added local PDF input via `--pdf` with text and image extraction for LLM chunking.
- Added PyMuPDF (`pymupdf`) dependency for PDF parsing.
- Local PDFs disable figure insertion (arXiv-only).
- Kept fixed-width progress bars and live chunk preview.

## 0.2.3 - 2026-02-07
- Fixed progress bar width with `ncols` so resizing the terminal does not shift bars.
- Added live chunk preview and previous summary preview during chunk summarization.

## 0.2.2 - 2026-02-07
- README updated with steps to obtain and set NVIDIA API keys.
- API key check now warns instead of exiting when missing.
- Added `paper2ppt help` helper command.

## 0.2.1 - 2026-02-07
- Added `paper2ppt help` helper command for quick terminal guidance.

## 0.2.0 - 2026-02-07
- Added `PAPER2PPT_ROOT_DIR` to set a default runs root from any working directory.
- Clarified output directory structure and precedence of `--root-dir`, `--work-dir`, and `--out-dir`.

## 0.1.0 - 2026-02-07
- Initial class-based refactor of the original PresentationAgent pipeline.
- CLI entry point `paper2ppt`.
- Outline drafts saved to `outline-*.json` in output directory.
- Run logs saved to `run.log` in output directory.
- Optional speaker notes (`--with-speaker-notes`, default off).
