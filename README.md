# Paper2ppt - An Agentic Workflow for Presentation Generation

# Author
Aditya Bagri  
Email: adityabagrii.work@gmail.com  
Academic: aditya22029@iiitd.ac.in
-----
Agentic CLI that turns arXiv papers and/or local PDFs into Beamer slide decks using LLMs. It supports query-guided presentations, optional web search with citations, figure insertion (arXiv only), speaker notes, and multi-source synthesis.

## Highlights
- arXiv and local PDF inputs (single or multiple)
- Query-guided decks that answer a user question (not just summaries)
- Optional web search with citations
- Speaker notes and figure suggestions
- Robust slide generation with retries
- Organized run directories with logs and outlines

## Requirements
- Python 3.10+
- NVIDIA NIM API key
- `pymupdf` for local PDF parsing
- `pdflatex` for PDF output (optional but recommended)

## Installation
From the repository root:
```bash
cd Paper2ppt
pip install -e .
```

Update dependencies after changes:
```bash
pip install -r requirements.txt
```

Verify:
```bash
paper2ppt --version
```

Help:
```bash
paper2ppt help
```

## Get an NVIDIA API Key
1. Create or sign in to your NVIDIA account.
2. Open the NVIDIA NIM portal and generate a new API key.
3. Copy the key and store it securely.

## Set Your NVIDIA API Key
Export the key in your terminal so the CLI can read it:
```bash
export NVIDIA_API_KEY="YOUR_KEY_HERE"
```

To verify it is set:
```bash
echo $NVIDIA_API_KEY
```

To persist across sessions:
```bash
echo 'export NVIDIA_API_KEY="YOUR_KEY_HERE"' >> ~/.zshrc
source ~/.zshrc
```

If you use bash, replace `~/.zshrc` with `~/.bashrc`.

## Quick Start
Basic arXiv run:
```bash
paper2ppt \
  -a "https://arxiv.org/abs/2602.05883" \
  --slides 10 \
  --bullets 4
```

Local PDF run:
```bash
paper2ppt \
  -p "/path/to/paper.pdf" \
  --slides 10 \
  --bullets 4
```

PDF URL run:
```bash
paper2ppt \
  -u "https://example.com/paper.pdf" \
  --slides 10 \
  --bullets 4
```

Query-guided run (web search enabled by default):
```bash
paper2ppt \
  -a 1811.12432 \
  --query "Compare this approach to prior work" \
  --slides 10 \
  --bullets 4
```

Multiple arXiv IDs:
```bash
paper2ppt \
  -a "1811.12432,1707.06347" \
  --query "Compare key frame detection approaches" \
  --slides 12 \
  --bullets 4
```

Multiple PDFs from a directory:
```bash
paper2ppt \
  -d "/path/to/pdfs" \
  --query "Compare methods across papers" \
  --slides 12 \
  --bullets 4
```

Mixed sources (arXiv + PDFs):
```bash
paper2ppt \
  -a 1811.12432 \
  -p "/path/to/paper.pdf" \
  --query "Compare approaches" \
  --slides 12 \
  --bullets 4
```

## Multi-PDF and Multi-Source Workflow
When you provide multiple arXiv IDs and/or multiple PDFs, Paper2ppt:
- Parses each source separately
- Prints a source list with titles and paths/IDs
- Merges all extracted content into a single summarization pipeline
- Generates a unified deck that answers the user query across sources

Source input options:
- Repeatable args: `-p file1.pdf -p file2.pdf`
- Comma-separated lists: `-a "1811.12432,1707.06347"`
- Directory scanning: `-d "/path/to/pdfs"`
- Direct URLs: `-u "https://example.com/paper.pdf"`
- Mixed inputs: any combination of `-a`, `-p`, `-d`, and `-u`

Notes:
- Local PDF parsing uses text extraction (no OCR). Scanned PDFs with no embedded text require OCR.
- Figure insertion is only supported for a single arXiv source.

## Default Directories
By default, Paper2ppt stores all runs under:
`~/paper2ppt_runs/<paper_title_slug>/`

Inside that folder it creates:
- `work/` for intermediate files (downloaded arXiv source, flattened TeX, extracted PDF images)
- `outputs/` for final artifacts

Example output structure:
```text
~/paper2ppt_runs/Adaptive_Frame_Interpolation_for_Fast_Video_Processing/
  work/
    arxiv_1811.12432/...
    pdf_<name>/pdf_images/...
  outputs/
    Adaptive_Frame_Interpolation_for_Fast_Video_Processing.tex
    Adaptive_Frame_Interpolation_for_Fast_Video_Processing.pdf
    run.log
    query.txt
    outline-1.json
    outline-2.json
```

Override the default root:
```bash
paper2ppt --root-dir "/path/to/runs" -a 1811.12432 --slides 10 --bullets 4
```

Set a default root once:
```bash
export PAPER2PPT_ROOT_DIR="/path/to/runs"
```

Override work/output directories directly:
```bash
paper2ppt --work-dir "/tmp/p2p_work" --out-dir "/tmp/p2p_outputs" -a 1811.12432 --slides 10 --bullets 4
```

Notes on structure:
- If `--root-dir` or `PAPER2PPT_ROOT_DIR` is used, each run gets its own subfolder named after a slugified paper title.
- If `--work-dir` or `--out-dir` is set, those paths are used directly and no run subfolder is created.
- For local PDFs, extracted images are saved under `work/pdf_images/` and their paths are included in the LLM input.
- For query-guided runs, the user query is saved to `outputs/query.txt` and web sources appear in the References slide.
- For multi-source runs, all PDFs and arXiv titles are listed in the console output, and the deck title is generated from the user query plus source titles.

## Outputs
- `work/` for intermediate files
- `outputs/` for final outputs
  - `<paper_title>.tex`
  - `<paper_title>.pdf` (if `pdflatex` is installed)
  - `run.log` (full run log)
  - `outline-1.json`, `outline-2.json`, ... (all outline drafts)

## CLI Options
- `-a`, `--arxiv` arXiv link or ID (repeatable or comma-separated list)
- `-p`, `--pdf` path to a local PDF (repeatable or comma-separated list)
- `-d`, `--pdf-dir` directory containing PDFs (repeatable)
- `-u`, `--pdf-url` direct PDF URL (repeatable or comma-separated list)
- `--slides` number of slides (required)
- `--bullets` bullets per slide (required)
- `--query` user query to guide the presentation theme (enables web search by default)
- `--no-web-search` disable web search even if `--query` is provided
- `--retry-slides` retry count for slide generation (default `3`)
- `--root-dir` root directory for all runs (default `$PAPER2PPT_ROOT_DIR` or `~/paper2ppt_runs`)
- `--work-dir` working directory (overrides `--root-dir`)
- `--out-dir` output directory (overrides `--root-dir`)
- `--max-summary-chunks` cap for LLM summary chunks (default `30`)
- `--no-approve` skip outline approval loop
- `--skip-llm-sanity` skip LLM sanity check
- `--model` NVIDIA NIM model name
- `--use-figures` enable figure selection and insertion (single arXiv source only)
- `--with-speaker-notes` generate speaker notes for each slide
- `--verbose` verbose logs
- `--version` show version and exit

## Use Cases
- Quick paper summary for a talk or class
- Comparative literature review across multiple papers
- Query-driven decks like "Compare methods" or "What are the tradeoffs?"
- Generate speaker notes for rehearsals

## Workflow Diagram
```text
User CLI
  |
  v
Parse args + load env
  |
  v
Initialize LLM
  |
  v
Sanity checks
  |
  v
Collect sources (arXiv/PDFs)
  |
  v
Extract + flatten text
  |
  v
Chunk + summarize (LLM)
  |
  v
Generate slide titles (LLM)
  |
  v
Generate slides (LLM)
  |
  v
Optional approval loop (user feedback -> LLM updates)
  |
  v
Optional figures (single arXiv only)
  |
  v
Render Beamer LaTeX
  |
  v
Compile PDF
```

## Project Directory
```text
Paper2ppt/
  arxiv_utils.py
  llm.py
  logging_utils.py
  main.py
  models.py
  pdf_utils.py
  pipeline.py
  tex_utils.py
  web_utils.py
  requirements.txt
  pyproject.toml
  README.md
  CHANGELOG.md
```

## Maintenance
- After any version upgrade, run: `pip install -r requirements.txt` from the codebase directory.
- Paper2ppt auto-installs dependencies when `requirements.txt` changes, but manual updates are safer for reproducibility.

## Changelog
See `CHANGELOG.md` for version history and changes.

## Install `pdflatex`

### macOS
Option 1 (Homebrew, minimal):
```bash
brew install --cask basictex
```
Then restart your terminal.

Option 2 (full distribution):
```bash
brew install --cask mactex
```

### Windows
Option 1 (recommended, MiKTeX):
1. Download and install MiKTeX from the official site.
2. Make sure the MiKTeX `bin` folder is on your PATH.
3. Open a new terminal and run `pdflatex --version` to verify.

Option 2 (full distribution, TeX Live):
1. Install TeX Live.
2. Add the TeX Live `bin` directory to PATH.
