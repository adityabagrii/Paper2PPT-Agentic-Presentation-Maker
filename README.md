# Paper2ppt - An Agentic Workflow for Presentation Generation

# Author
Aditya Bagri  
Email: adityabagrii.work@gmail.com  
Academic: aditya22029@iiitd.ac.in
-----
Agentic CLI that turns arXiv papers and/or local PDFs into Beamer slide decks using LLMs. It supports query-guided presentations, optional web search with citations, figure insertion (arXiv only), speaker notes, and multi-source synthesis.

## Highlights
- arXiv, local PDF, and PDF URL inputs (single or multiple)
- Query-guided decks that answer a user question (not just summaries)
- Optional web search with citations
- Speaker notes and figure suggestions
- Robust slide generation with retries and interactive fallbacks
- Organized run directories with logs, outlines, and resume support

## Requirements
- Python 3.10+
- NVIDIA NIM API key
- `pymupdf` for local PDF parsing
- `pdflatex` for PDF output (optional but recommended)
- Optional: `NVIDIA_API_KEY` for diagram/image generation (default)

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
If a PDF URL fails to download, the CLI will prompt you to either skip that URL or quit.

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

## Streamlit GUI
Launch the interactive GUI to configure inputs and run the pipeline:
```bash
cd Paper2ppt
streamlit run gui_streamlit.py
```
The GUI supports arXiv IDs, local PDFs, PDF directories, PDF URLs, and file uploads.
You can save a default root directory from the sidebar for future runs.
For image generation, set `NVIDIA_API_KEY` in your environment (default provider).

## Diagram/Image Generation
Paper2ppt can generate diagrams or illustrative images from the slide outline's figure ideas.
Default provider uses NVIDIA's `black-forest-labs/flux.1-kontext-dev`.
To enable:
```bash
paper2ppt -a 1811.12432 --slides 10 --bullets 4 --generate-images --max-generated-images 6
```
Images are saved to `outputs/generated_images/` and included in slides automatically.
Set your NVIDIA key:
```bash
export NVIDIA_API_KEY="YOUR_KEY_HERE"
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
- `--name` custom run name for the output directory
- `--no-web-search` disable web search even if `--query` is provided
- `--retry-slides` retry count for slide generation (default `3`)
- `--retry-empty` retry count for empty LLM outputs (default `3`)
- `--interactive` enable interactive checkpoints to allow aborting
- `--check-interval` how often to prompt during interactive runs (default `5`)
- `--resume` resume from a previous run directory or outputs directory
- `--generate-images` generate diagrams/images from figure ideas
- `--image-provider` image generation provider (default `nvidia`)
- `--image-model` image generation model name (default `black-forest-labs/flux.1-kontext-dev`)
- `--max-generated-images` max generated images per run (default `6`)
- `--image-size` image size (default `1024x1024`)
- `--image-quality` image quality (low/medium/high)
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

## Interactive Feedback and Resume
- With `--interactive`, the CLI pauses at key stages and accepts optional guidance text.
- This guidance is injected into slide title and slide generation prompts.
- Resume a stopped run with `--resume /path/to/run` (or `.../outputs`).
- If the model returns the wrong number of slide titles, the CLI can display the current titles and accept user feedback before auto-fixing.

### Title Mismatch Handling
When the LLM returns the wrong number of slide titles:
1. Paper2ppt re-prompts the LLM to fix the count.
2. If still mismatched and `--interactive` is enabled, it prints the current titles and asks for guidance.
3. It then retries with your feedback or falls back to padding/truncation to meet the exact count.

### Resume Flow
When `--resume` is provided, the run loads `outputs/progress.json`, restores titles and slides generated so far, and continues from the next missing slide.

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
Collect sources (arXiv/PDFs/URLs)
  |
  v
Extract + flatten text per source
  |
  v
Chunk + summarize (LLM)
  - Multi-source: parallel chunk summarization
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
  gui_streamlit.py
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

### Optimization Updates (Summary)
- 0.5.5: GUI caches LLM client, de-duplicates uploads/downloads, and uses saved default root for faster setup.
- 0.5.4: Multi-source chunk summaries run in parallel for lower latency.
- 0.4.4: Slide generation retries avoid hard failures when JSON is malformed.
- 0.4.2: Logging falls back to temp/console on filesystem timeouts.

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
