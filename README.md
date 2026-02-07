# Paper2ppt
Generate a Beamer slide deck from an arXiv paper or a local PDF using an LLM. The input is an arXiv link/ID or a PDF path, number of slides, and bullets per slide.

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

## Set Your NVIDIA API Key
Get an NVIDIA API key:
1. Create or sign in to your NVIDIA account.
2. Open the NVIDIA NIM portal and generate a new API key.
3. Copy the key and store it securely.

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

## Usage
Basic run:
```bash
paper2ppt \
  --arxiv "https://arxiv.org/abs/2602.05883" \
  --slides 10 \
  --bullets 4
```

Query-guided run (web search enabled by default):
```bash
paper2ppt \
  --arxiv 1811.12432 \
  --query "Compare this approach to prior work" \
  --slides 10 \
  --bullets 4
```

Note: Web search uses DuckDuckGo HTML results and injects top sources into the LLM context.
Note: Paper2ppt will auto-install updated dependencies if `requirements.txt` changes.

Local PDF run:
```bash
paper2ppt \
  --pdf "/path/to/paper.pdf" \
  --slides 10 \
  --bullets 4
```

Note: Local PDF parsing uses text extraction (no OCR). Scanned PDFs with no embedded text will need OCR first.

## Default Directories
By default, Paper2ppt stores all runs under:
`~/paper2ppt_runs/<paper_title_slug>/`

You can change the root run location from anywhere:
- `--root-dir "/path/to/runs"` for a one-off override
- `PAPER2PPT_ROOT_DIR="/path/to/runs"` to set a default for all runs

Inside that folder it creates:
- `work/` for intermediate files (downloaded arXiv source, flattened TeX, etc.)
- `outputs/` for final artifacts

Example output structure:
```text
~/paper2ppt_runs/Adaptive_Frame_Interpolation_for_Fast_Video_Processing/
  work/
    arxiv_source/...
  outputs/
    Adaptive_Frame_Interpolation_for_Fast_Video_Processing.tex
    Adaptive_Frame_Interpolation_for_Fast_Video_Processing.pdf
    run.log
    outline-1.json
    outline-2.json
```

Override the default root:
```bash
paper2ppt --root-dir "/path/to/runs" --arxiv 1811.12432 --slides 10 --bullets 4
```

Set a default root once:
```bash
export PAPER2PPT_ROOT_DIR="/path/to/runs"
```

Override work/output directories directly:
```bash
paper2ppt --work-dir "/tmp/p2p_work" --out-dir "/tmp/p2p_outputs" --arxiv 1811.12432 --slides 10 --bullets 4
```

Enable figures and skip the LLM sanity check:
```bash
paper2ppt \
  --arxiv 1811.12432 \
  --slides 10 \
  --bullets 4 \
  --use-figures \
  --skip-llm-sanity
```

Enable speaker notes:
```bash
paper2ppt \
  --arxiv 1811.12432 \
  --slides 10 \
  --bullets 4 \
  --with-speaker-notes
```

## Outputs
- `work/` for intermediate files
- `outputs/` for final outputs
  - `<paper_title>.tex`
  - `<paper_title>.pdf` (if `pdflatex` is installed)
  - `run.log` (full run log)
  - `outline-1.json`, `outline-2.json`, ... (all outline drafts)

Notes on structure:
- If `--root-dir` or `PAPER2PPT_ROOT_DIR` is used, each run gets its own subfolder named after a slugified paper title.
- If `--work-dir` or `--out-dir` is set, those paths are used directly and no run subfolder is created.
- Figure assets (when `--use-figures` is enabled) are copied into the `outputs/` directory alongside the generated `.tex` and `.pdf`.
- For local PDFs, extracted images are saved under `work/pdf_images/` and their paths are included in the LLM input.
- For query-guided runs, the user query is saved to `outputs/query.txt` and web sources appear in the References slide.

## CLI Options
- `--arxiv` arXiv link or ID (required if `--pdf` is not provided)
- `--pdf` path to a local PDF (required if `--arxiv` is not provided)
- `--slides` number of slides (required)
- `--bullets` bullets per slide (required)
- `--query` user query to guide the presentation theme (enables web search by default)
- `--no-web-search` disable web search even if `--query` is provided
- `--root-dir` root directory for all runs (default `$PAPER2PPT_ROOT_DIR` or `~/paper2ppt_runs`)
- `--work-dir` working directory (overrides `--root-dir`)
- `--out-dir` output directory (overrides `--root-dir`)
- `--max-summary-chunks` cap for LLM summary chunks (default `30`)
- `--no-approve` skip outline approval loop
- `--skip-llm-sanity` skip LLM sanity check
- `--model` NVIDIA NIM model name
- `--use-figures` enable figure selection and insertion
- `--with-speaker-notes` generate speaker notes for each slide
- `--verbose` verbose logs
- `--version` show version and exit

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
Download arXiv source  ->  Find main TeX  ->  Flatten TeX
  |
  v
Strip LaTeX -> Build paper text
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
Optional figures:
  Extract figures -> Plan figure placement (LLM) -> Copy figures
  |
  v
Render Beamer LaTeX
  |
  v
Compile PDF
```

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
