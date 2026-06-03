#!/bin/bash
set -e  # Exit on error

# Always operate from the 3d_generator/ directory so the relative
# `scripts/...`, `src/...` and `output/...` paths below resolve the same way
# no matter where this script is launched from. (The previous version mixed
# workspace-root paths like `3d_generator/main.py` with repo-relative
# `scripts/...` paths, so it could only ever work from one cwd — and pointed
# at a `main.py` that does not exist; the CLI lives at `src/main.py`.)
cd "$(dirname "$(readlink -f "$0")")"

echo "========================================================"
echo "  Reproducing 3D Object Generation Paper Results"
echo "========================================================"

# Activate the project virtual environment.  Prefer the canonical
# ~/venv/3d_cem location (see README §Installation); fall back to a
# legacy in-repo ./venv if a developer still keeps one around.
if [ -d "$HOME/venv/3d_cem" ]; then
    source "$HOME/venv/3d_cem/bin/activate"
elif [ -d "venv" ]; then
    source venv/bin/activate
else
    echo "Error: Virtual environment not found." \
         "Create one with 'python3 -m venv ~/venv/3d_cem' and run" \
         "'pip install -r requirements.txt' inside it before retrying."
    exit 1
fi

echo "[1/5] Generating figures for Method section..."
python scripts/generate_figures.py

echo "[2/5] Running Experiments (Batch Generation)..."
# Force training to ensure high quality. NOTE: the CLI module is src/main.py;
# running `python src/main.py` puts src/ on sys.path[0] so its flat
# `from generator import ...` imports resolve.
python src/main.py generate -n 100 -o output/objects --train --iterations 50

echo "[3/5] Evaluating Methods (Baseline vs CEM)..."
python scripts/evaluate_methods.py

echo "[4/5] Running Ablation Study..."
python scripts/run_ablation.py

echo "[5/5] Compiling Paper..."
# The LaTeX sources live in the (separate, private) papers/ tree, not in this
# repo. Point PAPER_DIR at the directory containing main.tex to compile it;
# otherwise this step is skipped so the data-generation steps above still count
# as a successful run.
PAPER_DIR="${PAPER_DIR:-}"
if [ -n "$PAPER_DIR" ] && [ -f "$PAPER_DIR/main.tex" ]; then
    ( cd "$PAPER_DIR" \
        && pdflatex -interaction=nonstopmode main \
        && bibtex main \
        && pdflatex -interaction=nonstopmode main \
        && pdflatex -interaction=nonstopmode main )
    echo "  Paper compiled to $PAPER_DIR/main.pdf"
else
    echo "  Skipping LaTeX build (set PAPER_DIR=/path/to/paper containing main.tex to enable)."
fi

echo "========================================================"
echo "  Done. Generated data is under output/ and images/."
echo "========================================================"
