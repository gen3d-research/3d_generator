#!/bin/bash
set -e  # Exit on error

echo "========================================================"
echo "  Reproducing 3D Object Generation Paper Results"
echo "========================================================"

# Activate virtual environment
if [ -d "venv" ]; then
    source venv/bin/activate
else
    echo "Error: Virtual environment not found. Please run installation first."
    exit 1
fi

echo "[1/5] Generating figures for Method section..."
python scripts/generate_figures.py

echo "[2/5] Running Experiments (Batch Generation)..."
# Force training to ensure high quality
python 3d_generator/main.py generate -n 100 -o 3d_generator/output/objects --train --iterations 50

echo "[3/5] Evaluating Methods (Baseline vs CEM)..."
python scripts/evaluate_methods.py

echo "[4/5] Running Ablation Study..."
python scripts/run_ablation.py

echo "[5/5] Compiling Paper..."
# Run pdflatex/bibtex sequence
pdflatex -interaction=nonstopmode main
bibtex main
pdflatex -interaction=nonstopmode main
pdflatex -interaction=nonstopmode main

echo "========================================================"
echo "  Success! Paper compiled to main.pdf"
echo "========================================================"
