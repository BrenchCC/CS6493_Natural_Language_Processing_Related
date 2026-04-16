# CS6493 Assignment 02 (Code Package + Bilingual LaTeX Reports)

This folder contains a code-package style submission for CS6493 Assignment 02.
No notebook is used.

## Structure

- `docs/`: original assignment PDF and figures.
- `src/main.py`: CLI entry point.
- `src/nlp_project/constants.py`: centralized constants and prompt data.
- `src/nlp_project/solvers/`: layered solvers (`q1_solver.py`, `q2_solver.py`).
- `src/nlp_project/exporters/`: output exporters (`latex_exporter.py`).
- `src/nlp_project/pipeline.py`: orchestration layer.
- `src/nlp_project/utils/`: utility functions.
- `results/`: generated JSON outputs for Q1/Q2 and summary files.
- `reports/report_en.tex`: English report source.
- `reports/report_zh.tex`: Chinese report source.
- `reports/generated/results_macros.tex`: auto-generated LaTeX macros from code.
- `reports/output/`: compiled PDF reports.

## Install

```bash
pip install -r Assignments/assignment02/requirements.txt
```

## Run Calculations

```bash
python Assignments/assignment02/src/main.py
```

Optional arguments:

```bash
python Assignments/assignment02/src/main.py \
  --output_dir Assignments/assignment02/results \
  --generated_tex_path Assignments/assignment02/reports/generated/results_macros.tex \
  --seed 42
```

## Compile Reports

Compile English report:

```bash
latexmk -xelatex -interaction=nonstopmode -halt-on-error \
  -output-directory=Assignments/assignment02/reports/output \
  Assignments/assignment02/reports/report_en.tex
```

Compile Chinese report:

```bash
latexmk -xelatex -interaction=nonstopmode -halt-on-error \
  -output-directory=Assignments/assignment02/reports/output \
  Assignments/assignment02/reports/report_zh.tex
```

Clean LaTeX build artifacts:

```bash
latexmk -C -output-directory=Assignments/assignment02/reports/output
```

## Reproducible Workflow

1. Run `main.py` to refresh `results/*.json` and `reports/generated/results_macros.tex`.
2. Compile both LaTeX reports.
3. Submit PDF report together with this code package.

## Torch Usage Scope

- Q1.3 decoding probabilities (greedy vs beam) are computed programmatically.
- Q1.4 BLEU components are computed from tokenized n-gram counts.
- Q2.1 matrix symmetry property under `W^Q = W^K` is numerically verified.
- Q2.2 beta scaling behavior is simulated with tensor-based attention.
- Q2.3 causal mask matrix is generated explicitly.
