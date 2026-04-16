# NLP Project Code Package

This package contains runnable code and generated outputs.

## Run

```bash
pip install -r requirements.txt
python src/main.py
```

Optional:

```bash
python src/main.py --output_dir results --seed 42
```

## Structure

- `src/main.py`: CLI entry.
- `src/nlp_project/pipeline.py`: task orchestration.
- `src/nlp_project/solvers/`: core computations for Q1 and Q2.
- `src/nlp_project/exporters/`: result export helpers.
- `src/nlp_project/utils/`: utility helpers.
- `src/nlp_project/constants.py`: fixed inputs and settings.

## Outputs

- `results/q1_results.json`
- `results/q2_results.json`
- `results/run_summary.json`
- `results/run_log.txt`
