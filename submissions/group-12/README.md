# CS5785 Kaggle Competition

Human or AI? text classification (AUROC). Notebooks hold the exploration; `src/` holds the
shared, re-runnable pieces that the report experiments build on.

## Setup

```bash
python -m venv .venv            # or: uv venv .venv
.venv/Scripts/activate          # Windows; use source .venv/bin/activate on Mac/Linux
pip install -r requirements.txt
kaggle competitions download -c cornell-tech-cs-5785-combined-2026-ai-vs-human-text -p data
# unzip into data/ so that data/train.csv, data/unlabeled.csv, data/test.csv exist (data/ is git-ignored)
```

## Shared validation split (use this for every report experiment)

The rubric requires one fixed stratified 2,000-row validation set from `train.csv`, reused across
all comparisons. It is committed as `splits/val_ids.csv` (seed 42) and was created with:

```bash
python -m src.split
```

Load it in any script or notebook with:

```python
from src.data import load_split
train_df, val_df = load_split()   # 8,000 / 2,000 rows, both 50% AI
```

## Modules

| File | What it has |
|---|---|
| `src/data.py` | paths, `load_train/unlabeled/test`, `load_split` |
| `src/models.py` | model factories from notebook 02: `make_tfidf_lr`, `make_masked_lr` (and flat variant); each takes raw text |
| `src/clusters.py` | the 10 topic clusters from notebook 01 (no labels used) and the fiction cluster id |
| `src/split.py` | creates `splits/val_ids.csv` |
| `src/q3_unlabeled.py` | report section 3: self-training with pseudo-labels from `unlabeled.csv` vs supervised |
| `src/q4_analysis.py` | report section 4: train vs val, subgroup AUROC, FP/FN tables |

## Report experiments

```bash
python -m src.q3_unlabeled --model masked_lr    # writes results/q3/masked_lr/
python -m src.q4_analysis --model masked_lr     # writes results/q4/masked_lr/
```

Outputs are small CSVs plus `summary.md` under `results/`, so the report tables can be copied from there. Draft LaTeX answers for sections 3 and 4 are in `report/`.
