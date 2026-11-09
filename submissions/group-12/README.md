# Group 12: Human or AI?

**Team members:** Reilly Koren (rk856), Hazel Xue (hx342), Melinda Xu (qx234)
**Kaggle team name:** `group-12`

**Final model:** Qwen2.5-1.5B fine-tuned with LoRA on all 10,000 training passages (pending)
**Pretrained weights:** [`Qwen/Qwen2.5-0.5B`](https://huggingface.co/Qwen/Qwen2.5-0.5B) and [`Qwen/Qwen2.5-1.5B`](https://huggingface.co/Qwen/Qwen2.5-1.5B)

## Setup

Python 3.12 locally, GPU runs used Kaggle's image: Python 3.13, torch 2.11, transformers 5.20.2, peft 0.21.2, on a T4

```bash
cd submissions/group-12
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
kaggle competitions download -c cornell-tech-cs-5785-combined-2026-ai-vs-human-text -p data
unzip data/*.zip -d data         # data/train.csv, unlabeled.csv, test.csv, sample_submission.csv (not committed)
```

## Files

| Path | Contents |
|---|---|
| `notebooks/01_eda.ipynb` | EDA: duplicates, length, style features, validation, topic clusters |
| `notebooks/02_init_model.ipynb` | Non-LM models: TF-IDF, style features, masked n-grams, and a variant with linebreaks removed |
| `notebooks/03_qwen_experiment.ipynb` | Qwen2.5-0.5B LoRA fine-tune|
| `notebooks/04_qwen_final.ipynb` | One model on all training data with test pseudo-labeling, Qwen2.5-1.5B |
| `src/data.py` | Paths, `load_train` / `load_unlabeled` / `load_test`, `load_split` |
| `src/models.py` | Model factories from notebook 02: `make_tfidf_lr`, `make_masked_lr` (and the flat variant), each taking raw text |
| `src/clusters.py` | The 10 topic clusters from notebook 01 (no labels used) and the fiction cluster id |
| `src/split.py` | Creates `splits/val_ids.csv` |
| `src/q2_learning_curve.py` | Report Q2: nested, class-balanced learning-curve experiment from 500 to 8,000 rows |
| `src/q3_unlabeled.py`, `src/q4_analysis.py` | Report Q3 (self-training with `unlabeled.csv`) and Q4 (train vs validation, subgroups, errors) |
| `results/` | Saved Qwen predictions (`03/`, `04/`) and report tables/figures (`q2/`, `q3/`, `q4/`) |
| `report/` | Draft LaTeX answers for report sections 2, 3, and 4 |

## Train and evaluate on the validation protocol

Every comparison in the report uses the fixed stratified split of `train.csv`: 8,000 training and 2,000 validation rows. It's committed as `splits/val_ids.csv` (seed 42, created by `python -m src.split`) and loads with:

```python
from src.data import load_split
train_df, val_df = load_split()
```

```bash
python -m src.q2_learning_curve --model masked_lr # report Q2, writes results/q2/masked_lr/
python -m src.q3_unlabeled --model masked_lr   # report Q3, writes results/q3/masked_lr/
python -m src.q4_analysis --model masked_lr    # report Q4, writes results/q4/masked_lr/
cd notebooks
jupyter nbconvert --to notebook --execute --inplace 02_init_model.ipynb
jupyter nbconvert --to notebook --execute --inplace 03_qwen_experiment.ipynb
```

Notebooks 03 and 04 load the saved Qwen predictions in `results/03/` and `results/04/`, so they run locally. To retrain, delete those files and run the notebook on a GPU. We used Kaggle notebooks with the competition data attached, a T4 GPU and internet on, then copied the CSVs from `/kaggle/working` back into `results/`

## Write submission.csv

```bash
cd notebooks
jupyter nbconvert --to notebook --execute --inplace 04_qwen_final.ipynb
cp ../submissions/04_qwen15b_all_train.csv ../submission.csv   # TODO: or 04_qwen05b_plus_15b_rank_avg.csv
```

04 writes two candidates:
- `04_qwen15b_all_train.csv`: the 1.5B model alone, public 0.98523
- `04_qwen05b_plus_15b_rank_avg.csv`: its rank average with the 0.5B model, public 0.98529

## Seeds and config

- **Seed 77:** topic clusters (KMeans), leave-cluster-out folds, Qwen training (weights init and batch order), and the pseudo-label test halves
- **Seed 42:** the fixed validation split
- **Qwen settings:**
  - LoRA: r=16, alpha=32, dropout 0, on all attention and MLP projections
  - Optimizer: AdamW, lr 1e-4, 10% warmup then linear decay
  - Training: 1 epoch, batch 8, max 512 tokens, fp16 autocast on GPU
  - The 1.5B model also uses gradient checkpointing

## AI assistance

We used Claude Code to develop the jupyter notebooks used for our models, as well as exploring potential experiments
based on similar Kaggle competitions. We also used it to fill in information/documentation and port over our work
that we began in a separate repo.
