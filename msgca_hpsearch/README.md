# msgca_hpsearch — MSGCA_FV Hyperparameter Search Package

Standalone package to tune hyperparameters for **MSGCA_FV** (the Sequential Gated Cross-Attention model in CE/fair-comparison mode). Runs independently from the main `deep-finance-kggen` project — compatible with local machines and **Google Colab**.

---

## Requirements

```bash
pip install torch scikit-learn pandas numpy
```

Or use the minimal requirements file:
```bash
pip install -r requirements_minimal.txt
```

---

## Setup

### 1. You need the dataset file

Copy `unified_dataset_test.pkl` from the main project:

```
deep-finance-kggen/data/processed/unified_dataset_test.pkl
```

Place it anywhere accessible — you'll pass the path via `--data`.

### 2. Directory structure (do not move files)

```
msgca_hpsearch/
├── run_hpsearch.py          ← entry point
├── requirements_minimal.txt
├── model/                   ← MSGCA model (self-contained)
├── data/                    ← data loading (self-contained)
├── trainer/                 ← training logic (self-contained)
└── results/                 ← outputs (auto-created)
```

---

## Usage

### Full flow (search → final evaluation)

```bash
python run_hpsearch.py --data /path/to/unified_dataset_test.pkl
```

### Grid search only (fast, uses seed=42)

```bash
python run_hpsearch.py --data data.pkl --mode search
```

### Final evaluation only (with HP from previous search)

```bash
python run_hpsearch.py --data data.pkl --mode eval
```

### Custom search space

```bash
python run_hpsearch.py --data data.pkl \
    --lr 5e-5 1e-4 3e-4 5e-4 \
    --dropout 0.1 0.2 0.3
```

### Custom model architecture

```bash
python run_hpsearch.py --data data.pkl \
    --dim 64 --num-head 2 --window-size 20
```

### Resume from crash

```bash
python run_hpsearch.py --data data.pkl --mode search --resume
```

### Quick smoke test (1 combo, 1 seed, 20 epochs)

```bash
python run_hpsearch.py --data data.pkl \
    --lr 1e-4 --dropout 0.1 --n-seeds 1 --max-epochs 20
```

---

## All CLI Arguments

| Argument | Default | Description |
|---|---|---|
| `--data` | *required* | Path to `.pkl` dataset |
| `--mode` | `full` | `search` / `eval` / `full` |
| `--resume` | off | Resume search from existing results |
| `--lr` | [5e-5, 1e-4, 3e-4, 5e-4] | Learning rates to search |
| `--dropout` | [0.1, 0.2, 0.3] | Dropout values to search |
| `--dim` | `64` | Model hidden dimension |
| `--num-head` | `2` | Attention heads |
| `--window-size` | `20` | Rolling window size (days) |
| `--news-dim` | auto | News embedding dim (768=FinBERT, 1024=Voyage) |
| `--quality-dim` | `4` | Quality stats dimension |
| `--n-seeds` | `5` | Number of seeds for final eval |
| `--max-epochs` | `150` | Max training epochs |
| `--patience` | `30` | Early stopping patience |
| `--warmup` | `15` | Warmup epochs |
| `--mod-dropout` | `0.30` | News modality dropout probability |
| `--train-ratio` | `0.70` | Training data ratio |
| `--valid-ratio` | `0.15` | Validation data ratio |
| `--price-mode` | `vol_adjusted` | Price feature mode |
| `--label-mode` | `rolling` | Label generation mode |
| `--verbose` | off | Print per-epoch training progress |

---

## Output Files

All results are saved to `results/`:

| File | Content |
|---|---|
| `msgca_best_hparams.json` | Best HP from grid search |
| `msgca_all_results.json` | All combo results (for `--resume`) |
| `msgca_final_eval.json` | Final ACC/MCC mean±std across seeds |

---

## Google Colab

See `colab_hpsearch.ipynb` for a ready-to-run Colab notebook.

Quick version:
```python
# Upload msgca_hpsearch/ folder and data file, then:
!pip install torch scikit-learn pandas numpy
!python run_hpsearch.py --data unified_dataset_test.pkl --mode search --n-seeds 1
```

---

## Model Configuration

The MSGCA_FV model uses:
- **Loss**: Cross-Entropy (no focal loss, no class weights)
- **Optimizer**: AdamW with separate LN/bias no-decay group
- **Scheduler**: LinearLR warmup → CosineAnnealingLR
- **Early stopping**: on val MCC, starts after `max(warmup, 40)` epochs
- **2-phase training**: Phase 1 finds best_epoch, Phase 2 retrains on full inner split

This matches exactly the MSGCA_FV row in the baseline comparison table.

---

## Tickers

Default tickers (must match dataset):
```
TSLA, AAPL, AMZN, MSFT, GOOGL, META, BA, JPM, WMT
```

Override with `--tickers TSLA AAPL MSFT`.
