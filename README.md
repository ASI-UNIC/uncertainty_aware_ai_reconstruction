# Uncertainty-Aware AI Reconstruction of Wall-Pressure Dynamics from Sparse Data in Extreme Flows

This repository accompanies the paper:

> Fung et al., "Uncertainty-aware artificial intelligence reconstruction of wall-pressure dynamics from sparse data in extreme flows," *Physics of Fluids* **38**, 066110 (2026).
> DOI: [10.1063/5.0332704](https://doi.org/10.1063/5.0332704) — AIP Featured Article (also highlighted by [AIP Scilight](https://www.aip.org/scilights/hybrid-framework-provides-high-resolution-reconstruction-of-hypersonic-wall-pressure-signals)).

A hybrid AI framework that reconstructs high-resolution hypersonic wall-pressure signals from sparse measurements. It combines cubic-spline interpolation with a transformer model that refines the reconstruction by learning spatial relationships. The approach is evaluated across 4 turbulence cases (perturbation levels 0.09, 0.130, 0.155, 0.190), multiple sparsity levels, and multiple sequence lengths. As reported in the paper, MAPE is below 1% in minimally sparse setups and below 4% in moderately sparse setups.

---

## 1. Requirements

- Python 3.10+
- Packages: `torch`, `pytorch-lightning`, `numpy`, `pandas`, `scipy`, `scikit-learn`, `matplotlib` (`argparse`, `os`, `pickle` are standard library).
- There is no `requirements.txt` in this repository — install the packages directly. A virtualenv is recommended:

```bash
python -m venv .venv && source .venv/bin/activate
pip install torch pytorch-lightning numpy pandas scipy scikit-learn matplotlib
```

## 2. Repository Layout

| Path / Script | Description |
|---|---|
| `data/` | Raw wall-pressure CSVs (gitignored; must be re-added locally) |
| `data/strips/` | Spatial-strip-averaged data written by `get_strips.py` (transposed on disk: 384 features, one per row, plus a header row) |
| `data/mean_strips/` | Per-feature time-averaged, single-column pressure data written by `get_mean_strips.py`; used for training and prediction |
| `intermediate_files/` | Per-sparsity sequence arrays written by `dataset.py` (write-only artifact; never read back — see Dataset Preparation) |
| `outputs/` | Model checkpoints and data-analysis figures |
| `visual/` | Prediction plots written by `predict.py` |
| `results/` | Paper figure output written by `plot3D.py` |
| `metrics.pkl` | Prediction metrics dumped by `predict.py` |
| `global_min_max.pkl` | Normalization statistics (min/max/mean/std) written by `dataset.py` (write-only in practice; the read-back is commented out) |
| `get_strips.py` | Step 1 of the data pipeline: average spatial strips from raw CSVs |
| `get_mean_strips.py` | Step 2 of the data pipeline: time-average the strip features |
| `get_attractors.py` | Phase portraits p(t) vs p(t+dt) per case |
| `attractors.py` | Pedagogical demo of fixed-point, limit-cycle, and Lorenz strange attractors |
| `analyze.py` | Rolling-mean / moving-average analysis of the raw signal |
| `mean_analyze.py` | Per-feature percentage deviation (std/mean) plots |
| `train.py` | Transformer training (interpolation task) |
| `predict.py` | Prediction and evaluation of trained checkpoints vs dense ground truth (plots, `metrics.pkl`, LaTeX tables) |
| `dataset.py` | Dataset / dataloader construction, sparsity + spline interpolation, normalization |
| `model.py` | Model definitions (hybrid / LSTM / transformer) |
| `plot3D.py` | Paper figures (3D surfaces, heatmaps, scatter plots of MAPE) |

## 3. Dataset Preparation

### 3.1 Raw data

Place the 4 raw CSV files in `data/` (each ~105 MB, wall-pressure time series with thousands of columns per row; `data/` is gitignored):

```
data/0.09_case.csv
data/0.130_case.csv
data/0.155_case.csv
data/0.190_case.csv
```

### 3.2 Step 1 — spatial strip averaging

```bash
python get_strips.py
```

Reads the raw CSVs from `data/` via `get_dataloaders(input_dim=384, get_strips=True)`, averages the spatial strips, transposes the result, and writes `data/strips/case_{0..3}.csv`. On disk each row is one spatial feature — 384 features, one per row (385 rows including the header).

### 3.3 Step 2 — mean strips

```bash
python get_mean_strips.py
```

Reads `data/strips/case_{i}.csv`. Because the strips CSVs are stored transposed (one feature per row, time along the columns), `df.mean(1)` averages across columns (time), yielding one per-feature time-mean per row. It writes `data/mean_strips/case_{0..3}.csv` — a single column of per-feature mean pressure. Training and prediction use `data/mean_strips`.

> **Note on caches:** `get_dataloaders` writes normalization statistics to `global_min_max.pkl` and per-sparsity sequence arrays to `intermediate_files/{sparsity}/`. These are write-only artifacts: the dataset always recomputes from the CSVs on every run (the reload gate in `dataset.py` is never triggered, and the `global_min_max.pkl` read-back is commented out). Deleting them is harmless but has no effect on subsequent runs.

## 4. Training

```bash
python train.py
```

Trains a transformer model (default `--model_type transformer`) for the interpolation task (default `--task interpolation`): the model takes a sparse, cubic-spline-interpolated sequence and reconstructs the dense ground truth.

### Important: the default command trains a full sweep

The `if __name__ == "__main__"` block at the bottom of `train.py` **hardcodes** a full sweep:

```python
for case_index in [0, 1, 2, 3]:
    for sparsity in [2, 4, 8, 16, 64]:
        for seq_len in [4, 8, 16, 32]:
            train(args, args.start_dim)
```

Running `python train.py` therefore trains 4 × 5 × 4 = **80 models sequentially**. To train a single configuration, edit these loops (or comment them out), or pass CLI args — but note that the loops **override** any `--case_index` / `--sparsity` / `--seq_len` values passed on the command line.

### Key CLI flags

| Flag | Default | Description |
|---|---|---|
| `--data_dir` | `data/mean_strips` | Data directory |
| `--output_dir` | `outputs` | Output (checkpoint) directory |
| `--model_type` | `transformer` | `hybrid` \| `lstm` \| `transformer` |
| `--task` | `interpolation` | `regression` \| `classification` \| `interpolation` |
| `--input_dim` | `100` | Feature dimension per timestep |
| `--sparsity` | `8` | Sparsity level (keep every n-th timestep) |
| `--seq_len` | `16` | Sequence length |
| `--step_size` | `2` | Step size |
| `--batch_size` | `32` | Batch size |
| `--epochs` | `100` | Number of epochs |
| `--lr` | `0.01` | Learning rate |
| `--num_workers` | `32` | DataLoader workers (lower this on small machines) |
| `--case_index` | `0` | Case index (0–3) |
| `--seed` | `42` | Random seed |
| `--dropout` | `0.3` | Dropout rate |
| `--dim_feedforward` | `32` | Transformer feedforward dimension |
| `--num_layers` | `1` | Number of transformer layers |
| `--compressed_dim` | `32` | Compressed embedding dimension |
| `--hidden_dim` | `32` | Hidden dimension |

### Checkpoints

Checkpoints are saved to

```
outputs/{get_strips}/{seed}/case {case_index}/{start_dim}/sparsity {sparsity}/seq_len {seq_len}/
```

as `epoch={epoch}-val_loss={val_loss:.4f}.ckpt` plus `last.ckpt` (with `save_top_k=3` monitoring `train_loss`). For example, with defaults:

```
outputs/False/42/case 0/0/sparsity 2/seq_len 4/last.ckpt
```

> **Note:** the trainer is configured with `accelerator="cpu"` (always CPU) and `deterministic=True`. On a fresh clone there are **no** checkpoints — `outputs/` is untracked, so you must train before predicting (see [Troubleshooting](#8-troubleshooting)).

## 5. Prediction & Evaluation

```bash
python predict.py --checkpoint <path> --perturbation <value>
```

For each checkpoint, the script:

1. Loads the checkpoint and rebuilds the dataset with the same hyperparameters stored in the checkpoint.
2. Predicts the dense wall-pressure from the sparse/interpolated input.
3. Computes MSE / R2 / MAPE / MAE / Max-MAE of the **predictions vs the dense ground-truth targets**. The interpolated-baseline errors (spline sequence vs targets) are also computed and printed, but they are **not saved**.
4. Saves per-case prediction plots to `visual/False/{sparsity}/{seq_len}/predictions_{case}.png` and `predictions_{case}.with_sequences.png`, and bar charts to `visual/{case}/{sparsity}/{error}_{case}.png`.
5. Dumps the stored metrics to `metrics.pkl`.
6. Prints ready-to-paste LaTeX tables of MAE / MAPE / Max-MAE.

### Flags

| Flag | Default | Description |
|---|---|---|
| `--checkpoint` | *(stale, machine-specific path — see below)* | Path to a checkpoint |
| `--sparsity` | `2` | Sparsity level |
| `--perturbation` | `0.13` | Perturbation value (case) — accepted but not consumed by the transformer path (see note below) |
| `--device` | `cpu` | `cpu` \| `cuda` (falls back to CPU if CUDA unavailable) |

> **Warning:** the default `--checkpoint` is a stale, machine-specific path (`/Users/darylfung/Documents/Work/Nicosia/hypersonic/outputs/False/case 0/0/sparsity 2/seq_len 4/last.ckpt`). It **must be overridden** to point at a checkpoint on your machine.

> **Note on `--perturbation`:** the flag is accepted but not consumed by the transformer path — `predict.py` builds a perturbation tensor but calls `model(sequence)` without it, and `ConditionalTransformerModel.forward(x)` ignores perturbation (its encoder is commented out). The hybrid/LSTM models expose `forward(x, perturbation)` and would crash on that call, so the current predict path works only with transformer checkpoints.

### Important: grid mismatch

`predict.py` loops over cases `[0,1,2,3]` × sparsity `[2,4,8,16,32,64]` × seq_len `[4,8,16,32]`, building each checkpoint path by string-replacing `"sparsity 2"`, `"seq_len 4"`, and `"case 0"` in the given `--checkpoint`. However, `train.py`'s sweep only covers sparsity `[2,4,8,16,64]` (no `32`). If a checkpoint is missing, the script raises `FileNotFoundError` when loading it — you must either train the missing configurations or edit the loops in `predict.py` to match your trained grid.

## 6. Paper Figures

```bash
python plot3D.py
```

Reads the MAPE values hardcoded at the top of the script (from the paper) and writes, for each case, to `results/{case}/3d/`:

- `MAPE_3D_Surface.png`
- `MAPE_Heatmap.png`
- `MAPE_3D_Scatter.png`
- `MAPE_2D_Scatter.png`

Runs without any training.

## 7. Data Analysis Scripts

| Script | What it does | Output |
|---|---|---|
| `analyze.py` | Rolling-mean / moving-average of the raw signal for window sizes 5, 10, 15, 20 | `outputs/data_analysis/rolling_mean_{w}.png`, `outputs/data_analysis/moving_average_{w}.png` |
| `mean_analyze.py` | Per-feature percentage deviation (std/mean × 100) plots | `outputs/data_analysis/percentage_deviated_case_{i}.png` |
| `get_attractors.py` | Phase portraits p(t) vs p(t+dt) per case from `data/mean_strips` | Interactive `plt.show()` windows |
| `attractors.py` | Pedagogical demo of fixed-point, limit-cycle (Van der Pol), and Lorenz strange attractors using `scipy.integrate.solve_ivp` | Interactive `plt.show()` windows |

> **Note:** `analyze.py` and `mean_analyze.py` save to `outputs/data_analysis/` but do not create it — on a fresh clone (where that directory does not exist) they fail. Run `mkdir -p outputs/data_analysis` first.

## 8. Troubleshooting

- **Missing checkpoint when running `predict.py`** — grid mismatch (see [Prediction & Evaluation](#5-prediction--evaluation)): train the missing configuration or trim the loops in `predict.py` to match your trained grid.
- **Out-of-memory / slow training** — lower `--num_workers`, `--batch_size`, or `--input_dim`.
- **Training always uses CPU** — `accelerator="cpu"` is hardcoded in the trainer inside `train.py`.
- **Raw data missing** — `data/` is gitignored; the raw CSVs must be re-added locally. `intermediate_files/` and `global_min_max.pkl` are recomputed from the CSVs on every run, so no cache invalidation is needed; `data/strips/` and `data/mean_strips/` are produced only by running `get_strips.py` and `get_mean_strips.py` (§3) — re-run those steps after restoring the raw data.
- **No checkpoints on a fresh clone** — `outputs/`, `visual/`, `results/`, `intermediate_files/`, and `metrics.pkl` are untracked, so you must train first. On this machine checkpoints exist under `outputs/False/{102392,593892}/case {i}/0/...` (older seed / `start_dim` naming), under `outputs/False/case {i}/0/...` (this matches the stale default `--checkpoint` path and exists on disk), and under `outputs/True/...` — point `--checkpoint` at one of these, or at your own training output.

## 9. Citation

```bibtex
@article{Fung2026,
  author  = {Fung, D. and Christakis, N. and Kokkinakis, I. W. and Drikakis, D. and Spottswood, S. M. and Brouwer, K. R. and Riley, Z. B.},
  title   = {Uncertainty-aware artificial intelligence reconstruction of wall-pressure dynamics from sparse data in extreme flows},
  journal = {Physics of Fluids},
  volume  = {38},
  pages   = {066110},
  year    = {2026},
  doi     = {10.1063/5.0332704}
}
```

- Paper DOI: [https://doi.org/10.1063/5.0332704](https://doi.org/10.1063/5.0332704)
- AIP Scilight highlight: [Hybrid framework provides high-resolution reconstruction of hypersonic wall-pressure signals](https://www.aip.org/scilights/hybrid-framework-provides-high-resolution-reconstruction-of-hypersonic-wall-pressure-signals)
