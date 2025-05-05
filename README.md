# Kaggle_RNA_2025  
**Stanford RNA 3D Folding**

---

## RNA 3D Folding Transformer + EGNN

### Overview

This repository contains Jupyter notebooks that implement a hybrid **Transformer + Equivariant Graph Neural Network (EGNN)** for predicting **C1′ RNA backbone coordinates** from sequence. The workflow covers:

- Building a coordinate cache from provided CSV/parquet labels  
- Dataset & dataloader definitions, with one-hot encoding and coordinate normalization  
- Model definition (Transformer encoder layers + EGNN layers + coordinate & distogram heads)  
- Loss functions including RMS-error, TM-score proxy, spread penalty, backbone regularization, and distogram cross-entropy  
- Training & validation loop, logging a suite of metrics  
- Plotting metrics over epochs  
- Visualizing predicted vs. true C1′ trajectories (with Kabsch alignment)  
- Generating test-set predictions and writing `submission.csv`

---

## Notebook

### `final-submission_5_epochs.ipynb`

- **Preset “quick”**: small model (D=32), batch size 6, 5 epochs; trains in ~1–2 hours on one GPU.  
- Loads competition & external data, builds a normalized coordinate cache (CAP=100 Å).  
- Trains the model for 5 epochs; logs and plots six metrics:
  - Soft-TM loss (1 – TM proxy)
  - Spread penalty (std vs. target)
  - Backbone regularization (bond-length squared)
  - Distogram CE loss
  - RMSE on C1′
  - Total weighted loss
- Visualizes one validation example in 3D (red = pred, blue = true) after rigid alignment.  
- Writes scaled test predictions to `submission.csv`.

---

## Usage

Prepare your environment by running:

```bash
pip install torch torchvision pandas pyarrow tqdm matplotlib
```

Organize input data:

- `COMP_DIR` should point to the Stanford RNA 3D Folding input (sequences + labels).
- `EXT_DIR_UW` / `EXT_DIR_RIBO` point at any external datasets you wish to sample.

Build the coordinate cache (auto-runs if empty):  
In notebook cell 2, add:

```python
if not os.listdir(CACHE_DIR):
    build_cache()
```

Run end-to-end training & validation:

- Choose your preset in the very first cell (`quick`, `highacc`, or `custom`).  
- Execute through to cell 10.

Plot metrics:

- Cell 11 plots six curves (train vs. val) and final train/val loss.

Visualize a validation example:

- Cell 12 overlays predicted vs. true C1′ in 3D using Kabsch alignment.

Generate test predictions:

- Cell 14 writes `submission.csv` scaled back to Ångström.

---

## Model & Loss Design

- **Transformer encoder** captures long-range context on one-hot base embeddings plus learned positional embeddings.
- **EGNN layers** propagate inter-residue messages and update 3D coords equivariantly.
- **Coordinate head** outputs 40 conformers per residue; we select the first and scale back to Å.
- **Distogram head** encourages correct inter-atomic distance distributions.

### Loss terms

| Loss Term              | Description                                                                 |
|------------------------|-----------------------------------------------------------------------------|
| Soft-TM proxy          | A differentiable approximation to TM-score that encourages correct global topology |
| Spread penalty         | Ensures the point cloud does not collapse (maintains realistic spread)     |
| Backbone regularization| Penalizes deviation from the ideal C1′–C1′ bond length (~3.3 Å)             |
| Distogram CE           | Cross-entropy on predicted vs. true pairwise distance bins                 |
| RMSE                   | Direct point-wise root-mean-square error on C1′ positions                   |

> **Weights (e.g. W_SOFT_TM_LOSS=1.5, W_RMSE=3.0)** are tuned to balance global vs. local accuracy.

---

## Sample Results

**Quick (5 epochs):**

```
Epoch 5/5
train_rmse : 0.216 Å
val_rmse   : 1.234 Å
train_loss : 3.48
val_loss   : 7.29
```

---

## Next Steps & Improvements

- **Longer training**: run 20–50 epochs with cosine or cyclical learning-rate schedules  
- **More data**: include all external sequences (UW, RiboZoo) and sample proportionally  
- **Stronger EGNN**: increase hidden dimensions or number of layers  
- **Data augmentation**: apply random rotations or noise on input coords to regularize  
- **Multi-task**: predict additional atoms (e.g. C2′) or torsion angles  
- **Mixed precision & gradient accumulation** to enable larger batch sizes within GPU memory  

> With more compute and data, this architecture can reach sub-Å accuracy and robust global fold prediction.
