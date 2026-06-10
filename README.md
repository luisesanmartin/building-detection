# Building Detection from Satellite Imagery

Semantic segmentation pipeline to detect buildings in aerial imagery of African cities, built on the [Open Cities AI Challenge](https://www.drivendata.org/competitions/60/building-segmentation-disaster-resilience/) dataset.

## Overview

The pipeline takes high-resolution aerial TIFs with paired GeoJSON building annotations and produces per-pixel building masks. It covers the full workflow from raw data to georeferenced predictions on unseen chips.

**Best validation IoU: 0.5741** (epoch 10, early stopping)

## Pipeline

```
data/raw/
  train_tier_1/   ← labelled chips (city/chip pairs)
  test/           ← unlabelled chips for inference

scripts/
  data_prep/      1. Rasterize GeoJSON labels → mask TIFs
                  2. Tile city rasters → 1024×1024 HDF5 patches
  training/       Train U-Net on patches, save best checkpoint
  inference/      Run model on full test chips, save prediction TIFs

data/predictions/ ← georeferenced binary masks (one per test chip)
```

## Model

- **Architecture:** U-Net with ResNet34 encoder (pretrained on ImageNet)
- **Loss:** BCE + Dice
- **Input:** 1024×1024 RGB patches, normalized to ImageNet statistics
- **Augmentation:** horizontal/vertical flips, 90° rotations, color jitter
- **Training split:** train on tier 1, validate on tier 2 (eliminates leakage between correlated patches from the same city raster)
- **Patch edges:** zero-padded to match the nodata borders present on test chips
- **Inference:** full 1024×1024 chips fed directly through the model (no tiling needed — model is fully convolutional)

## Training Curves

| Epoch | Train Loss | Train IoU | Val Loss | Val IoU |
|-------|-----------|-----------|----------|---------|
| 1     | 0.5208    | 0.6965    | 0.7612   | 0.5633  |
| 5     | 0.4262    | 0.7430    | 0.7753   | 0.5609  |
| 10    | 0.4038    | 0.7543    | 0.7302   | **0.5741** |
| 15    | 0.3889    | 0.7619    | 0.7639   | 0.5692  |
| 20    | 0.3775    | 0.7674    | 0.7395   | 0.5716  |

Training stopped at epoch 20 via early stopping (patience=10, no improvement since epoch 10). Validation is on tier_2 (different cities than tier_1), a harder generalization test than the previous raster-level split within tier_1 — these numbers aren't directly comparable to earlier runs.

## Data Examples

The dataset covers a wide range of urban density and land use across African cities:

| Sparse / industrial | Dense urban |
|---|---|
| ![sparse chip](exploration/acc-ca041a.png) | ![dense chip](exploration/ptn-abe1a3.png) |

## Predictions

Sample predictions on held-out test chips (red overlay = predicted building):

![sample predictions](results/sample_predictions.png)

## How to Run

**1. Download data**
```bash
cd scripts/data_collection
bash 1_get_test_data.sh
bash 2_get_train_data.sh
```

**2. Decompress archives**
```bash
python 3_decompress.py
```

**3. Rasterize labels**
```bash
cd ../data_prep
python 1_rasterize_labels.py
```

**4. Tile training rasters into HDF5 patches**
```bash
python 2_tile.py
```

**5. Train**
```bash
cd ../training
python train.py
```
Checkpoint saved to `models/best.pth`. Metrics saved to `results/metrics.csv`.

**6. Run inference on test set**
```bash
cd ../inference
python inference.py
```
Outputs georeferenced GeoTIFFs to `data/predictions/`.

## Requirements

```bash
conda activate building-detection
```

Key dependencies: `pytorch`, `segmentation-models-pytorch`, `rasterio`, `fiona`, `albumentations`, `h5py`.
