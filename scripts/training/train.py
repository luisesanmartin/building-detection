#!/usr/bin/env python3
import csv
from pathlib import Path
import torch
from torch.utils.data import DataLoader, random_split
from utils.loader import BuildingDataset, TransformSubset, get_transforms
from utils.model import build_model
from utils.utils import compute_loss, compute_iou, run_epoch

PATCHES_DIR    = Path(__file__).parent / '../../data/processed/patches'
CHECKPOINT_DIR = Path(__file__).parent / '../../models'
RESULTS_DIR    = Path(__file__).parent / '../../results'

BATCH_SIZE  = 16
NUM_WORKERS = 4
VAL_SPLIT   = 0.2
MAX_EPOCHS  = 200
PATIENCE    = 10
LR          = 1e-4
DEVICE      = 'cuda' if torch.cuda.is_available() else 'cpu'


if __name__ == '__main__':
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    results_csv = RESULTS_DIR / 'metrics.csv'

    full_ds  = BuildingDataset(PATCHES_DIR, tiers=['train_tier_1'])
    n_val    = int(len(full_ds) * VAL_SPLIT)
    n_train  = len(full_ds) - n_val
    train_subset, val_subset = random_split(full_ds, [n_train, n_val])

    train_ds = TransformSubset(train_subset, get_transforms(train=True))
    val_ds   = TransformSubset(val_subset,   get_transforms(train=False))

    print(f'Train patches: {len(train_ds):,}  |  Val patches: {len(val_ds):,}')

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=NUM_WORKERS, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

    model     = build_model().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    best_iou     = 0.0
    epochs_no_improvement = 0

    with open(results_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['epoch', 'train_loss', 'train_iou', 'val_loss', 'val_iou'])

        for epoch in range(1, MAX_EPOCHS + 1):
            train_loss, train_iou = run_epoch(model, train_loader, optimizer, DEVICE, train=True)
            val_loss,   val_iou   = run_epoch(model, val_loader,   optimizer, DEVICE, train=False)

            writer.writerow([epoch, f'{train_loss:.4f}', f'{train_iou:.4f}', f'{val_loss:.4f}', f'{val_iou:.4f}'])
            f.flush()

            print(f'Epoch {epoch:03d} | train loss {train_loss:.4f} iou {train_iou:.4f} | val loss {val_loss:.4f} iou {val_iou:.4f}')

            if val_iou > best_iou:
                best_iou = val_iou
                epochs_no_improvement = 0
                torch.save(model.state_dict(), CHECKPOINT_DIR / 'best.pth')
                print(f'  -> checkpoint saved (val iou {best_iou:.4f})')
            else:
                epochs_no_improvement += 1
                if epochs_no_improvement >= PATIENCE:
                    print(f'Early stopping: no improvement for {PATIENCE} epochs')
                    break
