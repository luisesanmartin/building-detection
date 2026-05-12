#!/usr/bin/env python3

'''
Run building detection inference on all test chips.

Feeds each 1024x1024 chip directly through the model (no tiling).
Outputs a georeferenced binary GeoTIFF per chip aligned to the source raster.

Output: data/predictions/{chip_id}_pred.tif
'''

import numpy as np
import torch
from pathlib import Path
from torch.utils.data import DataLoader
from utils.utils import find_test_tifs, load_model, save_prediction, TestDataset

RAW_DIR         = Path(__file__).parent / '../../data/raw'
CHECKPOINT      = Path(__file__).parent / '../../models/best.pth'
PREDICTIONS_DIR = Path(__file__).parent / '../../data/predictions'
DEVICE          = 'cuda' if torch.cuda.is_available() else 'cpu'
BATCH_SIZE      = 4
NUM_WORKERS     = 4


if __name__ == '__main__':
    all_tifs = find_test_tifs(RAW_DIR)
    pending  = [(p, c) for p, c in all_tifs if not (PREDICTIONS_DIR / f'{c}_pred.tif').exists()]
    print(f'test: {len(all_tifs)} chips total | {len(pending)} to process | device: {DEVICE}')

    model   = load_model(CHECKPOINT, DEVICE)
    dataset = TestDataset(pending)
    loader  = DataLoader(dataset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, pin_memory=True)

    done = 0
    for images, alphas, chip_ids, tif_paths in loader:
        images = images.to(DEVICE)

        with torch.no_grad():
            logits = model(images)                               # (B, 1, H, W)

        preds  = torch.sigmoid(logits).squeeze(1).cpu().numpy() # (B, H, W)
        alphas = alphas.numpy()

        for pred, alpha, chip_id, tif_path in zip(preds, alphas, chip_ids, tif_paths):
            mask = (pred > 0.5).astype(np.uint8)
            mask[alpha == 0] = 0
            save_prediction(mask, Path(tif_path), PREDICTIONS_DIR / f'{chip_id}_pred.tif')
            done += 1
            print(f'  [{done}/{len(pending)}] {chip_id} done')
