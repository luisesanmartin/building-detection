#!/usr/bin/env python3

'''
Tile (TIF, mask) pairs into 1024×1024 patches for CNN training.

Patches are non-overlapping (stride = patch size). Edge patches that extend
beyond the raster boundary are zero-padded to match the test chip format.

Output: data/processed/patches/{tier}/{city}/{chip_id}.h5
Each file has datasets 'images' (N, 1024, 1024, 3) uint8 and 'masks' (N, 1024, 1024) uint8.
'''

from pathlib import Path
from utils.utils import find_tif_mask_pairs, tile_chip

RAW_DIR        = Path(__file__).parent / '../../data/raw'
LABELS_DIR     = Path(__file__).parent / '../../data/processed/rasterized_labels'
PATCHES_DIR    = Path(__file__).parent / '../../data/processed/patches'
TIERS         = ['train_tier_1', 'train_tier_2']

PATCH_SIZE = 1024
STRIDE     = PATCH_SIZE        # no overlap
MAX_NODATA_RATIO = 0.5

if __name__ == '__main__':
    for tier in TIERS:
        pairs = find_tif_mask_pairs(RAW_DIR, LABELS_DIR, tier)
        print(f'\n{tier}: {len(pairs)} rasters')

        for tif_path, mask_path, city, chip_id in pairs:
            output_path = PATCHES_DIR / tier / city / f'{chip_id}.h5'

            if output_path.exists():
                print(f'  [{city}/{chip_id}] already exists, skipping')
                continue

            print(f'  [{city}/{chip_id}] tiling...', end=' ', flush=True)
            n = tile_chip(
                tif_path, mask_path, output_path,
                patch_size=PATCH_SIZE,
                stride=STRIDE,
                max_nodata_ratio=MAX_NODATA_RATIO,
            )
            print(f'{n} patches saved')
