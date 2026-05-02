#!/usr/bin/env python3

'''
Tile (TIF, mask) pairs into fixed-size patches for CNN training.

Patches are saved to a single HDF5 file per chip with datasets 'images' (N, H, W, 3) uint8 and 'masks' (N, H, W) uint8.
Output: data/processed/patches/{tier}/{city}/{chip_id}.h5
'''

from pathlib import Path
from utils.utils import find_tif_mask_pairs, tile_chip

RAW_DIR = Path(__file__).parent / '../../data/raw'
PROCESSED_DIR = Path(__file__).parent / '../../data/processed'
PATCHES_DIR = PROCESSED_DIR / 'patches'
TIERS = ['train_tier_1', 'train_tier_2', 'test']

PATCH_SIZE = 512
STRIDE = PATCH_SIZE // 2
MAX_NODATA_RATIO = 0.5
MIN_BUILDING_RATIO = 0.0

if __name__ == '__main__':
    for tier in TIERS:
        pairs = find_tif_mask_pairs(RAW_DIR, PROCESSED_DIR, tier)
        print(f'\n{tier}: {len(pairs)} chips')

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
                min_building_ratio=MIN_BUILDING_RATIO,
            )
            print(f'{n} patches saved')
