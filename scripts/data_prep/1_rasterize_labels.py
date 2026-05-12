#!/usr/bin/env python3

'''
Rasterize GeoJSON building labels into binary mask TIFs aligned to their paired rasters.

Output masks are saved to data/processed/{tier}/{city}/{chip_id}_mask.tif.
Pixels with a building polygon are 1; background is 0.


It processes each TIF in its native block windows so the full
raster array is never loaded at once and the memory is not overflowed.
'''

from pathlib import Path
from utils.utils import find_chip_pairs, rasterize_chip

RAW_DIR = Path(__file__).parent / '../../data/raw'
PROCESSED_DIR = Path(__file__).parent / '../../data/processed/rasterized_labels'
TIERS = ['train_tier_1', 'train_tier_2', 'test']

if __name__ == '__main__':
    for tier in TIERS:
        tier_dir = RAW_DIR / tier
        if not tier_dir.exists():
            print(f'Skipping {tier} (not found)')
            continue

        pairs = find_chip_pairs(tier_dir)
        print(f'\n{tier}: {len(pairs)} chips')

        for tif_path, geojson_path, city, chip_id in pairs:
            output_path = PROCESSED_DIR / tier / city / f'{chip_id}_mask.tif'

            if output_path.exists():
                print(f'  [{city}/{chip_id}] already exists, skipping')
                continue

            print(f'  [{city}/{chip_id}] rasterizing...', end=' ', flush=True)
            rasterize_chip(tif_path, geojson_path, output_path)
            print('done')
