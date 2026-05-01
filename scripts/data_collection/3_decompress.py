#!/usr/bin/env python3
import subprocess
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
DEST = str(SCRIPT_DIR / '../../data/raw')
ARCHIVES = [
    str(SCRIPT_DIR / '../../data/raw/train_tier_1.tgz'),
    str(SCRIPT_DIR / '../../data/raw/train_tier_2.tgz'),
    str(SCRIPT_DIR / '../../data/raw/test.tgz'),
]

if __name__ == '__main__':
    for archive in ARCHIVES:
        print(f'Extracting {archive}...')
        subprocess.run(['tar', '-xvf', archive, '-C', DEST], check=True)
        print(f'\tDone: {archive}')
