import bisect
import h5py
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2


class BuildingDataset(Dataset):
    '''
    PyTorch Dataset that streams patches from HDF5 files produced by 2_tile.py.

    Builds a cumulative index at init (one file-open per .h5) so __getitem__
    only opens the file it needs. No file handles are held open between calls.
    '''

    def __init__(self, files: list[Path], transform=None):
        self.transform   = transform
        self._files      = []
        self._cumulative = []

        total = 0
        for h5_path in files:
            with h5py.File(h5_path, 'r') as f:
                n = f['images'].shape[0]
            if n == 0:
                continue
            self._files.append(h5_path)
            total += n
            self._cumulative.append(total)

    def __len__(self) -> int:
        return self._cumulative[-1] if self._cumulative else 0

    def __getitem__(self, idx: int):
        file_idx  = bisect.bisect_right(self._cumulative, idx)
        local_idx = idx - (self._cumulative[file_idx - 1] if file_idx > 0 else 0)

        with h5py.File(self._files[file_idx], 'r') as f:
            image = f['images'][local_idx][:]  # (H, W, 3) uint8
            mask  = f['masks'][local_idx][:]   # (H, W)    uint8

        if self.transform:
            out   = self.transform(image=image, mask=mask)
            image = out['image']
            mask  = out['mask']

        return image, mask


def get_transforms(train: bool = True):
    if train:
        return A.Compose([

            # Geometric augmentations
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),

            # Photometric augmentation
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5),

            # Normalize to ImageNet stats (pre-trained CNNs expect this)
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),

            # Convert to PyTorch tensors
            ToTensorV2(),
        ])
    return A.Compose([
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])
