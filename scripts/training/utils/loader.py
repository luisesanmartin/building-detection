import numpy as np
import rasterio
from pathlib import Path
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2


def find_chip_pairs(raw_dir: Path, processed_dir: Path, tier: str) -> list[tuple[Path, Path]]:
    pairs = []
    tier_processed = processed_dir / tier

    if not tier_processed.exists():
        return pairs

    for city_dir in sorted(tier_processed.iterdir()):
        if not city_dir.is_dir():
            continue
        for mask_path in sorted(city_dir.glob('*_mask.tif')):
            chip_id  = mask_path.stem.replace('_mask', '')
            tif_path = raw_dir / tier / city_dir.name / chip_id / f'{chip_id}.tif'
            if tif_path.exists():
                pairs.append((tif_path, mask_path))

    return pairs


class BuildingDataset(Dataset):
    def __init__(self, pairs: list[tuple[Path, Path]], transform=None):
        self.pairs     = pairs
        self.transform = transform

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int):
        tif_path, mask_path = self.pairs[idx]

        with rasterio.open(tif_path) as src:
            rgba = src.read([1, 2, 3, 4])           # (4, H, W) uint8

        image       = np.moveaxis(rgba[:3], 0, -1)  # (H, W, 3)
        alpha       = rgba[3]                        # (H, W)
        image[alpha == 0] = 0                        # zero out nodata pixels

        with rasterio.open(mask_path) as src:
            mask = src.read(1)                       # (H, W) uint8

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
