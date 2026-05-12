import numpy as np
import rasterio
import segmentation_models_pytorch as smp
import torch
from pathlib import Path
from torch.utils.data import Dataset


MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def find_test_tifs(raw_dir: Path) -> list[tuple[Path, str]]:

    '''
    Return (tif_path, chip_id) for all chips in the test set.
    '''

    pairs = []
    test_dir = raw_dir / 'test'

    if not test_dir.exists():
        return pairs

    for chip_dir in sorted(test_dir.iterdir()):
        if not chip_dir.is_dir():
            continue
        chip_id  = chip_dir.name
        tif_path = chip_dir / f'{chip_id}.tif'
        if tif_path.exists():
            pairs.append((tif_path, chip_id))

    return pairs


def load_model(checkpoint_path: Path, device: str):

    '''
    Build a U-Net with ResNet34 encoder and load weights from checkpoint.
    '''

    model = smp.Unet(
        encoder_name='resnet34',
        encoder_weights=None,
        in_channels=3,
        classes=1,
        activation=None,
    )
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)
    model.eval()
    return model


class TestDataset(Dataset):

    '''Dataset that reads test TIFs and returns normalized image tensors.'''

    def __init__(self, pairs: list[tuple[Path, str]]):
        self.pairs = pairs

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int):
        tif_path, chip_id = self.pairs[idx]

        with rasterio.open(tif_path) as src:
            rgba = src.read([1, 2, 3, 4]).astype(np.float32)

        rgb   = rgba[:3]
        alpha = rgba[3]

        rgb /= 255.0
        rgb  = (rgb - MEAN[:, None, None]) / STD[:, None, None]

        return torch.from_numpy(rgb), torch.from_numpy(alpha), chip_id, str(tif_path)


def save_prediction(mask: np.ndarray, tif_path: Path, output_path: Path):

    '''
    Save binary mask as a georeferenced GeoTIFF aligned to the source chip.
    '''

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with rasterio.open(tif_path) as src:
        profile = src.profile.copy()

    profile.update(dtype='uint8', count=1, compress='lzw', nodata=None)
    profile.pop('photometric', None)

    with rasterio.open(output_path, 'w', **profile) as dst:
        dst.write(mask, 1)
