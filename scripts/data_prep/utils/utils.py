import sys
import numpy as np
from pathlib import Path
from shapely.geometry import shape, box
from shapely.ops import transform as shapely_transform
from shapely.strtree import STRtree
from pyproj import Transformer
import fiona
import rasterio
from rasterio.features import rasterize
from rasterio.windows import transform as window_transform, bounds as window_bounds


def find_chip_pairs(tier_dir: Path) -> list[tuple[Path, Path, str, str]]:
    
    '''Return list of (tif_path, geojson_path, city, chip_id) for a tier directory.'''
    
    pairs = []
    
    for city_dir in sorted(tier_dir.iterdir()):
        if not city_dir.is_dir():
            continue
        for chip_dir in sorted(city_dir.iterdir()):
            if not chip_dir.is_dir() or chip_dir.name.endswith('-labels'):
                continue
            chip_id = chip_dir.name
            tif_path = chip_dir / f'{chip_id}.tif'
            geojson_path = city_dir / f'{chip_id}-labels' / f'{chip_id}.geojson'
            if tif_path.exists() and geojson_path.exists():
                pairs.append((tif_path, geojson_path, city_dir.name, chip_id))
    
    return pairs


def load_geometries(geojson_path: Path, src_crs: rasterio.crs.CRS):

    '''Load and reproject all polygon geometries from GeoJSON to the TIF's CRS.'''
    
    transformer = Transformer.from_crs('EPSG:4326', src_crs.to_wkt(), always_xy=True)

    geoms = []
    
    with fiona.open(geojson_path) as labels:
        for feature in labels:
            if feature['geometry'] is None:
                continue
            geom = shape(feature['geometry'])
            if not geom.is_valid:
                geom = geom.buffer(0)
            projected = shapely_transform(transformer.transform, geom)
            geoms.append(projected)
    
    return geoms


def rasterize_chip(tif_path: Path, geojson_path: Path, output_path: Path):
    
    '''Rasterize building polygons from a GeoJSON into a binary mask TIF aligned to the paired raster.'''
    
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with rasterio.open(tif_path) as src:
        crs = src.crs
        transform = src.transform
        height, width = src.height, src.width
        block_shapes = src.block_shapes

    geoms = load_geometries(geojson_path, crs)

    if not geoms:
        print(f'    Warning: no geometries found in {geojson_path.name}', file=sys.stderr)

    spatial_index = STRtree(geoms)

    profile = {
        'driver': 'GTiff',
        'dtype': np.uint8,
        'width': width,
        'height': height,
        'count': 1,
        'crs': crs,
        'transform': transform,
        'compress': 'lzw',
        'tiled': True,
        'blockxsize': block_shapes[0][1] if block_shapes else 512,
        'blockysize': block_shapes[0][0] if block_shapes else 512,
    }

    with rasterio.open(output_path, 'w', **profile) as dst:
        for _, window in dst.block_windows(1):
            win_transform = window_transform(window, transform)
            win_bounds = window_bounds(window, transform)
            win_box = box(*win_bounds)

            candidate_indices = spatial_index.query(win_box)
            intersecting = [
                (geoms[i].__geo_interface__, 1)
                for i in candidate_indices
                if geoms[i].intersects(win_box)
            ]

            if intersecting:
                block = rasterize(
                    intersecting,
                    out_shape=(window.height, window.width),
                    transform=win_transform,
                    fill=0,
                    dtype=np.uint8,
                )
            else:
                block = np.zeros((window.height, window.width), dtype=np.uint8)

            dst.write(block, 1, window=window)


