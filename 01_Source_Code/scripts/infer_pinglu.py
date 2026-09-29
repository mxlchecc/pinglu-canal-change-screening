from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import rasterio
import shapefile
import torch
from pyproj import Transformer
from rasterio.enums import Resampling
from rasterio.features import geometry_mask
from rasterio.vrt import WarpedVRT
from rasterio.windows import Window, transform
from shapely.geometry import mapping, shape
from shapely.ops import transform as transform_geometry
from tqdm import tqdm

SCRIPT_DIR = Path(__file__).resolve().parent
PACKAGE_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PACKAGE_ROOT))

from src.dataset import IMAGENET_MEAN, IMAGENET_STD  # noqa: E402
from src.models import build_model  # noqa: E402


def load_corridor(path: Path, target_crs):
    reader = shapefile.Reader(str(path))
    source_wkt = path.with_suffix(".prj").read_text(encoding="utf-8", errors="replace")
    transformer = Transformer.from_crs(source_wkt, target_crs, always_xy=True)
    geometries = [
        transform_geometry(transformer.transform, shape(item.__geo_interface__))
        for item in reader.shapes()
    ]
    result = geometries[0]
    for geometry in geometries[1:]:
        result = result.union(geometry)
    return result


def normalize(array: np.ndarray, statistics: list[dict]) -> torch.Tensor:
    array = array.astype(np.float32)
    for band, values in enumerate(statistics):
        low, high = values["p02"], values["p98"]
        array[band] = np.clip((array[band] - low) / max(high - low, 1e-6), 0, 1)
    tensor = torch.from_numpy(array)
    return (tensor - IMAGENET_MEAN) / IMAGENET_STD


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("profile", type=Path)
    parser.add_argument("corridor", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()

    pair_profile = json.loads(args.profile.read_text(encoding="utf-8"))
    with args.manifest.open(encoding="utf-8-sig") as handle:
        manifest = list(csv.DictReader(handle))
    integer_fields = {
        "tile_id", "row", "column", "height", "width",
        "core_top", "core_bottom", "core_left", "core_right",
    }
    for item in manifest:
        for field in integer_fields:
            item[field] = int(item[field])

    saved = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model = build_model(saved["model_name"], saved["checkpoint"], **saved["model_kwargs"])
    model.load_state_dict(saved["model"])
    model.cuda().eval()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    histogram = np.zeros(101, dtype=np.int64)
    candidate_pixels = 0
    corridor_pixels = 0
    with rasterio.open(pair_profile["mosaic_a_source"]) as mosaic_a:
        corridor = load_corridor(args.corridor, mosaic_a.crs)
        output_profile = mosaic_a.profile.copy()
        output_profile.update(
            driver="GTiff",
            count=1,
            dtype="uint8",
            nodata=0,
            compress="DEFLATE",
            predictor=2,
            tiled=True,
            blockxsize=256,
            blockysize=256,
            BIGTIFF="YES",
            SPARSE_OK="TRUE",
        )
        with rasterio.open(pair_profile["mosaic_b_source"]) as mosaic_b_source:
            with WarpedVRT(
                mosaic_b_source,
                crs=mosaic_a.crs,
                transform=mosaic_a.transform,
                width=mosaic_a.width,
                height=mosaic_a.height,
                src_nodata=0,
                nodata=0,
                resampling=Resampling.bilinear,
            ) as mosaic_b, rasterio.open(args.output, "w", **output_profile) as destination:
                for batch_start in tqdm(range(0, len(manifest), args.batch_size), desc="Pinglu inference"):
                    items = manifest[batch_start : batch_start + args.batch_size]
                    tensors = []
                    masks = []
                    for item in items:
                        window = Window(item["column"], item["row"], item["width"], item["height"])
                        first = mosaic_a.read(pair_profile["bands"], window=window)
                        second = mosaic_b.read(pair_profile["bands"], window=window)
                        local_transform = transform(window, mosaic_a.transform)
                        inside = ~geometry_mask(
                            [mapping(corridor)],
                            out_shape=(item["height"], item["width"]),
                            transform=local_transform,
                            invert=False,
                        )
                        valid = inside & np.any(first != 0, axis=0) & np.any(second != 0, axis=0)
                        tensors.append(
                            torch.cat(
                                [
                                    normalize(first, pair_profile["stretch"]["mosaic_a"]),
                                    normalize(second, pair_profile["stretch"]["mosaic_b"]),
                                ],
                                dim=0,
                            )
                        )
                        masks.append(valid)
                    values = torch.stack(tensors).cuda(non_blocking=True)
                    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
                        swapped_values = torch.cat([values[:, 3:6], values[:, :3]], dim=1)
                        paired_logits = model(torch.cat([values, swapped_values], dim=0)).logits
                        forward_logits, reverse_logits = paired_logits.chunk(2, dim=0)
                        probabilities = (
                            0.5
                            * (
                                torch.softmax(forward_logits, dim=1)[:, 1]
                                + torch.softmax(reverse_logits, dim=1)[:, 1]
                            )
                        ).float().cpu().numpy()

                    for item, probability, valid in zip(items, probabilities, masks):
                        top, bottom = item["core_top"], item["core_bottom"]
                        left, right = item["core_left"], item["core_right"]
                        core_probability = probability[top:bottom, left:right]
                        core_valid = valid[top:bottom, left:right]
                        encoded = np.zeros(core_probability.shape, dtype=np.uint8)
                        # Floor to the stored one-percent bin so that applying an
                        # integer threshold to the GeoTIFF never promotes a value
                        # that was below the corresponding floating-point cutoff.
                        encoded[core_valid] = np.clip(
                            np.floor(core_probability[core_valid] * 100), 1, 100
                        ).astype(np.uint8)
                        output_window = Window(
                            item["column"] + left,
                            item["row"] + top,
                            right - left,
                            bottom - top,
                        )
                        destination.write(encoded, 1, window=output_window)
                        values_uint = encoded[core_valid]
                        histogram += np.bincount(values_uint, minlength=101)
                        corridor_pixels += int(core_valid.sum())
                        candidate_pixels += int(np.count_nonzero(core_probability[core_valid] >= args.threshold))

    elapsed = time.perf_counter() - start
    resolution_x, resolution_y = pair_profile["resolution"]
    summary = {
        "checkpoint": str(args.checkpoint.resolve()),
        "model_name": saved["model_name"],
        "model_seed": saved["seed"],
        "model_epoch": saved["epoch"],
        "manifest": str(args.manifest.resolve()),
        "probability_raster": str(args.output.resolve()),
        "probability_encoding": "uint8; values 1-100 represent probabilities 0.01-1.00; 0 is outside valid corridor",
        "temporal_order_aggregation": "mean probability from Mosaic A/B and Mosaic B/A inputs",
        "threshold": args.threshold,
        "tile_count": len(manifest),
        "corridor_valid_pixels": corridor_pixels,
        "candidate_pixels": candidate_pixels,
        "candidate_area_m2": candidate_pixels * abs(resolution_x * resolution_y),
        "candidate_area_ha": candidate_pixels * abs(resolution_x * resolution_y) / 10000,
        "elapsed_seconds": elapsed,
        "tiles_per_second": len(manifest) / max(elapsed, 1e-9),
        "probability_histogram": histogram.tolist(),
    }
    args.output.with_suffix(".json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
