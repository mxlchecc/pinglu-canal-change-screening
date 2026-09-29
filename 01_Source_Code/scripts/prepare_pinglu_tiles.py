from __future__ import annotations

import argparse
import csv
import json
import math
import random
from pathlib import Path

import numpy as np
import rasterio
import shapefile
from pyproj import Transformer
from rasterio.enums import Resampling
from rasterio.features import geometry_mask
from rasterio.vrt import WarpedVRT
from rasterio.windows import Window, bounds, transform
from shapely.geometry import box, mapping, shape
from shapely.ops import transform as transform_geometry


def origins(length: int, tile: int, stride: int) -> list[int]:
    values = list(range(0, max(1, length - tile + 1), stride))
    last = max(0, length - tile)
    if not values or values[-1] != last:
        values.append(last)
    return values


def core_offsets(values: list[int], index: int, tile: int, length: int) -> tuple[int, int]:
    current = values[index]
    absolute_start = 0 if index == 0 else round((values[index - 1] + current + tile) / 2)
    absolute_end = length if index == len(values) - 1 else round((current + values[index + 1] + tile) / 2)
    return absolute_start - current, absolute_end - current


def load_corridor(path: Path, target_crs) -> object:
    reader = shapefile.Reader(str(path))
    geometries = [shape(item.__geo_interface__) for item in reader.shapes()]
    source_wkt = path.with_suffix(".prj").read_text(encoding="utf-8", errors="replace")
    transformer = Transformer.from_crs(source_wkt, target_crs, always_xy=True)
    projected = [transform_geometry(transformer.transform, geometry) for geometry in geometries]
    result = projected[0]
    for geometry in projected[1:]:
        result = result.union(geometry)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mosaic_a", type=Path)
    parser.add_argument("mosaic_b", type=Path)
    parser.add_argument("corridor", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--tile-size", type=int, default=256)
    parser.add_argument("--overlap", type=float, default=0.20)
    parser.add_argument("--statistics-windows", type=int, default=256)
    parser.add_argument("--seed", type=int, default=20260803)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stride = round(args.tile_size * (1 - args.overlap))

    with rasterio.open(args.mosaic_a) as mosaic_a:
        corridor = load_corridor(args.corridor, mosaic_a.crs)
        rows = origins(mosaic_a.height, args.tile_size, stride)
        columns = origins(mosaic_a.width, args.tile_size, stride)
        manifest = []
        for row_index, row in enumerate(rows):
            core_top, core_bottom = core_offsets(rows, row_index, args.tile_size, mosaic_a.height)
            for column_index, column in enumerate(columns):
                core_left, core_right = core_offsets(columns, column_index, args.tile_size, mosaic_a.width)
                read_window = Window(column, row, args.tile_size, args.tile_size)
                tile_bounds = bounds(read_window, mosaic_a.transform)
                intersection = corridor.intersection(box(*tile_bounds))
                if intersection.is_empty:
                    continue
                manifest.append(
                    {
                        "tile_id": len(manifest),
                        "row": row,
                        "column": column,
                        "height": args.tile_size,
                        "width": args.tile_size,
                        "core_top": core_top,
                        "core_bottom": core_bottom,
                        "core_left": core_left,
                        "core_right": core_right,
                        "corridor_coverage_fraction": intersection.area / max(box(*tile_bounds).area, 1e-9),
                        "left": tile_bounds[0],
                        "bottom": tile_bounds[1],
                        "right": tile_bounds[2],
                        "top": tile_bounds[3],
                    }
                )

        rng = random.Random(args.seed)
        selected = rng.sample(manifest, min(args.statistics_windows, len(manifest)))
        sample_values = {"mosaic_a": [[], [], []], "mosaic_b": [[], [], []]}
        valid_counts = {"mosaic_a": 0, "mosaic_b": 0, "joint": 0}
        with rasterio.open(args.mosaic_b) as mosaic_b_source:
            with WarpedVRT(
                mosaic_b_source,
                crs=mosaic_a.crs,
                transform=mosaic_a.transform,
                width=mosaic_a.width,
                height=mosaic_a.height,
                src_nodata=0,
                nodata=0,
                resampling=Resampling.bilinear,
            ) as mosaic_b:
                for item in selected:
                    window = Window(item["column"], item["row"], item["width"], item["height"])
                    first = mosaic_a.read([3, 2, 1], window=window)
                    second = mosaic_b.read([3, 2, 1], window=window)
                    local_transform = transform(window, mosaic_a.transform)
                    inside = ~geometry_mask(
                        [mapping(corridor)],
                        out_shape=(args.tile_size, args.tile_size),
                        transform=local_transform,
                        invert=False,
                    )
                    valid_first = inside & np.any(first != 0, axis=0)
                    valid_second = inside & np.any(second != 0, axis=0)
                    joint = valid_first & valid_second
                    valid_counts["mosaic_a"] += int(valid_first.sum())
                    valid_counts["mosaic_b"] += int(valid_second.sum())
                    valid_counts["joint"] += int(joint.sum())
                    for band in range(3):
                        sample_values["mosaic_a"][band].append(first[band][joint])
                        sample_values["mosaic_b"][band].append(second[band][joint])

        stretch = {}
        for date in ("mosaic_a", "mosaic_b"):
            date_stats = []
            for band in range(3):
                values = np.concatenate(sample_values[date][band])
                low, high = np.percentile(values, (2, 98)).astype(float)
                date_stats.append({"rgb_position": band, "source_band": [3, 2, 1][band], "p02": low, "p98": high})
            stretch[date] = date_stats

        profile = {
            "crs": str(mosaic_a.crs),
            "transform": list(mosaic_a.transform),
            "width": mosaic_a.width,
            "height": mosaic_a.height,
            "resolution": list(mosaic_a.res),
            "bounds": list(mosaic_a.bounds),
            "bands": [3, 2, 1],
            "mosaic_a_source": str(args.mosaic_a.resolve()),
            "mosaic_b_source": str(args.mosaic_b.resolve()),
            "chronology_status": "unresolved; A/B are file identifiers and inference averages both input orders",
            "corridor_source": str(args.corridor.resolve()),
            "tile_size": args.tile_size,
            "stride": stride,
            "overlap_fraction": args.overlap,
            "candidate_tile_count": len(manifest),
            "statistics_window_count": len(selected),
            "valid_pixel_counts": valid_counts,
            "stretch": stretch,
        }
    with (args.output_dir / "pinglu_tile_manifest.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(manifest[0]))
        writer.writeheader()
        writer.writerows(manifest)
    (args.output_dir / "pinglu_pair_profile.json").write_text(
        json.dumps(profile, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(profile, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
