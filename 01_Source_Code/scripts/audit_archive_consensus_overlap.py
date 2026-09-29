from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
import shapefile
from pyproj import CRS, Transformer
from rasterio.features import geometry_mask
from rasterio.windows import bounds as window_bounds
from rasterio.windows import transform as window_transform
from shapely.geometry import box, mapping, shape
from shapely.ops import transform as transform_geometry
from shapely.ops import unary_union
from shapely.prepared import prep


def load_grouped_geometries(audit_path: Path, target_crs: CRS):
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    periods: dict[str, list] = defaultdict(list)
    segments: dict[str, list] = defaultdict(list)
    all_geometries = []
    period_names = ["Oct. 2022-Oct. 2023", "Nov. 2023-Mar. 2024", "Apr. 2024-Apr. 2025"]
    for index, record in enumerate(audit["files"]):
        source = Path(record["source_file"])
        source_crs = CRS.from_user_input(record["crs"])
        transformer = Transformer.from_crs(source_crs, target_crs, always_xy=True)
        collection = shapefile.Reader(str(source), encoding=record.get("dbf_encoding") or "utf-8")
        field_names = [field[0] for field in collection.fields[1:]]
        try:
            for shape_record in collection.iterShapeRecords():
                geometry = transform_geometry(transformer.transform, shape(shape_record.shape.__geo_interface__))
                if geometry.is_empty:
                    continue
                all_geometries.append(geometry)
                periods[period_names[index]].append(geometry)
                properties = dict(zip(field_names, shape_record.record))
                segment = str(properties.get("BD") or "Unassigned segment")
                segments[segment].append(geometry)
        finally:
            collection.close()
    groups = {"All archived polygons": unary_union(all_geometries)}
    groups.update({name: unary_union(items) for name, items in periods.items()})
    groups.update({f"Segment: {name}": unary_union(items) for name, items in segments.items()})
    return groups, audit


def summarize_group(readers, thresholds, geometry, pixel_area_m2: float):
    prepared = prep(geometry)
    footprint_pixels = 0
    valid_footprint_pixels = 0
    any_candidate_pixels = 0
    majority_pixels = 0
    consensus_pixels = 0
    for _, window in readers[0].block_windows(1):
        if not prepared.intersects(box(*window_bounds(window, readers[0].transform))):
            continue
        local_transform = window_transform(window, readers[0].transform)
        footprint = geometry_mask(
            [mapping(geometry)],
            out_shape=(int(window.height), int(window.width)),
            transform=local_transform,
            invert=True,
            all_touched=False,
        )
        if not footprint.any():
            continue
        arrays = [reader.read(1, window=window) for reader in readers]
        valid = np.logical_and.reduce([array != reader.nodata for array, reader in zip(arrays, readers)])
        votes = sum((array.astype(np.float32) / 100.0 >= threshold) for array, threshold in zip(arrays, thresholds))
        footprint_pixels += int(footprint.sum())
        valid_footprint = footprint & valid
        valid_footprint_pixels += int(valid_footprint.sum())
        any_candidate_pixels += int(np.count_nonzero(valid_footprint & (votes >= 1)))
        majority_pixels += int(np.count_nonzero(valid_footprint & (votes >= 2)))
        consensus_pixels += int(np.count_nonzero(valid_footprint & (votes == 3)))
    to_ha = pixel_area_m2 / 10000.0
    return {
        "archived_polygon_area_vector_ha": geometry.area / 10000.0,
        "archived_polygon_footprint_pixels": footprint_pixels,
        "archived_polygon_footprint_ha": footprint_pixels * to_ha,
        "valid_archived_footprint_ha": valid_footprint_pixels * to_ha,
        "any_model_candidate_overlap_ha": any_candidate_pixels * to_ha,
        "two_or_more_model_overlap_ha": majority_pixels * to_ha,
        "three_model_consensus_overlap_ha": consensus_pixels * to_ha,
        "three_model_consensus_coverage_of_valid_archived_footprint": (
            consensus_pixels / valid_footprint_pixels if valid_footprint_pixels else None
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--vector-audit", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    keys = ["segformer", "swin_upernet", "mask2former_difference_swap"]
    paths = [Path(manifest["ensemble_outputs"][key]) for key in keys]
    thresholds = [float(manifest["ensemble_thresholds"][key]) for key in keys]
    readers = [rasterio.open(path) for path in paths]
    try:
        reference = readers[0]
        for reader in readers[1:]:
            if reader.crs != reference.crs or reader.transform != reference.transform or reader.shape != reference.shape:
                raise ValueError("Ensemble rasters do not share the same grid")
        groups, vector_audit = load_grouped_geometries(args.vector_audit, reference.crs)
        pixel_area_m2 = abs(reference.transform.a * reference.transform.e)
        rows = []
        for label, geometry in groups.items():
            row = {"group": label, **summarize_group(readers, thresholds, geometry, pixel_area_m2)}
            rows.append(row)
            print(json.dumps(row, ensure_ascii=False))

        args.output_dir.mkdir(parents=True, exist_ok=True)
        frame = pd.DataFrame(rows)
        frame.to_csv(args.output_dir / "archive_consensus_overlap.csv", index=False, encoding="utf-8-sig")
        payload = {
            "scope": (
                "Spatial overlay only. Coverage is the fraction of the valid archived-polygon footprint "
                "intersecting the three-model consensus mask; it is not accuracy, recall, validation, or ground-truth agreement."
            ),
            "raster_crs": str(reference.crs),
            "pixel_area_m2": pixel_area_m2,
            "threshold_rule": manifest["family_threshold"],
            "thresholds": dict(zip(keys, thresholds)),
            "vector_sources": [item["source_file"] for item in vector_audit["files"]],
            "rows": rows,
        }
        (args.output_dir / "archive_consensus_overlap.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    finally:
        for reader in readers:
            reader.close()


if __name__ == "__main__":
    main()
