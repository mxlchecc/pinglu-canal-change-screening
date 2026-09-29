from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import rasterio


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", action="append", required=True, type=Path, help="Seed probability GeoTIFF; repeat for each seed")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if len(args.input) < 2:
        raise ValueError("At least two seed probability rasters are required")

    readers = [rasterio.open(path) for path in args.input]
    first = readers[0]
    for reader in readers[1:]:
        if reader.shape != first.shape or reader.transform != first.transform or reader.crs != first.crs:
            raise ValueError(f"Raster grid mismatch: {reader.name}")
    profile = first.profile.copy()
    profile.update(dtype="uint8", count=1, nodata=0, compress="DEFLATE", predictor=2, tiled=True, BIGTIFF="YES", SPARSE_OK="TRUE")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    histogram = np.zeros(101, dtype=np.int64)
    valid_pixels = 0
    try:
        with rasterio.open(args.output, "w", **profile) as destination:
            for _, window in first.block_windows(1):
                arrays = [reader.read(1, window=window) for reader in readers]
                valid = np.logical_and.reduce([array > 0 for array in arrays])
                encoded = np.zeros(arrays[0].shape, dtype=np.uint8)
                if valid.any():
                    average = np.mean([array.astype(np.float32) / 100.0 for array in arrays], axis=0)
                    encoded[valid] = np.clip(np.floor(average[valid] * 100), 1, 100).astype(np.uint8)
                    histogram += np.bincount(encoded[valid], minlength=101)
                    valid_pixels += int(valid.sum())
                destination.write(encoded, 1, window=window)
    finally:
        for reader in readers:
            reader.close()

    threshold_records = []
    for path in args.input:
        summary_path = path.with_suffix(".json")
        inference = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.is_file() else {}
        checkpoint = Path(inference.get("checkpoint", ""))
        calibration_path = checkpoint.parent / "threshold_calibration.json" if checkpoint else Path()
        threshold = None
        if checkpoint and calibration_path.is_file():
            threshold = float(json.loads(calibration_path.read_text(encoding="utf-8"))["selected_threshold"])
        threshold_records.append(
            {
                "probability_raster": str(path.resolve()),
                "checkpoint": str(checkpoint) if checkpoint else "",
                "validation_selected_threshold": threshold,
            }
        )
    thresholds = [record["validation_selected_threshold"] for record in threshold_records if record["validation_selected_threshold"] is not None]
    ensemble_threshold = float(np.mean(thresholds)) if thresholds else 0.5
    payload = {
        "method": "arithmetic mean of order-symmetric seed probability rasters",
        "threshold_rule": "mean of seed-specific validation-selected thresholds",
        "inputs": threshold_records,
        "ensemble_threshold": ensemble_threshold,
        "valid_pixels": valid_pixels,
        "probability_encoding": "uint8; values 1-100 represent averaged probabilities 0.01-1.00; 0 is outside the common valid corridor",
        "probability_histogram": histogram.tolist(),
    }
    args.output.with_suffix(".json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
