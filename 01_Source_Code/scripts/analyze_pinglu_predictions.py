from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prediction", action="append", required=True, help="Label=probability.tif")
    parser.add_argument("--threshold", action="append", required=True, help="Label=value")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    predictions = {label: Path(path) for label, path in (item.split("=", 1) for item in args.prediction)}
    thresholds = {label: float(value) for label, value in (item.split("=", 1) for item in args.threshold)}
    if set(predictions) != set(thresholds):
        raise ValueError("Prediction and threshold labels must match")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    readers = {label: rasterio.open(path) for label, path in predictions.items()}
    first = next(iter(readers.values()))
    resolution_area = abs(first.transform.a * first.transform.e)
    candidate = {label: 0 for label in readers}
    valid_pixels = 0
    pairwise_counts = {
        pair: {"intersection": 0, "union": 0}
        for pair in itertools.combinations(readers, 2)
    }
    agreement_counts = np.zeros(len(readers) + 1, dtype=np.int64)
    sensitivity_thresholds = np.arange(0.30, 0.81, 0.05)
    sensitivity = {label: np.zeros(len(sensitivity_thresholds), dtype=np.int64) for label in readers}

    try:
        for _, window in first.block_windows(1):
            arrays = {label: reader.read(1, window=window) for label, reader in readers.items()}
            valid = np.logical_and.reduce([array > 0 for array in arrays.values()])
            if not valid.any():
                continue
            valid_pixels += int(valid.sum())
            masks = {}
            for label, array in arrays.items():
                probability = array.astype(np.float32) / 100.0
                masks[label] = valid & (probability >= thresholds[label])
                candidate[label] += int(masks[label].sum())
                for index, threshold in enumerate(sensitivity_thresholds):
                    sensitivity[label][index] += int(np.count_nonzero(valid & (probability >= threshold)))
            agreement = np.zeros(valid.shape, dtype=np.uint8)
            for mask in masks.values():
                agreement += mask.astype(np.uint8)
            agreement_counts += np.bincount(agreement[valid], minlength=len(readers) + 1)
            for pair, counts in pairwise_counts.items():
                left, right = masks[pair[0]], masks[pair[1]]
                counts["intersection"] += int(np.count_nonzero(left & right))
                counts["union"] += int(np.count_nonzero(left | right))
    finally:
        for reader in readers.values():
            reader.close()

    model_rows = [
        {
            "model": label,
            "selected_threshold": thresholds[label],
            "candidate_pixels": candidate[label],
            "candidate_area_ha": candidate[label] * resolution_area / 10000.0,
            "candidate_fraction_of_valid_corridor": candidate[label] / max(valid_pixels, 1),
        }
        for label in readers
    ]
    pairwise_rows = [
        {
            "model_a": pair[0],
            "model_b": pair[1],
            **counts,
            "candidate_iou": counts["intersection"] / max(counts["union"], 1),
        }
        for pair, counts in pairwise_counts.items()
    ]
    sensitivity_rows = []
    for label, counts in sensitivity.items():
        for threshold, count in zip(sensitivity_thresholds, counts):
            sensitivity_rows.append(
                {
                    "model": label,
                    "threshold": round(float(threshold), 2),
                    "candidate_pixels": int(count),
                    "candidate_area_ha": int(count) * resolution_area / 10000.0,
                }
            )

    pd.DataFrame(model_rows).to_csv(args.output_dir / "model_candidate_summary.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(pairwise_rows).to_csv(args.output_dir / "pairwise_agreement.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(sensitivity_rows).to_csv(args.output_dir / "threshold_sensitivity.csv", index=False, encoding="utf-8-sig")
    payload = {
        "valid_corridor_pixels": valid_pixels,
        "pixel_area_m2": resolution_area,
        "agreement_pixel_counts": {str(i): int(value) for i, value in enumerate(agreement_counts)},
        "agreement_area_ha": {
            str(i): float(value * resolution_area / 10000.0) for i, value in enumerate(agreement_counts)
        },
        "model_summary": model_rows,
        "pairwise_agreement": pairwise_rows,
    }
    (args.output_dir / "prediction_analysis.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
