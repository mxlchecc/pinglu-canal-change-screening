from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.vrt import WarpedVRT
from rasterio.windows import Window


def stretch(array: np.ndarray, statistics: list[dict]) -> np.ndarray:
    rgb = np.moveaxis(array.astype(np.float32), 0, -1)
    for channel, values in enumerate(statistics):
        low, high = float(values["p02"]), float(values["p98"])
        rgb[..., channel] = np.clip((rgb[..., channel] - low) / max(high - low, 1e-6), 0, 1)
    return rgb


def overlay(rgb: np.ndarray, mask: np.ndarray, color: tuple[float, float, float]) -> np.ndarray:
    result = rgb.copy()
    alpha = 0.62
    result[mask] = (1 - alpha) * result[mask] + alpha * np.asarray(color)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--prediction", action="append", required=True, help="Label=probability.tif")
    parser.add_argument("--threshold", action="append", required=True, help="Label=value")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rows", type=int, default=4)
    args = parser.parse_args()

    with args.manifest.open(encoding="utf-8-sig") as handle:
        manifest = list(csv.DictReader(handle))
    integer_fields = {
        "tile_id", "row", "column", "height", "width",
        "core_top", "core_bottom", "core_left", "core_right",
    }
    for item in manifest:
        for field in integer_fields:
            item[field] = int(item[field])
    profile = json.loads(args.profile.read_text(encoding="utf-8"))
    predictions = {label: Path(path) for label, path in (item.split("=", 1) for item in args.prediction)}
    thresholds = {label: float(value) for label, value in (item.split("=", 1) for item in args.threshold)}
    if set(predictions) != set(thresholds):
        raise ValueError("Prediction and threshold labels must match")

    probability_readers = {label: rasterio.open(path) for label, path in predictions.items()}
    scores = []
    try:
        for item in manifest:
            window = Window(item["column"], item["row"], item["width"], item["height"])
            agreement = np.zeros((item["height"], item["width"]), dtype=np.uint8)
            valid = np.ones_like(agreement, dtype=bool)
            for label, reader in probability_readers.items():
                encoded = reader.read(1, window=window)
                valid &= encoded > 0
                agreement += (encoded.astype(np.float32) / 100 >= thresholds[label]).astype(np.uint8)
            if not valid.any():
                continue
            unanimous = np.count_nonzero(valid & (agreement == len(probability_readers))) / valid.sum()
            majority = np.count_nonzero(valid & (agreement >= max(2, len(probability_readers) - 1))) / valid.sum()
            scores.append((unanimous + 0.25 * majority, item))

        selected = []
        for _, item in sorted(scores, key=lambda value: value[0], reverse=True):
            if all(abs(item["row"] - other["row"]) > 512 or abs(item["column"] - other["column"]) > 512 for other in selected):
                selected.append(item)
            if len(selected) >= args.rows:
                break

        columns = 3 + len(predictions)
        figure, axes = plt.subplots(len(selected), columns, figsize=(2.8 * columns, 2.75 * len(selected)))
        if len(selected) == 1:
            axes = np.expand_dims(axes, 0)
        titles = ["Mosaic A", "Mosaic B"] + list(predictions) + ["Model agreement"]
        for column, title in enumerate(titles):
            axes[0, column].set_title(title, fontsize=9.5, weight="bold")

        with rasterio.open(profile["mosaic_a_source"]) as mosaic_a, rasterio.open(profile["mosaic_b_source"]) as mosaic_b_source:
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
                for row, item in enumerate(selected):
                    window = Window(item["column"], item["row"], item["width"], item["height"])
                    rgb_a = stretch(mosaic_a.read(profile["bands"], window=window), profile["stretch"]["mosaic_a"])
                    rgb_b = stretch(mosaic_b.read(profile["bands"], window=window), profile["stretch"]["mosaic_b"])
                    panels = [rgb_a, rgb_b]
                    agreement = np.zeros((item["height"], item["width"]), dtype=np.uint8)
                    for label, reader in probability_readers.items():
                        encoded = reader.read(1, window=window)
                        mask = encoded.astype(np.float32) / 100 >= thresholds[label]
                        agreement += mask.astype(np.uint8)
                        panels.append(overlay(rgb_b, mask, (1.0, 0.15, 0.05)))
                    agreement_rgb = rgb_b.copy()
                    majority = agreement >= max(2, len(predictions) - 1)
                    unanimous = agreement == len(predictions)
                    agreement_rgb = overlay(agreement_rgb, majority, (1.0, 0.75, 0.0))
                    agreement_rgb = overlay(agreement_rgb, unanimous, (1.0, 0.0, 0.0))
                    panels.append(agreement_rgb)
                    for column, panel in enumerate(panels):
                        axes[row, column].imshow(panel)
                        axes[row, column].set_axis_off()
                    axes[row, 0].text(
                        0.02,
                        0.98,
                        f"Tile {item['tile_id']}",
                        transform=axes[row, 0].transAxes,
                        va="top",
                        color="white",
                        fontsize=8,
                        bbox={"facecolor": "black", "alpha": 0.55, "pad": 2, "edgecolor": "none"},
                    )
    finally:
        for reader in probability_readers.values():
            reader.close()

    figure.tight_layout(pad=0.5)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=400, bbox_inches="tight")
    figure.savefig(args.output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)
    args.output.with_suffix(".json").write_text(
        json.dumps({"selected_tiles": selected, "thresholds": thresholds}, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
