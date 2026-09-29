from __future__ import annotations

import argparse
import io
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from PIL import Image

SCRIPT_DIR = Path(__file__).resolve().parent
PACKAGE_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PACKAGE_ROOT))

from src.dataset import SysuChangeDataset, load_sysu_split  # noqa: E402
from src.utils import load_config
from src.models import build_model  # noqa: E402


def to_pil(value: object) -> Image.Image:
    if isinstance(value, Image.Image):
        return value
    if isinstance(value, dict) and value.get("bytes") is not None:
        return Image.open(io.BytesIO(value["bytes"]))
    if isinstance(value, dict) and value.get("path"):
        return Image.open(value["path"])
    raise TypeError(type(value))


def load_threshold(checkpoint: Path) -> float:
    path = checkpoint.parent / "threshold_calibration.json"
    if not path.exists():
        return 0.5
    return float(json.loads(path.read_text(encoding="utf-8"))["selected_threshold"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=PACKAGE_ROOT.parent / "02_Final_Experiment_Configs" / "runtime_config.yaml")
    parser.add_argument("--checkpoint", action="append", required=True, help="Label=path/to/best.pt")
    parser.add_argument("--indices", nargs="+", type=int, default=[13, 107, 811, 1776])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--error-output",
        type=Path,
        help="Optional TP/TN/FP/FN error-map figure. White=TP, black=TN, red=FP, blue=FN.",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    source = load_sysu_split(Path(config["data"]["sysu_root"]), "test")
    tensor_source = SysuChangeDataset(source, training=False, fixed_resolution_m=0.5)

    models: list[tuple[str, torch.nn.Module, float]] = []
    for specification in args.checkpoint:
        label, raw_path = specification.split("=", 1)
        path = Path(raw_path)
        saved = torch.load(path, map_location="cpu", weights_only=True)
        model = build_model(saved["model_name"], saved["checkpoint"], **saved["model_kwargs"])
        model.load_state_dict(saved["model"])
        models.append((label, model.cuda().eval(), load_threshold(path)))

    columns = 3 + len(models)
    figure, axes = plt.subplots(len(args.indices), columns, figsize=(3.05 * columns, 3.0 * len(args.indices)))
    if len(args.indices) == 1:
        axes = np.expand_dims(axes, 0)
    titles = ["Earlier image", "Later image", "Reference mask"] + [label for label, _, _ in models]
    for column, title in enumerate(titles):
        axes[0, column].set_title(title, fontsize=10, weight="bold")

    error_rows: list[tuple[int, np.ndarray, list[np.ndarray]]] = []
    for row_index, sample_index in enumerate(args.indices):
        raw = source[sample_index]
        inputs = tensor_source[sample_index]["pixel_values"].unsqueeze(0).cuda()
        panels: list[np.ndarray] = [
            np.asarray(to_pil(raw["imageA"]).convert("RGB")),
            np.asarray(to_pil(raw["imageB"]).convert("RGB")),
            np.asarray(to_pil(raw["label"]).convert("L")) > 127,
        ]
        predictions: list[np.ndarray] = []
        for _, model, threshold in models:
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
                probability = torch.softmax(model(inputs).logits, dim=1)[0, 1]
            prediction = (probability >= threshold).cpu().numpy()
            predictions.append(prediction)
            panels.append(prediction)
        error_rows.append((sample_index, panels[2].astype(bool), predictions))
        for column, panel in enumerate(panels):
            axis = axes[row_index, column]
            axis.imshow(panel, cmap="gray" if panel.ndim == 2 else None, vmin=0 if panel.ndim == 2 else None, vmax=1 if panel.ndim == 2 else None)
            axis.set_axis_off()
            if column == 0:
                axis.text(
                    0.02,
                    0.98,
                    f"Test #{sample_index}",
                    transform=axis.transAxes,
                    va="top",
                    ha="left",
                    fontsize=8,
                    color="white",
                    bbox={"facecolor": "black", "alpha": 0.55, "pad": 2, "edgecolor": "none"},
                )

    figure.tight_layout(pad=0.6)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=400, bbox_inches="tight")
    figure.savefig(args.output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)

    if args.error_output:
        error_figure, error_axes = plt.subplots(
            len(error_rows), len(models), figsize=(3.2 * len(models), 3.05 * len(error_rows))
        )
        if len(error_rows) == 1:
            error_axes = np.expand_dims(error_axes, 0)
        if len(models) == 1:
            error_axes = np.expand_dims(error_axes, 1)
        for column, (label, _, _) in enumerate(models):
            error_axes[0, column].set_title(label, fontsize=10, weight="bold")
        for row, (sample_index, reference, predictions) in enumerate(error_rows):
            for column, prediction in enumerate(predictions):
                rgb = np.zeros((*reference.shape, 3), dtype=np.float32)
                true_positive = reference & prediction
                false_positive = ~reference & prediction
                false_negative = reference & ~prediction
                rgb[true_positive] = (1.0, 1.0, 1.0)
                rgb[false_positive] = (0.92, 0.16, 0.16)
                rgb[false_negative] = (0.10, 0.35, 0.95)
                axis = error_axes[row, column]
                axis.imshow(rgb)
                axis.set_axis_off()
                if column == 0:
                    axis.text(
                        0.02,
                        0.98,
                        f"Test #{sample_index}",
                        transform=axis.transAxes,
                        va="top",
                        ha="left",
                        fontsize=8,
                        color="white",
                        bbox={"facecolor": "black", "alpha": 0.55, "pad": 2, "edgecolor": "none"},
                    )
        error_figure.text(
            0.5,
            0.005,
            "White: true positive   Black: true negative   Red: false positive   Blue: false negative",
            ha="center",
            fontsize=9,
        )
        error_figure.tight_layout(rect=(0, 0.025, 1, 1), pad=0.65)
        args.error_output.parent.mkdir(parents=True, exist_ok=True)
        error_figure.savefig(args.error_output, dpi=400, bbox_inches="tight")
        error_figure.savefig(args.error_output.with_suffix(".pdf"), bbox_inches="tight")
        plt.close(error_figure)


if __name__ == "__main__":
    main()
