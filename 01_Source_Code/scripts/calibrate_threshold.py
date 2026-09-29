from __future__ import annotations

import argparse
import csv
import json
import sys
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

SCRIPT_DIR = Path(__file__).resolve().parent
PACKAGE_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PACKAGE_ROOT))

from src.dataset import SysuChangeDataset, load_sysu_split  # noqa: E402
from src.metrics import ConfusionAccumulator  # noqa: E402
from src.utils import load_config
from src.models import build_model  # noqa: E402
from train import evaluate  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--config", type=Path, default=PACKAGE_ROOT.parent / "02_Final_Experiment_Configs" / "runtime_config.yaml")
    parser.add_argument("--minimum", type=float, default=0.10)
    parser.add_argument("--maximum", type=float, default=0.90)
    parser.add_argument("--step", type=float, default=0.05)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    config = load_config(args.config)
    saved = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model = build_model(saved["model_name"], saved["checkpoint"], **saved["model_kwargs"])
    model.load_state_dict(saved["model"])
    model.cuda().eval()
    thresholds = np.round(np.arange(args.minimum, args.maximum + args.step / 2, args.step), 4)
    accumulators = [ConfusionAccumulator() for _ in thresholds]

    validation_source = load_sysu_split(Path(config["data"]["sysu_root"]), "validation")
    validation = SysuChangeDataset(validation_source, training=False, fixed_resolution_m=0.5)
    validation_loader = DataLoader(
        validation,
        batch_size=config["training"]["batch_size"],
        shuffle=False,
        num_workers=config["training"]["num_workers"],
        pin_memory=True,
        persistent_workers=config["training"]["num_workers"] > 0,
    )
    for batch in tqdm(validation_loader, desc="threshold calibration"):
        values = batch["pixel_values"].cuda(non_blocking=True)
        labels = batch["labels"].numpy()
        context = torch.autocast("cuda", dtype=torch.float16) if config["training"]["amp"] else nullcontext()
        with torch.inference_mode(), context:
            probability = torch.softmax(model(values).logits, dim=1)[:, 1].float().cpu().numpy()
        for threshold, accumulator in zip(thresholds, accumulators):
            accumulator.update(probability >= threshold, labels)
    rows = [
        {"threshold": float(threshold), **accumulator.compute().to_dict()}
        for threshold, accumulator in zip(thresholds, accumulators)
    ]
    best = max(rows, key=lambda item: (item["f1"], item["iou"]))

    test_source = load_sysu_split(Path(config["data"]["sysu_root"]), "test")
    test = SysuChangeDataset(test_source, training=False, fixed_resolution_m=0.5)
    test_loader = DataLoader(
        test,
        batch_size=config["training"]["batch_size"],
        shuffle=False,
        num_workers=config["training"]["num_workers"],
        pin_memory=True,
        persistent_workers=config["training"]["num_workers"] > 0,
    )
    test_metrics = evaluate(
        model,
        test_loader,
        torch.device("cuda"),
        config["training"]["amp"],
        threshold=best["threshold"],
    )
    payload = {
        "selection_split": "SYSU-CD validation",
        "selected_threshold": best["threshold"],
        "validation_metrics": best,
        "test_split": "SYSU-CD test",
        "test_metrics": test_metrics,
        "grid": rows,
    }
    output = args.output or args.checkpoint.parent / "threshold_calibration.json"
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    with output.with_suffix(".csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
