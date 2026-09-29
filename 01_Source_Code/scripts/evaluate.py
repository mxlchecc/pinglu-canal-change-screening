from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader, Subset

SCRIPT_DIR = Path(__file__).resolve().parent
PACKAGE_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PACKAGE_ROOT))

from src.dataset import SysuChangeDataset, load_sysu_split  # noqa: E402
from src.utils import load_config
from src.models import build_model  # noqa: E402
from src.utils import write_json  # noqa: E402
from train import evaluate  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--config", type=Path, default=PACKAGE_ROOT.parent / "02_Final_Experiment_Configs" / "runtime_config.yaml")
    parser.add_argument("--split", choices=["validation", "test"], default="test")
    parser.add_argument("--resolutions", nargs="+", type=float, default=[0.5, 0.8, 2.0, 2.3])
    parser.add_argument("--threshold", type=float)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max-samples", type=int, help="Bounded functional check; omit for manuscript evaluation")
    args = parser.parse_args()

    config = load_config(args.config)
    saved = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model = build_model(saved["model_name"], saved["checkpoint"], **saved["model_kwargs"])
    model.load_state_dict(saved["model"])
    device = torch.device(args.device)
    model.to(device).eval()
    if args.threshold is None:
        calibration_path = args.checkpoint.parent / "threshold_calibration.json"
        if calibration_path.exists():
            threshold = float(json.loads(calibration_path.read_text(encoding="utf-8"))["selected_threshold"])
        else:
            threshold = 0.5
    else:
        threshold = args.threshold
    source = load_sysu_split(Path(config["data"]["sysu_root"]), args.split)
    rows = []
    for resolution in args.resolutions:
        dataset = SysuChangeDataset(source, training=False, fixed_resolution_m=resolution)
        if args.max_samples:
            dataset = Subset(dataset, range(min(args.max_samples, len(dataset))))
        loader = DataLoader(
            dataset,
            batch_size=config["training"]["batch_size"],
            shuffle=False,
            num_workers=config["training"]["num_workers"],
            pin_memory=True,
            persistent_workers=config["training"]["num_workers"] > 0,
        )
        metrics = evaluate(model, loader, device, config["training"]["amp"] and device.type == "cuda", threshold=threshold)
        rows.append({"resolution_m": resolution, "threshold": threshold, **metrics})
        print(json.dumps(rows[-1]))
    requested = args.output or args.checkpoint.parent / f"{args.split}_resolution_metrics.csv"
    if requested.suffix.lower() == ".json":
        csv_output = requested.with_suffix(".csv")
        json_output = requested
    else:
        csv_output = requested
        json_output = requested.with_suffix(".json")
    csv_output.parent.mkdir(parents=True, exist_ok=True)
    with csv_output.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    write_json(json_output, rows)


if __name__ == "__main__":
    main()
