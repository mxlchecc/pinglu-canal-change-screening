from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from contextlib import nullcontext
from pathlib import Path

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


def shift_later(values: torch.Tensor, dy: int, dx: int) -> torch.Tensor:
    result = values.clone()
    later = torch.roll(values[:, 3:6], shifts=(dy, dx), dims=(-2, -1))
    if dy > 0:
        later[..., :dy, :] = 0
    elif dy < 0:
        later[..., dy:, :] = 0
    if dx > 0:
        later[..., :, :dx] = 0
    elif dx < 0:
        later[..., :, dx:] = 0
    result[:, 3:6] = later
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--config", type=Path, default=PACKAGE_ROOT.parent / "02_Final_Experiment_Configs" / "runtime_config.yaml")
    parser.add_argument("--threshold", type=float)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    config = load_config(args.config)
    saved = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model = build_model(saved["model_name"], saved["checkpoint"], **saved["model_kwargs"])
    model.load_state_dict(saved["model"])
    model.cuda().eval()
    if args.threshold is None:
        calibration = json.loads((args.checkpoint.parent / "threshold_calibration.json").read_text(encoding="utf-8"))
        threshold = float(calibration["selected_threshold"])
    else:
        threshold = args.threshold

    source = load_sysu_split(Path(config["data"]["sysu_root"]), "test")
    dataset = SysuChangeDataset(source, training=False, fixed_resolution_m=0.5)
    loader = DataLoader(
        dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=False,
        num_workers=config["training"]["num_workers"],
        pin_memory=True,
        persistent_workers=config["training"]["num_workers"] > 0,
    )
    shifts = [(0, 0), (0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (2, 0)]
    accumulators = {shift: ConfusionAccumulator() for shift in shifts}
    start = time.perf_counter()
    for batch in tqdm(loader, desc="registration perturbation"):
        values = batch["pixel_values"].cuda(non_blocking=True)
        labels = batch["labels"].numpy()
        for dy, dx in shifts:
            shifted = shift_later(values, dy, dx)
            context = torch.autocast("cuda", dtype=torch.float16) if config["training"]["amp"] else nullcontext()
            with torch.inference_mode(), context:
                probability = torch.softmax(model(shifted).logits, dim=1)[:, 1]
            accumulators[(dy, dx)].update((probability >= threshold).cpu().numpy(), labels)

    rows = []
    for (dy, dx), accumulator in accumulators.items():
        metrics = accumulator.compute().to_dict()
        rows.append(
            {
                "shift_y_pixels": dy,
                "shift_x_pixels": dx,
                "shift_magnitude_pixels": (dy * dy + dx * dx) ** 0.5,
                "threshold": threshold,
                **metrics,
            }
        )
    baseline_f1 = rows[0]["f1"]
    for row in rows:
        row["f1_change_from_unshifted"] = row["f1"] - baseline_f1

    output = args.output or args.checkpoint.parent / "registration_perturbation.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    payload = {
        "checkpoint": str(args.checkpoint.resolve()),
        "model": saved["model_name"],
        "selected_threshold": threshold,
        "shift_definition": "Later RGB image shifted after normalization; uncovered borders set to zero; reference mask unchanged.",
        "elapsed_seconds": time.perf_counter() - start,
        "rows": rows,
    }
    output.with_suffix(".json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
