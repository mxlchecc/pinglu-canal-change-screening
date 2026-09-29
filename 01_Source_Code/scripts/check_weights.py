from __future__ import annotations
import argparse
import csv
import gc
import sys
from pathlib import Path
import torch

root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(root / "01_Source_Code"))
from src.models import build_model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--forward", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(4)
    with (root / "09_Trained_Weights/weight_manifest.csv").open(encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
    for row in rows:
        saved = torch.load(root / row["weight_file"], map_location="cpu", weights_only=True)
        assert saved["seed"] == int(row["seed"])
        assert saved["epoch"] == int(row["epoch"])
        model = build_model(saved["model_name"], saved["checkpoint"], **saved["model_kwargs"])
        model.load_state_dict(saved["model"], strict=True)
        assert sum(p.numel() for p in model.parameters()) == int(row["parameters_total"])
        if args.forward:
            model.to(args.device).eval()
            with torch.inference_mode():
                output = model(torch.zeros(1, 6, 256, 256, device=args.device)).logits
            assert tuple(output.shape) == (1, 2, 256, 256)
            assert torch.isfinite(output).all()
            del output
        print(f"PASS {row['run_name']}", flush=True)
        del saved, model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    print(f"Verified {len(rows)} trained checkpoints.")

if __name__ == "__main__":
    main()
