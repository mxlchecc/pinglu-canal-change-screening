from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


COLORS = ["#2E6F9E", "#D9893D", "#4D9974"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    sensitivity = pd.read_csv(args.analysis / "threshold_sensitivity.csv")
    payload = json.loads((args.analysis / "prediction_analysis.json").read_text(encoding="utf-8"))
    pairwise = pd.DataFrame(payload["pairwise_agreement"])
    figure, axes = plt.subplots(1, 2, figsize=(10.4, 4.1))
    for color, (label, group) in zip(COLORS, sensitivity.groupby("model", sort=False)):
        axes[0].plot(group["threshold"], group["candidate_area_ha"], marker="o", color=color, label=label)
    axes[0].set_xlabel("Probability threshold")
    axes[0].set_ylabel("Candidate-change area (ha)")
    axes[0].legend(frameon=False, fontsize=8)
    short_name = {
        "SegFormer-B0": "SegFormer",
        "Swin-Tiny-UPerNet": "Swin-UPerNet",
        "Mask2Former (Difference + swap)": "Mask2Former",
    }
    pairwise["label"] = pairwise.apply(
        lambda row: f"{short_name[row['model_a']]}\nvs\n{short_name[row['model_b']]}",
        axis=1,
    )
    bars = axes[1].bar(pairwise["label"], pairwise["candidate_iou"], color="#7866A8")
    for bar, value in zip(bars, pairwise["candidate_iou"]):
        axes[1].text(bar.get_x() + bar.get_width() / 2, value + 0.015, f"{value:.3f}", ha="center", fontsize=8)
    axes[1].set_xlabel("Model-family ensemble pair", labelpad=8)
    axes[1].set_ylabel("Candidate-mask IoU")
    axes[1].set_ylim(0, min(1.0, max(0.75, pairwise["candidate_iou"].max() + 0.12)))
    axes[1].tick_params(axis="x", labelsize=7.5, pad=3)
    for axis in axes:
        axis.grid(axis="y", color="#D8DEE6", linewidth=0.7, alpha=0.8)
        axis.set_axisbelow(True)
    figure.tight_layout(w_pad=2.2)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=400, bbox_inches="tight", facecolor="white")
    figure.savefig(args.output.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(figure)


if __name__ == "__main__":
    main()
