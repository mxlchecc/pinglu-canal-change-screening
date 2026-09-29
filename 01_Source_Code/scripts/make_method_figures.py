from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


def box(axis, x, y, width, height, text, color, fontsize=8.3, linewidth=1.0):
    patch = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.015,rounding_size=0.018",
        facecolor=color,
        edgecolor="#273746",
        linewidth=linewidth,
    )
    axis.add_patch(patch)
    axis.text(x + width / 2, y + height / 2, text, ha="center", va="center", fontsize=fontsize)
    return patch


def arrow(axis, start, end, color="#566573", linewidth=1.2, style="-|>"):
    axis.add_patch(FancyArrowPatch(start, end, arrowstyle=style, mutation_scale=10, color=color, linewidth=linewidth))


def workflow(output: Path) -> None:
    fig, ax = plt.subplots(figsize=(9.0, 5.2))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.text(0.16, 0.95, "Supervised benchmark", ha="center", fontsize=9.8, weight="bold", color="#1F4E79")
    ax.text(0.50, 0.95, "Controlled model comparison", ha="center", fontsize=9.8, weight="bold", color="#6A4C93")
    ax.text(0.84, 0.95, "Unlabeled corridor application", ha="center", fontsize=9.8, weight="bold", color="#6B5A2B")

    left_color, middle_color, right_color = "#DCEAF7", "#E9DFF3", "#F7EED2"
    box(ax, 0.05, 0.77, 0.24, 0.105, "SYSU-CD bi-temporal pairs\n0.5 m; release-defined patch subsets", left_color)
    box(ax, 0.05, 0.58, 0.24, 0.105, "Synchronized augmentation\n+ simulated multi-resolution inputs", left_color)
    box(ax, 0.05, 0.39, 0.24, 0.105, "Pixel-wise reference masks\nPrecision, Recall, F1, IoU, OA, Kappa", left_color)
    arrow(ax, (0.17, 0.77), (0.17, 0.685))
    arrow(ax, (0.17, 0.58), (0.17, 0.495))

    box(ax, 0.38, 0.77, 0.24, 0.105, "Shared input: RGB(t1) + RGB(t2)\nSix-channel temporal tensor", middle_color)
    box(ax, 0.38, 0.58, 0.24, 0.105, "SegFormer | Swin-UPerNet\nMask2Former factorial ablation\n(8 configurations)", middle_color, fontsize=7.1)
    box(ax, 0.38, 0.39, 0.24, 0.105, "Three prespecified seeds per configuration\nCommon data and training controls\nmean ± SD", middle_color, fontsize=7.0)
    box(ax, 0.38, 0.20, 0.24, 0.105, "Validation-only selection\nLocked held-out testing", middle_color, fontsize=7.7)
    arrow(ax, (0.50, 0.77), (0.50, 0.685))
    arrow(ax, (0.50, 0.58), (0.50, 0.495))
    arrow(ax, (0.50, 0.39), (0.50, 0.305))

    box(ax, 0.71, 0.77, 0.24, 0.105, "Two delivered Pinglu mosaics\nChronology unresolved from archive", right_color, fontsize=7.5)
    box(ax, 0.71, 0.58, 0.24, 0.105, "Common 2.3 m grid; 256 × 256 tiles\n20% contextual overlap", right_color, fontsize=7.7)
    box(ax, 0.71, 0.39, 0.24, 0.105, "Candidate-change probabilities\n(no project pixel-level reference mask)", right_color, fontsize=7.5)
    box(ax, 0.71, 0.20, 0.24, 0.105, "Order-symmetric model agreement,\nthreshold and perturbation sensitivity", right_color, fontsize=7.1)
    arrow(ax, (0.83, 0.77), (0.83, 0.685))
    arrow(ax, (0.83, 0.58), (0.83, 0.495))
    arrow(ax, (0.83, 0.39), (0.83, 0.305))

    arrow(ax, (0.29, 0.63), (0.38, 0.63), color="#1F4E79", linewidth=1.6)
    arrow(ax, (0.62, 0.63), (0.71, 0.63), color="#6A4C93", linewidth=1.6)
    ax.text(
        0.50,
        0.075,
        "Archived project vectors and multi-temporal site images are analyzed separately as engineering records, not as training labels.",
        ha="center",
        va="center",
        fontsize=8.7,
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "#F4F6F7", "edgecolor": "#85929E"},
    )
    fig.tight_layout()
    fig.savefig(output, dpi=450, bbox_inches="tight")
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def architecture(output: Path) -> None:
    fig, ax = plt.subplots(figsize=(9.0, 4.5))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    box(ax, 0.02, 0.69, 0.12, 0.15, "Time-1 RGB\nI(t1)", "#DCEAF7", fontsize=8.8)
    box(ax, 0.02, 0.24, 0.12, 0.15, "Time-2 RGB\nI(t2)", "#F7EED2", fontsize=8.8)
    box(ax, 0.19, 0.65, 0.14, 0.17, "Temporal pair\n[I(t1); I(t2)]\n6 channels", "#E8EEF3", fontsize=8.3)
    box(ax, 0.19, 0.25, 0.14, 0.17, "Absolute difference\n|I(t2) − I(t1)|\n3 channels", "#F9D6D5", fontsize=8.3)
    box(ax, 0.40, 0.45, 0.13, 0.18, "Optional difference-aware\nlearned projection\n9 → 6 channels", "#E9DFF3", fontsize=7.7)
    box(ax, 0.58, 0.45, 0.11, 0.18, "Optional residual\nresolution\nadapter", "#DDF2E4", fontsize=7.8)
    box(ax, 0.73, 0.45, 0.11, 0.18, "Swin-Tiny\nencoder +\npixel decoder", "#D9EAD3", fontsize=8.0)
    box(ax, 0.88, 0.45, 0.10, 0.18, "Masked-\nattention\ndecoder", "#FCE5CD", fontsize=8.0)
    arrow(ax, (0.14, 0.77), (0.19, 0.75))
    arrow(ax, (0.14, 0.32), (0.19, 0.72), style="-")
    arrow(ax, (0.14, 0.74), (0.19, 0.34), style="-")
    arrow(ax, (0.14, 0.32), (0.19, 0.34))
    arrow(ax, (0.33, 0.73), (0.40, 0.58))
    arrow(ax, (0.33, 0.34), (0.40, 0.50))
    arrow(ax, (0.53, 0.54), (0.58, 0.54))
    arrow(ax, (0.69, 0.54), (0.73, 0.54))
    arrow(ax, (0.84, 0.54), (0.88, 0.54))
    box(ax, 0.67, 0.16, 0.31, 0.13, "Output logits → change probability\nCE + Dice supervision", "#EEF2F3", fontsize=8.8)
    arrow(ax, (0.93, 0.45), (0.83, 0.29))
    ax.annotate(
        "Optional temporal-swap consistency: repeat with [I(t2); I(t1)] and minimize probability disagreement",
        xy=(0.66, 0.39),
        xytext=(0.51, 0.075),
        arrowprops={"arrowstyle": "-|>", "color": "#6A4C93", "linewidth": 1.2},
        fontsize=8.3,
        color="#6A4C93",
        ha="center",
    )
    ax.text(0.5, 0.94, "Mask2Former components evaluated in the factorial ablation", ha="center", fontsize=11.5, weight="bold")
    fig.tight_layout()
    fig.savefig(output, dpi=450, bbox_inches="tight")
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    workflow(args.output_dir / "workflow.png")
    architecture(args.output_dir / "improved_mask2former_architecture.png")


if __name__ == "__main__":
    main()
