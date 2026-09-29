from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = PACKAGE_ROOT.parent
COLORS = ["#2E6F9E", "#D9893D", "#4D9974"]
VARIANT_ORDER = [
    "base",
    "difference_only",
    "adapter_only",
    "swap_only",
    "difference_adapter",
    "difference_swap",
    "adapter_swap",
    "full",
]
VARIANT_LABELS = {
    "base": "Base",
    "difference_only": "Difference",
    "adapter_only": "Adapter",
    "swap_only": "Swap consistency",
    "difference_adapter": "Difference + adapter",
    "difference_swap": "Difference + swap",
    "adapter_swap": "Adapter + swap",
    "full": "Full",
}


def save(figure: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=400, bbox_inches="tight", facecolor="white")
    figure.savefig(path.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(figure)


def model_comparison(results: Path, output: Path) -> None:
    frame = pd.read_csv(results / "selected_model_comparison_mean_std.csv")
    selection = json.loads((results / "selection.json").read_text(encoding="utf-8"))
    selected = selection["selected_variant"]
    label_map = {
        "segformer": "SegFormer-B0",
        "swin_upernet": "Swin-Tiny-UPerNet",
        f"mask2former_{selected}": f"Mask2Former ({VARIANT_LABELS[selected]})",
    }
    frame["label"] = frame["comparison_label"].map(label_map)
    metrics = [("test_f1", "F1"), ("test_iou", "IoU"), ("test_kappa", "Kappa")]
    x = np.arange(len(metrics))
    width = 0.24
    figure, axis = plt.subplots(figsize=(8.2, 4.8))
    for index, (_, row) in enumerate(frame.iterrows()):
        means = [row[f"{metric}_mean"] for metric, _ in metrics]
        errors = [row[f"{metric}_std"] for metric, _ in metrics]
        axis.bar(
            x + (index - 1) * width,
            means,
            width,
            yerr=errors,
            capsize=3,
            color=COLORS[index],
            edgecolor="white",
            linewidth=0.6,
            label=row["label"],
        )
    axis.set_ylabel("Held-out test score (mean ± SD)")
    axis.set_xticks(x, [label for _, label in metrics])
    axis.set_ylim(0.62, 0.90)
    axis.grid(axis="y", color="#D8DEE6", linewidth=0.7, alpha=0.8)
    axis.set_axisbelow(True)
    axis.legend(frameon=False, ncol=1, loc="upper right")
    figure.tight_layout()
    save(figure, output / "multiseed_model_comparison.png")


def ablation(results: Path, output: Path) -> None:
    frame = pd.read_csv(results / "mask2former_ablation_mean_std.csv").set_index("variant").loc[VARIANT_ORDER].reset_index()
    selection = json.loads((results / "selection.json").read_text(encoding="utf-8"))
    selected = selection["selected_variant"]
    y = np.arange(len(frame))
    figure, axis = plt.subplots(figsize=(8.6, 5.4))
    test_colors = ["#C35A38" if value == selected else "#4E87B5" for value in frame["variant"]]
    axis.errorbar(
        frame["validation_f1_mean"],
        y - 0.11,
        xerr=frame["validation_f1_std"],
        fmt="o",
        color="#4D9974",
        ecolor="#4D9974",
        capsize=3,
        label="Validation F1",
    )
    for index, row in frame.iterrows():
        axis.errorbar(
            row["test_f1_mean"],
            y[index] + 0.11,
            xerr=row["test_f1_std"],
            fmt="s",
            color=test_colors[index],
            ecolor=test_colors[index],
            capsize=3,
        )
    axis.scatter([], [], marker="s", color="#4E87B5", label="Held-out test F1")
    axis.scatter([], [], marker="s", color="#C35A38", label="Validation-selected configuration")
    axis.set_yticks(y, [VARIANT_LABELS[value] for value in frame["variant"]])
    axis.invert_yaxis()
    axis.set_xlabel("F1 (mean ± SD across three seeds)")
    axis.set_xlim(0.73, 0.86)
    axis.grid(axis="x", color="#D8DEE6", linewidth=0.7, alpha=0.8)
    axis.set_axisbelow(True)
    axis.legend(
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.12),
        ncol=3,
        fontsize=8.5,
        handletextpad=0.5,
        columnspacing=1.1,
    )
    figure.tight_layout(rect=(0, 0.09, 1, 1))
    save(figure, output / "mask2former_ablation_8combo.png")


def confusion_matrices(results: Path, output: Path) -> None:
    frame = pd.read_csv(results / "selected_model_confusion_normalized_per_seed.csv")
    selection = json.loads((results / "selection.json").read_text(encoding="utf-8"))
    selected = selection["selected_variant"]
    groups = [
        ("segformer", "segformer", "SegFormer-B0"),
        ("swin_upernet", "swin_upernet", "Swin-Tiny-UPerNet"),
        ("mask2former", selected, f"Mask2Former\n({VARIANT_LABELS[selected]})"),
    ]
    figure, axes = plt.subplots(1, 3, figsize=(12.2, 3.7))
    for index, (axis, (family, variant, title)) in enumerate(zip(axes, groups)):
        subset = frame[(frame["family"] == family) & (frame["variant"] == variant)]
        pivot = subset.groupby(["actual", "predicted"])["proportion"].mean().unstack()
        matrix = np.array(
            [
                [pivot.loc["unchanged", "unchanged"], pivot.loc["unchanged", "changed"]],
                [pivot.loc["changed", "unchanged"], pivot.loc["changed", "changed"]],
            ]
        )
        image = axis.imshow(matrix, vmin=0, vmax=1, cmap="Blues")
        for row in range(2):
            for column in range(2):
                value = matrix[row, column]
                axis.text(column, row, f"{value * 100:.2f}%", ha="center", va="center", color="white" if value > 0.55 else "#243040", weight="bold")
        axis.set_xticks([0, 1], ["Unchanged", "Changed"])
        axis.set_yticks([0, 1], ["Unchanged", "Changed"])
        axis.set_xlabel("Predicted class")
        if index == 0:
            axis.set_ylabel("Reference class")
        axis.tick_params(labelsize=8.5)
        axis.set_title(title, fontsize=9.2, weight="bold", pad=7)
    figure.subplots_adjust(left=0.07, right=0.90, bottom=0.17, top=0.82, wspace=0.30)
    color_axis = figure.add_axes([0.925, 0.19, 0.014, 0.61])
    figure.colorbar(image, cax=color_axis, label="Row-normalized proportion")
    save(figure, output / "selected_confusion_matrices.png")


def training_curves(results: Path, output: Path) -> None:
    runs = pd.read_csv(results / "selected_model_comparison_per_seed.csv")
    selection = json.loads((results / "selection.json").read_text(encoding="utf-8"))
    selected = selection["selected_variant"]
    label_map = {
        "segformer": "SegFormer-B0",
        "swin_upernet": "Swin-Tiny-UPerNet",
        f"mask2former_{selected}": f"Mask2Former ({VARIANT_LABELS[selected]})",
    }
    figure, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    for color, (comparison_label, group) in zip(COLORS, runs.groupby("comparison_label", sort=False)):
        histories: list[pd.DataFrame] = []
        for directory in group["directory"]:
            frame = pd.read_csv(Path(directory) / "history.csv")
            histories.append(frame[["epoch", "train_loss", "f1"]])
        combined = pd.concat(histories, keys=range(len(histories)), names=["seed_index", "row"])
        means = combined.groupby("epoch").mean(numeric_only=True)
        stds = combined.groupby("epoch").std(numeric_only=True)
        label = label_map[comparison_label]
        axes[0].plot(means.index, means["train_loss"], color=color, label=label)
        axes[0].fill_between(means.index, means["train_loss"] - stds["train_loss"].fillna(0), means["train_loss"] + stds["train_loss"].fillna(0), color=color, alpha=0.16)
        axes[1].plot(means.index, means["f1"], color=color, label=label)
        axes[1].fill_between(means.index, means["f1"] - stds["f1"].fillna(0), means["f1"] + stds["f1"].fillna(0), color=color, alpha=0.16)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Training loss")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Validation F1")
    for axis in axes:
        axis.grid(color="#D8DEE6", linewidth=0.7, alpha=0.8)
        axis.set_axisbelow(True)
    axes[1].legend(frameon=False, loc="lower right")
    figure.tight_layout()
    save(figure, output / "multiseed_training_curves.png")


def robustness(results: Path, output: Path) -> None:
    resolution = pd.read_csv(results / "selected_resolution_per_seed.csv")
    registration = pd.read_csv(results / "selected_registration_per_seed.csv")
    selection = json.loads((results / "selection.json").read_text(encoding="utf-8"))
    selected = selection["selected_variant"]
    label_map = {
        "segformer": "SegFormer-B0",
        "swin_upernet": "Swin-Tiny-UPerNet",
        f"mask2former_{selected}": f"Mask2Former ({VARIANT_LABELS[selected]})",
    }
    figure, axes = plt.subplots(1, 2, figsize=(10.5, 4.1))
    for color, (label, group) in zip(COLORS, resolution.groupby("comparison_label", sort=False)):
        stats = group.groupby("resolution_m")["f1"].agg(["mean", "std"]).reset_index()
        axes[0].errorbar(stats["resolution_m"], stats["mean"], yerr=stats["std"], color=color, marker="o", capsize=3, label=label_map[label])
    cardinal = registration[np.isclose(registration["shift_magnitude_pixels"], 0) | np.isclose(registration["shift_magnitude_pixels"], 1)].copy()
    for color, (label, group) in zip(COLORS, cardinal.groupby("comparison_label", sort=False)):
        stats = group.groupby("shift_magnitude_pixels")["f1"].agg(["mean", "std"]).reset_index()
        axes[1].errorbar(stats["shift_magnitude_pixels"], stats["mean"], yerr=stats["std"], color=color, marker="o", capsize=3, label=label_map[label])
    axes[0].set_xlabel("Simulated ground sampling distance (m)")
    axes[0].set_ylabel("SYSU-CD test F1 (mean ± SD)")
    axes[0].set_xticks([0.5, 0.8, 2.0, 2.3])
    axes[1].set_xlabel("Controlled displacement (pixels)")
    axes[1].set_ylabel("SYSU-CD test F1 (mean ± SD)")
    axes[1].set_xticks([0, 1])
    for axis in axes:
        axis.grid(color="#D8DEE6", linewidth=0.7, alpha=0.8)
        axis.set_axisbelow(True)
    axes[0].legend(frameon=False, loc="lower left")
    figure.tight_layout()
    save(figure, output / "multiseed_robustness.png")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, default=WORKSPACE_ROOT / "outputs" / "multiseed_results")
    parser.add_argument("--output", type=Path, default=WORKSPACE_ROOT / "outputs" / "figures")
    args = parser.parse_args()
    required = [
        args.results / "selected_model_comparison_mean_std.csv",
        args.results / "mask2former_ablation_mean_std.csv",
        args.results / "selected_model_confusion_normalized_per_seed.csv",
        args.results / "selected_resolution_per_seed.csv",
        args.results / "selected_registration_per_seed.csv",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise SystemExit("Missing complete multi-seed outputs: " + ", ".join(missing))
    model_comparison(args.results, args.output)
    ablation(args.results, args.output)
    confusion_matrices(args.results, args.output)
    training_curves(args.results, args.output)
    robustness(args.results, args.output)


if __name__ == "__main__":
    main()
