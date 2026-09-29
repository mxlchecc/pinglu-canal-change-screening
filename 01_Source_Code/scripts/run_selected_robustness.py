from __future__ import annotations

import sys

import argparse
import subprocess
from pathlib import Path

import pandas as pd


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = PACKAGE_ROOT.parent


def execute(command: list[str]) -> None:
    print(subprocess.list2cmdline(command), flush=True)
    subprocess.run(command, cwd=WORKSPACE_ROOT, check=True)


def mean_std(frame: pd.DataFrame, groups: list[str], metrics: list[str]) -> pd.DataFrame:
    aggregation = {metric: ["mean", "std"] for metric in metrics}
    result = frame.groupby(groups, as_index=False).agg(aggregation)
    result.columns = ["_".join(item).rstrip("_") if isinstance(item, tuple) else item for item in result.columns]
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--config", type=Path, default=PACKAGE_ROOT.parent / "02_Final_Experiment_Configs" / "runtime_config.yaml")
    parser.add_argument("--results", type=Path, default=WORKSPACE_ROOT / "outputs" / "multiseed_results")
    args = parser.parse_args()
    runs = pd.read_csv(args.results / "selected_model_comparison_per_seed.csv")
    if len(runs) != 9:
        raise RuntimeError(f"Expected 9 selected model/seed runs, found {len(runs)}")

    resolution_rows = []
    registration_rows = []
    for _, item in runs.iterrows():
        directory = Path(item["directory"])
        checkpoint = directory / "best.pt"
        resolution_output = directory / "test_resolution_metrics.csv"
        registration_output = directory / "registration_perturbation.csv"
        if not resolution_output.is_file():
            execute(
                [
                    str(args.python),
                    str(PACKAGE_ROOT / "scripts" / "evaluate.py"),
                    str(checkpoint),
                    "--config",
                    str(args.config),
                    "--split",
                    "test",
                    "--resolutions",
                    "0.5",
                    "0.8",
                    "2.0",
                    "2.3",
                    "--output",
                    str(resolution_output),
                ]
            )
        if not registration_output.is_file():
            execute(
                [
                    str(args.python),
                    str(PACKAGE_ROOT / "scripts" / "evaluate_registration_perturbation.py"),
                    str(checkpoint),
                    "--config",
                    str(args.config),
                    "--output",
                    str(registration_output),
                ]
            )
        resolution = pd.read_csv(resolution_output)
        resolution["comparison_label"] = item["comparison_label"]
        resolution["seed"] = int(item["seed"])
        resolution_rows.append(resolution)
        registration = pd.read_csv(registration_output)
        registration["comparison_label"] = item["comparison_label"]
        registration["seed"] = int(item["seed"])
        registration_rows.append(registration)

    resolution = pd.concat(resolution_rows, ignore_index=True)
    registration = pd.concat(registration_rows, ignore_index=True)
    resolution.to_csv(args.results / "selected_resolution_per_seed.csv", index=False)
    registration.to_csv(args.results / "selected_registration_per_seed.csv", index=False)
    mean_std(
        resolution,
        ["comparison_label", "resolution_m"],
        ["precision", "recall", "f1", "iou", "overall_accuracy", "kappa"],
    ).to_csv(args.results / "selected_resolution_mean_std.csv", index=False)
    mean_std(
        registration,
        ["comparison_label", "shift_y_pixels", "shift_x_pixels", "shift_magnitude_pixels"],
        ["precision", "recall", "f1", "iou", "overall_accuracy", "kappa", "f1_change_from_unshifted"],
    ).to_csv(args.results / "selected_registration_mean_std.csv", index=False)


if __name__ == "__main__":
    main()
