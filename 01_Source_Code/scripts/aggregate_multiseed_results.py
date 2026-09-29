from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from run_multiseed_matrix import build_specs


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = PACKAGE_ROOT.parent
METRICS = ["precision", "recall", "f1", "iou", "overall_accuracy", "kappa"]
CONFUSION = ["true_negative", "false_positive", "false_negative", "true_positive"]


def resolve_directory(spec, new_root: Path) -> Path:
    legacy = spec.legacy_directory
    if legacy and (legacy / "threshold_calibration.json").is_file():
        return legacy
    return new_root / spec.run_name


def best_validation_epoch(history_path: Path) -> tuple[int | None, float | None]:
    if not history_path.is_file():
        return None, None
    frame = pd.read_csv(history_path)
    if frame.empty or "f1" not in frame:
        return None, None
    index = frame["f1"].astype(float).idxmax()
    return int(frame.loc[index, "epoch"]), float(frame.loc[index, "f1"])


def parameter_count(checkpoint_path: Path) -> int | None:
    # Parameter count is written by training in run_config/summary only when available.
    # Loading every multi-hundred-megabyte checkpoint here would make aggregation slow.
    for name in ("summary.json", "run_config.json"):
        candidate = checkpoint_path.parent / name
        if not candidate.is_file():
            continue
        payload = json.loads(candidate.read_text(encoding="utf-8"))
        for key in ("parameters_total", "parameter_count", "parameters_trainable", "trainable_parameters", "parameters"):
            if key in payload and isinstance(payload[key], (int, float)):
                return int(payload[key])
    return None


def collect(new_root: Path) -> pd.DataFrame:
    rows: list[dict] = []
    for spec in build_specs():
        directory = resolve_directory(spec, new_root)
        result_path = directory / "threshold_calibration.json"
        if not result_path.is_file():
            continue
        payload = json.loads(result_path.read_text(encoding="utf-8"))
        epoch, history_val_f1 = best_validation_epoch(directory / "history.csv")
        row = {
            "run_name": spec.run_name,
            "family": "mask2former" if spec.model == "mask2former_cd" else spec.model,
            "variant": spec.label,
            "seed": spec.seed,
            "directory": str(directory),
            "threshold": payload["selected_threshold"],
            "best_validation_epoch": epoch,
            "history_best_validation_f1": history_val_f1,
            "parameter_count": parameter_count(directory / "best.pt"),
        }
        for split_key, prefix in (("validation_metrics", "validation"), ("test_metrics", "test")):
            values = payload[split_key]
            for metric in METRICS + CONFUSION:
                if metric in values:
                    row[f"{prefix}_{metric}"] = values[metric]
            if "samples_per_second" in values:
                row[f"{prefix}_samples_per_second"] = values["samples_per_second"]
            if "elapsed_seconds" in values:
                row[f"{prefix}_elapsed_seconds"] = values["elapsed_seconds"]
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["family", "variant", "seed"]).reset_index(drop=True)


def summarize(frame: pd.DataFrame, group_columns: list[str]) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(columns=[*group_columns, "n_seeds"])
    numeric = [
        column
        for column in frame.columns
        if column.startswith("validation_") or column.startswith("test_") or column in {"threshold", "best_validation_epoch", "parameter_count"}
    ]
    rows: list[dict] = []
    for keys, group in frame.groupby(group_columns, sort=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(group_columns, keys))
        row["n_seeds"] = len(group)
        for column in numeric:
            values = pd.to_numeric(group[column], errors="coerce").dropna()
            if values.empty:
                continue
            row[f"{column}_mean"] = values.mean()
            row[f"{column}_std"] = values.std(ddof=1) if len(values) > 1 else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def normalized_confusion_rows(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for _, row in frame.iterrows():
        tn, fp = float(row["test_true_negative"]), float(row["test_false_positive"])
        fn, tp = float(row["test_false_negative"]), float(row["test_true_positive"])
        rows.extend(
            [
                {"family": row["family"], "variant": row["variant"], "seed": row["seed"], "actual": "unchanged", "predicted": "unchanged", "proportion": tn / (tn + fp)},
                {"family": row["family"], "variant": row["variant"], "seed": row["seed"], "actual": "unchanged", "predicted": "changed", "proportion": fp / (tn + fp)},
                {"family": row["family"], "variant": row["variant"], "seed": row["seed"], "actual": "changed", "predicted": "unchanged", "proportion": fn / (fn + tp)},
                {"family": row["family"], "variant": row["variant"], "seed": row["seed"], "actual": "changed", "predicted": "changed", "proportion": tp / (fn + tp)},
            ]
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--new-root", type=Path, default=WORKSPACE_ROOT / "outputs" / "experiments_multiseed")
    parser.add_argument("--output", type=Path, default=WORKSPACE_ROOT / "outputs" / "multiseed_results")
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    frame = collect(args.new_root)
    frame.to_csv(args.output / "all_runs_per_seed.csv", index=False)
    expected = len(build_specs())
    if len(frame) != expected and not args.allow_partial:
        raise SystemExit(f"Only {len(frame)}/{expected} calibrated runs are available. Use --allow-partial for an interim table.")

    baselines = frame[frame["family"].isin(["segformer", "swin_upernet"])].copy()
    ablations = frame[frame["family"].eq("mask2former")].copy()
    baseline_summary = summarize(baselines, ["family"])
    ablation_summary = summarize(ablations, ["variant"])
    baselines.to_csv(args.output / "baseline_per_seed.csv", index=False)
    ablations.to_csv(args.output / "mask2former_ablation_per_seed.csv", index=False)
    baseline_summary.to_csv(args.output / "baseline_mean_std.csv", index=False)
    ablation_summary.to_csv(args.output / "mask2former_ablation_mean_std.csv", index=False)

    complete_ablation = ablation_summary[ablation_summary["n_seeds"].eq(3)].copy()
    selection = {
        "rule": "maximum mean validation F1 across the three predeclared seeds",
        "expected_seeds": [20260803, 20260804, 20260805],
        "available_calibrated_runs": len(frame),
        "expected_calibrated_runs": expected,
        "selected_variant": None,
    }
    if not complete_ablation.empty:
        selected = complete_ablation.sort_values(
            ["validation_f1_mean", "validation_f1_std"], ascending=[False, True]
        ).iloc[0]
        selected_variant = str(selected["variant"])
        selection["selected_variant"] = selected_variant
        selection["selected_validation_f1_mean"] = float(selected["validation_f1_mean"])
        selection["selected_validation_f1_std"] = float(selected["validation_f1_std"])
        comparison_runs = pd.concat(
            [baselines, ablations[ablations["variant"].eq(selected_variant)]], ignore_index=True
        )
        comparison_runs["comparison_label"] = comparison_runs.apply(
            lambda row: row["family"] if row["family"] != "mask2former" else f"mask2former_{selected_variant}", axis=1
        )
        comparison_summary = summarize(comparison_runs, ["comparison_label"])
        comparison_runs.to_csv(args.output / "selected_model_comparison_per_seed.csv", index=False)
        comparison_summary.to_csv(args.output / "selected_model_comparison_mean_std.csv", index=False)
        normalized = normalized_confusion_rows(comparison_runs)
        normalized.to_csv(args.output / "selected_model_confusion_normalized_per_seed.csv", index=False)
        summarize(normalized.rename(columns={"proportion": "test_proportion"}), ["family", "variant", "actual", "predicted"]).to_csv(
            args.output / "selected_model_confusion_normalized_mean_std.csv", index=False
        )

    (args.output / "selection.json").write_text(json.dumps(selection, indent=2), encoding="utf-8")
    print(json.dumps(selection, indent=2))


if __name__ == "__main__":
    main()
