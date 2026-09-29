from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = PACKAGE_ROOT.parent


def run(command: list[str]) -> None:
    print(subprocess.list2cmdline(command), flush=True)
    subprocess.run(command, cwd=WORKSPACE_ROOT, check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--results", type=Path, default=WORKSPACE_ROOT / "outputs" / "multiseed_results")
    parser.add_argument("--tiles", type=Path, default=WORKSPACE_ROOT / "outputs" / "pinglu_tiles")
    parser.add_argument("--output", type=Path, default=WORKSPACE_ROOT / "outputs" / "pinglu_predictions_multiseed")
    parser.add_argument("--analysis", type=Path, default=WORKSPACE_ROOT / "outputs" / "pinglu_analysis_multiseed")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    args.analysis.mkdir(parents=True, exist_ok=True)

    runs = pd.read_csv(args.results / "selected_model_comparison_per_seed.csv")
    selection = json.loads((args.results / "selection.json").read_text(encoding="utf-8"))
    selected_variant = selection["selected_variant"]
    variant_display = {
        "base": "Base",
        "difference_only": "Difference",
        "adapter_only": "Adapter",
        "swap_only": "Swap consistency",
        "difference_adapter": "Difference + adapter",
        "difference_swap": "Difference + swap",
        "adapter_swap": "Adapter + swap",
        "full": "Full",
    }
    display_labels = {
        "segformer": "SegFormer-B0",
        "swin_upernet": "Swin-Tiny-UPerNet",
        f"mask2former_{selected_variant}": f"Mask2Former ({variant_display[selected_variant]})",
    }
    profile_path = args.tiles / "pinglu_pair_profile.json"
    profile = json.loads(profile_path.read_text(encoding="utf-8"))
    manifest_path = args.tiles / "pinglu_tile_manifest.csv"
    corridor_path = Path(profile["corridor_source"])
    if len(runs) != 9:
        raise RuntimeError(f"Expected 9 model/seed runs, found {len(runs)}")

    seed_outputs: dict[str, list[Path]] = {}
    for _, item in runs.iterrows():
        label = str(item["comparison_label"])
        seed = int(item["seed"])
        checkpoint = Path(item["directory"]) / "best.pt"
        threshold = float(item["threshold"])
        output = args.output / f"{label}_seed{seed}.tif"
        if not output.is_file() or not output.with_suffix(".json").is_file():
            run(
                [
                    str(args.python),
                    str(PACKAGE_ROOT / "scripts" / "infer_pinglu.py"),
                    str(checkpoint),
                    str(manifest_path),
                    str(profile_path),
                    str(corridor_path),
                    str(output),
                    "--threshold",
                    str(threshold),
                    "--batch-size",
                    "16",
                ]
            )
        seed_outputs.setdefault(label, []).append(output)

    ensemble_outputs: dict[str, Path] = {}
    ensemble_thresholds: dict[str, float] = {}
    for label, paths in seed_outputs.items():
        output = args.output / f"{label}_ensemble.tif"
        if not output.is_file() or not output.with_suffix(".json").is_file():
            command = [
                str(args.python),
                str(PACKAGE_ROOT / "scripts" / "ensemble_pinglu_seeds.py"),
            ]
            for path in sorted(paths):
                command.extend(["--input", str(path)])
            command.extend(["--output", str(output)])
            run(command)
        payload = json.loads(output.with_suffix(".json").read_text(encoding="utf-8"))
        ensemble_outputs[label] = output
        ensemble_thresholds[label] = float(payload["ensemble_threshold"])

    command = [
        str(args.python),
        str(PACKAGE_ROOT / "scripts" / "analyze_pinglu_predictions.py"),
    ]
    for label, path in ensemble_outputs.items():
        command.extend(["--prediction", f"{display_labels[label]}={path}"])
    for label, threshold in ensemble_thresholds.items():
        command.extend(["--threshold", f"{display_labels[label]}={threshold}"])
    command.extend(["--output-dir", str(args.analysis)])
    run(command)

    visual_command = [
        str(args.python),
        str(PACKAGE_ROOT / "scripts" / "visualize_pinglu_candidates.py"),
        "--manifest",
        str(manifest_path),
        "--profile",
        str(profile_path),
    ]
    for label, path in ensemble_outputs.items():
        visual_command.extend(["--prediction", f"{display_labels[label]}={path}"])
    for label, threshold in ensemble_thresholds.items():
        visual_command.extend(["--threshold", f"{display_labels[label]}={threshold}"])
    visual_command.extend(["--output", str(WORKSPACE_ROOT / "outputs" / "figures" / "pinglu_candidates_multiseed.png")])
    run(visual_command)
    run(
        [
            str(args.python),
            str(PACKAGE_ROOT / "scripts" / "plot_pinglu_multiseed_application.py"),
            "--analysis",
            str(args.analysis),
            "--output",
            str(WORKSPACE_ROOT / "outputs" / "figures" / "pinglu_sensitivity_multiseed.png"),
        ]
    )

    manifest = {
        "model_family_aggregation": "mean of three order-symmetric seed probability rasters",
        "family_threshold": "mean of three seed-specific validation-selected thresholds",
        "seed_outputs": {key: [str(path.resolve()) for path in value] for key, value in seed_outputs.items()},
        "ensemble_outputs": {key: str(value.resolve()) for key, value in ensemble_outputs.items()},
        "ensemble_thresholds": ensemble_thresholds,
        "display_labels": display_labels,
        "analysis": str((args.analysis / "prediction_analysis.json").resolve()),
    }
    (args.output / "application_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
