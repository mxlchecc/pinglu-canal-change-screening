from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = PACKAGE_ROOT.parent


@dataclass(frozen=True)
class RunSpec:
    label: str
    model: str
    seed: int
    run_name: str
    flags: tuple[str, ...] = ()
    legacy_directory: Path | None = None


VARIANTS = {
    "base": ("--disable-difference-fusion", "--disable-resolution-adapter", "--disable-temporal-consistency"),
    "difference_only": ("--disable-resolution-adapter", "--disable-temporal-consistency"),
    "adapter_only": ("--disable-difference-fusion", "--disable-temporal-consistency"),
    "swap_only": ("--disable-difference-fusion", "--disable-resolution-adapter"),
    "difference_adapter": ("--disable-temporal-consistency",),
    "difference_swap": ("--disable-resolution-adapter",),
    "adapter_swap": ("--disable-difference-fusion",),
    "full": (),
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def completed(directory: Path) -> bool:
    return all((directory / name).is_file() and (directory / name).stat().st_size > 0 for name in ("best.pt", "history.csv"))


def calibrated(directory: Path) -> bool:
    return (directory / "threshold_calibration.json").is_file()


def run_command(command: list[str], log_path: Path, env: dict[str, str]) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(f"\n[{utc_now()}] COMMAND: {subprocess.list2cmdline(command)}\n")
        handle.flush()
        process = subprocess.Popen(
            command,
            cwd=WORKSPACE_ROOT,
            stdout=handle,
            stderr=subprocess.STDOUT,
            text=True,
            env=env,
        )
        return process.wait()


def build_specs() -> list[RunSpec]:
    specs: list[RunSpec] = []
    old_root = WORKSPACE_ROOT / "outputs" / "experiments"
    for model in ("segformer", "swin_upernet"):
        for seed in (20260803, 20260804, 20260805):
            legacy = None
            specs.append(RunSpec(model, model, seed, f"{model}_seed{seed}", legacy_directory=legacy))
    # All eight factorial configurations use the common 15-epoch ceiling.
    legacy_mask: dict[str, Path] = {}
    for seed in (20260803, 20260804, 20260805):
        for label, flags in VARIANTS.items():
            run_name = f"mask2former_{label}_seed{seed}"
            specs.append(
                RunSpec(
                    label=label,
                    model="mask2former_cd",
                    seed=seed,
                    run_name=run_name,
                    flags=flags,
                    legacy_directory=legacy_mask.get(label) if seed == 20260803 else None,
                )
            )
    return specs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=PACKAGE_ROOT.parent / "02_Final_Experiment_Configs" / "runtime_config.yaml")
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--output-root", type=Path, default=WORKSPACE_ROOT / "outputs" / "experiments_multiseed")
    parser.add_argument("--log-root", type=Path, default=WORKSPACE_ROOT / "outputs" / "experiment_queue_logs")
    parser.add_argument("--status", type=Path, default=WORKSPACE_ROOT / "outputs" / "multiseed_queue_status.jsonl")
    args = parser.parse_args()
    import yaml
    sys.path.insert(0, str(PACKAGE_ROOT))
    from src.utils import load_config
    resolved = load_config(args.config)
    resolved["output"]["root"] = str(args.output_root.resolve())
    args.output_root = args.output_root.resolve()
    args.output_root = args.output_root.resolve()
    args.output_root.mkdir(parents=True, exist_ok=True)
    resolved_path = args.output_root / "runtime_config.yaml"
    resolved_path.write_text(yaml.safe_dump(resolved, sort_keys=False), encoding="utf-8")
    args.config = resolved_path

    args.output_root.mkdir(parents=True, exist_ok=True)
    args.log_root.mkdir(parents=True, exist_ok=True)
    args.status.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["PYTHONUTF8"] = "1"
    env["TOKENIZERS_PARALLELISM"] = "false"
    failures: list[str] = []

    for index, spec in enumerate(build_specs(), 1):
        directory = spec.legacy_directory if spec.legacy_directory and completed(spec.legacy_directory) else args.output_root / spec.run_name
        event = {
            "time": utc_now(),
            "index": index,
            "total": len(build_specs()),
            "run_name": spec.run_name,
            "label": spec.label,
            "model": spec.model,
            "seed": spec.seed,
            "directory": str(directory),
        }
        print(json.dumps(event), flush=True)
        with args.status.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event) + "\n")

        log_path = args.log_root / f"{spec.run_name}.log"
        if not completed(directory):
            command = [
                str(args.python),
                str(PACKAGE_ROOT / "scripts" / "train.py"),
                "--config", str(args.config),
                "--model", spec.model,
                "--seed", str(spec.seed),
                "--run-name", spec.run_name,
                *spec.flags,
            ]
            code = run_command(command, log_path, env)
            if code != 0 or not completed(directory):
                failures.append(f"{spec.run_name}: training exit {code}")
                continue

        if not calibrated(directory):
            command = [
                str(args.python),
                str(PACKAGE_ROOT / "scripts" / "calibrate_threshold.py"),
                str(directory / "best.pt"),
                "--config", str(args.config),
            ]
            code = run_command(command, log_path, env)
            if code != 0 or not calibrated(directory):
                failures.append(f"{spec.run_name}: calibration exit {code}")

        finish = event | {"time": utc_now(), "state": "complete" if spec.run_name not in failures else "failed"}
        with args.status.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(finish) + "\n")

    summary = {"time": utc_now(), "failures": failures, "completed": len(build_specs()) - len(failures), "total": len(build_specs())}
    (args.output_root / "queue_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary), flush=True)
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
