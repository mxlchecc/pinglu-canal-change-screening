from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
from contextlib import nullcontext
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

SCRIPT_DIR = Path(__file__).resolve().parent
PACKAGE_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PACKAGE_ROOT))

from src.dataset import SysuChangeDataset, load_sysu_split  # noqa: E402
from src.losses import ce_dice_loss  # noqa: E402
from src.metrics import ConfusionAccumulator  # noqa: E402
from src.utils import load_config
from src.models import build_model  # noqa: E402
from src.utils import seed_everything, write_json  # noqa: E402


def build_scheduler(optimizer: AdamW, total_steps: int, warmup_ratio: float) -> LambdaLR:
    warmup = max(1, round(total_steps * warmup_ratio))

    def schedule(step: int) -> float:
        if step < warmup:
            return step / warmup
        progress = (step - warmup) / max(1, total_steps - warmup)
        return 0.5 * (1 + math.cos(math.pi * progress))

    return LambdaLR(optimizer, schedule)


@torch.inference_mode()
def evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    amp: bool,
    threshold: float = 0.5,
) -> dict:
    model.eval()
    confusion = ConfusionAccumulator()
    loss_total = 0.0
    sample_total = 0
    start = time.perf_counter()
    for batch in tqdm(loader, desc="evaluate", leave=False):
        values = batch["pixel_values"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)
        context = torch.autocast(device_type="cuda", dtype=torch.float16) if amp else nullcontext()
        with context:
            logits = model(values).logits
            loss = ce_dice_loss(logits, labels)
        predictions = torch.softmax(logits, dim=1)[:, 1] >= threshold
        confusion.update(predictions.cpu().numpy(), labels.cpu().numpy())
        loss_total += float(loss) * len(values)
        sample_total += len(values)
    elapsed = time.perf_counter() - start
    metrics = confusion.compute().to_dict()
    metrics.update(
        {
            "loss": loss_total / max(1, sample_total),
            "samples": sample_total,
            "elapsed_seconds": elapsed,
            "samples_per_second": sample_total / max(elapsed, 1e-9),
        }
    )
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=PACKAGE_ROOT.parent / "02_Final_Experiment_Configs" / "runtime_config.yaml")
    parser.add_argument("--model", choices=["segformer", "swin_upernet", "mask2former_cd"], required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--max-train-samples", type=int)
    parser.add_argument("--max-validation-samples", type=int)
    parser.add_argument("--disable-difference-fusion", action="store_true")
    parser.add_argument("--disable-resolution-adapter", action="store_true")
    parser.add_argument("--disable-temporal-consistency", action="store_true")
    parser.add_argument("--run-name")
    args = parser.parse_args()

    config = load_config(args.config)
    train_cfg = config["training"]
    model_cfg = dict(config["models"][args.model])
    checkpoint = model_cfg.pop("checkpoint")
    temporal_consistency_weight = float(model_cfg.pop("temporal_swap_consistency_weight", 0.0))
    if args.disable_temporal_consistency:
        temporal_consistency_weight = 0.0
    if args.model == "mask2former_cd":
        if args.disable_difference_fusion:
            model_cfg["use_difference_fusion"] = False
        if args.disable_resolution_adapter:
            model_cfg["use_resolution_adapter"] = False

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the recorded manuscript experiments")
    device = torch.device("cuda")
    seed_everything(args.seed)
    torch.set_float32_matmul_precision("high")

    data_root = Path(config["data"]["sysu_root"])
    train_hf = load_sysu_split(data_root, "train")
    val_hf = load_sysu_split(data_root, "validation")
    train_data = SysuChangeDataset(
        train_hf,
        training=True,
        resolution_choices_m=config["data"]["degradation_scales_m"],
    )
    val_data = SysuChangeDataset(val_hf, training=False, fixed_resolution_m=0.5)
    if args.max_train_samples:
        train_data = Subset(train_data, range(min(args.max_train_samples, len(train_data))))
    if args.max_validation_samples:
        val_data = Subset(val_data, range(min(args.max_validation_samples, len(val_data))))

    loader_kwargs = {
        "num_workers": train_cfg["num_workers"],
        "pin_memory": True,
        "persistent_workers": train_cfg["num_workers"] > 0,
    }
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        train_data,
        batch_size=train_cfg["batch_size"],
        shuffle=True,
        drop_last=True,
        generator=generator,
        **loader_kwargs,
    )
    val_loader = DataLoader(
        val_data,
        batch_size=train_cfg["batch_size"],
        shuffle=False,
        **loader_kwargs,
    )

    run_name = args.run_name or f"{args.model}_seed{args.seed}"
    output_dir = Path(config["output"]["root"]) / run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(
        output_dir / "run_config.json",
        {
            "arguments": vars(args) | {"config": str(args.config)},
            "model_checkpoint": checkpoint,
            "model_kwargs": model_cfg,
            "temporal_swap_consistency_weight": temporal_consistency_weight,
            "resolved_config": config,
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0),
        },
    )

    model = build_model(args.model, checkpoint, **model_cfg).to(device)
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    total = sum(parameter.numel() for parameter in model.parameters())
    optimizer = AdamW(model.parameters(), lr=train_cfg["learning_rate"], weight_decay=train_cfg["weight_decay"])
    epochs = args.epochs or train_cfg["epochs"]
    accumulation = train_cfg["gradient_accumulation"]
    updates_per_epoch = math.ceil(len(train_loader) / accumulation)
    scheduler = build_scheduler(optimizer, epochs * updates_per_epoch, train_cfg["warmup_ratio"])
    scaler = torch.amp.GradScaler("cuda", enabled=train_cfg["amp"])

    history_path = output_dir / "history.csv"
    fieldnames = [
        "epoch", "train_loss", "learning_rate", "epoch_seconds", "gpu_peak_mib",
        "val_loss", "precision", "recall", "f1", "iou", "overall_accuracy", "kappa",
    ]
    with history_path.open("w", newline="", encoding="utf-8-sig") as handle:
        csv.DictWriter(handle, fieldnames=fieldnames).writeheader()

    best_f1 = -1.0
    best_epoch = 0
    stale = 0
    global_step = 0
    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        running_loss = 0.0
        seen = 0
        torch.cuda.reset_peak_memory_stats()
        epoch_start = time.perf_counter()
        progress = tqdm(train_loader, desc=f"epoch {epoch}/{epochs}")
        for batch_index, batch in enumerate(progress, 1):
            values = batch["pixel_values"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=train_cfg["amp"]):
                if temporal_consistency_weight > 0:
                    swapped_values = torch.cat([values[:, 3:6], values[:, :3]], dim=1)
                    # A single batched forward pass preserves the same objective
                    # while avoiding duplicated kernel-launch overhead.
                    paired_logits = model(torch.cat([values, swapped_values], dim=0)).logits
                    logits, swapped_logits = paired_logits.chunk(2, dim=0)
                else:
                    logits = model(values).logits
                loss = ce_dice_loss(
                    logits,
                    labels,
                    change_class_weight=train_cfg["loss"]["change_class_weight"],
                    ce_weight=train_cfg["loss"]["cross_entropy_weight"],
                    dice_weight=train_cfg["loss"]["dice_weight"],
                )
                if temporal_consistency_weight > 0:
                    probability = torch.softmax(logits, dim=1)
                    swapped_probability = torch.softmax(swapped_logits, dim=1)
                    consistency = F.mse_loss(probability, swapped_probability)
                    loss = loss + temporal_consistency_weight * consistency
                scaled_loss = loss / accumulation
            scaler.scale(scaled_loss).backward()
            if batch_index % accumulation == 0 or batch_index == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scale_before = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                # GradScaler skips optimizer.step() when non-finite gradients
                # are detected. Advance the scheduler only after a real update.
                if scaler.get_scale() >= scale_before:
                    scheduler.step()
                    global_step += 1
            running_loss += float(loss.detach()) * len(values)
            seen += len(values)
            progress.set_postfix(loss=f"{running_loss / seen:.4f}")

        validation = evaluate(model, val_loader, device, train_cfg["amp"])
        row = {
            "epoch": epoch,
            "train_loss": running_loss / max(1, seen),
            "learning_rate": optimizer.param_groups[0]["lr"],
            "epoch_seconds": time.perf_counter() - epoch_start,
            "gpu_peak_mib": torch.cuda.max_memory_allocated() / 1024**2,
            "val_loss": validation["loss"],
            "precision": validation["precision"],
            "recall": validation["recall"],
            "f1": validation["f1"],
            "iou": validation["iou"],
            "overall_accuracy": validation["overall_accuracy"],
            "kappa": validation["kappa"],
        }
        with history_path.open("a", newline="", encoding="utf-8-sig") as handle:
            csv.DictWriter(handle, fieldnames=fieldnames).writerow(row)
        print(json.dumps(row))

        if validation["f1"] > best_f1 + 1e-6:
            best_f1 = validation["f1"]
            best_epoch = epoch
            stale = 0
            torch.save(
                {
                    "model": model.state_dict(),
                    "model_name": args.model,
                    "checkpoint": checkpoint,
                    "model_kwargs": model_cfg,
                    "temporal_swap_consistency_weight": temporal_consistency_weight,
                    "seed": args.seed,
                    "epoch": epoch,
                    "validation": validation,
                    "parameters_total": total,
                    "parameters_trainable": trainable,
                },
                output_dir / "best.pt",
            )
        else:
            stale += 1
        if stale >= train_cfg["early_stopping_patience"]:
            print(f"Early stopping after epoch {epoch}; best epoch was {best_epoch}.")
            break

    write_json(
        output_dir / "summary.json",
        {
            "run_name": run_name,
            "best_epoch": best_epoch,
            "best_validation_f1": best_f1,
            "global_optimizer_steps": global_step,
            "parameters_total": total,
            "parameters_trainable": trainable,
        },
    )


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
