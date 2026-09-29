from __future__ import annotations

import io
import random
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
import torch.nn.functional as F
from datasets import Dataset, load_dataset
from PIL import Image, ImageEnhance
from torch.utils.data import Dataset as TorchDataset


IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(3, 1, 1)


def _to_pil(value: object) -> Image.Image:
    if isinstance(value, Image.Image):
        return value
    if isinstance(value, dict):
        if value.get("bytes") is not None:
            return Image.open(io.BytesIO(value["bytes"]))
        if value.get("path"):
            return Image.open(value["path"])
    raise TypeError(f"Unsupported image value: {type(value)!r}")


def load_sysu_split(root: Path, split: str) -> Dataset:
    aliases = {"validation": "val", "val": "val", "train": "train", "test": "test"}
    prefix = aliases[split]
    files = sorted(str(path) for path in root.glob(f"{prefix}-*.parquet"))
    if not files:
        raise FileNotFoundError(f"No {prefix} parquet shards in {root}")
    return load_dataset("parquet", data_files={split: files}, split=split)


def _pil_rgb_to_tensor(image: Image.Image) -> torch.Tensor:
    array = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(np.moveaxis(array, -1, 0).copy())


def _pil_mask_to_tensor(mask: Image.Image) -> torch.Tensor:
    array = np.asarray(mask.convert("L"), dtype=np.uint8)
    return torch.from_numpy((array > 127).astype(np.int64))


def degrade_resolution(image: torch.Tensor, source_m: float, target_m: float) -> torch.Tensor:
    if target_m <= source_m + 1e-9:
        return image
    height, width = image.shape[-2:]
    factor = source_m / target_m
    low_h = max(8, round(height * factor))
    low_w = max(8, round(width * factor))
    batch = image.unsqueeze(0)
    low = F.interpolate(batch, size=(low_h, low_w), mode="area")
    restored = F.interpolate(low, size=(height, width), mode="bicubic", align_corners=False)
    return restored.squeeze(0).clamp(0, 1)


class SysuChangeDataset(TorchDataset):
    """SYSU-CD paired-image dataset with synchronized geometric augmentation.

    The official subsets are preserved. Augmentation is applied only after a
    sample has been selected from a fixed subset.
    """

    def __init__(
        self,
        dataset: Dataset,
        training: bool,
        source_resolution_m: float = 0.5,
        resolution_choices_m: Sequence[float] = (0.5,),
        fixed_resolution_m: float | None = None,
    ) -> None:
        self.dataset = dataset
        self.training = training
        self.source_resolution_m = source_resolution_m
        self.resolution_choices_m = tuple(resolution_choices_m)
        self.fixed_resolution_m = fixed_resolution_m

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | int | float]:
        row = self.dataset[int(index)]
        image_a = _to_pil(row["imageA"]).convert("RGB")
        image_b = _to_pil(row["imageB"]).convert("RGB")
        mask = _to_pil(row["label"]).convert("L")

        if self.training:
            if random.random() < 0.5:
                image_a, image_b = image_b, image_a
            if random.random() < 0.5:
                image_a = image_a.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
                image_b = image_b.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
                mask = mask.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            if random.random() < 0.5:
                image_a = image_a.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
                image_b = image_b.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
                mask = mask.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
            turns = random.randrange(4)
            if turns:
                angle = 90 * turns
                image_a = image_a.rotate(angle)
                image_b = image_b.rotate(angle)
                mask = mask.rotate(angle)
            # Mild independent photometric perturbations model acquisition differences.
            image_a = ImageEnhance.Brightness(image_a).enhance(random.uniform(0.9, 1.1))
            image_b = ImageEnhance.Brightness(image_b).enhance(random.uniform(0.9, 1.1))
            image_a = ImageEnhance.Contrast(image_a).enhance(random.uniform(0.9, 1.1))
            image_b = ImageEnhance.Contrast(image_b).enhance(random.uniform(0.9, 1.1))

        tensor_a = _pil_rgb_to_tensor(image_a)
        tensor_b = _pil_rgb_to_tensor(image_b)
        target_resolution = (
            self.fixed_resolution_m
            if self.fixed_resolution_m is not None
            else random.choice(self.resolution_choices_m)
        )
        tensor_a = degrade_resolution(tensor_a, self.source_resolution_m, target_resolution)
        tensor_b = degrade_resolution(tensor_b, self.source_resolution_m, target_resolution)
        tensor_a = (tensor_a - IMAGENET_MEAN) / IMAGENET_STD
        tensor_b = (tensor_b - IMAGENET_MEAN) / IMAGENET_STD
        return {
            "pixel_values": torch.cat([tensor_a, tensor_b], dim=0),
            "labels": _pil_mask_to_tensor(mask),
            "index": int(index),
            "resolution_m": float(target_resolution),
        }
