from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    Mask2FormerConfig,
    SegformerConfig,
    UperNetConfig,
    Mask2FormerForUniversalSegmentation,
    SegformerForSemanticSegmentation,
    UperNetForSemanticSegmentation,
)


def _replace_first_rgb_conv(module: nn.Module, input_channels: int = 6) -> str:
    """Replace the first 3-channel convolution and duplicate its RGB weights.

    Each temporal branch receives half of the original pretrained response, so
    the activation scale is preserved at initialization.
    """

    for name, child in module.named_modules():
        if isinstance(child, nn.Conv2d) and child.in_channels == 3:
            parent = module
            parts = name.split(".")
            for part in parts[:-1]:
                parent = getattr(parent, part)
            old = getattr(parent, parts[-1])
            new = nn.Conv2d(
                input_channels,
                old.out_channels,
                kernel_size=old.kernel_size,
                stride=old.stride,
                padding=old.padding,
                dilation=old.dilation,
                groups=old.groups,
                bias=old.bias is not None,
                padding_mode=old.padding_mode,
            )
            with torch.no_grad():
                if input_channels == 6:
                    new.weight[:, :3].copy_(old.weight / 2)
                    new.weight[:, 3:].copy_(old.weight / 2)
                else:
                    repeats = (input_channels + 2) // 3
                    expanded = old.weight.repeat(1, repeats, 1, 1)[:, :input_channels]
                    new.weight.copy_(expanded * (3.0 / input_channels))
                if old.bias is not None:
                    new.bias.copy_(old.bias)
            setattr(parent, parts[-1], new)
            return name
    raise RuntimeError("No 3-channel convolution found")


class DifferenceFusion(nn.Module):
    """Fuse the two dates with an explicit absolute-difference pathway."""

    def __init__(self) -> None:
        super().__init__()
        self.projection = nn.Conv2d(9, 6, kernel_size=1, bias=True)
        with torch.no_grad():
            self.projection.weight.zero_()
            self.projection.bias.zero_()
            for channel in range(6):
                self.projection.weight[channel, channel, 0, 0] = 1.0
            # A small symmetric difference contribution avoids changing the
            # pretrained activation scale abruptly.
            for channel in range(3):
                self.projection.weight[channel, 6 + channel, 0, 0] = 0.05
                self.projection.weight[3 + channel, 6 + channel, 0, 0] = 0.05

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        first, second = values[:, :3], values[:, 3:6]
        difference = torch.abs(first - second)
        return self.projection(torch.cat([first, second, difference], dim=1))


class ResolutionAdapter(nn.Module):
    """Learn a residual, anti-alias-aware correction before the backbone."""

    def __init__(self, channels: int = 6) -> None:
        super().__init__()
        self.depthwise = nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False)
        self.pointwise = nn.Conv2d(channels, channels, 1, bias=True)
        self.gate = nn.Parameter(torch.full((1, channels, 1, 1), -2.0))
        nn.init.constant_(self.depthwise.weight, 1 / 9)
        nn.init.zeros_(self.pointwise.weight)
        nn.init.zeros_(self.pointwise.bias)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        correction = self.pointwise(self.depthwise(values))
        return values + torch.sigmoid(self.gate) * correction


@dataclass
class ModelOutput:
    logits: torch.Tensor
    raw: object


class ChangeModel(nn.Module):
    def __init__(self, use_difference_fusion: bool, use_resolution_adapter: bool) -> None:
        super().__init__()
        self.difference_fusion = DifferenceFusion() if use_difference_fusion else nn.Identity()
        self.resolution_adapter = ResolutionAdapter() if use_resolution_adapter else nn.Identity()

    def prepare_input(self, values: torch.Tensor) -> torch.Tensor:
        return self.resolution_adapter(self.difference_fusion(values))


class SegFormerCD(ChangeModel):
    def __init__(self, checkpoint: str, revision: str | None = None, network_config: dict | None = None) -> None:
        super().__init__(False, False)
        if network_config is None:
            self.network = SegformerForSemanticSegmentation.from_pretrained(
                checkpoint, revision=revision, num_labels=2, ignore_mismatched_sizes=True
            )
        else:
            # Released trained tensors supply all parameters; no network access is needed.
            self.network = SegformerForSemanticSegmentation(SegformerConfig.from_dict(network_config))
        self.input_conv_name = _replace_first_rgb_conv(self.network, 6)

    def forward(self, pixel_values: torch.Tensor) -> ModelOutput:
        output = self.network(pixel_values=self.prepare_input(pixel_values))
        logits = F.interpolate(output.logits, size=pixel_values.shape[-2:], mode="bilinear", align_corners=False)
        return ModelOutput(logits, output)


class SwinUPerNetCD(ChangeModel):
    def __init__(self, checkpoint: str, revision: str | None = None, network_config: dict | None = None) -> None:
        super().__init__(False, False)
        if network_config is None:
            self.network = UperNetForSemanticSegmentation.from_pretrained(
                checkpoint, revision=revision, num_labels=2, ignore_mismatched_sizes=True
            )
        else:
            # Released trained tensors supply all parameters; no network access is needed.
            self.network = UperNetForSemanticSegmentation(UperNetConfig.from_dict(network_config))
        self.input_conv_name = _replace_first_rgb_conv(self.network, 6)

    def forward(self, pixel_values: torch.Tensor) -> ModelOutput:
        output = self.network(pixel_values=self.prepare_input(pixel_values))
        logits = F.interpolate(output.logits, size=pixel_values.shape[-2:], mode="bilinear", align_corners=False)
        return ModelOutput(logits, output)


class Mask2FormerCD(ChangeModel):
    def __init__(
        self,
        checkpoint: str,
        use_difference_fusion: bool = True,
        use_resolution_adapter: bool = True,
        revision: str | None = None,
        network_config: dict | None = None,
    ) -> None:
        super().__init__(use_difference_fusion, use_resolution_adapter)
        if network_config is None:
            self.network = Mask2FormerForUniversalSegmentation.from_pretrained(
                checkpoint, revision=revision, num_labels=2, ignore_mismatched_sizes=True
            )
        else:
            # Released trained tensors supply all parameters; no network access is needed.
            self.network = Mask2FormerForUniversalSegmentation(Mask2FormerConfig.from_dict(network_config))
        self.input_conv_name = _replace_first_rgb_conv(self.network, 6)

    def forward(self, pixel_values: torch.Tensor) -> ModelOutput:
        output = self.network(pixel_values=self.prepare_input(pixel_values))
        class_probability = torch.softmax(output.class_queries_logits, dim=-1)[..., :-1]
        mask_probability = torch.sigmoid(output.masks_queries_logits)
        semantic_probability = torch.einsum("bqc,bqhw->bchw", class_probability, mask_probability)
        semantic_probability = semantic_probability / semantic_probability.sum(dim=1, keepdim=True).clamp_min(1e-6)
        logits = torch.log(semantic_probability.clamp_min(1e-6))
        logits = F.interpolate(logits, size=pixel_values.shape[-2:], mode="bilinear", align_corners=False)
        return ModelOutput(logits, output)


def build_model(name: str, checkpoint: str, **kwargs: object) -> ChangeModel:
    if name == "segformer":
        return SegFormerCD(checkpoint, **kwargs)
    if name == "swin_upernet":
        return SwinUPerNetCD(checkpoint, **kwargs)
    if name == "mask2former_cd":
        return Mask2FormerCD(checkpoint, **kwargs)
    raise ValueError(f"Unknown model: {name}")
