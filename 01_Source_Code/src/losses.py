from __future__ import annotations

import torch
import torch.nn.functional as F


def soft_dice_loss(logits: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    probability = torch.softmax(logits, dim=1)[:, 1]
    target = target.float()
    intersection = torch.sum(probability * target, dim=(1, 2))
    denominator = torch.sum(probability, dim=(1, 2)) + torch.sum(target, dim=(1, 2))
    return (1 - (2 * intersection + eps) / (denominator + eps)).mean()


def ce_dice_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    change_class_weight: float = 3.0,
    ce_weight: float = 1.0,
    dice_weight: float = 1.0,
) -> torch.Tensor:
    weights = torch.tensor([1.0, change_class_weight], device=logits.device, dtype=logits.dtype)
    ce = F.cross_entropy(logits, target.long(), weight=weights)
    dice = soft_dice_loss(logits, target)
    return ce_weight * ce + dice_weight * dice
