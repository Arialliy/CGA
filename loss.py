"""Losses for MSHNet / MSHNetCGA.

This file is intentionally self-contained and keeps CGA supervision strictly in
training.  Evaluation should use only the final logit.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.cga_targets import CGATargetConfig, build_cga_targets


def _resize_like(target: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    if target.dim() == 3:
        target = target[:, None]
    target = target.float()
    if target.shape[-2:] != ref.shape[-2:]:
        target = F.interpolate(target, size=ref.shape[-2:], mode="nearest")
    return target


def extract_final_logit(output: Any) -> torch.Tensor:
    if isinstance(output, dict):
        for key in ("final_logit", "final_logits", "base_logits", "base_logit", "logits"):
            if key in output:
                return output[key]
        raise KeyError(f"Could not find final logit in output keys: {sorted(output.keys())}")
    if isinstance(output, (tuple, list)):
        if len(output) >= 2 and torch.is_tensor(output[1]):
            return output[1]
        if len(output) >= 1 and torch.is_tensor(output[-1]):
            return output[-1]
    if torch.is_tensor(output):
        return output
    raise TypeError(f"Unsupported output type: {type(output)!r}")


class SoftIoULoss(nn.Module):
    def forward(self, pred_prob: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        target = _resize_like(target, pred_prob)
        pred_prob = pred_prob.float()
        smooth = 1.0
        inter = (pred_prob * target).flatten(1).sum(dim=1)
        union = (pred_prob + target - pred_prob * target).flatten(1).sum(dim=1)
        return (1.0 - (inter + smooth) / (union + smooth)).mean()


class OHEMBCEWithLogitsLoss(nn.Module):
    def __init__(self, topk_ratio: float = 0.01) -> None:
        super().__init__()
        self.topk_ratio = float(topk_ratio)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        target = _resize_like(target, logits)
        loss_map = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
        losses = []
        for b in range(logits.shape[0]):
            flat_loss = loss_map[b].flatten()
            flat_target = target[b].flatten()
            pos = flat_loss[flat_target > 0]
            neg = flat_loss[flat_target <= 0]
            if neg.numel() > 0:
                k = max(1, int(neg.numel() * self.topk_ratio))
                neg = torch.topk(neg, k=min(k, neg.numel()), largest=True).values
            if pos.numel() > 0:
                losses.append(torch.cat([pos, neg]).mean() if neg.numel() else pos.mean())
            elif neg.numel() > 0:
                losses.append(neg.mean())
        if not losses:
            return logits.sum() * 0.0
        return torch.stack(losses).mean()


class MSHNetOHEMLoss(nn.Module):
    def __init__(self, ohem_ratio: float = 0.01, lambda_iou: float = 1.0, warm_epoch: int = 5) -> None:
        super().__init__()
        self.ohem = OHEMBCEWithLogitsLoss(ohem_ratio)
        self.soft_iou = SoftIoULoss()
        self.lambda_iou = float(lambda_iou)
        self.warm_epoch = int(warm_epoch)

    def forward(self, output: Any, target: torch.Tensor, epoch: int = 0) -> dict[str, torch.Tensor]:
        final_logit = extract_final_logit(output)
        target = _resize_like(target, final_logit)
        loss_ohem = self.ohem(final_logit, target)
        loss_iou = self.soft_iou(torch.sigmoid(final_logit), target)
        loss_total = loss_ohem + self.lambda_iou * loss_iou
        # Warm-scale supervision if multi-scale masks exist.
        masks = output.get("masks", []) if isinstance(output, dict) else (output[0] if isinstance(output, (tuple, list)) else [])
        if masks and epoch <= self.warm_epoch:
            scale_loss = sum(self.ohem(m, target) for m in masks) / max(1, len(masks))
            loss_total = loss_total + 0.2 * scale_loss
        else:
            scale_loss = final_logit.sum() * 0.0
        return {
            "total": loss_total,
            "ohem": loss_ohem.detach(),
            "soft_iou": loss_iou.detach(),
            "scale": scale_loss.detach(),
        }


@dataclass(frozen=True)
class CGALossConfig:
    lambda_center: float = 0.05
    lambda_boundary: float = 0.03
    lambda_scale: float = 0.02
    lambda_peak: float = 0.03
    start_epoch: int = 20
    ramp_epochs: int = 40
    ohem_ratio: float = 0.01
    lambda_iou: float = 1.0
    warm_epoch: int = 5


def _ramp_weight(epoch: int, start_epoch: int, ramp_epochs: int) -> float:
    if epoch < start_epoch:
        return 0.0
    if ramp_epochs <= 0:
        return 1.0
    return min(1.0, max(0.0, float(epoch - start_epoch + 1) / float(ramp_epochs)))


class MSHNetCGALoss(nn.Module):
    def __init__(self, cfg: CGALossConfig | None = None, target_cfg: CGATargetConfig | None = None) -> None:
        super().__init__()
        self.cfg = cfg or CGALossConfig()
        self.target_cfg = target_cfg or CGATargetConfig()
        self.base_loss = MSHNetOHEMLoss(
            ohem_ratio=self.cfg.ohem_ratio,
            lambda_iou=self.cfg.lambda_iou,
            warm_epoch=self.cfg.warm_epoch,
        )

    @staticmethod
    def _bce(logit: torch.Tensor | None, target: torch.Tensor) -> torch.Tensor:
        if logit is None:
            # Preserve graph with zero if a head is intentionally disabled.
            return target.sum() * 0.0
        target = _resize_like(target, logit)
        return F.binary_cross_entropy_with_logits(logit, target)

    def forward(self, output: dict[str, torch.Tensor], target: torch.Tensor, epoch: int = 0) -> dict[str, torch.Tensor]:
        if not isinstance(output, dict):
            return self.base_loss(output, target, epoch=epoch)
        final_logit = extract_final_logit(output)
        target = _resize_like(target, final_logit)
        base = self.base_loss(output, target, epoch=epoch)
        targets = build_cga_targets(target, self.target_cfg)
        loss_center = self._bce(output.get("cga_center_logit"), targets["cga_center_target"])
        loss_boundary = self._bce(output.get("cga_boundary_logit"), targets["cga_boundary_target"])
        loss_scale = self._bce(output.get("cga_scale_logit"), targets["cga_scale_target"])
        loss_peak = self._bce(output.get("cga_peak_logit"), targets["cga_peak_target"])
        w = _ramp_weight(epoch, self.cfg.start_epoch, self.cfg.ramp_epochs)
        aux_total = (
            self.cfg.lambda_center * loss_center
            + self.cfg.lambda_boundary * loss_boundary
            + self.cfg.lambda_scale * loss_scale
            + self.cfg.lambda_peak * loss_peak
        )
        total = base["total"] + w * aux_total
        return {
            "total": total,
            "base_total": base["total"].detach(),
            "ohem": base["ohem"],
            "soft_iou": base["soft_iou"],
            "scale": base["scale"],
            "cga_w": torch.tensor(w, device=final_logit.device, dtype=final_logit.dtype),
            "cga_center": loss_center.detach(),
            "cga_boundary": loss_boundary.detach(),
            "cga_scale": loss_scale.detach(),
            "cga_peak": loss_peak.detach(),
        }


def build_loss(name: str = "MSHNetCGA", **kwargs) -> nn.Module:
    name_l = name.lower()
    if "cga" in name_l:
        cfg = CGALossConfig(**{k: v for k, v in kwargs.items() if k in CGALossConfig.__annotations__})
        return MSHNetCGALoss(cfg)
    return MSHNetOHEMLoss(
        ohem_ratio=float(kwargs.get("ohem_ratio", 0.01)),
        lambda_iou=float(kwargs.get("lambda_iou", 1.0)),
        warm_epoch=int(kwargs.get("warm_epoch", 5)),
    )
