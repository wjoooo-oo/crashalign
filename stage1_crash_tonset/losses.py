#!/usr/bin/env python3
"""Losses for Task-1 cumulative phase model."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def masked_mean(x: torch.Tensor, mask: torch.Tensor, dim: int = 1, eps: float = 1e-6) -> torch.Tensor:
    """x: [B,T,...] or [B,T]; mask: [B,T] bool."""
    m = mask.float()
    while m.ndim < x.ndim:
        m = m.unsqueeze(-1)
    num = (x * m).sum(dim=dim)
    den = m.sum(dim=dim).clamp(min=eps)
    return num / den


def last_valid(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """x [B,T] or [B,T,H]; return value at last True mask step."""
    lengths = mask.sum(dim=1).clamp(min=1).long() - 1  # [B]
    idx = lengths.view(-1, 1)
    if x.ndim == 2:
        return x.gather(1, idx).squeeze(1)
    # [B,T,H]
    idx = idx.unsqueeze(-1).expand(-1, 1, x.size(-1))
    return x.gather(1, idx).squeeze(1)


def balanced_ce_with_logits(
    logits: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Masked cross-entropy averaged over the classes present in the batch.

    Same class-balancing convention as `balanced_bce_with_logits`, extended to
    the 4-way phase head. `logits` [B,T,C], `targets` [B,T] long, `mask` [B,T].
    """
    c = logits.size(-1)
    logits = logits.reshape(-1, c)[mask.reshape(-1)]
    targets = targets.reshape(-1)[mask.reshape(-1)]
    if logits.numel() == 0:
        return logits.new_zeros(())

    per = F.cross_entropy(logits, targets, reduction="none")
    losses = [per[targets == k].mean() for k in range(c) if bool((targets == k).any())]
    if not losses:
        return logits.new_zeros(())
    return torch.stack(losses).mean()


def balanced_bce_with_logits(
    logits: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """BBCE over masked positions: 0.5 * mean(pos) + 0.5 * mean(neg).

    If only one class is present, use that class mean.
    If no valid positions, return 0 (not NaN).
    """
    logits = logits[mask]
    targets = targets[mask]
    if logits.numel() == 0:
        return logits.new_zeros(())

    pos = targets > 0.5
    neg = ~pos
    losses = []
    if pos.any():
        losses.append(F.binary_cross_entropy_with_logits(logits[pos], targets[pos]))
    if neg.any():
        losses.append(F.binary_cross_entropy_with_logits(logits[neg], targets[neg]))
    if not losses:
        return logits.new_zeros(())
    if len(losses) == 1:
        return losses[0]
    return 0.5 * (losses[0] + losses[1])
