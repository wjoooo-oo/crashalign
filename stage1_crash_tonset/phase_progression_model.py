#!/usr/bin/env python3
"""Phase progression head used by 0728_full_accident_phase_t (E2 v1 recipe).

Standalone module so checkpoints remain loadable after the original training
script was removed from the repo.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class PhaseProgressionModel(nn.Module):
    """BiGRU + 4-way softmax over normal / pre / impact / post."""

    def __init__(self, emb_dim: int = 768, gru_hidden: int = 256, dropout: float = 0.3):
        super().__init__()
        self.gru = nn.GRU(
            emb_dim,
            gru_hidden,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if dropout > 0 else 0.0,
        )
        self.out = nn.Linear(gru_hidden * 2, 4)

    def forward(self, emb: torch.Tensor, mask: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        h, _ = self.gru(emb)
        logits = self.out(h)
        phase_prob = torch.softmax(logits, dim=-1)
        return {"phase_logits": logits, "phase_prob": phase_prob}
