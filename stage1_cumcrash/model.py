#!/usr/bin/env python3
"""Task1 model: frozen VideoMAE embeddings → GRU → cumulative q_pre/q_post + crash."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import VideoMAEForVideoClassification


class FrozenVideoMAE(nn.Module):
    def __init__(self, pretrained: str):
        super().__init__()
        base = VideoMAEForVideoClassification.from_pretrained(pretrained, num_labels=1)
        self.videomae = base.videomae
        self.fc_norm = base.fc_norm
        self.hidden = base.config.hidden_size
        for p in self.parameters():
            p.requires_grad = False
        self.eval()

    @torch.no_grad()
    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        out = self.videomae(pixel_values)
        seq = out.last_hidden_state
        if self.fc_norm is not None:
            return self.fc_norm(seq.mean(1))
        return seq[:, 0]


class CumCrashModel(nn.Module):
    """Sequence model on window embeddings.

    Monotone cumulative heads:
      delta = softplus(raw); q = sigmoid(cumsum(delta))
      q_post = q_pre * sigmoid(cumsum(delta_post))  # cannot precede pre
    """

    def __init__(self, emb_dim: int = 768, gru_hidden: int = 256, dropout: float = 0.2):
        super().__init__()
        self.gru = nn.GRU(
            input_size=emb_dim,
            hidden_size=gru_hidden,
            num_layers=1,
            batch_first=True,
            bidirectional=False,
        )
        self.cum_pre = nn.Linear(gru_hidden, 1)
        self.cum_post = nn.Linear(gru_hidden, 1)
        self.crash_head = nn.Sequential(
            nn.Linear(gru_hidden * 2 + 4, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
        )

    def forward(self, emb: torch.Tensor, valid_mask: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        emb: [B, T, D], valid_mask: [B, T] bool
        """
        # zero-pad invalid steps so GRU does not see garbage
        emb = emb * valid_mask.unsqueeze(-1).float()
        lengths = valid_mask.sum(dim=1).clamp(min=1).cpu()
        packed = nn.utils.rnn.pack_padded_sequence(
            emb, lengths, batch_first=True, enforce_sorted=False
        )
        packed_out, h_n = self.gru(packed)
        h, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True, total_length=emb.size(1))

        d_pre = F.softplus(self.cum_pre(h).squeeze(-1))  # [B,T]
        d_post = F.softplus(self.cum_post(h).squeeze(-1))
        d_pre = d_pre * valid_mask.float()
        d_post = d_post * valid_mask.float()
        q_pre = torch.sigmoid(torch.cumsum(d_pre, dim=1))
        # post is gated by pre so trajectory respects pre→post order
        q_post = q_pre * torch.sigmoid(torch.cumsum(d_post, dim=1))

        # video-level features from pattern + states
        mask_f = valid_mask.float()
        denom = mask_f.sum(dim=1, keepdim=True).clamp(min=1.0)
        h_masked = h * mask_f.unsqueeze(-1)
        h_max = h_masked.max(dim=1).values
        h_avg = h_masked.sum(dim=1) / denom
        qpre_max = (q_pre * mask_f).max(dim=1).values
        qpost_max = (q_post * mask_f).max(dim=1).values
        # rise strength: last valid - first
        idx = (lengths - 1).long().to(emb.device)
        qpre_last = q_pre.gather(1, idx.unsqueeze(1)).squeeze(1)
        qpost_last = q_post.gather(1, idx.unsqueeze(1)).squeeze(1)
        feat = torch.cat(
            [
                h_max,
                h_avg,
                qpre_max.unsqueeze(1),
                qpost_max.unsqueeze(1),
                qpre_last.unsqueeze(1),
                qpost_last.unsqueeze(1),
            ],
            dim=-1,
        )
        crash_logit = self.crash_head(feat).squeeze(-1)
        return {
            "h": h,
            "q_pre": q_pre,
            "q_post": q_post,
            "crash_logit": crash_logit,
        }
