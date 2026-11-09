#!/usr/bin/env python3
"""Task-1 model: frozen VideoMAE embeddings → BiGRU → a/b phase + crash MLP."""

from __future__ import annotations

import torch
import torch.nn as nn
from transformers import VideoMAEForVideoClassification

from losses import last_valid, masked_mean


class FrozenVideoMAE(nn.Module):
    """Frozen VideoMAE window encoder.

    pooling="auto" reproduces HF classification pooling: fc_norm(mean) when the
    config uses mean pooling, otherwise token 0. VideoMAE has no CLS token, so for
    checkpoints with use_mean_pooling=False (e.g. checkpoints_temporal_clean) token 0
    is the top-left patch of the first tubelet. pooling="mean" averages all tokens.
    pooling="auto" 复现 HF 分类头的池化：均值池化配置下用 fc_norm(mean)，否则取第 0 个 token。
    VideoMAE 没有 CLS token，use_mean_pooling=False 时第 0 个 token 是首个 tubelet 的左上角 patch；
    pooling="mean" 对全部 token 取平均。
    """

    def __init__(self, pretrained: str, pooling: str = "auto"):
        super().__init__()
        if pooling not in ("auto", "mean"):
            raise ValueError(pooling)
        base = VideoMAEForVideoClassification.from_pretrained(pretrained, num_labels=1)
        self.videomae = base.videomae
        self.fc_norm = base.fc_norm
        self.pooling = pooling
        self.hidden_size = base.config.hidden_size
        for p in self.parameters():
            p.requires_grad = False
        self.eval()

    @torch.no_grad()
    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        out = self.videomae(pixel_values)
        seq = out.last_hidden_state
        if self.fc_norm is not None:
            return self.fc_norm(seq.mean(1))
        if self.pooling == "mean":
            return seq.mean(1)
        return seq[:, 0]


CRASH_FEAT_CHOICES = (
    "full",
    "phase_only",  # sequential phase summaries (+ last, order)
    "h_only",
    "phase_unordered",  # bag-of-phase: mean/max/std only (no last/order)
    "pattern_gated",  # sequential phase + soft pre→post gate (cum)
    "fp_gated",  # h crash score + phase FP-suppressing gate (softmax4)
)

PHASE_MODES = (
    "cum",  # legacy: independent cumulative a_t / b_t sigmoid heads
    "softmax4",  # 4-way softmax over normal / pre / impact / post
)
N_PHASES = 4


class Task1CumPhaseModel(nn.Module):
    def __init__(
        self,
        emb_dim: int = 768,
        gru_hidden: int = 256,
        dropout: float = 0.3,
        bidirectional: bool = True,
        crash_feat: str = "full",
        phase_mode: str = "cum",
    ):
        """
        crash_feat:
          full             — h_mean + phase summaries (default; can ignore phase)
          phase_only       — phase summaries only (no contextual features)
          h_only           — ONLY h_mean (no phase summaries; ablation)
          phase_unordered  — unordered phase bag (cum mode only)
          pattern_gated    — sequential phase + soft pattern gate (cum mode only)
          fp_gated         — h score gated by phase (FP-suppressing; softmax4)

        phase_mode:
          cum       — legacy cumulative a_t/b_t heads (impact ignored)
          softmax4  — 4-way phase logits; crash head reads per-class mean/last/max
        """
        super().__init__()
        assert crash_feat in CRASH_FEAT_CHOICES
        assert phase_mode in PHASE_MODES
        if phase_mode == "softmax4":
            assert crash_feat in ("full", "h_only", "phase_only", "fp_gated"), (
                "softmax4 supports full | h_only | phase_only | fp_gated"
            )
        if crash_feat == "fp_gated":
            assert phase_mode == "softmax4", "fp_gated requires phase_mode=softmax4"
        self.crash_feat = crash_feat
        self.phase_mode = phase_mode
        self.bidirectional = bidirectional
        self.gru = nn.GRU(
            input_size=emb_dim,
            hidden_size=gru_hidden,
            num_layers=1,
            batch_first=True,
            bidirectional=bidirectional,
        )
        h_dim = gru_hidden * (2 if bidirectional else 1)
        self.phase_a = nn.Linear(h_dim, 1)
        self.phase_b = nn.Linear(h_dim, 1)
        self.phase4 = nn.Linear(h_dim, N_PHASES) if phase_mode == "softmax4" else None
        # per-class mean/last/max over the 4 phase probabilities
        n_phase_summary = 3 * N_PHASES if phase_mode == "softmax4" else None
        if crash_feat in ("h_only", "fp_gated"):
            # fp_gated: crash MLP reads features only; phase only drives the gate
            # fp_gated：crash MLP 只看特征；phase 仅通过 gate 抑制假阳
            crash_in = h_dim
        elif phase_mode == "softmax4":
            crash_in = (h_dim + n_phase_summary) if crash_feat == "full" else n_phase_summary
        elif crash_feat == "full":
            crash_in = h_dim + 4
        elif crash_feat == "phase_unordered":
            crash_in = 6  # a/b mean,max,std
        else:
            # phase_only / pattern_gated
            crash_in = 7  # a_mean,b_mean,a_last,b_last,a_max,b_max,order
        self.crash_head = nn.Sequential(
            nn.Linear(crash_in, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 1),
        )
        if crash_feat == "pattern_gated" and phase_mode == "cum":
            # soft gate MLP: high when pre→post progression is present
            self.gate_head = nn.Sequential(
                nn.Linear(7, 32),
                nn.GELU(),
                nn.Linear(32, 1),
            )
            # logit offset when gate≈0 (push toward non-crash)
            self.noncrash_bias = nn.Parameter(torch.tensor(-4.0))
        elif crash_feat == "fp_gated":
            # Phase-summary gate with soft floor: never fully kill the h pathway.
            # High gate → keep h; low gate → shrink toward non-crash but keep ≥β·σ(h).
            # 带 soft floor 的 phase 门控：高则保留 h；低则压向非事故，但仍保留 ≥β·σ(h)。
            self.gate_head = nn.Sequential(
                nn.Linear(n_phase_summary, 64),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(64, 1),
            )
            # Fixed soft floor in (0,1); stored as buffer so ckpts stay compatible.
            # 固定 soft floor；用 buffer 保存，便于 checkpoint 兼容。
            self.register_buffer("fp_gate_beta", torch.tensor(0.5))
            self.noncrash_bias = None  # unused in soft-floor form / soft-floor 形式不再使用
            nn.init.constant_(self.gate_head[-1].bias, 2.0)
        else:
            self.gate_head = None
            self.noncrash_bias = None
        self.h_dim = h_dim

    def forward(self, emb: torch.Tensor, valid_mask: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        emb: [B,T,D], valid_mask: [B,T] bool
        """
        emb = emb * valid_mask.unsqueeze(-1).float()
        lengths = valid_mask.sum(dim=1).clamp(min=1).cpu()
        packed = nn.utils.rnn.pack_padded_sequence(
            emb, lengths, batch_first=True, enforce_sorted=False
        )
        packed_out, _ = self.gru(packed)
        h, _ = nn.utils.rnn.pad_packed_sequence(
            packed_out, batch_first=True, total_length=emb.size(1)
        )

        a_logits = self.phase_a(h).squeeze(-1)  # [B,T]
        b_logits = self.phase_b(h).squeeze(-1)
        a_prob = torch.sigmoid(a_logits)
        b_prob = torch.sigmoid(b_logits)

        h_mean = masked_mean(h, valid_mask, dim=1)
        a_mean = masked_mean(a_prob, valid_mask, dim=1)
        b_mean = masked_mean(b_prob, valid_mask, dim=1)
        a_last = last_valid(a_prob, valid_mask)
        b_last = last_valid(b_prob, valid_mask)

        # peak times for order feature (masked argmax)
        neg_inf = torch.finfo(a_prob.dtype).min
        a_masked = a_prob.masked_fill(~valid_mask, neg_inf)
        b_masked = b_prob.masked_fill(~valid_mask, neg_inf)
        a_max = a_masked.max(dim=1).values
        b_max = b_masked.max(dim=1).values
        t_a = a_masked.argmax(dim=1).float()
        t_b = b_masked.argmax(dim=1).float()
        T = valid_mask.sum(dim=1).clamp(min=1).float()
        order = ((t_b - t_a) / T).clamp(-1, 1)  # >0 if post peak after pre peak

        # masked std for unordered bag
        def _std(x):
            mu = masked_mean(x, valid_mask, dim=1)
            var = masked_mean((x - mu.unsqueeze(1)) ** 2, valid_mask, dim=1)
            return var.clamp(min=0).sqrt()

        a_std = _std(a_prob)
        b_std = _std(b_prob)

        if self.phase_mode == "softmax4":
            return self._forward_softmax4(h, h_mean, valid_mask)

        gate = None
        if self.crash_feat == "full":
            g = torch.cat(
                [
                    h_mean,
                    a_mean.unsqueeze(1),
                    b_mean.unsqueeze(1),
                    a_last.unsqueeze(1),
                    b_last.unsqueeze(1),
                ],
                dim=-1,
            )
            crash_logit = self.crash_head(g).squeeze(-1)
        elif self.crash_feat == "h_only":
            crash_logit = self.crash_head(h_mean).squeeze(-1)
        elif self.crash_feat == "phase_unordered":
            g = torch.stack([a_mean, b_mean, a_max, b_max, a_std, b_std], dim=-1)
            crash_logit = self.crash_head(g).squeeze(-1)
        else:
            g = torch.stack(
                [a_mean, b_mean, a_last, b_last, a_max, b_max, order], dim=-1
            )
            base = self.crash_head(g).squeeze(-1)
            if self.crash_feat == "pattern_gated":
                gate = torch.sigmoid(self.gate_head(g).squeeze(-1))
                crash_logit = gate * base + (1.0 - gate) * self.noncrash_bias
            else:
                crash_logit = base

        return {
            "h": h,
            "a_logits": a_logits,
            "b_logits": b_logits,
            "a_prob": a_prob,
            "b_prob": b_prob,
            "crash_logit": crash_logit,
            "order": order,
            "gate": gate,
        }

    def _forward_softmax4(
        self, h: torch.Tensor, h_mean: torch.Tensor, valid_mask: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        phase_logits = self.phase4(h)  # [B,T,4]
        phase_prob = torch.softmax(phase_logits, dim=-1)

        # per-class temporal aggregation: mean / last / max over valid windows
        p_mean = masked_mean(phase_prob, valid_mask, dim=1)  # [B,4]
        p_last = last_valid(phase_prob, valid_mask)  # [B,4]
        # probabilities are non-negative, so 0 is a safe mask value and keeps the
        # summary finite when a sequence has no valid windows at all
        p_max = phase_prob.masked_fill(~valid_mask.unsqueeze(-1), 0.0).max(dim=1).values
        phase_summary = torch.cat([p_mean, p_last, p_max], dim=-1)  # [B,12]

        gate = None
        h_logit = None
        if self.crash_feat == "h_only":
            crash_logit = self.crash_head(h_mean).squeeze(-1)
        elif self.crash_feat == "full":
            g = torch.cat([h_mean, phase_summary], dim=-1)
            crash_logit = self.crash_head(g).squeeze(-1)
        elif self.crash_feat == "fp_gated":
            # Soft-floor FP gate in probability space (β keeps a fraction of σ(h)).
            # 概率空间 soft-floor：即使 gate=0 也保留 β·σ(h)，避免召回被掐死。
            h_logit = self.crash_head(h_mean).squeeze(-1)
            gate = torch.sigmoid(self.gate_head(phase_summary).squeeze(-1))
            h_prob = torch.sigmoid(h_logit)
            beta = self.fp_gate_beta
            p = (h_prob * (beta + (1.0 - beta) * gate)).clamp(1e-4, 1.0 - 1e-4)
            crash_logit = torch.logit(p)
        else:  # phase_only — contextual features never reach the crash head
            crash_logit = self.crash_head(phase_summary).squeeze(-1)

        # cumulative views kept for backward-compatible diagnostics
        a_prob = phase_prob[..., 1:].sum(dim=-1)  # crash process started
        b_prob = phase_prob[..., 3]  # post-impact
        return {
            "h": h,
            "phase_logits": phase_logits,
            "phase_prob": phase_prob,
            "phase_pred": phase_logits.argmax(dim=-1),
            "phase_summary": phase_summary,
            "a_prob": a_prob,
            "b_prob": b_prob,
            "crash_logit": crash_logit,
            "h_logit": h_logit,
            "gate": gate,
        }
