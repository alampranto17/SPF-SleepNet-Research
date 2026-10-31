"""SleepQualityNet v2 detection network.

Backbone kept from v1 (3 conv blocks 16->32->48 channels with BatchNorm +
LayerNorm, additive attention pooling over 32 time steps), with kernel sizes
halved because the input is now 100 Hz (3000 samples) instead of 200 Hz.

Switches used by the RQ4 component analysis (exactly one changes at a time):
  use_feat      : fuse the handcrafted feature vector with the CNN embedding
  use_attention : additive attention pooling (True) vs mean pooling (False)
  context       : dilated residual temporal CNN over neighbouring epoch
                  embeddings (receptive field +/-15 epochs = 7.5 min, the
                  sequence-context idea used by state-of-the-art single-channel
                  stagers such as L-SeqSleepNet / TinySleepNet, without an LSTM);
                  False = epoch-wise

The quality-regression head of v1 is removed: its target (0.6*delta+0.4*SWA)
was a deterministic function of model inputs (circular).  Factor scores are
now computed transparently outside the network (see scores.py).
"""
from __future__ import annotations

import torch
import torch.nn as nn

CONTEXT_DILATIONS = (1, 2, 4, 8)
CONTEXT_RADIUS = sum(CONTEXT_DILATIONS)   # epochs on each side


class EpochEncoder(nn.Module):
    def __init__(self, use_attention=True):
        super().__init__()
        self.use_attention = use_attention
        self.conv1 = nn.Sequential(nn.Conv1d(1, 16, 32, stride=2, padding=16),
                                   nn.BatchNorm1d(16), nn.ReLU(), nn.MaxPool1d(4))
        self.ln1 = nn.LayerNorm(16)
        self.conv2 = nn.Sequential(nn.Conv1d(16, 32, 16, stride=2, padding=8),
                                   nn.BatchNorm1d(32), nn.ReLU(), nn.MaxPool1d(4))
        self.ln2 = nn.LayerNorm(32)
        self.conv3 = nn.Sequential(nn.Conv1d(32, 48, 8, stride=1, padding=4),
                                   nn.BatchNorm1d(48), nn.ReLU(), nn.AdaptiveAvgPool1d(32))
        self.ln3 = nn.LayerNorm(48)
        if use_attention:   # not instantiated for the mean-pooling ablation
            self.attention = nn.Sequential(nn.Linear(48, 32), nn.Tanh(), nn.Linear(32, 1))

    @staticmethod
    def _ln(x, ln):
        return ln(x.transpose(1, 2)).transpose(1, 2)

    def forward(self, x, return_attn=False):          # x: (N, 1, 3000)
        x = self._ln(self.conv1(x), self.ln1)
        x = self._ln(self.conv2(x), self.ln2)
        x = self._ln(self.conv3(x), self.ln3)            # (N, 48, 32)
        h = x.transpose(1, 2)                            # (N, 32, 48)
        if self.use_attention:
            w = torch.softmax(self.attention(h).float(), dim=1).to(h.dtype)
        else:
            w = torch.full_like(h[..., :1], 1.0 / h.shape[1])
        z = (h * w).sum(1)
        return (z, w.squeeze(-1)) if return_attn else z


class SleepQualityNetV2(nn.Module):
    def __init__(self, n_feat=19, use_feat=True, use_attention=True, context=False,
                 emb=64, dropout=0.3, dilations=CONTEXT_DILATIONS):
        super().__init__()
        self.n_feat, self.use_feat, self.context = n_feat, use_feat, context
        dilations = tuple(dilations)
        self.encoder = EpochEncoder(use_attention)
        self.fuse = nn.Sequential(nn.Linear(48 + (n_feat if use_feat else 0), emb),
                                  nn.LayerNorm(emb), nn.ReLU(), nn.Dropout(dropout))
        if context:
            # dilated residual temporal CNN over epoch embeddings:
            # dilations 1,2,4,8 with kernel 3 -> receptive field +/-15 epochs (7.5 min);
            # the receptive-field ablation uses dilations 1,2 (+/-3 epochs = 1.5 min)
            self.ctx = nn.ModuleList([nn.Sequential(
                nn.Conv1d(emb, emb, 3, padding=d, dilation=d), nn.ReLU(), nn.Dropout(dropout),
                nn.Conv1d(emb, emb, 1)) for d in dilations])
            self.ctx_norm = nn.ModuleList([nn.LayerNorm(emb) for _ in dilations])
        self.head = nn.Sequential(nn.Linear(emb, 32), nn.ReLU(), nn.Dropout(dropout),
                                  nn.Linear(32, 1))

    def embed(self, x, f):
        """x: (N, 3000), f: (N, n_feat) -> (N, emb) epoch embeddings."""
        z = self.encoder(x.unsqueeze(1))
        if self.use_feat:
            z = torch.cat([z, f.to(z.dtype)], dim=1)
        return self.fuse(z)

    def contextualise(self, e):
        """e: (B, L, emb) -> (B, L, emb)."""
        if not self.context:
            return e
        for block, norm in zip(self.ctx, self.ctx_norm):
            e = norm(e + block(e.transpose(1, 2)).transpose(1, 2))
        return e

    def forward(self, x, f):
        """x: (B, L, 3000), f: (B, L, n_feat) -> logits (B, L)."""
        B, L, T = x.shape
        e = self.embed(x.reshape(B * L, T), f.reshape(B * L, -1)).reshape(B, L, -1)
        e = self.contextualise(e)
        return self.head(e).squeeze(-1)

    @torch.no_grad()
    def predict_sequence(self, x, f, batch=1024, return_emb=False):
        """Whole recording in chronological order: x (n, 3000), f (n, n_feat)."""
        embs = [self.embed(x[i:i + batch].float(), f[i:i + batch])
                for i in range(0, len(x), batch)]
        e = torch.cat(embs, 0).unsqueeze(0)
        e = self.contextualise(e)
        logits = self.head(e).squeeze(-1).squeeze(0)
        return (logits, e.squeeze(0)) if return_emb else logits


def count_params(m):
    return int(sum(p.numel() for p in m.parameters()))
