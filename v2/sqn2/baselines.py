"""Published single-channel sleep-staging architectures, re-implemented as
same-protocol baselines for binary N3 detection.

Every baseline is trained and evaluated exactly like SleepQualityNet v2
(same participant folds, inner-validation early stopping / threshold /
temperature, binary N3 loss with the same positive-class weight, same
optimiser, chunks of 64 consecutive 30-s epochs at 100 Hz, raw EEG only,
no handcrafted features).  Only the network changes.  Architectures follow
the original papers at 100 Hz; the deviations needed for the common protocol
are listed per class and in ``BASELINE_NOTES`` (reported in Methods).

  DeepSleepNet   Supratak et al., IEEE TNSRE 2017
  TinySleepNet   Supratak & Guo, IEEE EMBC 2020
  AttnSleep      Eldele et al., IEEE TNSRE 2021
  L-SeqSleepNet  Phan et al., IEEE JBHI 2023

All classes expose the SleepQualityNetV2 interface used by the trainer:
  forward(x (B, L, 3000), f) -> logits (B, L)
  predict_sequence(x (n, 3000), f, return_emb) -> logits (n,) [, embeddings (n, d)]
Recurrent models are applied to whole recordings with overlapping windows of
the training length (64 epochs, stride 32, centre-weighted blending), so
training and inference see the same context length.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as Fnn

SEQ_LEN = 64            # training chunk length (epochs), = cfg["chunk_len"]

BASELINE_NOTES = {
    "deepsleepnet": "Dual-branch CNN (small filters fs/2, large filters 4*fs) + 2-layer BiLSTM(512) with "
                    "residual FC shortcut, as published; trained end-to-end on 64-epoch sequences "
                    "(the original two-step pre-training + fine-tuning schedule is replaced by the "
                    "common protocol); binary N3 output.",
    "tinysleepnet": "CNN (128 filters, fs/2 kernel, fs/16 stride, 4 conv layers) + 1-layer LSTM(128), as "
                    "published; 64-epoch sequences instead of 15-20; binary N3 output; the original "
                    "signal-shift augmentation is replaced by the common amplitude/noise augmentation.",
    "attnsleep": "Multi-resolution CNN + adaptive feature recalibration (SE, 30 channels) + temporal "
                 "context encoder (2 layers, 5 heads, causal convolutions, d_model 80, d_ff 120), as "
                 "published for 100 Hz; epoch-wise model (no inter-epoch context, as published); binary "
                 "N3 output; class-weighted BCE instead of class-weighted CE; computed in float32 "
                 "(mixed precision overflowed in the attention scores).",
    "lseqsleepnet": "Log-magnitude STFT input (2-s Hamming window, 1-s hop, 256-point FFT, 129 bins), "
                    "learnable filterbank (32), epoch encoder BiGRU(64) + attention pooling, long-sequence "
                    "encoder with subsequence-wise and epoch-wise BiGRUs (64) and residual connections, as "
                    "published; sequence of 64 epochs folded into 8 subsequences of 8 (published: 200 "
                    "epochs as 20 x 10) because the common protocol trains on 64-epoch chunks.",
}


# ---------------------------------------------------------------------------
# shared sequence-wrapper
# ---------------------------------------------------------------------------
class _SeqBaseline(nn.Module):
    recurrent = True                       # whole-night inference uses overlapping windows
    seq_multiple = 1                       # window length must be a multiple of this

    def epoch_embed(self, x):              # (N, 3000) -> (N, d)
        raise NotImplementedError

    def seq(self, e):                      # (B, L, d) -> (B, L, h)
        return e

    def head(self, h):                     # (B, L, h) -> (B, L)
        return self.out(h).squeeze(-1)

    def forward(self, x, f=None):
        B, L, T = x.shape
        e = self.epoch_embed(x.reshape(B * L, T)).reshape(B, L, -1)
        return self.head(self.seq(e))

    @torch.no_grad()
    def predict_sequence(self, x, f=None, batch=512, return_emb=False):
        e = torch.cat([self.epoch_embed(x[i:i + batch].float()) for i in range(0, len(x), batch)], 0)
        n = e.shape[0]
        if not self.recurrent:
            h = self.seq(e.unsqueeze(0)).squeeze(0)
            lg = self.head(h.unsqueeze(0)).squeeze(0)
            return (lg, h) if return_emb else lg
        L = SEQ_LEN
        if n <= L:
            m = self.seq_multiple
            pad = (-n) % m
            ee = Fnn.pad(e, (0, 0, 0, pad)) if pad else e
            h = self.seq(ee.unsqueeze(0)).squeeze(0)[:n]
            lg = self.head(h.unsqueeze(0)).squeeze(0)
            return (lg, h) if return_emb else lg
        starts = list(range(0, n - L + 1, L // 2))
        if starts[-1] != n - L:
            starts.append(n - L)
        w = torch.hann_window(L + 2, periodic=False, device=e.device)[1:-1].clamp_min(1e-3)
        acc_h, acc_l, acc_w = None, torch.zeros(n, device=e.device), torch.zeros(n, device=e.device)
        for s in starts:
            h = self.seq(e[s:s + L].unsqueeze(0)).squeeze(0).float()
            lg = self.head(h.unsqueeze(0).to(e.dtype)).squeeze(0).float()
            if acc_h is None:
                acc_h = torch.zeros(n, h.shape[1], device=e.device)
            acc_h[s:s + L] += h * w[:, None]
            acc_l[s:s + L] += lg * w
            acc_w[s:s + L] += w
        lg, h = acc_l / acc_w, acc_h / acc_w[:, None]
        return (lg, h) if return_emb else lg


def _conv_bn_relu(cin, cout, k, s=1, p=None):
    p = k // 2 if p is None else p
    return [nn.Conv1d(cin, cout, k, stride=s, padding=p, bias=False), nn.BatchNorm1d(cout), nn.ReLU()]


# ---------------------------------------------------------------------------
# DeepSleepNet (2017)
# ---------------------------------------------------------------------------
class DeepSleepNet(_SeqBaseline):
    def __init__(self, fs=100, dropout=0.5):
        super().__init__()
        self.small = nn.Sequential(
            *_conv_bn_relu(1, 64, fs // 2, s=fs // 16, p=fs // 4), nn.MaxPool1d(8, 8), nn.Dropout(dropout),
            *_conv_bn_relu(64, 128, 8), *_conv_bn_relu(128, 128, 8), *_conv_bn_relu(128, 128, 8),
            nn.MaxPool1d(4, 4), nn.Flatten())
        self.large = nn.Sequential(
            *_conv_bn_relu(1, 64, fs * 4, s=fs // 2, p=fs * 2), nn.MaxPool1d(4, 4), nn.Dropout(dropout),
            *_conv_bn_relu(64, 128, 6), *_conv_bn_relu(128, 128, 6), *_conv_bn_relu(128, 128, 6),
            nn.MaxPool1d(2, 2), nn.Flatten())
        with torch.no_grad():
            d = self._cnn(torch.zeros(2, 3000)).shape[1]
        self.drop = nn.Dropout(dropout)
        self.lstm = nn.LSTM(d, 512, num_layers=2, batch_first=True, bidirectional=True, dropout=dropout)
        self.shortcut = nn.Sequential(nn.Linear(d, 1024), nn.BatchNorm1d(1024), nn.ReLU())
        self.drop2 = nn.Dropout(dropout)
        self.out = nn.Linear(1024, 1)

    def _cnn(self, x):
        x = x.unsqueeze(1)
        return torch.cat([self.small(x), self.large(x)], 1)

    def epoch_embed(self, x):
        return self.drop(self._cnn(x))

    def seq(self, e):
        B, L, d = e.shape
        h, _ = self.lstm(e)
        sc = self.shortcut(e.reshape(B * L, d)).reshape(B, L, -1)
        return self.drop2(h + sc)


# ---------------------------------------------------------------------------
# TinySleepNet (2020)
# ---------------------------------------------------------------------------
class TinySleepNet(_SeqBaseline):
    def __init__(self, fs=100, dropout=0.5):
        super().__init__()
        self.cnn = nn.Sequential(
            *_conv_bn_relu(1, 128, fs // 2, s=fs // 16, p=fs // 4), nn.MaxPool1d(8, 8), nn.Dropout(dropout),
            *_conv_bn_relu(128, 128, 8), *_conv_bn_relu(128, 128, 8), *_conv_bn_relu(128, 128, 8),
            nn.MaxPool1d(4, 4), nn.Flatten(), nn.Dropout(dropout))
        with torch.no_grad():
            d = self.cnn(torch.zeros(2, 1, 3000)).shape[1]
        self.lstm = nn.LSTM(d, 128, batch_first=True)
        self.drop = nn.Dropout(dropout)
        self.out = nn.Linear(128, 1)

    def epoch_embed(self, x):
        return self.cnn(x.unsqueeze(1))

    def seq(self, e):
        h, _ = self.lstm(e)
        return self.drop(h)


# ---------------------------------------------------------------------------
# AttnSleep (2021)
# ---------------------------------------------------------------------------
class _SE(nn.Module):
    def __init__(self, c, r=16):
        super().__init__()
        self.fc = nn.Sequential(nn.Linear(c, max(c // r, 1)), nn.ReLU(), nn.Linear(max(c // r, 1), c),
                                nn.Sigmoid())

    def forward(self, x):
        return x * self.fc(x.mean(-1)).unsqueeze(-1)


class _AFR(nn.Module):
    """Adaptive feature recalibration: SE residual block 128 -> 30 channels."""

    def __init__(self, cin=128, c=30):
        super().__init__()
        self.body = nn.Sequential(nn.Conv1d(cin, c, 1), nn.BatchNorm1d(c), nn.ReLU(),
                                  nn.Conv1d(c, c, 1), nn.BatchNorm1d(c), _SE(c))
        self.down = nn.Sequential(nn.Conv1d(cin, c, 1, bias=False), nn.BatchNorm1d(c))

    def forward(self, x):
        return Fnn.relu(self.body(x) + self.down(x))


class _CausalConv(nn.Conv1d):
    def __init__(self, c, k=7):
        super().__init__(c, c, k, padding=k - 1)

    def forward(self, x):
        return super().forward(x)[..., :x.shape[-1]]


class _TCELayer(nn.Module):
    def __init__(self, tokens=30, d_model=80, heads=5, d_ff=120, dropout=0.1):
        super().__init__()
        self.h, self.dk = heads, d_model // heads
        self.q, self.k, self.v = (_CausalConv(tokens) for _ in range(3))
        self.qconv = _CausalConv(tokens)
        self.proj = nn.Linear(d_model, d_model)
        self.n1, self.n2 = nn.LayerNorm(d_model), nn.LayerNorm(d_model)
        self.ff = nn.Sequential(nn.Linear(d_model, d_ff), nn.ReLU(), nn.Dropout(dropout),
                                nn.Linear(d_ff, d_model))
        self.drop = nn.Dropout(dropout)

    def _attn(self, q_in, kv):
        N, T, D = kv.shape                              # T tokens (channels), D = d_model (time)
        q = self.q(q_in).view(N, T, self.h, self.dk).transpose(1, 2)
        k = self.k(kv).view(N, T, self.h, self.dk).transpose(1, 2)
        v = self.v(kv).view(N, T, self.h, self.dk).transpose(1, 2)
        a = torch.softmax((q.float() @ k.float().transpose(-2, -1)) / math.sqrt(self.dk), -1).to(v.dtype)
        return self.proj((self.drop(a) @ v).transpose(1, 2).reshape(N, T, D))

    def forward(self, x):
        q = self.qconv(x)
        x = self.n1(q + self.drop(self._attn(q, x)))
        return self.n2(x + self.drop(self.ff(x)))


class AttnSleep(_SeqBaseline):
    recurrent = False

    def __init__(self, fs=100, dropout=0.5):
        super().__init__()
        g = nn.GELU
        self.b1 = nn.Sequential(
            nn.Conv1d(1, 64, 50, 6, 24, bias=False), nn.BatchNorm1d(64), g(), nn.MaxPool1d(8, 2, 4),
            nn.Dropout(dropout),
            nn.Conv1d(64, 128, 8, 1, 4, bias=False), nn.BatchNorm1d(128), g(),
            nn.Conv1d(128, 128, 8, 1, 4, bias=False), nn.BatchNorm1d(128), g(), nn.MaxPool1d(4, 4, 2))
        self.b2 = nn.Sequential(
            nn.Conv1d(1, 64, 400, 50, 200, bias=False), nn.BatchNorm1d(64), g(), nn.MaxPool1d(4, 2, 2),
            nn.Dropout(dropout),
            nn.Conv1d(64, 128, 7, 1, 3, bias=False), nn.BatchNorm1d(128), g(),
            nn.Conv1d(128, 128, 7, 1, 3, bias=False), nn.BatchNorm1d(128), g(), nn.MaxPool1d(2, 2, 1))
        self.drop = nn.Dropout(dropout)
        self.afr = _AFR(128, 30)
        with torch.no_grad():
            t = torch.cat([self.b1(torch.zeros(2, 1, 3000)), self.b2(torch.zeros(2, 1, 3000))], -1).shape[-1]
        self.fix = nn.Identity() if t == 80 else nn.AdaptiveAvgPool1d(80)   # published: 80 at 100 Hz
        self.tce = nn.Sequential(_TCELayer(), _TCELayer())
        self.out = nn.Linear(30 * 80, 1)

    def epoch_embed(self, x):
        # float32 throughout: under fp16 autocast the attention scores overflowed (NaN on T4)
        with torch.autocast(x.device.type, enabled=False):
            x = x.float().unsqueeze(1)
            z = self.drop(torch.cat([self.b1(x), self.b2(x)], -1))
            z = self.tce(self.fix(self.afr(z)))
            return z.flatten(1)                            # (N, 2400)

    def head(self, h):
        with torch.autocast(h.device.type, enabled=False):
            return self.out(h.float()).squeeze(-1)


# ---------------------------------------------------------------------------
# L-SeqSleepNet (2023)
# ---------------------------------------------------------------------------
class LSeqSleepNet(_SeqBaseline):
    def __init__(self, fs=100, n_filt=32, hid=64, sub=8, dropout=0.25):
        super().__init__()
        self.sub = sub
        self.seq_multiple = sub
        self.nfft, self.win, self.hop = 256, 2 * fs, fs
        self.register_buffer("window", torch.hamming_window(self.win), persistent=False)
        nb = self.nfft // 2 + 1
        self.bn_in = nn.BatchNorm1d(nb)
        self.fb = nn.Parameter(torch.randn(nb, n_filt) * 0.1)     # filterbank (sigmoid-constrained)
        self.enc_rnn = nn.GRU(n_filt, hid, batch_first=True, bidirectional=True)
        self.att = nn.Sequential(nn.Linear(2 * hid, 2 * hid), nn.Tanh(), nn.Linear(2 * hid, 1, bias=False))
        self.intra = nn.GRU(2 * hid, hid, batch_first=True, bidirectional=True)
        self.inter = nn.GRU(2 * hid, hid, batch_first=True, bidirectional=True)
        self.n1, self.n2 = nn.LayerNorm(2 * hid), nn.LayerNorm(2 * hid)
        self.drop = nn.Dropout(dropout)
        self.out = nn.Linear(2 * hid, 1)

    def epoch_embed(self, x):                              # (N, 3000)
        with torch.autocast(x.device.type, enabled=False):
            S = torch.stft(x.float(), self.nfft, hop_length=self.hop, win_length=self.win,
                           window=self.window, center=False, return_complex=True)
            S = 20 * torch.log10(S.abs() + 1e-6)          # (N, 129, frames)
            S = self.bn_in(S).transpose(1, 2)              # (N, frames, 129)
            S = S @ torch.sigmoid(self.fb)                  # (N, frames, 32)
        h, _ = self.enc_rnn(S.to(x.dtype) if x.dtype != torch.float32 else S)
        w = torch.softmax(self.att(h).float(), 1).to(h.dtype)
        return self.drop((h * w).sum(1))                   # (N, 128)

    def seq(self, e):
        B, L, d = e.shape
        pad = (-L) % self.sub
        if pad:
            e = Fnn.pad(e, (0, 0, 0, pad))
        Lp = e.shape[1]
        K = Lp // self.sub
        z = e.reshape(B * K, self.sub, d)
        h1, _ = self.intra(z)
        h1 = self.n1(h1 + z).reshape(B, K, self.sub, d)
        z2 = h1.transpose(1, 2).reshape(B * self.sub, K, d)
        h2, _ = self.inter(z2)
        h2 = self.n2(h2 + z2).reshape(B, self.sub, K, d).transpose(1, 2).reshape(B, Lp, d)
        return self.drop(h2[:, :L])


BASELINES = {"deepsleepnet": DeepSleepNet, "tinysleepnet": TinySleepNet,
             "attnsleep": AttnSleep, "lseqsleepnet": LSeqSleepNet}


def build(arch, **kw):
    return BASELINES[arch](**kw)
