import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.signal import butter, sosfilt, find_peaks
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                              recall_score, roc_auc_score,
                              mean_squared_error, r2_score)

try:
    import mne
    _HAS_MNE = True
except Exception:
    _HAS_MNE = False


# ============================================================================
# 1. PREPROCESSING
# ============================================================================

class EEGPreprocessor:
    """EEG preprocessing + 6-feature extraction for N3 detection and quality."""

    def __init__(self, sampling_rate=100, epoch_duration=30):
        self.sampling_rate  = sampling_rate
        self.epoch_duration = epoch_duration
        self.epoch_samples  = sampling_rate * epoch_duration

    def bandpass_filter(self, signal_data, low_freq=0.5, high_freq=40):
        sos = butter(4, [low_freq, high_freq], btype='band',
                     fs=self.sampling_rate, output='sos')
        return sosfilt(sos, signal_data)

    def extract_frequency_bands(self, eeg_epoch):
        bands = {'delta':(0.5,4),'theta':(4,8),'alpha':(8,12),
                 'sigma':(12,16),'beta':(16,30)}
        out = {}
        for name,(lo,hi) in bands.items():
            sos = butter(4,[lo,hi],btype='band',fs=self.sampling_rate,output='sos')
            out[name] = float(np.mean(sosfilt(sos, eeg_epoch)**2))
        return out

    # ------------------------------------------------------------------
    # DETECTION FEATURES (4) — for n3_head
    # ------------------------------------------------------------------

    def extract_delta_power(self, eeg_epoch):
        """Feature 1: Delta band power (0.5-4 Hz) — primary N3 energy."""
        sos = butter(4, [0.5, 4], btype='band', fs=self.sampling_rate, output='sos')
        filtered = sosfilt(sos, eeg_epoch)
        return float(np.mean(filtered ** 2))

    def extract_slow_wave_activity(self, eeg_epoch):
        """Feature 2: Slow Wave Activity — integral of delta power over time.
        Stronger than delta power: captures both amplitude and duration."""
        sos = butter(4, [0.5, 4], btype='band', fs=self.sampling_rate, output='sos')
        filtered = sosfilt(sos, eeg_epoch)
        # RMS amplitude of slow waves (energy per sample, not mean square)
        return float(np.sqrt(np.mean(filtered ** 2)))

    def extract_slow_wave_density(self, eeg_epoch, amp_threshold_uv=75.0):
        """Feature 3: Slow Wave Density — AASM clinical standard.
        Count of slow waves with peak-to-peak amplitude > 75 uV per minute.
        High in N3 by definition (AASM 2007)."""
        sos = butter(4, [0.5, 2], btype='band', fs=self.sampling_rate, output='sos')
        filtered = sosfilt(sos, eeg_epoch)
        # find positive peaks
        peaks_pos, _ = find_peaks(filtered,
                                   height=amp_threshold_uv / 2,
                                   distance=int(self.sampling_rate * 0.25))
        peaks_neg, _ = find_peaks(-filtered,
                                   height=amp_threshold_uv / 2,
                                   distance=int(self.sampling_rate * 0.25))
        n_waves = (len(peaks_pos) + len(peaks_neg)) / 2.0
        epoch_minutes = self.epoch_duration / 60.0
        return float(n_waves / max(epoch_minutes, 1e-6))

    def extract_relative_delta_power(self, eeg_epoch):
        """Feature 4: Relative delta power = delta / total power.
        Normalizes for subject-to-subject amplitude differences."""
        bands = self.extract_frequency_bands(eeg_epoch)
        total = sum(bands.values()) + 1e-8
        return float(bands['delta'] / total)

    # ------------------------------------------------------------------
    # QUALITY FEATURES (2) — for quality_head
    # ------------------------------------------------------------------

    def extract_peak_to_peak_amplitude(self, eeg_epoch):
        """Quality Feature 1: Peak-to-peak amplitude of slow waves.
        Higher amplitude = deeper, higher quality N3."""
        sos = butter(4, [0.5, 4], btype='band', fs=self.sampling_rate, output='sos')
        filtered = sosfilt(sos, eeg_epoch)
        return float(np.max(filtered) - np.min(filtered))

    def extract_fragmentation_index(self, eeg_epoch):
        """Quality Feature 2: Fragmentation index = zero crossing rate.
        Low in high-quality N3 (smooth slow waves).
        High in fragmented sleep (irregular signal)."""
        zero_crossings = np.sum(np.diff(np.sign(eeg_epoch)) != 0)
        return float(zero_crossings / len(eeg_epoch))

    # ------------------------------------------------------------------
    # COMBINED EXTRACTION
    # ------------------------------------------------------------------

    def extract_all_features(self, eeg_epoch):
        """Extract all 6 features. Returns (det_feat, qual_feat) as numpy arrays.

        Detection features (4): delta_power, SWA, SWD, relative_delta
        Quality features   (2): peak_to_peak, fragmentation_index
        """
        det = np.array([
            self.extract_delta_power(eeg_epoch),
            self.extract_slow_wave_activity(eeg_epoch),
            self.extract_slow_wave_density(eeg_epoch),
            self.extract_relative_delta_power(eeg_epoch),
        ], dtype=np.float32)

        qual = np.array([
            self.extract_peak_to_peak_amplitude(eeg_epoch),
            self.extract_fragmentation_index(eeg_epoch),
        ], dtype=np.float32)

        return det, qual

    def normalize_epoch(self, eeg_epoch):
        return (eeg_epoch - np.mean(eeg_epoch)) / (np.std(eeg_epoch) + 1e-8)

    def create_epochs(self, raw_signal, labels=None):
        """Slice signal into epochs. Returns (epochs, det_features, qual_features, labels)."""
        epochs, det_feats, qual_feats, epoch_labels = [], [], [], []
        n = len(raw_signal) // self.epoch_samples
        for i in range(n):
            start = i * self.epoch_samples
            ep    = raw_signal[start:start+self.epoch_samples]
            ep    = self.bandpass_filter(ep)
            ep    = self.normalize_epoch(ep)
            det, qual = self.extract_all_features(ep)
            epochs.append(ep)
            det_feats.append(det)
            qual_feats.append(qual)
            if labels is not None and i < len(labels):
                epoch_labels.append(labels[i])
        return (np.array(epochs),
                np.array(det_feats),
                np.array(qual_feats),
                np.array(epoch_labels))


# ============================================================================
# 2. N3-ONLY QUALITY LABEL BUILDER
# ============================================================================

def make_n3_quality_labels(det_features, stage_labels):
    """
    Build quality labels non-zero ONLY for N3 epochs.
    Uses delta power (det_features[:,0]) for N3-only min-max normalization.
    Returns shape (N, 1), float32.
    """
    dp = np.asarray(det_features, dtype=np.float32)
    if dp.ndim == 2:
        dp = dp[:, 0]   # delta power is feature index 0
    sl = np.asarray(stage_labels, dtype=np.int64).flatten()
    q  = np.zeros(len(dp), dtype=np.float32)
    n3_mask = (sl == 3)
    if n3_mask.sum() > 1:
        n3_dp = dp[n3_mask]
        rng   = n3_dp.max() - n3_dp.min()
        if rng > 1e-8:
            q[n3_mask] = (n3_dp - n3_dp.min()) / rng
        else:
            q[n3_mask] = 0.5
    return q.reshape(-1, 1)


# ============================================================================
# 3. FEATURE NORMALIZER — per-feature standardization across dataset
# ============================================================================

class FeatureNormalizer:
    """Standardize detection and quality features to zero mean, unit variance.
    Fit on training set, apply to val/test."""

    def __init__(self):
        self.det_mean  = None
        self.det_std   = None
        self.qual_mean = None
        self.qual_std  = None

    def fit(self, det_features, qual_features):
        self.det_mean  = det_features.mean(axis=0, keepdims=True)
        self.det_std   = det_features.std(axis=0,  keepdims=True) + 1e-8
        self.qual_mean = qual_features.mean(axis=0, keepdims=True)
        self.qual_std  = qual_features.std(axis=0,  keepdims=True) + 1e-8

    def transform_det(self, det_features):
        return (det_features - self.det_mean) / self.det_std

    def transform_qual(self, qual_features):
        return (qual_features - self.qual_mean) / self.qual_std


# ============================================================================
# 4. MODEL — Dual-path feature fusion
#
#   Detection path : CNN-BiLSTM-Attention + 4 detection features -> n3_head
#   Quality path   : CNN-BiLSTM-Attention + 2 quality features   -> quality_head
#
#   RQ1: both heads together answer primary question
#   RQ2: explicit physiological features in both paths
#   RQ3: BCEWithLogitsLoss(pos_weight) on n3_head
# ============================================================================

class SleepQualityNet(nn.Module):
    """
    Dual-path multi-feature N3 detector + quality regressor.

    Detection features (4): delta_power, SWA, slow_wave_density, relative_delta
    Quality features   (2): peak_to_peak_amplitude, fragmentation_index

    Forward inputs:
        raw_eeg    : (B, 1, L)
        det_feat   : (B, 4)  — detection features
        qual_feat  : (B, 2)  — quality features

    Forward outputs:
        n3_logit   : (B, 1)  — raw logit, sigmoid -> P(N3)
        quality    : (B, 1)  — quality score [0,1], x100 = score
    """

    N_DET_FEAT  = 4
    N_QUAL_FEAT = 2

    def __init__(self, input_length=6000, use_det_feat=True, use_qual_feat=True):
        super().__init__()
        self.use_det_feat  = use_det_feat
        self.use_qual_feat = use_qual_feat

        # ── 1D-CNN backbone (shared) ─────────────────────────────────────────
        self.conv1 = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=64, stride=2, padding=32),
            nn.BatchNorm1d(32), nn.ReLU(), nn.MaxPool1d(4))
        self.conv2 = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=32, stride=2, padding=16),
            nn.BatchNorm1d(64), nn.ReLU(), nn.MaxPool1d(4))
        self.conv3 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=16, stride=1, padding=8),
            nn.BatchNorm1d(128), nn.ReLU(), nn.AdaptiveAvgPool1d(1))

        # ── BiLSTM (shared) ──────────────────────────────────────────────────
        self.lstm = nn.LSTM(input_size=128, hidden_size=128, num_layers=2,
                            bidirectional=True, batch_first=True, dropout=0.3)

        # ── Attention (shared) ───────────────────────────────────────────────
        self.attention = nn.Sequential(
            nn.Linear(256, 64), nn.Tanh(), nn.Linear(64, 1), nn.Softmax(dim=1))

        # ── Detection path fusion: attended(256) + det_feat(4) ───────────────
        self.det_fusion = nn.Sequential(
            nn.Linear(256 + self.N_DET_FEAT, 128),
            nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.3))
        self.det_fc = nn.Sequential(
            nn.Linear(128, 64), nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(0.2))

        # ── Quality path fusion: attended(256) + qual_feat(2) ────────────────
        self.qual_fusion = nn.Sequential(
            nn.Linear(256 + self.N_QUAL_FEAT, 128),
            nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.3))
        self.qual_fc = nn.Sequential(
            nn.Linear(128, 64), nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(0.2))

        # ── Head 1: binary N3 detector ───────────────────────────────────────
        self.n3_head = nn.Sequential(
            nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 1))

        # ── Head 2: N3 quality regressor ─────────────────────────────────────
        self.quality_head = nn.Sequential(
            nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 1), nn.Sigmoid())

    def _encode(self, raw_eeg):
        """Shared backbone: CNN -> BiLSTM -> attention -> attended (B, 256)."""
        x = self.conv1(raw_eeg)
        x = self.conv2(x)
        x = self.conv3(x)
        cnn_feat    = x.squeeze(-1)
        lstm_out, _ = self.lstm(cnn_feat.unsqueeze(1))
        attn_w      = self.attention(lstm_out)
        attended    = (lstm_out * attn_w).squeeze(1)
        return attended

    def forward(self, raw_eeg, det_feat, qual_feat):
        attended = self._encode(raw_eeg)          # (B, 256)

        # Detection path
        d = det_feat if self.use_det_feat else torch.zeros_like(det_feat)
        det_fused = self.det_fusion(torch.cat([attended, d], dim=1))
        det_out   = self.det_fc(det_fused)

        # Quality path
        q = qual_feat if self.use_qual_feat else torch.zeros_like(qual_feat)
        qual_fused = self.qual_fusion(torch.cat([attended, q], dim=1))
        qual_out   = self.qual_fc(qual_fused)

        return self.n3_head(det_out), self.quality_head(qual_out)


# ============================================================================
# 5. MULTI-TASK LOSS
#
#   L_cls  : BCEWithLogitsLoss(pos_weight)  — N3 upweighting (RQ3)
#   L_qual : SmoothL1 on N3 epochs only     — quality regression (RQ1)
#   L_phys : delta power <-> quality corr   — physiological consistency (RQ2)
# ============================================================================

class MultiTaskLoss(nn.Module):

    def __init__(self, lambda_cls=1.0, lambda_qual=0.5, lambda_phys=0.2,
                 n3_pos_weight=2.0):
        super().__init__()
        self.lambda_cls  = lambda_cls
        self.lambda_qual = lambda_qual
        self.lambda_phys = lambda_phys
        self.register_buffer('pos_weight',
                             torch.tensor([n3_pos_weight], dtype=torch.float32))

    def forward(self, n3_logit, quality_pred, n3_labels, quality_labels, det_feat):
        bce   = nn.BCEWithLogitsLoss(pos_weight=self.pos_weight.to(n3_logit.device))
        L_cls = bce(n3_logit, n3_labels.float().unsqueeze(1))

        n3_mask = (n3_labels == 1)

        if n3_mask.any():
            L_qual = F.smooth_l1_loss(quality_pred[n3_mask],
                                       quality_labels[n3_mask])
        else:
            L_qual = torch.tensor(0.0, device=n3_logit.device)

        # L_phys: delta power (det_feat[:,0]) should correlate with quality on N3
        if n3_mask.sum() > 1:
            dp_n3 = det_feat[n3_mask, 0:1]
            qp_n3 = quality_pred[n3_mask]
            d = (dp_n3 - dp_n3.mean()) / (dp_n3.std() + 1e-8)
            q = (qp_n3 - qp_n3.mean()) / (qp_n3.std() + 1e-8)
            L_phys = -torch.mean(d * q)
        else:
            L_phys = torch.tensor(0.0, device=n3_logit.device)

        L_total = (self.lambda_cls*L_cls + self.lambda_qual*L_qual
                   + self.lambda_phys*L_phys)
        return L_total, L_cls, L_qual, L_phys


# ============================================================================
# 6. TRAINER
# ============================================================================

class SleepQualityTrainer:

    def __init__(self, model, device=None, checkpoint_path='best_model.pth',
                 n3_pos_weight=2.0):
        self.device    = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model     = model.to(self.device)
        self.ckpt      = checkpoint_path
        self.criterion = MultiTaskLoss(n3_pos_weight=n3_pos_weight)
        self.optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=10)
        self.train_losses, self.val_losses = [], []
        self.train_accs,   self.val_accs   = [], []

    @staticmethod
    def _to_binary(stage_labels):
        return (stage_labels == 3).long()

    def train_epoch(self, train_loader, verbose=True):
        self.model.train()
        total_loss = 0.0
        all_preds, all_labels = [], []
        for i, (raw_eeg, det_feat, qual_feat, stage_labels, quality_labels) in enumerate(train_loader):
            raw_eeg        = raw_eeg.to(self.device)
            det_feat       = det_feat.to(self.device)
            qual_feat      = qual_feat.to(self.device)
            n3_labels      = self._to_binary(stage_labels).to(self.device)
            quality_labels = quality_labels.to(self.device)

            n3_logit, quality_pred = self.model(raw_eeg, det_feat, qual_feat)
            loss, l_cls, l_qual, l_phys = self.criterion(
                n3_logit, quality_pred, n3_labels, quality_labels, det_feat)

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

            total_loss += loss.item()
            preds = (torch.sigmoid(n3_logit).squeeze(1) > 0.5).long().cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(n3_labels.cpu().numpy())
            if verbose and (i+1) % 10 == 0:
                print(f"    batch {i+1:>3}: loss={loss.item():.4f} "
                      f"[cls={l_cls.item():.4f} qual={l_qual.item():.4f} "
                      f"phys={l_phys.item():.4f}]")

        avg = total_loss / len(train_loader)
        self.train_losses.append(avg)
        self.train_accs.append(accuracy_score(all_labels, all_preds))
        return avg

    def evaluate(self, val_loader):
        self.model.eval()
        total_loss = 0.0
        n3_preds, n3_true, n3_probs = [], [], []
        q_preds, q_true = [], []

        with torch.no_grad():
            for raw_eeg, det_feat, qual_feat, stage_labels, quality_labels in val_loader:
                raw_eeg        = raw_eeg.to(self.device)
                det_feat       = det_feat.to(self.device)
                qual_feat      = qual_feat.to(self.device)
                n3_labels      = self._to_binary(stage_labels).to(self.device)
                quality_labels = quality_labels.to(self.device)

                n3_logit, quality_pred = self.model(raw_eeg, det_feat, qual_feat)
                loss, *_ = self.criterion(
                    n3_logit, quality_pred, n3_labels, quality_labels, det_feat)
                total_loss += loss.item()

                prob = torch.sigmoid(n3_logit).squeeze(1).cpu().numpy()
                pred = (prob > 0.5).astype(int)
                n3_probs.extend(prob)
                n3_preds.extend(pred)
                n3_true.extend(n3_labels.cpu().numpy())

                mask = (n3_labels.cpu().numpy() == 1)
                if mask.any():
                    q_preds.extend(quality_pred.cpu().numpy().flatten()[mask])
                    q_true.extend(quality_labels.cpu().numpy().flatten()[mask])

        avg = total_loss / len(val_loader)
        self.val_losses.append(avg)

        yt       = np.array(n3_true)
        yp       = np.array(n3_preds)
        prob_arr = np.array(n3_probs)
        acc = float((yt == yp).mean())
        self.val_accs.append(acc)

        metrics = {
            'loss'         : avg,
            'accuracy'     : acc,
            'n3_recall'    : recall_score(yt, yp, zero_division=0),
            'n3_precision' : precision_score(yt, yp, zero_division=0),
            'n3_f1'        : f1_score(yt, yp, zero_division=0),
            'auc'          : roc_auc_score(yt, prob_arr) if len(np.unique(yt))>1 else 0.0,
        }
        if q_preds:
            qt = np.array(q_true); qp = np.array(q_preds)
            metrics['quality_rmse'] = float(np.sqrt(mean_squared_error(qt, qp)))
            metrics['quality_r2']   = float(r2_score(qt, qp))
        else:
            metrics['quality_rmse'] = 0.0
            metrics['quality_r2']   = 0.0
        return metrics

    def train(self, train_loader, val_loader, epochs=50, patience=15, verbose=True):
        best_val = float('inf')
        wait = 0
        for epoch in range(epochs):
            if verbose:
                print(f"\nEpoch {epoch+1}/{epochs}")
            tr_loss = self.train_epoch(train_loader, verbose=verbose)
            m = self.evaluate(val_loader)
            print(f"  loss={tr_loss:.4f}/{m['loss']:.4f}  "
                  f"N3-Recall={m['n3_recall']:.3f}  "
                  f"N3-F1={m['n3_f1']:.3f}  AUC={m['auc']:.3f}  "
                  f"Q-RMSE={m['quality_rmse']:.3f}")
            self.scheduler.step(m['loss'])
            if m['loss'] < best_val:
                best_val = m['loss']; wait = 0
                torch.save(self.model.state_dict(), self.ckpt)
            else:
                wait += 1
                if wait >= patience:
                    print(f"  early stopping at epoch {epoch+1}")
                    break
        return self.model


# ============================================================================
# 7. DATA LOADER
# ============================================================================

class SleepDataLoader:

    STAGE_MAP = {'W':0,'N1':1,'N2':2,'N3':3,'R':4,'REM':4}

    @staticmethod
    def numpy_to_tensor_dataset(raw_eegs, det_features, qual_features,
                                 stage_labels, quality_labels):
        """Wrap arrays into TensorDataset.
        det_features : (N, 4) — detection features
        qual_features: (N, 2) — quality features
        """
        return torch.utils.data.TensorDataset(
            torch.FloatTensor(raw_eegs).unsqueeze(1),
            torch.FloatTensor(det_features),
            torch.FloatTensor(qual_features),
            torch.LongTensor(stage_labels),
            torch.FloatTensor(quality_labels))

    @staticmethod
    def create_synthetic_dataset(num_samples=600, sampling_rate=100,
                                  epoch_duration=30, seed=42):
        rng = np.random.default_rng(seed)
        n   = sampling_rate * epoch_duration
        t   = np.arange(n) / sampling_rate
        pre = EEGPreprocessor(sampling_rate=sampling_rate,
                              epoch_duration=epoch_duration)
        eegs, det_list, qual_list, stages = [], [], [], []
        for _ in range(num_samples):
            stage = int(rng.integers(0, 5))
            x = np.zeros(n)
            if   stage==0: x += 20*np.sin(2*np.pi*10*t)+15*np.sin(2*np.pi*8*t)
            elif stage==1: x += 30*np.sin(2*np.pi*5*t)+10*np.sin(2*np.pi*1*t)
            elif stage==2: x += 40*np.sin(2*np.pi*2*t)+20*np.sin(2*np.pi*13*t)
            elif stage==3: x += 80*np.sin(2*np.pi*0.8*t)+40*np.sin(2*np.pi*1.5*t)
            else:          x += 15*np.sin(2*np.pi*6*t)+10*np.sin(2*np.pi*12*t)
            x += rng.normal(0, 5, n)
            x = x.astype(np.float32)
            det, qual = pre.extract_all_features(x)
            eegs.append(x)
            det_list.append(det)
            qual_list.append(qual)
            stages.append(stage)
        eegs   = np.array(eegs,     dtype=np.float32)
        det_f  = np.array(det_list, dtype=np.float32)
        qual_f = np.array(qual_list,dtype=np.float32)
        stages = np.array(stages,   dtype=np.int64)
        quality = make_n3_quality_labels(det_f, stages)
        return eegs, det_f, qual_f, stages, quality
