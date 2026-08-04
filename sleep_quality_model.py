


import torch

import torch.nn as nn

import torch.nn.functional as F

import numpy as np

import math

from scipy.signal import butter, sosfilt, sosfiltfilt, find_peaks

from sklearn.metrics import (accuracy_score, f1_score, precision_score,

                              recall_score, roc_auc_score,

                              mean_squared_error, r2_score)



try:

    import mne

    _HAS_MNE = True

except Exception:

    _HAS_MNE = False





# ============================================================================

# 1. PREPROCESSING  (FIX: no double-clipping, robust IQR norm)

# ============================================================================



class EEGPreprocessor:

    """EEG preprocessing + 6-feature extraction for N3 detection and quality."""



    def __init__(self, sampling_rate=200, epoch_duration=30):

        self.sampling_rate  = sampling_rate

        self.epoch_duration = epoch_duration

        self.epoch_samples  = sampling_rate * epoch_duration



    def bandpass_filter(self, signal_data, low_freq=0.5, high_freq=40):

        sos = butter(4, [low_freq, high_freq], btype='band',

                     fs=self.sampling_rate, output='sos')

        return sosfiltfilt(sos, signal_data)



    def extract_frequency_bands(self, eeg_epoch):

        bands = {'delta':(0.5,4),'theta':(4,8),'alpha':(8,12),

                 'sigma':(12,16),'beta':(16,30)}

        out = {}

        for name,(lo,hi) in bands.items():

            sos = butter(4,[lo,hi],btype='band',fs=self.sampling_rate,output='sos')

            out[name] = float(np.mean(sosfiltfilt(sos, eeg_epoch)**2))

        return out



    def extract_delta_power(self, eeg_epoch):

        """Feature 1: Delta band power (0.5-4 Hz)."""

        sos = butter(4, [0.5, 4], btype='band', fs=self.sampling_rate, output='sos')

        filtered = sosfiltfilt(sos, eeg_epoch)

        return float(np.mean(filtered ** 2))



    def extract_slow_wave_activity(self, eeg_epoch):

        """Feature 2: Slow Wave Activity - RMS amplitude of delta band."""

        sos = butter(4, [0.5, 4], btype='band', fs=self.sampling_rate, output='sos')

        filtered = sosfiltfilt(sos, eeg_epoch)

        return float(np.sqrt(np.mean(filtered ** 2)))



    def extract_slow_wave_density(self, eeg_epoch, amp_threshold_uv=75.0):

        """Feature 3: Slow Wave Density - AASM clinical standard."""

        sos = butter(4, [0.5, 2], btype='band', fs=self.sampling_rate, output='sos')

        filtered = sosfiltfilt(sos, eeg_epoch)

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

        """Feature 4: Relative delta power = delta / total power."""

        bands = self.extract_frequency_bands(eeg_epoch)

        total = sum(bands.values()) + 1e-8

        return float(bands['delta'] / total)



    def extract_peak_to_peak_amplitude(self, eeg_epoch):

        """Quality Feature 1: Peak-to-peak amplitude of slow waves."""

        sos = butter(4, [0.5, 4], btype='band', fs=self.sampling_rate, output='sos')

        filtered = sosfiltfilt(sos, eeg_epoch)

        return float(np.max(filtered) - np.min(filtered))



    def extract_fragmentation_index(self, eeg_epoch):

        """Quality Feature 2: Fragmentation index = zero crossing rate."""

        zero_crossings = np.sum(np.diff(np.sign(eeg_epoch)) != 0)

        return float(zero_crossings / len(eeg_epoch))



    def extract_all_features(self, eeg_epoch):

        """Extract all 6 features. Returns (det_feat, qual_feat)."""

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



    def notch_filter(self, signal_data, freq=50, Q=30):

        """Remove powerline noise (50 or 60 Hz)."""

        from scipy.signal import iirnotch, tf2sos, sosfiltfilt

        b, a = iirnotch(freq, Q, fs=self.sampling_rate)

        sos = tf2sos(b, a)

        return sosfiltfilt(sos, signal_data)



    def detect_artifacts(self, eeg_epoch, threshold_std=5.0, threshold_amp_uv=300):

        """Detect EMG, movement, and electrode artifacts."""

        from scipy.signal import butter, sosfiltfilt

        peak_amplitude = np.max(np.abs(eeg_epoch))

        if peak_amplitude > threshold_amp_uv:

            return True, 0.9

        z_scores = np.abs((eeg_epoch - np.mean(eeg_epoch)) / (np.std(eeg_epoch) + 1e-8))

        if np.max(z_scores) > threshold_std:

            return True, 0.8

        sos_hf = butter(4, [30, 40], btype='band', fs=self.sampling_rate, output='sos')

        hf_signal = sosfiltfilt(sos_hf, eeg_epoch)

        hf_power = np.mean(hf_signal ** 2)

        sos_delta = butter(4, [0.5, 4], btype='band', fs=self.sampling_rate, output='sos')

        delta_signal = sosfiltfilt(sos_delta, eeg_epoch)

        delta_power = np.mean(delta_signal ** 2)

        if hf_power > 0.5 * delta_power:

            return True, 0.7

        return False, 0.0



    def clip_outliers(self, eeg_epoch, n_std=4.0):

        """Clip values beyond n standard deviations."""

        mean = np.mean(eeg_epoch)

        std = np.std(eeg_epoch)

        lower_bound = mean - n_std * std

        upper_bound = mean + n_std * std

        clipped = np.clip(eeg_epoch, lower_bound, upper_bound)

        n_clipped = np.sum((eeg_epoch < lower_bound) | (eeg_epoch > upper_bound))

        return clipped, n_clipped



    def compute_signal_quality(self, eeg_epoch):

        """Compute signal quality score (0-1)."""

        from scipy.signal import butter, sosfiltfilt

        rms = np.sqrt(np.mean(eeg_epoch ** 2))

        flatness_score = 1.0 if rms < 1.0 else 0.0

        sos_lf = butter(4, [0.5, 15], btype='band', fs=self.sampling_rate, output='sos')

        sos_hf = butter(4, [30, 40], btype='band', fs=self.sampling_rate, output='sos')

        lf_signal = sosfiltfilt(sos_lf, eeg_epoch)

        hf_signal = sosfiltfilt(sos_hf, eeg_epoch)

        lf_power = np.mean(lf_signal ** 2)

        hf_power = np.mean(hf_signal ** 2)

        noise_ratio = hf_power / (lf_power + 1e-8)

        sos_delta = butter(4, [0.5, 4], btype='band', fs=self.sampling_rate, output='sos')

        sos_hf2 = butter(4, [30, 45], btype='band', fs=self.sampling_rate, output='sos')

        delta = sosfiltfilt(sos_delta, eeg_epoch)

        hf_noise = sosfiltfilt(sos_hf2, eeg_epoch)

        delta_power = np.mean(delta ** 2)

        noise_power = np.mean(hf_noise ** 2)

        snr = 10 * np.log10(delta_power / (noise_power + 1e-8))

        quality_score = 1.0

        if noise_ratio > 0.3:

            quality_score -= 0.3

        if snr < 3:

            quality_score -= 0.3

        if flatness_score > 0.5:

            quality_score -= 0.4

        quality_score = np.clip(quality_score, 0, 1)

        return {'flatness': flatness_score, 'noise_ratio': noise_ratio, 'snr_db': snr,

                'quality_score': quality_score, 'is_good': quality_score > 0.6}



    def normalize_epoch(self, eeg_epoch):

        """Robust normalization using median & IQR."""

        q75, q25 = np.percentile(eeg_epoch, [75, 25])

        iqr = q75 - q25

        if iqr == 0:

            iqr = 1.0

        median = np.median(eeg_epoch)

        return (eeg_epoch - median) / iqr



    def create_epochs(self, raw_signal, labels=None, remove_artifacts=True,

                      remove_low_quality=True, notch_freq=50):

        """Create epochs with ROBUST preprocessing pipeline."""

        epochs, det_feats, qual_feats, epoch_labels = [], [], [], []

        n_removed = {'artifact': 0, 'quality': 0, 'clipped_too_much': 0}

        n = len(raw_signal) // self.epoch_samples

        for i in range(n):

            start = i * self.epoch_samples

            ep = raw_signal[start:start+self.epoch_samples]

            ep = self.notch_filter(ep, freq=notch_freq)

            ep = self.bandpass_filter(ep)

            ep, n_clipped = self.clip_outliers(ep, n_std=4.0)

            if n_clipped > 100:

                n_removed['clipped_too_much'] += 1

                continue

            if remove_artifacts:

                is_artifact, score = self.detect_artifacts(ep)

                if is_artifact:

                    n_removed['artifact'] += 1

                    continue

            quality = self.compute_signal_quality(ep)

            if remove_low_quality and not quality['is_good']:

                n_removed['quality'] += 1

                continue

            det, qual = self.extract_all_features(ep)

            ep = self.normalize_epoch(ep)

            epochs.append(ep)

            det_feats.append(det)

            qual_feats.append(qual)

            if labels is not None and i < len(labels):

                epoch_labels.append(labels[i])

        print("\n" + "="*70)

        print("PREPROCESSING QUALITY REPORT")

        print("="*70)

        print(f"Total epochs attempted:        {n:,}")

        print(f"Artifact epochs removed:       {n_removed['artifact']:,}")

        print(f"Low-quality epochs removed:    {n_removed['quality']:,}")

        print(f"Heavily clipped epochs removed: {n_removed['clipped_too_much']:,}")

        print("-"*70)

        print(f"Final epochs retained:         {len(epochs):,}")

        if n > 0:

            print(f"Retention rate:                {100*len(epochs)/n:.1f}%")

        print("="*70 + "\n")

        return (np.array(epochs), np.array(det_feats),

                np.array(qual_feats), np.array(epoch_labels))





# ============================================================================

# 2. DATA AUGMENTATION

# ============================================================================



class EEGAugmentor:

    """Realistic EEG augmentations that PRESERVE slow-wave (N3) features."""



    def __init__(self, sampling_rate=200):

        self.sr = sampling_rate



    def __call__(self, raw_eeg):

        """raw_eeg: (B, 1, L) tensor - returns augmented tensor of same shape."""

        aug = raw_eeg.clone()



        # 1. Gaussian noise injection (lower std for stability)

        if torch.rand(1).item() < 0.4:

            noise_std = torch.rand(1).item() * 0.02  # FIX: reduced from 0.03

            aug = aug + torch.randn_like(aug) * noise_std * aug.std()



        # 2. Random time shift (+/-0.5 s) with REFLECTION padding

        if torch.rand(1).item() < 0.25:  # FIX: reduced prob from 0.3

            shift = int(torch.randint(-100, 100, (1,)).item())

            L = aug.shape[-1]

            if shift > 0:

                aug = F.pad(aug[..., :-shift], (0, shift), mode='reflect')

            elif shift < 0:

                aug = F.pad(aug[..., -shift:], (-shift, 0), mode='reflect')



        # 3. Amplitude scaling (narrower range for stability)

        if torch.rand(1).item() < 0.25:  # FIX: reduced prob from 0.3

            scale = 0.90 + 0.20 * torch.rand(1).item()  # FIX: narrower 0.90-1.10

            aug = aug * scale



        return aug





# ============================================================================

# 3. MODEL — Dual-path CNN-Attention (FIXED: LayerNorm, warmup-ready, EMA)

# ============================================================================



class SleepQualityNet(nn.Module):

    """Dual-path multi-feature N3 detector + quality regressor.



    FIX v7 architecture changes:

      * Added LayerNorm after each conv block (more stable than only BatchNorm

        with small effective batches and subject-level split).

      * Kept the V6 reduced channel sizes (16->32->48).

      * Kept constrained quality head (scaled Sigmoid in [0.02, 0.98]).

      * Removed LSTM (same as V6).

    """



    N_DET_FEAT  = 4

    N_QUAL_FEAT = 2



    def __init__(self, input_length=6000, use_det_feat=True, use_qual_feat=True):

        super().__init__()

        self.use_det_feat  = use_det_feat

        self.use_qual_feat = use_qual_feat



        # CNN backbone (shared)

        self.conv1 = nn.Sequential(

            nn.Conv1d(1, 16, kernel_size=64, stride=2, padding=32),

            nn.BatchNorm1d(16), nn.ReLU(), nn.MaxPool1d(4))

        # FIX: LayerNorm over the channel dim for gradient stability

        self.ln1 = nn.LayerNorm(16)



        self.conv2 = nn.Sequential(

            nn.Conv1d(16, 32, kernel_size=32, stride=2, padding=16),

            nn.BatchNorm1d(32), nn.ReLU(), nn.MaxPool1d(4))

        self.ln2 = nn.LayerNorm(32)



        self.conv3 = nn.Sequential(

            nn.Conv1d(32, 48, kernel_size=16, stride=1, padding=8),

            nn.BatchNorm1d(48), nn.ReLU(), nn.AdaptiveAvgPool1d(32))

        self.ln3 = nn.LayerNorm(48)



        # Attention (shared)

        self.attention = nn.Sequential(

            nn.Linear(48, 32), nn.Tanh(), nn.Linear(32, 1), nn.Softmax(dim=1))



        # Detection path fusion

        self.det_fusion = nn.Sequential(

            nn.Linear(48 + self.N_DET_FEAT, 64),

            nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(0.4))

        self.det_fc = nn.Sequential(

            nn.Linear(64, 32), nn.BatchNorm1d(32), nn.ReLU(), nn.Dropout(0.3))



        # Quality path fusion

        self.qual_fusion = nn.Sequential(

            nn.Linear(48 + self.N_QUAL_FEAT, 64),

            nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(0.4))

        self.qual_fc = nn.Sequential(

            nn.Linear(64, 32), nn.BatchNorm1d(32), nn.ReLU(), nn.Dropout(0.3))



        # Head 1: binary N3 detector

        self.n3_head = nn.Sequential(

            nn.Linear(32, 16), nn.ReLU(), nn.Linear(16, 1))



        # Head 2: N3 quality regressor (constrained to [0.02, 0.98])

        self.quality_head_inner = nn.Sequential(

            nn.Linear(32, 16), nn.ReLU(), nn.Linear(16, 1))

        self.quality_temp = nn.Parameter(torch.tensor(1.0))



    def _encode(self, raw_eeg):

        """Shared backbone: CNN -> LayerNorm -> attention -> attended (B, 48)."""

        x = self.conv1(raw_eeg)  # (B, 16, T1)

        x = x.permute(0, 2, 1)   # (B, T1, 16) for LayerNorm

        x = self.ln1(x)

        x = x.permute(0, 2, 1)   # (B, 16, T1) back



        x = self.conv2(x)         # (B, 32, T2)

        x = x.permute(0, 2, 1)

        x = self.ln2(x)

        x = x.permute(0, 2, 1)



        x = self.conv3(x)         # (B, 48, 32)

        x = x.permute(0, 2, 1)

        x = self.ln3(x)

        x = x.permute(0, 2, 1)   # (B, 48, 32)



        cnn_feat = x.permute(0, 2, 1)  # (B, 32, 48)

        attn_w   = self.attention(cnn_feat)        # (B, 32, 1)

        attended = (cnn_feat * attn_w).sum(dim=1)  # (B, 48)

        return attended



    def forward(self, raw_eeg, det_feat, qual_feat):

        attended = self._encode(raw_eeg)

        d = det_feat if self.use_det_feat else torch.zeros_like(det_feat)

        det_fused = self.det_fusion(torch.cat([attended, d], dim=1))

        det_out   = self.det_fc(det_fused)

        q = qual_feat if self.use_qual_feat else torch.zeros_like(qual_feat)

        qual_fused = self.qual_fusion(torch.cat([attended, q], dim=1))

        qual_out   = self.qual_fc(qual_fused)

        temp = F.softplus(self.quality_temp) + 0.5

        q_logit = self.quality_head_inner(qual_out) / temp

        q_pred  = 0.02 + 0.96 * torch.sigmoid(q_logit)

        return self.n3_head(det_out), q_pred





# ============================================================================

# 3b. LOG-TRANSFORM + WINSORIZATION (same as V5)

# ============================================================================



LOG_TRANSFORM_DET_INDICES  = [0, 1]

LOG_TRANSFORM_QUAL_INDICES = [0]





def log_transform_features(det_features, qual_features):

    det_features  = np.asarray(det_features,  dtype=np.float32).copy()

    qual_features = np.asarray(qual_features, dtype=np.float32).copy()

    for i in LOG_TRANSFORM_DET_INDICES:

        det_features[:, i] = np.log1p(np.clip(det_features[:, i], 0, None))

    for i in LOG_TRANSFORM_QUAL_INDICES:

        qual_features[:, i] = np.log1p(np.clip(qual_features[:, i], 0, None))

    return det_features, qual_features





class Winsorizer:

    def __init__(self, k=5.0):

        self.k = k

        self.det_lo = self.det_hi = None

        self.qual_lo = self.qual_hi = None



    def fit(self, det_features, qual_features):

        det_features  = np.asarray(det_features,  dtype=np.float32)

        qual_features = np.asarray(qual_features, dtype=np.float32)

        det_median = np.median(det_features, axis=0)

        det_q75, det_q25 = np.percentile(det_features, [75, 25], axis=0)

        det_iqr = det_q75 - det_q25 + 1e-8

        self.det_lo = det_median - self.k * det_iqr

        self.det_hi = det_median + self.k * det_iqr

        qual_median = np.median(qual_features, axis=0)

        qual_q75, qual_q25 = np.percentile(qual_features, [75, 25], axis=0)

        qual_iqr = qual_q75 - qual_q25 + 1e-8

        self.qual_lo = qual_median - self.k * qual_iqr

        self.qual_hi = qual_median + self.k * qual_iqr

        return self



    def transform(self, det_features, qual_features):

        if self.det_lo is None:

            raise RuntimeError("Winsorizer.fit() must be called before transform()")

        det_features  = np.asarray(det_features,  dtype=np.float32)

        qual_features = np.asarray(qual_features, dtype=np.float32)

        det_clipped  = np.clip(det_features,  self.det_lo,  self.det_hi)

        qual_clipped = np.clip(qual_features, self.qual_lo, self.qual_hi)

        return det_clipped, qual_clipped





def clip_features_outliers(det_features, qual_features, lower_pct=2, upper_pct=98):

    det_features  = np.asarray(det_features,  dtype=np.float32)

    qual_features = np.asarray(qual_features, dtype=np.float32)

    det_lo  = np.percentile(det_features,  lower_pct, axis=0)

    det_hi  = np.percentile(det_features,  upper_pct, axis=0)

    qual_lo = np.percentile(qual_features, lower_pct, axis=0)

    qual_hi = np.percentile(qual_features, upper_pct, axis=0)

    det_clipped  = np.clip(det_features,  det_lo,  det_hi)

    qual_clipped = np.clip(qual_features, qual_lo, qual_hi)

    clip_bounds = {'det_lo': det_lo, 'det_hi': det_hi,

                   'qual_lo': qual_lo, 'qual_hi': qual_hi}

    return det_clipped, qual_clipped, clip_bounds





def apply_clip_bounds(det_features, qual_features, clip_bounds):

    det_features  = np.asarray(det_features,  dtype=np.float32)

    qual_features = np.asarray(qual_features, dtype=np.float32)

    det_clipped  = np.clip(det_features,  clip_bounds['det_lo'],  clip_bounds['det_hi'])

    qual_clipped = np.clip(qual_features, clip_bounds['qual_lo'], clip_bounds['qual_hi'])

    return det_clipped, qual_clipped





# ============================================================================

# 4. MULTI-TASK LOSS  (FIX: removed variance penalty, added label smoothing)

# ============================================================================



class MultiTaskLoss(nn.Module):

    """FIX v7 multi-task loss:



    Changes vs V5:

      1. VARIANCE PENALTY REMOVED — it caused loss fluctuation because

         batch composition (N3 vs non-N3 ratio) changes every epoch due

         to shuffle=True. The penalty magnitude was inconsistent across

         batches, creating noisy gradients.

      2. LABEL SMOOTHING for BCE — instead of hard 0/1 targets, use

         0.05/0.95. This prevents the model from becoming overconfident

         early (which caused the "best at epoch 4" problem).

      3. Quality loss weight reduced to 0.3 (from 0.5) — the quality head

         was competing too strongly with detection, causing the detection

         loss to oscillate as the quality loss swung.

    """



    def __init__(self, lambda_cls=1.0, lambda_qual=0.3, n3_pos_weight=2.0,

                 non_n3_qual_weight=0.2, label_smooth=0.05):

        super().__init__()

        self.lambda_cls  = lambda_cls

        self.lambda_qual = lambda_qual

        self.non_n3_qual_weight = non_n3_qual_weight

        self.label_smooth = label_smooth

        self.register_buffer('pos_weight',

                             torch.tensor([n3_pos_weight], dtype=torch.float32))



    def forward(self, n3_logit, quality_pred, n3_labels, quality_labels, det_feat):

        # FIX: Label-smoothed BCE targets

        n3_target = n3_labels.float().unsqueeze(1)

        n3_target_smooth = n3_target * (1.0 - self.label_smooth) + self.label_smooth * (1.0 - n3_target)



        bce   = nn.BCEWithLogitsLoss(pos_weight=self.pos_weight.to(n3_logit.device))

        L_cls = bce(n3_logit, n3_target_smooth)



        # Quality loss on ALL epochs, weighted by N3 vs non-N3

        n3_mask     = (n3_labels == 1)

        non_n3_mask = (~n3_mask)

        target      = quality_labels.unsqueeze(1)

        per_sample_sq_err = (quality_pred - target) ** 2



        L_qual_n3     = per_sample_sq_err[n3_mask].mean() if n3_mask.any()     else torch.tensor(0.0, device=n3_logit.device)

        L_qual_non_n3 = per_sample_sq_err[non_n3_mask].mean() if non_n3_mask.any() else torch.tensor(0.0, device=n3_logit.device)

        L_qual = L_qual_n3 + self.non_n3_qual_weight * L_qual_non_n3



        # FIX: No variance penalty — it was the #1 cause of loss fluctuation



        L_total = self.lambda_cls * L_cls + self.lambda_qual * L_qual

        return L_total, L_cls, L_qual, torch.tensor(0.0, device=n3_logit.device)





# ============================================================================

# 5. EMA (Exponential Moving Average) — FIX for smooth loss curves

# ============================================================================



class ModelEMA:

    """Exponential Moving Average of model parameters.



    FIX: EMA smooths the validation loss curve by evaluating on a

    time-averaged version of the weights instead of the last mini-batch

    update. This is the single most effective technique for reducing

    loss fluctuation in small-data regimes.

    """



    def __init__(self, model, decay=0.999):

        self.decay = decay

        self.shadow = {}

        for name, param in model.named_parameters():

            if param.requires_grad:

                self.shadow[name] = param.data.clone()



    def update(self, model):

        for name, param in model.named_parameters():

            if param.requires_grad:

                self.shadow[name].mul_(self.decay).add_(param.data, alpha=1.0 - self.decay)



    def state_dict(self):

        return self.shadow



    def load_into(self, model):

        model.load_state_dict(self.shadow, strict=False)

        return model





# ============================================================================

# 6. TRAINER (FIX v7: warmup + cosine, EMA, gradient accumulation, no variance penalty)

# ============================================================================



class SleepQualityTrainer:

    """FIX v7 trainer — addresses loss fluctuation and early-epoch overfitting.



    Key fixes:

      1. LR WARMUP for 5 epochs before cosine decay — prevents the aggressive

         initial LR from causing rapid overfitting (the "best at epoch 4" bug).

      2. EMA (Exponential Moving Average) — smooths validation metrics.

      3. Gradient accumulation (effective_batch=96) — reduces batch-to-batch

         variance in the quality loss (fewer N3 per batch = noisier gradients).

      4. No early stopping — runs all epochs; best EMA checkpoint saved.

      5. Higher weight_decay default (1e-2) — stronger L2 regularization

         prevents the model from memorizing training subjects too fast.

    """



    def __init__(self, model, device=None, checkpoint_path='best_model.pth',

                 n3_pos_weight=2.0, augmentor=None,

                 lr=1e-4, weight_decay=1e-2,     # FIX: lower lr, higher wd

                 scheduler='cosine', cosine_T_max=40, cosine_eta_min=1e-6,

                 grad_clip=1.0, lambda_var=0.0,     # FIX: variance penalty OFF

                 warmup_epochs=5,                   # FIX: NEW — warmup

                 ema_decay=0.999,                   # FIX: NEW — EMA

                 accum_steps=2):                    # FIX: NEW — gradient accumulation

        if device is not None:

            self.device = torch.device(device)

        elif torch.cuda.is_available():

            self.device = torch.device('cuda')

        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():

            self.device = torch.device('mps')

        else:

            self.device = torch.device('cpu')

        self.model     = model.to(self.device)

        self.ckpt      = checkpoint_path

        self.criterion = MultiTaskLoss(n3_pos_weight=n3_pos_weight)
        # NOTE: lambda_var removed — MultiTaskLoss no longer has a variance penalty (FIX v7)

        self.optimizer = torch.optim.AdamW(

            model.parameters(), lr=lr, weight_decay=weight_decay)

        self.grad_clip = grad_clip



        # FIX: Warmup + Cosine scheduler

        self.warmup_epochs = warmup_epochs

        self.cosine_T_max = cosine_T_max

        self.cosine_eta_min = cosine_eta_min

        self.base_lr = lr

        self._current_lr = lr



        # Legacy scheduler (kept for compatibility)

        self._sched_is_plateau = False



        # History

        self.train_losses       = []

        self.train_losses_clean = []

        self.val_losses         = []

        self.train_accs         = []

        self.val_accs           = []

        self.train_grad_norms   = []

        self.augmentor          = augmentor

        self.best_epoch         = -1



        # FIX: EMA

        self.ema = ModelEMA(model, decay=ema_decay) if ema_decay > 0 else None



        # FIX: Gradient accumulation

        self.accum_steps = accum_steps



    def _get_lr(self, epoch):

        """FIX: Linear warmup for warmup_epochs, then cosine decay."""

        if epoch < self.warmup_epochs:

            # Linear warmup

            return self.base_lr * (epoch + 1) / self.warmup_epochs

        else:

            # Cosine annealing from warmup end to T_max

            progress = (epoch - self.warmup_epochs) / max(self.cosine_T_max - self.warmup_epochs, 1)

            progress = min(progress, 1.0)

            return self.cosine_eta_min + 0.5 * (self.base_lr - self.cosine_eta_min) * (

                1.0 + math.cos(math.pi * progress))



    @staticmethod

    def _to_binary(stage_labels):

        return (stage_labels == 3).long()



    def train_epoch(self, train_loader, verbose=True):

        self.model.train()

        total_loss       = 0.0

        total_loss_clean = 0.0

        total_samples    = 0

        total_grad_norm  = 0.0

        all_preds_clean, all_labels = [], []



        self.optimizer.zero_grad(set_to_none=True)



        for i, (raw_eeg, det_feat, qual_feat, stage_labels, quality_labels) in enumerate(train_loader):

            raw_eeg        = raw_eeg.to(self.device, non_blocking=True)

            det_feat       = det_feat.to(self.device, non_blocking=True)

            qual_feat      = qual_feat.to(self.device, non_blocking=True)

            n3_labels      = self._to_binary(stage_labels).to(self.device, non_blocking=True)

            quality_labels = quality_labels.to(self.device, non_blocking=True)



            if self.augmentor is not None:

                raw_eeg_aug = self.augmentor(raw_eeg)

            else:

                raw_eeg_aug = raw_eeg



            n3_logit, quality_pred = self.model(raw_eeg_aug, det_feat, qual_feat)

            loss, l_cls, l_qual, l_phys = self.criterion(

                n3_logit, quality_pred, n3_labels, quality_labels, det_feat)



            # FIX: Gradient accumulation — scale loss by accum_steps

            scaled_loss = loss / self.accum_steps

            scaled_loss.backward()



            batch_size = raw_eeg.size(0)

            total_loss += loss.item() * batch_size



            # FIX: Step optimizer and update EMA only every accum_steps

            if (i + 1) % self.accum_steps == 0 or (i + 1) == len(train_loader):

                grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)

                total_grad_norm += float(grad_norm)

                self.optimizer.step()

                self.optimizer.zero_grad(set_to_none=True)

                if self.ema is not None:

                    self.ema.update(self.model)



            # Clean forward pass (no aug)

            with torch.no_grad():

                n3_logit_c, quality_pred_c = self.model(raw_eeg, det_feat, qual_feat)

                loss_c, *_ = self.criterion(

                    n3_logit_c, quality_pred_c, n3_labels, quality_labels, det_feat)

                total_loss_clean += loss_c.item() * batch_size

                preds_c = (torch.sigmoid(n3_logit_c).squeeze(1) > 0.5).long().cpu().numpy()

                all_preds_clean.extend(preds_c)

            stage_np = n3_labels.detach().cpu().numpy()
            all_labels.extend(stage_np)

            total_samples += batch_size



            if verbose and (i+1) % 10 == 0:

                print(f"    batch {i+1:>3}: loss={loss.item():.4f} "

                      f"[cls={l_cls.item():.4f} qual={l_qual.item():.4f}]")



        n_samples = max(total_samples, 1)

        avg       = total_loss / n_samples

        avg_clean = total_loss_clean / n_samples

        avg_grad_norm = total_grad_norm / max(len(train_loader) // self.accum_steps, 1)

        self.train_grad_norms.append(avg_grad_norm)

        self.train_losses.append(avg)

        self.train_losses_clean.append(avg_clean)

        self.train_accs.append(accuracy_score(all_labels, all_preds_clean)

                                if all_labels else 0.0)

        return avg, avg_clean



    def evaluate(self, val_loader, use_ema=False):

        """Evaluate — optionally using EMA weights for smoother metrics."""

        if use_ema and self.ema is not None:

            self.ema.load_into(self.model)

        self.model.eval()

        total_loss = 0.0

        total_samples = 0

        n3_preds, n3_true, n3_probs = [], [], []

        q_preds, q_true = [], []

        with torch.no_grad():

            for raw_eeg, det_feat, qual_feat, stage_labels, quality_labels in val_loader:

                raw_eeg        = raw_eeg.to(self.device, non_blocking=True)

                det_feat       = det_feat.to(self.device, non_blocking=True)

                qual_feat      = qual_feat.to(self.device, non_blocking=True)

                n3_labels      = self._to_binary(stage_labels).to(self.device, non_blocking=True)

                quality_labels = quality_labels.to(self.device, non_blocking=True)

                n3_logit, quality_pred = self.model(raw_eeg, det_feat, qual_feat)

                loss, *_ = self.criterion(

                    n3_logit, quality_pred, n3_labels, quality_labels, det_feat)

                batch_size = raw_eeg.size(0)

                total_loss += loss.item() * batch_size

                total_samples += batch_size

                prob = torch.sigmoid(n3_logit).squeeze(1).cpu().numpy()

                pred = (prob > 0.5).astype(int)

                n3_probs.extend(prob)

                n3_preds.extend(pred)

                n3_np = n3_labels.detach().cpu().numpy()
                n3_true.extend(n3_np)

                mask = (n3_np == 1)

                if mask.any():

                    q_preds.extend(quality_pred.cpu().numpy().flatten()[mask])

                    q_true.extend(quality_labels.cpu().numpy().flatten()[mask])

        avg = total_loss / max(total_samples, 1)

        self.val_losses.append(avg)

        yt       = np.array(n3_true)

        yp       = np.array(n3_preds)

        prob_arr = np.array(n3_probs)

        acc = float((yt == yp).mean())

        self.val_accs.append(acc)

        metrics = {

            'loss': avg, 'accuracy': acc,

            'n3_recall': recall_score(yt, yp, zero_division=0),

            'n3_precision': precision_score(yt, yp, zero_division=0),

            'n3_f1': f1_score(yt, yp, zero_division=0),

            'auc': roc_auc_score(yt, prob_arr) if len(np.unique(yt))>1 else 0.0,

        }

        if q_preds:

            qt = np.array(q_true); qp = np.array(q_preds)

            metrics['quality_rmse'] = float(np.sqrt(mean_squared_error(qt, qp)))

            metrics['quality_r2']   = float(r2_score(qt, qp))

        else:

            metrics['quality_rmse'] = 0.0

            metrics['quality_r2']   = 0.0

        return metrics



    def train(self, train_loader, val_loader, epochs=40, verbose=True):

        """FIX v7 — warmup + cosine LR, EMA evaluation, gradient accumulation."""

        best_val = float('inf')

        for epoch in range(epochs):

            # FIX: Manual LR scheduling (warmup + cosine)

            lr_now = self._get_lr(epoch)

            for pg in self.optimizer.param_groups:

                pg['lr'] = lr_now

            self._current_lr = lr_now



            if verbose:

                print(f"\nEpoch {epoch+1}/{epochs}  lr={lr_now:.2e}"

                      f"{'  [WARMUP]' if epoch < self.warmup_epochs else ''}")

            tr_loss, tr_loss_clean = self.train_epoch(train_loader, verbose=verbose)



            # FIX: Evaluate using EMA weights for smoother validation loss

            m = self.evaluate(val_loader, use_ema=True)



            print(f"  train={tr_loss:.4f}  train(clean)={tr_loss_clean:.4f}  "

                  f"val={m['loss']:.4f}  "

                  f"N3-F1={m['n3_f1']:.3f}  AUC={m['auc']:.3f}  "

                  f"Q-R2={m['quality_r2']:.3f}")



            # Save best EMA checkpoint on every val-loss improvement

            if m['loss'] < best_val:

                best_val = m['loss']

                self.best_epoch = epoch

                # Save EMA weights as the best checkpoint

                if self.ema is not None:

                    self.ema.load_into(self.model)

                torch.save(self.model.state_dict(), self.ckpt)

                if verbose:

                    print(f"  -> new best val loss = {best_val:.4f} (epoch {epoch+1})")



        # Restore best checkpoint at end of training

        import os

        if os.path.exists(self.ckpt):

            self.model.load_state_dict(

                torch.load(self.ckpt, map_location=self.device))

        return self.model



    def plot_training_curves(self, figsize=(10, 6)):

        """Plot train vs val loss + accuracy (IEEE publication quality)."""

        import matplotlib.pyplot as plt

        import os

        if not self.train_losses:

            print("No training history to plot.")

            return

        epochs = np.arange(1, len(self.train_losses)+1)

        fig, axes = plt.subplots(1, 2, figsize=figsize, constrained_layout=True)



        # --- Loss ---

        axes[0].plot(epochs, self.train_losses,        'o-',  color='#1f77b4',

                     label='Train (augmented)',  lw=2.0, ms=6)

        if self.train_losses_clean:

            axes[0].plot(epochs, self.train_losses_clean, 's--', color='#17becf',

                         label='Train (clean)',  lw=1.6, ms=5)

        axes[0].plot(epochs, self.val_losses,          '^-',  color='#d62728',

                     label='Validation',         lw=2.0, ms=6)

        if self.best_epoch >= 0:

            axes[0].axvline(self.best_epoch+1, color='#27ae60', ls=':',

                            lw=1.8, alpha=0.8,

                            label=f'Best epoch={self.best_epoch+1}')

        # FIX: Mark warmup region

        if self.warmup_epochs > 0:

            axes[0].axvspan(0.5, self.warmup_epochs + 0.5, alpha=0.08, color='orange',

                            label=f'Warmup ({self.warmup_epochs} ep)')

        axes[0].set_xlabel('Epoch', fontsize=14)

        axes[0].set_ylabel('Loss', fontsize=14)

        axes[0].set_title('Training vs Validation Loss',

                          fontsize=15, fontweight='bold')

        axes[0].legend(loc='best', fontsize=11)

        axes[0].grid(alpha=0.3, linestyle='--')

        axes[0].tick_params(axis='both', labelsize=12)



        # --- Accuracy ---

        axes[1].plot(epochs, self.train_accs, 'o-', color='#1f77b4',

                     label='Train',  lw=2.0, ms=6)

        axes[1].plot(epochs, self.val_accs,   '^-', color='#d62728',

                     label='Validation', lw=2.0, ms=6)

        if self.best_epoch >= 0:

            axes[1].axvline(self.best_epoch+1, color='#27ae60', ls=':',

                            lw=1.8, alpha=0.8,

                            label=f'Best epoch={self.best_epoch+1}')

        if self.warmup_epochs > 0:

            axes[1].axvspan(0.5, self.warmup_epochs + 0.5, alpha=0.08, color='orange',

                            label=f'Warmup ({self.warmup_epochs} ep)')

        axes[1].set_xlabel('Epoch', fontsize=14)

        axes[1].set_ylabel('Binary N3 Accuracy', fontsize=14)

        axes[1].set_title('Training vs Validation Accuracy',

                          fontsize=15, fontweight='bold')

        axes[1].legend(loc='best', fontsize=11)

        axes[1].grid(alpha=0.3, linestyle='--')

        axes[1].set_ylim(0, 1.05)

        axes[1].tick_params(axis='both', labelsize=12)



        os.makedirs('outputs', exist_ok=True)

        fig.savefig('outputs/training_curves.png', dpi=300,

                    bbox_inches='tight', pad_inches=0.1)

        print(f"  [saved] outputs/training_curves.png")

        plt.show()



        # Print final gap summary

        print()

        if self.train_losses_clean and self.val_losses:

            gap = self.val_losses[-1] - self.train_losses_clean[-1]

            print(f"  Final-epoch loss  | train(clean)={self.train_losses_clean[-1]:.4f}  "

                  f"val={self.val_losses[-1]:.4f}  ->  gap = {gap:+.4f}")

        if self.train_accs and self.val_accs:

            gap = self.val_accs[-1] - self.train_accs[-1]

            print(f"  Final-epoch acc   | train={self.train_accs[-1]:.4f}  "

                  f"val={self.val_accs[-1]:.4f}  ->  gap = {gap:+.4f}")

        if self.best_epoch >= 0:

            print(f"  Best epoch: {self.best_epoch+1}  "

                  f"(val loss = {min(self.val_losses):.4f}, "

                  f"val acc = {self.val_accs[self.best_epoch]:.4f})")

        print("\n  Reading guide:")

        print("   * train(aug) > train(clean)  -> augmentation is hardening training (good)")

        print("   * train(clean) ~ val          -> well-regularized (target)")

        print("   * train(clean) << val         -> overfitting")

        print("   * train(clean) >> val         -> underfitting (raise lr / add capacity)")

        print(f"\n  FIXES APPLIED:")

        print(f"   * Warmup {self.warmup_epochs} epochs + cosine decay (prevents early overfitting)")

        print(f"   * EMA decay=0.999 (smooths validation loss)")

        print(f"   * Gradient accumulation x{self.accum_steps} (reduces batch variance)")

        print(f"   * No variance penalty (was causing loss oscillation)")

        print(f"   * Label smoothing 0.05 (prevents early overconfidence)")





# ============================================================================

# 7. DATA LOADER

# ============================================================================



class SleepDataLoader:



    STAGE_MAP = {'W':0,'N1':1,'N2':2,'N3':3,'R':4,'REM':4}



    @staticmethod

    def numpy_to_tensor_dataset(raw_eegs, det_features, qual_features,

                                 stage_labels, quality_labels):

        return torch.utils.data.TensorDataset(

            torch.from_numpy(np.asarray(raw_eegs, dtype=np.float32)).unsqueeze(1),

            torch.from_numpy(np.asarray(det_features, dtype=np.float32)),

            torch.from_numpy(np.asarray(qual_features, dtype=np.float32)),

            torch.from_numpy(np.asarray(stage_labels, dtype=np.int64)),

            torch.from_numpy(np.asarray(quality_labels, dtype=np.float32)))



    @staticmethod

    def create_synthetic_dataset(num_samples=600, sampling_rate=200,

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

            eegs.append(x); det_list.append(det)

            qual_list.append(qual); stages.append(stage)

        eegs   = np.array(eegs,     dtype=np.float32)

        det_f  = np.array(det_list, dtype=np.float32)

        qual_f = np.array(qual_list,dtype=np.float32)

        stages = np.array(stages,   dtype=np.int64)

        quality = make_n3_quality_labels(det_f, stages)

        return eegs, det_f, qual_f, stages, quality





# ============================================================================

# 8. N3 QUALITY LABEL GENERATION + FEATURE NORMALIZATION (FIX: no double-clipping)

# ============================================================================



def make_n3_quality_labels(det_features, stage_labels, smooth_non_n3=True):

    det_features = np.asarray(det_features, dtype=np.float32)

    stage_labels = np.asarray(stage_labels)

    n3_mask = (stage_labels == 3)

    quality = np.full(len(stage_labels), 0.05, dtype=np.float32)

    if not n3_mask.any():

        return quality

    delta_power = det_features[:, 0]

    swa         = det_features[:, 1]

    raw_score = 0.6 * delta_power + 0.4 * swa

    n3_raw = raw_score[n3_mask]

    lo = float(np.percentile(n3_raw, 5))

    hi = float(np.percentile(n3_raw, 95))

    if hi - lo < 1e-8:

        quality[n3_mask] = 0.5

    else:

        quality[n3_mask] = 0.10 + 0.85 * np.clip((n3_raw - lo) / (hi - lo), 0.0, 1.0)

    if smooth_non_n3:

        non_n3_raw = raw_score[~n3_mask]

        if len(non_n3_raw) > 0:

            nn_lo = float(np.percentile(non_n3_raw, 5))

            nn_hi = float(np.percentile(non_n3_raw, 95))

            if nn_hi - nn_lo > 1e-8:

                nn_score = 0.02 + 0.08 * np.clip((non_n3_raw - nn_lo) / (nn_hi - nn_lo), 0.0, 1.0)

                quality[~n3_mask] = nn_score.astype(np.float32)

    return quality.astype(np.float32)





def fit_quality_label_bounds(det_features, stage_labels, smooth_non_n3=True):

    det_features = np.asarray(det_features, dtype=np.float32)

    stage_labels = np.asarray(stage_labels)

    n3_mask = (stage_labels == 3)

    bounds = {'smooth_non_n3': smooth_non_n3}

    if n3_mask.any():

        raw_score_n3 = 0.6 * det_features[n3_mask, 0] + 0.4 * det_features[n3_mask, 1]

        bounds['n3_lo'] = float(np.percentile(raw_score_n3, 5))

        bounds['n3_hi'] = float(np.percentile(raw_score_n3, 95))

    else:

        bounds['n3_lo'] = 0.0

        bounds['n3_hi'] = 1.0

    if smooth_non_n3 and (~n3_mask).any():

        raw_score_nn = 0.6 * det_features[~n3_mask, 0] + 0.4 * det_features[~n3_mask, 1]

        bounds['nn_lo'] = float(np.percentile(raw_score_nn, 5))

        bounds['nn_hi'] = float(np.percentile(raw_score_nn, 95))

    else:

        bounds['nn_lo'] = 0.0

        bounds['nn_hi'] = 1.0

    return bounds





def apply_quality_label_bounds(det_features, stage_labels, bounds):

    det_features = np.asarray(det_features, dtype=np.float32)

    stage_labels = np.asarray(stage_labels)

    n3_mask = (stage_labels == 3)

    quality = np.full(len(stage_labels), 0.05, dtype=np.float32)

    delta_power = det_features[:, 0]

    swa = det_features[:, 1]

    raw_score = 0.6 * delta_power + 0.4 * swa

    if n3_mask.any():

        n3_raw = raw_score[n3_mask]

        lo, hi = bounds['n3_lo'], bounds['n3_hi']

        if hi - lo < 1e-8:

            quality[n3_mask] = 0.5

        else:

            quality[n3_mask] = 0.10 + 0.85 * np.clip((n3_raw - lo) / (hi - lo), 0.0, 1.0)

    if bounds.get('smooth_non_n3', True) and (~n3_mask).any():

        non_n3_raw = raw_score[~n3_mask]

        if len(non_n3_raw) > 0:

            nn_lo, nn_hi = bounds['nn_lo'], bounds['nn_hi']

            if nn_hi - nn_lo > 1e-8:

                nn_score = 0.02 + 0.08 * np.clip((non_n3_raw - nn_lo) / (nn_hi - nn_lo), 0.0, 1.0)

                quality[~n3_mask] = nn_score.astype(np.float32)

    return quality.astype(np.float32)





class FeatureNormalizer:

    """FIX v7: Z-score normaliser WITHOUT double-clipping.



    Original bug: transform_det/transform_qual clipped at median +/- 2.5*IQR

    INSIDE the transform step. But features were ALREADY clipped + Winsorized

    before calling transform. This double-clipping created gradient

    discontinuities that caused loss fluctuation.



    Fix: Just do (x - median) / IQR. No additional clipping.

    """



    def __init__(self):

        self.det_median  = None

        self.det_iqr     = None

        self.qual_median = None

        self.qual_iqr    = None



    def fit(self, det_features, qual_features):

        det_features  = np.asarray(det_features,  dtype=np.float32)

        qual_features = np.asarray(qual_features, dtype=np.float32)

        self.det_median  = np.median(det_features, axis=0)

        q75, q25 = np.percentile(det_features, [75, 25], axis=0)

        self.det_iqr     = (q75 - q25) + 1e-8

        self.qual_median = np.median(qual_features, axis=0)

        q75, q25 = np.percentile(qual_features, [75, 25], axis=0)

        self.qual_iqr    = (q75 - q25) + 1e-8

        return self



    def transform_det(self, det_features):

        if self.det_median is None:

            raise RuntimeError("FeatureNormalizer.fit() must be called before transform_det()")

        det_features = np.asarray(det_features, dtype=np.float32)

        # FIX: NO double-clipping — features are already preprocessed

        return ((det_features - self.det_median) / self.det_iqr).astype(np.float32)



    def transform_qual(self, qual_features):

        if self.qual_median is None:

            raise RuntimeError("FeatureNormalizer.fit() must be called before transform_qual()")

        qual_features = np.asarray(qual_features, dtype=np.float32)

        # FIX: NO double-clipping — features are already preprocessed

        return ((qual_features - self.qual_median) / self.qual_iqr).astype(np.float32)



    def fit_transform(self, det_features, qual_features):

        self.fit(det_features, qual_features)

        return self.transform_det(det_features), self.transform_qual(qual_features)





# ============================================================================

# 9. INFERENCE HELPERS

# ============================================================================



def clip_quality_predictions(q_pred, lo=0.02, hi=0.98):

    if isinstance(q_pred, torch.Tensor):

        return torch.clamp(q_pred, lo, hi)

    else:

        return np.clip(q_pred, lo, hi)





def compute_prediction_stability_metrics(y_true, y_pred):

    y_true = np.asarray(y_true, dtype=np.float64)

    y_pred = np.asarray(y_pred, dtype=np.float64)

    if len(y_true) == 0:

        return {}

    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

    ss_res = np.sum((y_true - y_pred) ** 2)

    ss_tot = np.sum((y_true - y_true.mean()) ** 2)

    r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0

    corr = float(np.corrcoef(y_true, y_pred)[0, 1]) if len(y_true) > 2 else 0.0

    pred_std = float(y_pred.std())

    t_lo, t_hi = np.percentile(y_true, [5, 95])

    outlier_rate = float(np.mean((y_pred < t_lo) | (y_pred > t_hi)))

    extreme_rate = float(np.mean((y_pred < 5) | (y_pred > 95)))

    true_iqr = float(np.subtract(*np.percentile(y_true, [75, 25])))

    pred_iqr = float(np.subtract(*np.percentile(y_pred, [75, 25])))

    iqr_ratio = float(pred_iqr / true_iqr) if true_iqr > 0 else 0.0

    return {

        'rmse': rmse, 'r2': r2, 'corr': corr,

        'pred_std': pred_std,

        'outlier_rate': outlier_rate,

        'extreme_rate': extreme_rate,

        'iqr_ratio': iqr_ratio,

    }

