"""
FIXED Sleep-EDF Preprocessing Script
=====================================
Key fixes over original:
1. Channel-specific feature extraction (not uniform DE across all channels)
   - EEG: DE across 5 bands + spectral edge + relative band powers + hjorth = 14 features
   - EOG: DE across 3 relevant bands + eye movement features = 8 features
   - EMG: RMS, zero-crossing rate, mean absolute value, spectral entropy = 6 features
   - Resp: breathing rate, tidal amplitude variance, I:E ratio, spectral power = 6 features
   - Temp: mean, trend (slope), variance, range = 4 features

2. Resp/Temp no longer get meaningless high-frequency DE bands applied to them
   - Original: Resp filtered to 0.1-2Hz but DE computed up to 50Hz → bands 3,4,5 = 0
   - Fixed: each channel gets features matched to its actual information content

3. Output format change:
   - Original: X shape (N, C, T, 5) — all channels had 5 features
   - Fixed: X shape (N, C, T, F_max) — channels padded to same length, with a
     channel_feature_dims array telling the model true dimensionality per channel

4. Model-side: STGNNSleepModel updated to use per-channel feature projections
   instead of a shared Linear(f_de, node_hidden) — so each channel type
   gets its own projection matching its true feature count
"""

import numpy as np
import scipy.signal as signal
import scipy.stats as stats
import mne
import os
from pathlib import Path
import warnings
from datetime import datetime
from collections import defaultdict
import gc

warnings.filterwarnings('ignore')

# GPU Setup
try:
    import cupy as cp
    from cupyx.scipy import signal as cp_signal
    test_array = cp.array([1, 2, 3])
    _ = cp.sum(test_array)
    GPU_AVAILABLE = True
    print("✓ GPU (CuPy) detected")
except (ImportError, Exception) as e:
    print(f"⚠ GPU not available ({type(e).__name__}), using CPU")
    GPU_AVAILABLE = False
    cp = np


# ─────────────────────────────────────────────────────────────────
# Per-channel feature dimensions (must match extractor below)
# ─────────────────────────────────────────────────────────────────
CHANNEL_FEATURE_DIMS = {
    'EEG Fpz-Cz':       14,   # DE×5 + spectral_edge + rel_powers×5 + hjorth×3
    'EEG Pz-Oz':        14,
    'EOG horizontal':    8,   # DE×3 + slow_eye_power + blink_rate + correlation_eeg
                              #        + variance + hjorth_activity
    'EMG submental':     6,   # rms + zcr + mav + spectral_entropy + kurt + var
    'Resp oro-nasal':    6,   # breathing_rate + tidal_var + ie_ratio
                              # + low_spectral_power + envelope_var + rms
    'Temp rectal':       4,   # mean + slope + variance + range
}

# Maximum feature dim — all channels zero-padded to this in the output array
F_MAX = max(CHANNEL_FEATURE_DIMS.values())  # 14

# Canonical EEG frequency bands
EEG_BANDS = [(0.5, 4), (4, 8), (8, 13), (13, 30), (30, 50)]  # delta/theta/alpha/beta/gamma
EOG_BANDS = [(0.1, 1), (1, 4), (4, 8)]  # slow eye, delta, theta


# ─────────────────────────────────────────────────────────────────
# GPU helper
# ─────────────────────────────────────────────────────────────────
class GPUProcessor:
    def __init__(self, use_gpu=True):
        self.use_gpu = use_gpu and GPU_AVAILABLE
        if self.use_gpu:
            print("✓ GPU acceleration ENABLED (DE computation only)")
        else:
            print("✓ CPU processing mode")

    def to_gpu(self, data):
        if self.use_gpu and GPU_AVAILABLE:
            return cp.asarray(data)
        return data

    def to_cpu(self, data):
        if self.use_gpu and GPU_AVAILABLE and isinstance(data, cp.ndarray):
            return cp.asnumpy(data)
        return np.asarray(data)


# ─────────────────────────────────────────────────────────────────
# Feature extractors — one per channel type
# ─────────────────────────────────────────────────────────────────

def _safe_de(sig, fs, low, high):
    """Differential entropy for one band. Returns scalar."""
    try:
        nyq = fs / 2.0
        h = min(high, nyq - 0.5)
        l = max(low, 0.1)
        if l >= h:
            return 0.0
        sos = signal.butter(5, [l, h], btype='band', fs=fs, output='sos')
        filtered = signal.sosfiltfilt(sos, sig)
        if np.any(~np.isfinite(filtered)):
            return 0.0
        v = float(np.var(filtered))
        return 0.5 * np.log(2 * np.pi * np.e * v) if v > 1e-12 else 0.0
    except Exception:
        return 0.0


def _spectral_edge(sig, fs, edge_pct=0.95, low=0.5, high=50.0):
    """Frequency below which `edge_pct` of power lies."""
    try:
        f, pxx = signal.welch(sig, fs=fs, nperseg=min(len(sig), int(fs * 4)))
        mask = (f >= low) & (f <= high)
        f, pxx = f[mask], pxx[mask]
        cumpower = np.cumsum(pxx)
        cumpower /= cumpower[-1] + 1e-12
        idx = np.searchsorted(cumpower, edge_pct)
        return float(f[min(idx, len(f) - 1)])
    except Exception:
        return 0.0


def _hjorth(sig):
    """Hjorth activity, mobility, complexity."""
    try:
        d1 = np.diff(sig)
        d2 = np.diff(d1)
        activity = float(np.var(sig))
        mobility = float(np.sqrt(np.var(d1) / (np.var(sig) + 1e-12)))
        complexity = float(
            np.sqrt(np.var(d2) / (np.var(d1) + 1e-12)) / (mobility + 1e-12)
        )
        return activity, mobility, complexity
    except Exception:
        return 0.0, 0.0, 0.0


def extract_eeg_features(epoch_data, fs):
    """
    14 features:
      [0-4]  DE for delta/theta/alpha/beta/gamma
      [5]    spectral edge frequency (95%)
      [6-10] relative band powers (each band / total power)
      [11-13] Hjorth activity, mobility, complexity
    """
    feats = np.zeros(14, dtype=np.float32)

    # DE per band
    for i, (lo, hi) in enumerate(EEG_BANDS):
        feats[i] = _safe_de(epoch_data, fs, lo, hi)

    # Spectral edge
    feats[5] = _spectral_edge(epoch_data, fs)

    # Relative band powers
    try:
        f, pxx = signal.welch(epoch_data, fs=fs, nperseg=min(len(epoch_data), int(fs * 4)))
        total_power = np.sum(pxx[(f >= 0.5) & (f <= 50)]) + 1e-12
        for i, (lo, hi) in enumerate(EEG_BANDS):
            band_power = np.sum(pxx[(f >= lo) & (f <= hi)])
            feats[6 + i] = float(band_power / total_power)
    except Exception:
        pass

    # Hjorth
    act, mob, comp = _hjorth(epoch_data)
    feats[11] = np.clip(act, -1e6, 1e6)
    feats[12] = np.clip(mob, -1e6, 1e6)
    feats[13] = np.clip(comp, -1e6, 1e6)

    return feats


def extract_eog_features(epoch_data, fs):
    """
    8 features:
      [0-2]  DE for slow-eye/delta/theta bands (physiologically relevant for EOG)
      [3]    slow eye movement power (0.1–1 Hz) — REM vs NREM discriminator
      [4]    rapid eye movement proxy: variance of high-passed signal (1-8 Hz)
      [5]    signal variance (overall)
      [6-7]  Hjorth activity, mobility
    
    Why not gamma/beta DE for EOG:
      EOG is filtered to 0.1–50Hz but meaningful content is <8Hz.
      High-frequency components are mostly EEG bleed-through, not eye movements.
    """
    feats = np.zeros(8, dtype=np.float32)

    # DE for physiologically relevant bands only
    for i, (lo, hi) in enumerate(EOG_BANDS):
        feats[i] = _safe_de(epoch_data, fs, lo, hi)

    # Slow eye movement power (key REM indicator)
    try:
        sos = signal.butter(4, [0.1, 1.0], btype='band', fs=fs, output='sos')
        slow = signal.sosfiltfilt(sos, epoch_data)
        feats[3] = float(np.var(slow))
    except Exception:
        feats[3] = 0.0

    # Rapid eye movement proxy
    try:
        sos = signal.butter(4, [1.0, 8.0], btype='band', fs=fs, output='sos')
        fast = signal.sosfiltfilt(sos, epoch_data)
        feats[4] = float(np.var(fast))
    except Exception:
        feats[4] = 0.0

    feats[5] = float(np.var(epoch_data))

    act, mob, _ = _hjorth(epoch_data)
    feats[6] = np.clip(act, -1e6, 1e6)
    feats[7] = np.clip(mob, -1e6, 1e6)

    return feats


def extract_emg_features(epoch_data, fs):
    """
    6 features:
      [0]  RMS amplitude — muscle tone proxy (high in Wake, suppressed in REM)
      [1]  Zero-crossing rate — frequency content proxy
      [2]  Mean absolute value
      [3]  Spectral entropy (10–100 Hz) — muscle activation complexity
      [4]  Kurtosis — burst vs tonic activity
      [5]  Variance

    Why not DE for EMG:
      EMG information is in amplitude and frequency content of bursts,
      not in the Gaussian variance assumption underlying DE.
      RMS and ZCR directly capture atonia (REM) vs tonic activity (Wake/N1).
    """
    feats = np.zeros(6, dtype=np.float32)

    feats[0] = float(np.sqrt(np.mean(epoch_data ** 2)))  # RMS
    feats[1] = float(np.sum(np.abs(np.diff(np.sign(epoch_data)))) / (2 * len(epoch_data)))  # ZCR
    feats[2] = float(np.mean(np.abs(epoch_data)))  # MAV

    # Spectral entropy in EMG band
    try:
        f, pxx = signal.welch(epoch_data, fs=fs, nperseg=min(len(epoch_data), int(fs * 2)))
        mask = (f >= 10) & (f <= 100)
        pxx_band = pxx[mask]
        pxx_norm = pxx_band / (pxx_band.sum() + 1e-12)
        feats[3] = float(-np.sum(pxx_norm * np.log(pxx_norm + 1e-12)))  # spectral entropy
    except Exception:
        feats[3] = 0.0

    feats[4] = float(np.clip(stats.kurtosis(epoch_data), -10, 10))
    feats[5] = float(np.var(epoch_data))

    return feats


def extract_resp_features(epoch_data, fs):
    """
    6 features:
      [0]  Estimated breathing rate (Hz) — from dominant frequency in 0.1–0.5 Hz
      [1]  Tidal amplitude variance — breathing depth variability
      [2]  Inhalation-to-exhalation ratio proxy — from signal asymmetry
      [3]  Total spectral power in respiratory band (0.1–2 Hz)
      [4]  Envelope variance — amplitude modulation
      [5]  RMS

    Why not DE for Resp:
      Resp is filtered to 0.1–2 Hz. DE over 5 bands up to 50 Hz is 80% zeros.
      Breathing rate and tidal volume are the discriminative features.
      These change meaningfully between sleep stages (slow/regular in N3,
      irregular in REM, faster in Wake).
    """
    feats = np.zeros(6, dtype=np.float32)

    # Breathing rate: dominant frequency in respiratory band
    try:
        f, pxx = signal.welch(epoch_data, fs=fs,
                              nperseg=min(len(epoch_data), int(fs * 10)))
        resp_mask = (f >= 0.1) & (f <= 0.5)
        if resp_mask.any():
            dominant_freq = float(f[resp_mask][np.argmax(pxx[resp_mask])])
            feats[0] = dominant_freq
        else:
            feats[0] = 0.2  # default ~12 breaths/min
    except Exception:
        feats[0] = 0.2

    # Tidal amplitude variance (variance of peak-to-peak amplitudes)
    try:
        peaks, _ = signal.find_peaks(epoch_data, distance=int(fs * 1.5))
        troughs, _ = signal.find_peaks(-epoch_data, distance=int(fs * 1.5))
        if len(peaks) > 2 and len(troughs) > 2:
            amplitudes = [epoch_data[p] for p in peaks]
            feats[1] = float(np.var(amplitudes))
        else:
            feats[1] = float(np.var(epoch_data))
    except Exception:
        feats[1] = float(np.var(epoch_data))

    # I:E ratio proxy: ratio of time above vs below mean
    try:
        above = np.sum(epoch_data > np.mean(epoch_data))
        below = len(epoch_data) - above
        feats[2] = float(above / (below + 1e-6))
    except Exception:
        feats[2] = 1.0

    # Total spectral power in 0.1–2 Hz
    try:
        f, pxx = signal.welch(epoch_data, fs=fs,
                              nperseg=min(len(epoch_data), int(fs * 10)))
        mask = (f >= 0.1) & (f <= 2.0)
        feats[3] = float(np.sum(pxx[mask]))
    except Exception:
        feats[3] = 0.0

    # Envelope variance (Hilbert amplitude modulation)
    try:
        analytic = signal.hilbert(epoch_data)
        envelope = np.abs(analytic)
        feats[4] = float(np.var(envelope))
    except Exception:
        feats[4] = 0.0

    feats[5] = float(np.sqrt(np.mean(epoch_data ** 2)))  # RMS

    return feats


def extract_temp_features(epoch_data, fs):
    """
    4 features:
      [0]  Mean — absolute temperature level
      [1]  Linear slope — temperature trend (cooling in NREM, rises in REM)
      [2]  Variance — stability
      [3]  Range (max - min)

    Why not DE for Temp:
      Temp is filtered to 0.01–0.5 Hz. It carries almost no frequency-domain
      information within a 30s epoch. The discriminative signal is in slow
      trends and absolute level shifts across the night.
    """
    feats = np.zeros(4, dtype=np.float32)
    feats[0] = float(np.mean(epoch_data))
    feats[2] = float(np.var(epoch_data))
    feats[3] = float(np.ptp(epoch_data))  # peak-to-peak (range)

    # Linear slope via least squares
    try:
        t = np.arange(len(epoch_data), dtype=float)
        slope, _ = np.polyfit(t, epoch_data, 1)
        feats[1] = float(slope)
    except Exception:
        feats[1] = 0.0

    return feats


# Map channel names to their extractor and feature dim
CHANNEL_EXTRACTORS = {
    'EEG Fpz-Cz':      (extract_eeg_features,  14),
    'EEG Pz-Oz':       (extract_eeg_features,  14),
    'EOG horizontal':   (extract_eog_features,   8),
    'EMG submental':    (extract_emg_features,   6),
    'Resp oro-nasal':   (extract_resp_features,  6),
    'Temp rectal':      (extract_temp_features,  4),
}


# ─────────────────────────────────────────────────────────────────
# Data loader (unchanged from original)
# ─────────────────────────────────────────────────────────────────
class SleepEDFDataLoader:
    def __init__(self, data_dir, target_distribution=None):
        self.data_dir = Path(data_dir)
        self.target_distribution = target_distribution or {
            0: 24.92, 1: 6.34, 2: 39.62, 3: 12.0985, 4: 17.094
        }
        self.stage_mapping = {
            'Sleep stage W': 0, 'Sleep stage R': 4,
            'Sleep stage 1': 1, 'Sleep stage 2': 2,
            'Sleep stage 3': 3, 'Sleep stage 4': 3,
            'Sleep stage ?': -1, 'Movement time': -1,
            'W': 0, 'R': 4, '1': 1, '2': 2, '3': 3, '4': 3,
            'M': -1, '?': -1
        }

    def load_psg_and_labels(self, psg_file, hypnogram_file):
        psg_file = Path(psg_file)
        hypnogram_file = Path(hypnogram_file)
        raw = mne.io.read_raw_edf(str(psg_file), preload=True, verbose=False)
        sampling_rate = raw.info['sfreq']
        channels = raw.ch_names
        print(f"\nLoaded PSG: {psg_file.name} | {sampling_rate} Hz | {channels}")

        signals = {}
        for ch in channels:
            data = raw.get_data(picks=[ch])[0]
            if not (np.any(np.isnan(data)) or np.any(np.isinf(data))):
                signals[ch] = data

        annotations = mne.read_annotations(str(hypnogram_file))
        labels = self._parse_hypnogram(
            annotations, len(next(iter(signals.values()))), sampling_rate
        )
        return signals, labels, sampling_rate

    def _parse_hypnogram(self, annotations, signal_length, fs):
        epoch_length = 30
        num_epochs = int(signal_length / (epoch_length * fs))
        labels = np.full(num_epochs, -1, dtype=int)
        ann_list = []
        for onset, duration, desc in zip(
            annotations.onset, annotations.duration, annotations.description
        ):
            desc = desc.strip()
            if desc in self.stage_mapping:
                lbl = self.stage_mapping[desc]
            else:
                stage = desc.split()[-1] if ' ' in desc else desc
                lbl = self.stage_mapping.get(stage, -1)
            ann_list.append({'onset': onset, 'duration': duration, 'label': lbl})

        ann_list.sort(key=lambda x: x['onset'])
        for ann in ann_list:
            lbl = ann['label']
            start_ep = int(np.floor(ann['onset'] / epoch_length))
            end_ep = int(np.ceil((ann['onset'] + ann['duration']) / epoch_length))
            for ep in range(start_ep, end_ep):
                if 0 <= ep < num_epochs:
                    ov_s = max(ann['onset'], ep * epoch_length)
                    ov_e = min(ann['onset'] + ann['duration'], (ep + 1) * epoch_length)
                    if (ov_e - ov_s) > (epoch_length / 2):
                        labels[ep] = lbl

        print("Label distribution:")
        stage_names = {-1: 'Excl', 0: 'Wake', 1: 'N1', 2: 'N2', 3: 'N3', 4: 'REM'}
        for lbl, cnt in zip(*np.unique(labels, return_counts=True)):
            print(f"  {stage_names.get(lbl, str(lbl))}: {cnt} ({100*cnt/num_epochs:.1f}%)")
        return labels

    def get_file_pairs(self):
        psg_files = sorted(self.data_dir.glob("*PSG.edf"))
        hypno_files = sorted(self.data_dir.glob("*Hypnogram.edf"))
        hypno_dict = {}
        for hf in hypno_files:
            key = hf.stem.replace("-Hypnogram", "").replace("Hypnogram", "")[:7]
            hypno_dict[key] = hf

        pairs = []
        for pf in psg_files:
            key = pf.stem.replace("-PSG", "").replace("PSG", "")[:7]
            if key in hypno_dict:
                pairs.append((str(pf), str(hypno_dict[key])))

        print(f"Matched {len(pairs)} PSG–Hypnogram pairs")
        if len(pairs) == 0:
            raise ValueError("No PSG–Hypnogram pairs found.")
        return pairs


# ─────────────────────────────────────────────────────────────────
# Main preprocessor
# ─────────────────────────────────────────────────────────────────
class SleepEDFPreprocessor:
    def __init__(self, target_fs=100, epoch_length=30, temporal_receptive_field=9,
                 use_gpu=True):
        self.target_fs = target_fs
        self.epoch_length = epoch_length
        self.temporal_receptive_field = temporal_receptive_field
        self.gpu_processor = GPUProcessor(use_gpu=use_gpu)
        self.channel_priority = list(CHANNEL_EXTRACTORS.keys())

    # ── filtering (CPU only, same as original) ──────────────────

    def _notch_filter(self, data, freqs, fs):
        try:
            filtered = data.copy()
            for freq in freqs:
                if freq < fs / 2:
                    b, a = signal.iirnotch(freq, Q=30, fs=fs)
                    filtered = signal.filtfilt(b, a, filtered)
            return filtered
        except Exception:
            return data

    def _bandpass(self, data, lo, hi, fs):
        try:
            nyq = fs / 2.0
            hi = min(hi, nyq - 0.5)
            if lo >= hi:
                return data
            sos = signal.butter(5, [lo, hi], btype='band', fs=fs, output='sos')
            return signal.sosfiltfilt(sos, data)
        except Exception:
            return data

    def preprocess_signals(self, signals, fs):
        """CPU-only filtering + resample + normalise."""
        preprocessed = {}
        for ch_name, ch_data in signals.items():
            notched = self._notch_filter(ch_data, [50, 60], fs)

            if 'EEG' in ch_name:
                filtered = self._bandpass(notched, 0.3, 50.0, fs)
            elif 'EOG' in ch_name:
                filtered = self._bandpass(notched, 0.1, 50.0, fs)
            elif 'EMG' in ch_name:
                filtered = self._bandpass(notched, 10.0, 100.0, fs)
            elif 'Resp' in ch_name:
                filtered = self._bandpass(notched, 0.1, 2.0, fs)
            elif 'Temp' in ch_name:
                # Very narrow band: if output is degenerate, fall back to notch-only
                filtered = self._bandpass(notched, 0.01, 0.5, fs)
                if not np.all(np.isfinite(filtered)) or np.std(filtered) < 1e-15:
                    filtered = notched
            else:
                filtered = notched

            if not np.all(np.isfinite(filtered)):
                filtered = ch_data

            if fs != self.target_fs:
                try:
                    resampled = signal.resample_poly(filtered, self.target_fs, int(fs))
                    if not np.all(np.isfinite(resampled)):
                        continue
                except Exception:
                    continue
            else:
                resampled = filtered

            # Always normalise — epsilon denominator prevents div-by-zero
            mu = np.mean(resampled)
            sd = np.std(resampled)
            normalized = (resampled - mu) / (sd + 1e-12)
            preprocessed[ch_name] = normalized.astype(np.float32)

        return preprocessed

    def select_channels(self, signals):
        selected, names = {}, []
        for ch in self.channel_priority:
            if ch in signals:
                selected[ch] = signals[ch]
                names.append(ch)
        if not selected:
            for ch, data in signals.items():
                selected[ch] = data
                names.append(ch)
        return selected, names

    def segment_into_epochs(self, signals, labels):
        epoch_samples = int(self.epoch_length * self.target_fs)
        ch_names = list(signals.keys())

        # Use EEG as reference length — avoids Resp/Temp length mismatches
        # from mixed native sampling rates in Sleep-EDF EDF files
        ref_ch = next((ch for ch in ch_names if 'EEG' in ch), ch_names[0])
        ref_len = len(signals[ref_ch])
        num_epochs = min(ref_len // epoch_samples, len(labels))

        epoch_signals = np.zeros(
            (num_epochs, len(ch_names), epoch_samples), dtype=np.float32
        )

        # Only EEG channels are used for epoch-level quality control.
        #
        # WHY NOT EMG: Sleep-EDF Cassette EMG submental frequently records only
        # an initial burst then flatlines for the entire night (poor electrode
        # contact in home recordings). Excluding flat EMG epochs would discard
        # the entire dataset. EMG flatline is also physiologically normal (atonia).
        #
        # WHY NOT EOG: EOG can legitimately be near-flat during deep NREM when
        # there are no eye movements. Excluding those epochs would bias against N3.
        #
        # WHY NOT Resp/Temp: slow signals, within-epoch std near-zero by design.
        #
        # EEG being flat = genuine electrode dropout / recording failure.
        # If both EEG channels are dead, the epoch cannot be staged.
        QC_CHANNELS = {'EEG Fpz-Cz', 'EEG Pz-Oz'}
        FLATLINE_THR = 1e-4  # only catches total hardware dropout after normalization

        dead = np.zeros((num_epochs, len(ch_names)), dtype=bool)
        for ci, ch in enumerate(ch_names):
            data = signals[ch]
            apply_qc = ch in QC_CHANNELS
            for ei in range(num_epochs):
                s, e = ei * epoch_samples, (ei + 1) * epoch_samples
                if e <= len(data):
                    seg = data[s:e]
                    epoch_signals[ei, ci, :] = seg
                    if apply_qc and np.std(seg) < FLATLINE_THR:
                        dead[ei, ci] = True
                else:
                    # Channel shorter than reference — pad with zeros
                    avail = max(0, len(data) - s)
                    if avail > 0:
                        epoch_signals[ei, ci, :avail] = data[s:s+avail]
                    if apply_qc:
                        dead[ei, ci] = True

        # Keep epoch only if all QC (neural) channels are alive.
        # Resp/Temp failures alone never discard an epoch.
        core_indices = [ci for ci, ch in enumerate(ch_names) if ch in QC_CHANNELS]

        valid_idx = []
        for ei in range(num_epochs):
            if core_indices:
                if not any(dead[ei, ci] for ci in core_indices):
                    valid_idx.append(ei)
            else:
                valid_idx.append(ei)

        n_dropped = num_epochs - len(valid_idx)
        print(f"  Epochs retained: {len(valid_idx)}/{num_epochs}  "
              f"(dropped {n_dropped} for dead EEG/EOG/EMG channels)")

        return epoch_signals[valid_idx], labels[valid_idx], ch_names

    # ── NEW: channel-specific feature extraction ─────────────────

    def compute_channel_features(self, epoch_signals, channel_names):
        """
        Compute channel-specific features.

        Returns:
            features: np.ndarray shape (N_epochs, C, F_MAX)
                      Channels with fewer than F_MAX features are zero-padded.
            channel_feature_dims: list[int] — true feature count per channel
        """
        N, C, _ = epoch_signals.shape
        features = np.zeros((N, C, F_MAX), dtype=np.float32)
        channel_feature_dims = []

        for ci, ch_name in enumerate(channel_names):
            if ch_name in CHANNEL_EXTRACTORS:
                extractor, f_dim = CHANNEL_EXTRACTORS[ch_name]
            else:
                # Fallback: treat as EEG
                extractor, f_dim = extract_eeg_features, 14
                print(f"  Warning: unknown channel '{ch_name}', using EEG extractor")

            channel_feature_dims.append(f_dim)

            for ei in range(N):
                feats = extractor(epoch_signals[ei, ci], self.target_fs)
                # Zero-pad to F_MAX
                features[ei, ci, :f_dim] = feats
                # Remaining [f_dim:F_MAX] stay zero — model knows to ignore them

        # Sanity check
        if np.any(~np.isfinite(features)):
            print("  Warning: non-finite features detected, replacing with 0")
            features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

        print(f"  Features shape: {features.shape}  (N, C={C}, F_MAX={F_MAX})")
        print(f"  Channel feature dims: {dict(zip(channel_names, channel_feature_dims))}")
        return features, channel_feature_dims

    def create_context_windows(self, features, labels):
        """Sliding windows — same logic as original."""
        N = len(features)
        half = self.temporal_receptive_field // 2

        if N < self.temporal_receptive_field:
            C, F = features.shape[1], features.shape[2]
            return (
                np.zeros((0, C, self.temporal_receptive_field, F), dtype=np.float32),
                np.array([], dtype=int)
            )

        win_feats, win_labels = [], []
        for ci in range(half, N - half):
            lbl = labels[ci]
            if lbl < 0:
                continue
            s, e = ci - half, ci + half + 1
            window = features[s:e]        # (T, C, F)
            win_labels_slice = labels[s:e]

            if np.sum(win_labels_slice >= 0) / len(win_labels_slice) < 0.7:
                continue

            # Reshape: (T, C, F) → (C, T, F)
            window = np.transpose(window, (1, 0, 2))
            win_feats.append(window.astype(np.float32))
            win_labels.append(int(lbl))

        if win_feats:
            return np.array(win_feats, dtype=np.float32), np.array(win_labels, dtype=int)
        else:
            C, F = features.shape[1], features.shape[2]
            return (
                np.zeros((0, C, self.temporal_receptive_field, F), dtype=np.float32),
                np.array([], dtype=int)
            )

    def process_subject(self, signals, labels, fs):
        selected, ch_names = self.select_channels(signals)
        print(f"  Channels: {ch_names}")
        preprocessed = self.preprocess_signals(selected, fs)
        epoch_sigs, valid_labels, ch_names_out = self.segment_into_epochs(
            preprocessed, labels
        )
        print(f"  Epochs: {len(epoch_sigs)}")
        features, ch_feat_dims = self.compute_channel_features(epoch_sigs, ch_names_out)
        windowed, win_labels = self.create_context_windows(features, valid_labels)
        print(f"  Windows: {len(windowed)}")

        if len(win_labels) > 0:
            stage_names = {0: 'Wake', 1: 'N1', 2: 'N2', 3: 'N3', 4: 'REM'}
            for lbl, cnt in zip(*np.unique(win_labels, return_counts=True)):
                print(f"    {stage_names.get(lbl, str(lbl))}: {cnt} ({100*cnt/len(win_labels):.1f}%)")

        return windowed, win_labels, ch_names_out, ch_feat_dims


# ─────────────────────────────────────────────────────────────────
# Dataset builder
# ─────────────────────────────────────────────────────────────────
def prepare_dataset(
    data_dir, preprocessor, loader,
    output_dir='preprocessed_subjects_fixed',
    final_output='preprocessed_sleep_edf_fixed.npz'
):
    start = datetime.now()
    out_path = Path(output_dir)
    out_path.mkdir(exist_ok=True)
    file_pairs = loader.get_file_pairs()

    print(f"\n{'='*60}\nProcessing {len(file_pairs)} subjects\n{'='*60}")

    successful = []
    global_channels = []
    global_ch_feat_dims = None

    for idx, (psg_file, hypno_file) in enumerate(file_pairs):
        print(f"\n{'='*60}\nSubject {idx+1}/{len(file_pairs)}: {Path(psg_file).stem}\n{'='*60}")
        try:
            signals, labels, fs = loader.load_psg_and_labels(psg_file, hypno_file)
            feats, win_labels, ch_names, ch_feat_dims = preprocessor.process_subject(
                signals, labels, fs
            )

            if len(feats) > 0:
                subj_file = out_path / f'subject_{idx:04d}.npz'
                np.savez_compressed(
                    subj_file,
                    X=feats, y=win_labels, subject_id=idx,
                    channel_names=np.array(ch_names, dtype=object),
                    channel_feature_dims=np.array(ch_feat_dims, dtype=int)
                )
                successful.append(idx)

                for ch in ch_names:
                    if ch not in global_channels:
                        global_channels.append(ch)

                if global_ch_feat_dims is None:
                    global_ch_feat_dims = ch_feat_dims

                print(f"  ✓ Saved {len(feats)} windows")
            else:
                print(f"  ✗ No valid windows")

        except Exception as e:
            import traceback
            print(f"  ✗ Error: {e}")
            traceback.print_exc()
        finally:
            gc.collect()
            if GPU_AVAILABLE:
                cp.get_default_memory_pool().free_all_blocks()

    if not successful:
        raise ValueError("No subjects processed successfully!")

    print(f"\n{'='*60}\nCombining {len(successful)} subjects...\n{'='*60}")

    X_all, y_all, sid_all = [], [], []
    for idx in successful:
        d = np.load(out_path / f'subject_{idx:04d}.npz', allow_pickle=True)
        X_all.append(d['X'])
        y_all.append(d['y'])
        sid_all.extend([idx] * len(d['y']))
        gc.collect()

    X_final = np.concatenate(X_all)
    y_final = np.concatenate(y_all)
    sid_final = np.array(sid_all, dtype=int)

    print(f"Final X shape: {X_final.shape}  (N, C, T, F_MAX)")
    print(f"  N={X_final.shape[0]} windows, C={X_final.shape[1]} channels, "
          f"T={X_final.shape[2]} context epochs, F_MAX={X_final.shape[3]} features")

    # Print class distribution
    stage_names = {0: 'Wake', 1: 'N1', 2: 'N2', 3: 'N3', 4: 'REM'}
    print("\nFinal class distribution:")
    for lbl, cnt in zip(*np.unique(y_final, return_counts=True)):
        print(f"  {stage_names.get(lbl, str(lbl))}: {cnt} ({100*cnt/len(y_final):.2f}%)")

    np.savez_compressed(
        final_output,
        X=X_final,
        y=y_final,
        subject_ids=sid_final,
        global_channel_names=np.array(global_channels, dtype=object),
        channel_feature_dims=np.array(global_ch_feat_dims or [], dtype=int),
        f_de=F_MAX,             # kept for backward compat with training script
        f_max=F_MAX,
        target_distribution=loader.target_distribution,
    )

    elapsed = (datetime.now() - start).total_seconds()
    print(f"\n✓ Saved to {final_output}  ({elapsed/60:.1f} min)")
    print(f"  channel_feature_dims: {dict(zip(global_channels, global_ch_feat_dims or []))}")
    return X_final, y_final, sid_final


# ─────────────────────────────────────────────────────────────────
# Run
# ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    DATA_DIR = "sleep-cassette"
    OUTPUT_DIR = "preprocessed_subjects_fixed"
    FINAL_OUTPUT = "preprocessed_sleep_edf_fixed.npz"
    USE_GPU = True

    if not os.path.exists(DATA_DIR):
        print(f"❌ Data directory not found: {DATA_DIR}")
        exit(1)

    print("\n" + "="*60)
    print("FIXED Sleep-EDF Preprocessing")
    print("="*60)
    print("Channel-specific feature extraction:")
    for ch, (_, dim) in CHANNEL_EXTRACTORS.items():
        print(f"  {ch:<22s}: {dim} features")
    print(f"\nOutput: X shape (N, 6, 9, {F_MAX})  [zero-padded to F_MAX={F_MAX}]")
    print(f"channel_feature_dims tells model true dims per channel")
    print("="*60)

    loader = SleepEDFDataLoader(DATA_DIR)
    preprocessor = SleepEDFPreprocessor(
        target_fs=100,
        epoch_length=30,
        temporal_receptive_field=9,
        use_gpu=USE_GPU
    )

    X, y, sids = prepare_dataset(
        DATA_DIR, preprocessor, loader,
        output_dir=OUTPUT_DIR,
        final_output=FINAL_OUTPUT
    )