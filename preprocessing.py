"""
GPU-Accelerated Memory-Efficient Sleep-EDF Preprocessing Script
- Uses CuPy for GPU acceleration ONLY for DE computation (not filtering)
- Aligned with ASTGSleep paper specifications
- Research-grade biosignal processing
- Automatic GPU detection and fallback to CPU if GPU unavailable
"""

import numpy as np
import scipy.signal as signal
import mne
import os
from pathlib import Path
import warnings
from datetime import datetime
from collections import defaultdict
import gc

warnings.filterwarnings('ignore')

# GPU Setup - Try to import CuPy
try:
    import cupy as cp
    from cupyx.scipy import signal as cp_signal
    # Test if GPU actually works
    test_array = cp.array([1, 2, 3])
    _ = cp.sum(test_array)
    GPU_AVAILABLE = True
    print("✓ GPU (CuPy) detected and available for acceleration")
    # Print GPU info
    try:
        print(f"  GPU Device: {cp.cuda.Device().compute_capability}")
        print(f"  GPU Memory: {cp.cuda.Device().mem_info[1] / 1e9:.2f} GB total")
    except:
        print("  GPU info partially available")
except (ImportError, Exception) as e:
    print("⚠ GPU not available. Reasons could be:")
    print("  - CuPy not installed: pip install cupy-cuda11x")
    print("  - CUDA Toolkit not installed")
    print("  - CUDA runtime libraries not found")
    print(f"  Error: {type(e).__name__}")
    print("  Falling back to CPU processing (script will still work!)")
    GPU_AVAILABLE = False
    cp = np


class GPUProcessor:
    """Helper class for GPU/CPU operations"""
    
    def __init__(self, use_gpu=True):
        self.use_gpu = use_gpu and GPU_AVAILABLE
        # GPU acceleration: DE only (not filtering) to avoid numerical inconsistencies
        self.use_gpu_for_de = self.use_gpu
        self.use_gpu_for_filtering = False  # Force CPU for filtering
        
        if self.use_gpu:
            print("✓ GPU acceleration ENABLED (DE computation only)")
        else:
            print("✓ CPU processing mode")
    
    def to_gpu(self, data):
        """Move data to GPU if available"""
        if self.use_gpu and GPU_AVAILABLE:
            return cp.asarray(data)
        return data
    
    def to_cpu(self, data):
        """Move data back to CPU"""
        if self.use_gpu and GPU_AVAILABLE and isinstance(data, cp.ndarray):
            return cp.asnumpy(data)
        return np.asarray(data)
    
    def get_array_module(self, data):
        """Get appropriate array module (cupy or numpy)"""
        if self.use_gpu and GPU_AVAILABLE:
            return cp.get_array_module(data)
        return np


class SleepEDFDataLoader:
    """Data loader for Sleep-EDF dataset with custom distribution"""
    
    def __init__(self, data_dir, target_distribution=None):
        """
        Args:
            data_dir: Path to dataset (folder containing PSG and Hypnogram EDFs)
            target_distribution: Dict with target percentages for each class
                                 (used only for reporting, not resampling)
        """
        self.data_dir = Path(data_dir)
        # target_distribution used only for reporting, not resampling
        self.target_distribution = target_distribution or {
            0: 24.92,    # Wake
            1: 6.34,     # N1
            2: 39.62,    # N2
            3: 12.0985,  # N3
            4: 17.094    # REM
        }
        
        # Robust stage mapping
        self.stage_mapping = {
            'Sleep stage W': 0, 'Sleep stage R': 4,
            'Sleep stage 1': 1, 'Sleep stage 2': 2,
            'Sleep stage 3': 3, 'Sleep stage 4': 3,
            'Sleep stage ?': -1, 'Movement time': -1,
            'W': 0, 'R': 4, '1': 1, '2': 2, '3': 3, '4': 3,
            'M': -1, '?': -1
        }
    
    def load_psg_and_labels(self, psg_file, hypnogram_file):
        """Load PSG recording and corresponding hypnogram"""
        psg_file = Path(psg_file)
        hypnogram_file = Path(hypnogram_file)
        
        if not psg_file.exists():
            raise FileNotFoundError(f"PSG file not found: {psg_file}")
        if not hypnogram_file.exists():
            raise FileNotFoundError(f"Hypnogram file not found: {hypnogram_file}")
        
        raw = mne.io.read_raw_edf(str(psg_file), preload=True, verbose=False)
        sampling_rate = raw.info['sfreq']
        channels = raw.ch_names
        
        print(f"\nLoaded PSG: {psg_file.name} | Sampling rate: {sampling_rate} Hz | Channels: {channels}")
        
        if len(channels) == 0:
            raise ValueError(f"No channels found in {psg_file}")
        
        signals = {}
        for ch in channels:
            data = raw.get_data(picks=[ch])[0]
            if np.any(np.isnan(data)) or np.any(np.isinf(data)):
                print(f"  Warning: Channel {ch} contains NaN/Inf values, skipping...")
                continue
            signals[ch] = data
        
        if len(signals) == 0:
            raise ValueError(f"No valid signals extracted from {psg_file}")
        
        annotations = mne.read_annotations(str(hypnogram_file))
        if len(annotations) == 0:
            raise ValueError(f"No annotations found in {hypnogram_file}")
        
        labels = self._parse_hypnogram(annotations, len(next(iter(signals.values()))), sampling_rate)
        
        return signals, labels, sampling_rate
    
    def _parse_hypnogram(self, annotations, signal_length, fs):
        """Parse hypnogram keeping all epochs initially"""
        epoch_length = 30  # seconds
        num_epochs = int(signal_length / (epoch_length * fs))
        labels = np.full(num_epochs, -1, dtype=int)
        
        print(f"  Signal length: {signal_length} samples")
        print(f"  Sampling rate: {fs} Hz")
        print(f"  Expected epochs: {num_epochs}")
        print(f"  Total annotations: {len(annotations)}")
        
        annotation_list = []
        for onset, duration, description in zip(annotations.onset, annotations.duration, annotations.description):
            desc = description.strip()
            if desc in self.stage_mapping:
                label = self.stage_mapping[desc]
            else:
                stage = desc.split()[-1] if ' ' in desc else desc
                label = self.stage_mapping.get(stage, -1)
            
            annotation_list.append({
                'onset': onset, 'duration': duration,
                'description': desc, 'label': label
            })
        
        annotation_list.sort(key=lambda x: x['onset'])
        
        for ann in annotation_list:
            label = ann['label']
            start_time = ann['onset']
            end_time = ann['onset'] + ann['duration']
            start_epoch = int(np.floor(start_time / epoch_length))
            end_epoch = int(np.ceil(end_time / epoch_length))
            
            for epoch_idx in range(start_epoch, end_epoch):
                if 0 <= epoch_idx < num_epochs:
                    epoch_start = epoch_idx * epoch_length
                    epoch_end = (epoch_idx + 1) * epoch_length
                    overlap_start = max(start_time, epoch_start)
                    overlap_end = min(end_time, epoch_end)
                    overlap = overlap_end - overlap_start
                    
                    if overlap > (epoch_length / 2):
                        labels[epoch_idx] = label
        
        print("\nOriginal (per-file) label distribution:")
        unique, counts = np.unique(labels, return_counts=True)
        stage_names = {-1: 'Unlabeled/Exclude', 0: 'Wake', 1: 'N1', 2: 'N2', 3: 'N3', 4: 'REM'}
        for lbl, cnt in zip(unique, counts):
            stage_name = stage_names.get(lbl, f'Unknown({lbl})')
            percentage = 100 * cnt / max(1, num_epochs)
            print(f"  {stage_name}: {cnt} ({percentage:.2f}%)")
        
        return labels
    
    def get_file_pairs(self):
        """Find all PSG and Hypnogram file pairs"""
        psg_files = sorted(self.data_dir.glob("*PSG.edf"))
        hypno_files = sorted(self.data_dir.glob("*Hypnogram.edf"))
        
        print(f"\nFound {len(psg_files)} PSG files and {len(hypno_files)} Hypnogram files in {self.data_dir}")
        
        if len(psg_files) == 0 or len(hypno_files) == 0:
            raise ValueError("No PSG or Hypnogram files found in dataset folder.")
        
        hypno_dict = {}
        for hf in hypno_files:
            name = hf.stem.replace("-Hypnogram", "").replace("Hypnogram", "")
            key = name[:7] if len(name) >= 7 else name
            hypno_dict[key] = hf
        
        file_pairs = []
        unmatched_psg = []
        
        for pf in psg_files:
            name = pf.stem.replace("-PSG", "").replace("PSG", "")
            key = name[:7] if len(name) >= 7 else name
            
            if key in hypno_dict:
                file_pairs.append((str(pf), str(hypno_dict[key])))
            else:
                unmatched_psg.append(pf.name)
        
        print(f"\nMatched {len(file_pairs)} PSG–Hypnogram pairs")
        if unmatched_psg:
            print(f"Unmatched PSG examples: {unmatched_psg[:10]}")
        
        if len(file_pairs) == 0:
            raise ValueError("No PSG–Hypnogram pairs matched. Check filename structure!")
        
        return file_pairs


class DEPSDModule:
    """
    GPU-Accelerated Differential Entropy - Power Spectral Density Module
    
    Differential Entropy assumes approximate Gaussianity within short (30s) 
    stationary windows. This assumption is standard in sleep EEG literature.
    """
    
    def __init__(self, frequency_bands=None, gpu_processor=None):
        if frequency_bands is None:
            # Canonical non-overlapping frequency bands for reduced redundancy
            self.frequency_bands = [
                (0.5, 4),    # Delta
                (4, 8),      # Theta
                (8, 13),     # Alpha
                (13, 30),    # Beta
                (30, 50)     # Gamma (low)
            ]
        else:
            self.frequency_bands = frequency_bands
        
        self.gpu_processor = gpu_processor or GPUProcessor(use_gpu=True)
    
    def compute_de(self, signal_data, fs):
        """
        GPU-accelerated DE computation
        
        Differential Entropy assumes approximate Gaussianity
        within short (30s) stationary windows.
        This assumption is standard in sleep EEG literature.
        
        Args:
            signal_data: shape (C, epoch_samples)
            fs: sampling frequency (Hz)
        
        Returns:
            de_features: shape (C, len(frequency_bands))
        """
        C, T = signal_data.shape
        de_features = np.zeros((C, len(self.frequency_bands)), dtype=np.float32)
        
        # GPU acceleration is used for feature extraction, not signal cleaning,
        # to avoid numerical inconsistencies.
        if self.gpu_processor.use_gpu_for_de:
            signal_data_gpu = self.gpu_processor.to_gpu(signal_data)
            xp = cp
        else:
            signal_data_gpu = signal_data
            xp = np
        
        for c in range(C):
            for i, (low, high) in enumerate(self.frequency_bands):
                try:
                    nyquist = fs / 2.0
                    _high = high if high < nyquist else (nyquist - 1e-3)
                    _low = low if low < _high else max(0.0, _high - 1.0)
                    
                    if _low >= _high:
                        de_features[c, i] = 0.0
                        continue
                    
                    # Use GPU-accelerated butter filter for DE only
                    if self.gpu_processor.use_gpu_for_de:
                        sos = signal.butter(5, [_low, _high], btype='band', fs=fs, output='sos')
                        sos_gpu = cp.asarray(sos)
                        filtered = cp_signal.sosfiltfilt(sos_gpu, signal_data_gpu[c])
                    else:
                        sos = signal.butter(5, [_low, _high], btype='band', fs=fs, output='sos')
                        filtered = signal.sosfiltfilt(sos, signal_data_gpu[c])
                    
                    if xp.any(xp.isnan(filtered)) or xp.any(xp.isinf(filtered)):
                        de_features[c, i] = 0.0
                        continue
                    
                    variance = float(xp.var(filtered))
                    if variance > 1e-12:
                        de = 0.5 * np.log(2.0 * np.pi * np.e * variance)
                    else:
                        de = 0.0
                    
                    de_features[c, i] = float(de)
                    
                except Exception:
                    de_features[c, i] = 0.0
        
        return de_features


class SleepEDFPreprocessor:
    """GPU-Accelerated Preprocessor for Sleep-EDF dataset"""
    
    def __init__(self, target_fs=100, epoch_length=30, temporal_receptive_field=9, 
                 use_gpu=True, save_raw_epochs=False):
        """
        Args:
            target_fs: Target sampling frequency (Hz)
            epoch_length: Epoch duration (seconds)
            temporal_receptive_field: Number of context epochs (renamed from num_context_epochs)
            use_gpu: Enable GPU acceleration for DE computation
            save_raw_epochs: Save raw epoch signals for debugging (default: False)
        """
        self.target_fs = target_fs
        self.epoch_length = epoch_length
        self.temporal_receptive_field = temporal_receptive_field
        self.save_raw_epochs = save_raw_epochs
        self.gpu_processor = GPUProcessor(use_gpu=use_gpu)
        self.de_module = DEPSDModule(gpu_processor=self.gpu_processor)
        
        self.channel_priority = [
            'EEG Fpz-Cz', 'EEG Pz-Oz', 
            'EOG horizontal', 'EMG submental',
            'Resp oro-nasal', 'Temp rectal'
        ]
    
    def select_channels(self, signals):
        """Select and order channels"""
        selected_signals = {}
        channel_names = []
        
        for ch_name in self.channel_priority:
            if ch_name in signals:
                selected_signals[ch_name] = signals[ch_name]
                channel_names.append(ch_name)
        
        if len(selected_signals) == 0:
            for ch_name, ch_data in signals.items():
                if any(x in ch_name for x in ['EEG', 'EOG', 'EMG', 'EEG ']):
                    selected_signals[ch_name] = ch_data
                    channel_names.append(ch_name)
        
        if len(selected_signals) == 0:
            for ch_name, ch_data in signals.items():
                selected_signals[ch_name] = ch_data
                channel_names.append(ch_name)
        
        return selected_signals, channel_names
    
    def _notch_filter_cpu(self, data, freqs, fs):
        """CPU-only notch filter (avoid GPU numerical inconsistencies)"""
        try:
            filtered = data.copy()
            for freq in freqs:
                if freq < fs / 2:
                    b, a = signal.iirnotch(freq, Q=30, fs=fs)
                    filtered = signal.filtfilt(b, a, filtered)
            return filtered
        except Exception:
            return data
    
    def _bandpass_filter_cpu(self, data, lowcut, highcut, fs):
        """CPU-only bandpass filter (avoid GPU numerical inconsistencies)"""
        try:
            nyquist = fs / 2.0
            if highcut >= nyquist:
                highcut = nyquist - 1e-3
            if lowcut >= highcut:
                return data
            
            sos = signal.butter(5, [lowcut, highcut], btype='band', fs=fs, output='sos')
            filtered = signal.sosfiltfilt(sos, data)
            return filtered
        except Exception:
            return data
    
    def preprocess_signals(self, signals, fs):
        """CPU-based signal preprocessing (research-grade biosignal processing)"""
        preprocessed = {}
        
        for ch_name, ch_data in signals.items():
            # Notch filter (CPU only)
            notched = self._notch_filter_cpu(ch_data, [50, 60], fs)
            
            # Bandpass filter (CPU only)
            if 'EEG' in ch_name:
                filtered = self._bandpass_filter_cpu(notched, 0.3, 50.0, fs)
            elif 'EOG' in ch_name:
                filtered = self._bandpass_filter_cpu(notched, 0.1, 50.0, fs)
            elif 'EMG' in ch_name:
                # EMG frequency content extends into higher bands
                # Truncating at 50 Hz risks loss of discriminative muscle tone
                filtered = self._bandpass_filter_cpu(notched, 10.0, 100.0, fs)
            elif 'Resp' in ch_name or 'resp' in ch_name:
                # Respiration signal: preserve slow breathing patterns
                filtered = self._bandpass_filter_cpu(notched, 0.1, 2.0, fs)
            elif 'Temp' in ch_name or 'temp' in ch_name:
                # Temperature: very slow changes, minimal filtering
                filtered = self._bandpass_filter_cpu(notched, 0.01, 0.5, fs)
            else:
                filtered = notched
            
            if np.any(np.isnan(filtered)) or np.any(np.isinf(filtered)):
                filtered = ch_data
            
            # Resample using polyphase filter (research-grade for biosignals)
            if fs != self.target_fs:
                try:
                    # Polyphase resampling (superior to FFT-based for biosignals)
                    resampled = signal.resample_poly(filtered, self.target_fs, int(fs))
                    
                    if np.any(np.isnan(resampled)) or np.any(np.isinf(resampled)):
                        continue
                except Exception:
                    continue
            else:
                resampled = filtered
            
            # Subject-wise, channel-wise normalization
            # Prevents inter-subject amplitude leakage
            subject_channel_mean = np.mean(resampled)
            subject_channel_std = np.std(resampled)
            
            if subject_channel_std < 1e-12:
                normalized = resampled
            else:
                normalized = (resampled - subject_channel_mean) / subject_channel_std
            
            preprocessed[ch_name] = normalized.astype(np.float32)
        
        if len(preprocessed) == 0:
            raise ValueError("No channels successfully preprocessed!")
        
        return preprocessed
    
    def segment_into_epochs(self, signals, labels):
        """Segment continuous signals into epochs with quality control"""
        epoch_samples = int(self.epoch_length * self.target_fs)
        channel_names = list(signals.keys())
        num_channels = len(channel_names)
        
        min_length = min(len(signals[ch]) for ch in channel_names)
        num_epochs = min(min_length // epoch_samples, len(labels))
        
        if num_epochs == 0:
            raise ValueError("No complete epochs can be created!")
        
        epoch_signals = np.zeros((num_epochs, num_channels, epoch_samples), dtype=np.float32)
        epoch_valid_mask = np.ones(num_epochs, dtype=bool)
        
        for ch_idx, ch_name in enumerate(channel_names):
            ch_data = signals[ch_name]
            for epoch_idx in range(num_epochs):
                start = epoch_idx * epoch_samples
                end = start + epoch_samples
                
                if end <= len(ch_data):
                    epoch_data = ch_data[start:end]
                    
                    # Epoch quality control: detect flatline/low-quality epochs
                    # We explicitly control for low-quality epochs rather than 
                    # implicitly discarding them
                    if np.std(epoch_data) < 1e-6:
                        epoch_signals[epoch_idx, ch_idx, :] = 0.0
                        epoch_valid_mask[epoch_idx] = False
                    else:
                        epoch_signals[epoch_idx, ch_idx, :] = epoch_data
                else:
                    epoch_signals[epoch_idx, ch_idx, :] = 0.0
                    epoch_valid_mask[epoch_idx] = False
        
        # Filter out epochs where too many channels are invalid
        valid_epochs_idx = []
        for epoch_idx in range(num_epochs):
            num_valid_channels = np.sum(
                [np.std(epoch_signals[epoch_idx, ch_idx]) > 1e-6 
                 for ch_idx in range(num_channels)]
            )
            # Require at least 50% of channels to be valid
            if num_valid_channels >= (num_channels * 0.5):
                valid_epochs_idx.append(epoch_idx)
        
        if len(valid_epochs_idx) == 0:
            raise ValueError("No valid epochs after quality control!")
        
        epoch_signals = epoch_signals[valid_epochs_idx]
        valid_labels = labels[valid_epochs_idx]
        
        raw_epochs = epoch_signals.copy() if self.save_raw_epochs else None
        
        print(f"  Quality control: {len(valid_epochs_idx)}/{num_epochs} epochs retained")
        
        return epoch_signals, valid_labels, channel_names, raw_epochs
    
    def compute_de_features(self, epoch_signals):
        """Compute DE features (GPU-accelerated)"""
        num_epochs, num_channels, _ = epoch_signals.shape
        num_bands = len(self.de_module.frequency_bands)
        
        de_features = np.zeros((num_epochs, num_channels, num_bands), dtype=np.float32)
        
        for epoch_idx in range(num_epochs):
            de_features[epoch_idx] = self.de_module.compute_de(
                epoch_signals[epoch_idx], self.target_fs
            )
        
        if np.any(np.isnan(de_features)) or np.any(np.isinf(de_features)):
            de_features = np.nan_to_num(de_features, nan=0.0, posinf=0.0, neginf=0.0)
        
        print(f"  DEBUG: DE features computed: {de_features.shape} (epochs, channels, bands={num_bands})")
        
        return de_features
    
    def create_context_windows(self, de_features, labels):
        """Create sliding windows with temporal receptive field"""
        num_epochs = len(de_features)
        half_context = self.temporal_receptive_field // 2
        
        if num_epochs < self.temporal_receptive_field:
            return (np.zeros((0, de_features.shape[1], self.temporal_receptive_field, 
                            de_features.shape[2]), dtype=np.float32), 
                   np.array([]))
        
        windowed_features = []
        windowed_labels = []
        
        for center_idx in range(half_context, num_epochs - half_context):
            label = labels[center_idx]
            
            if label >= 0:
                start_idx = center_idx - half_context
                end_idx = center_idx + half_context + 1
                window = de_features[start_idx:end_idx]
                window_labels = labels[start_idx:end_idx]
                
                valid_count = np.sum(window_labels >= 0)
                valid_ratio = valid_count / len(window_labels)
                
                if valid_ratio >= 0.7:
                    window = np.transpose(window, (1, 0, 2))
                    windowed_features.append(window.astype(np.float32))
                    windowed_labels.append(int(label))
        
        if len(windowed_features) > 0:
            windowed_features = np.array(windowed_features, dtype=np.float32)
            windowed_labels = np.array(windowed_labels, dtype=int)
        else:
            windowed_features = np.zeros((0, de_features.shape[1], self.temporal_receptive_field, 
                                         de_features.shape[2]), dtype=np.float32)
            windowed_labels = np.array([])
        
        return windowed_features, windowed_labels
    
    def process_subject(self, signals, labels, fs):
        """Complete preprocessing pipeline for one subject"""
        selected_signals, channel_names = self.select_channels(signals)
        print(f"  Selected {len(channel_names)} channels: {channel_names}")
        
        preprocessed = self.preprocess_signals(selected_signals, fs)
        print(f"  Preprocessed {len(preprocessed)} channels")
        
        epoch_signals, valid_labels, channel_names_out, raw_epochs = self.segment_into_epochs(
            preprocessed, labels
        )
        print(f"  Created {len(epoch_signals)} epochs")
        
        de_features = self.compute_de_features(epoch_signals)
        print(f"  Computed DE features: {de_features.shape}")
        
        windowed_features, windowed_labels = self.create_context_windows(de_features, valid_labels)
        print(f"  Created {len(windowed_features)} windows for subject")
        
        if len(windowed_labels) > 0:
            print("  Subject label distribution (windows):")
            unique, counts = np.unique(windowed_labels, return_counts=True)
            stage_names = {0: 'Wake', 1: 'N1', 2: 'N2', 3: 'N3', 4: 'REM'}
            for lbl, cnt in zip(unique, counts):
                stage_name = stage_names.get(lbl, f'Unknown({lbl})')
                percentage = 100 * cnt / len(windowed_labels)
                print(f"    {stage_name}: {cnt} ({percentage:.2f}%)")
        
        return windowed_features, windowed_labels, channel_names_out, raw_epochs


def prepare_sleep_edf_dataset_memory_efficient(
    data_dir, preprocessor, loader, 
    output_dir='preprocessed_subjects', 
    final_output='preprocessed_data_combined.npz'
):
    """Memory-efficient dataset preparation with GPU acceleration"""
    start_time = datetime.now()
    
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    file_pairs = loader.get_file_pairs()
    
    print(f"\n{'='*70}\nProcessing {len(file_pairs)} subjects with GPU acceleration\n{'='*70}")
    
    successful_subjects = []
    global_channel_set = []
    
    for idx, (psg_file, hypno_file) in enumerate(file_pairs):
        print(f"\n{'='*70}\nSubject {idx+1}/{len(file_pairs)}: {Path(psg_file).stem}\n{'='*70}")
        
        try:
            signals, labels, fs = loader.load_psg_and_labels(psg_file, hypno_file)
            features, valid_labels, channel_names, raw_epochs = preprocessor.process_subject(
                signals, labels, fs
            )
            
            if len(features) > 0 and len(valid_labels) > 0:
                subject_file = output_path / f'subject_{idx:04d}.npz'
                
                save_dict = {
                    'X': features,
                    'y': valid_labels,
                    'subject_id': idx,
                    'channel_names': np.array(channel_names, dtype=object)
                }
                
                # Only save raw epochs if explicitly requested (for debugging)
                if preprocessor.save_raw_epochs and raw_epochs is not None:
                    save_dict['raw_epochs'] = raw_epochs
                
                np.savez_compressed(subject_file, **save_dict)
                
                successful_subjects.append(idx)
                for ch in channel_names:
                    if ch not in global_channel_set:
                        global_channel_set.append(ch)
                
                print(f"  ✓ Saved {len(features)} windows from subject {idx} to {subject_file}")
            else:
                print(f"  ✗ No valid windows for subject {idx}")
        
        except Exception as e:
            print(f"  ✗ Error processing subject {idx}: {e}")
            continue
        
        gc.collect()
        if GPU_AVAILABLE:
            cp.get_default_memory_pool().free_all_blocks()
    
    if len(successful_subjects) == 0:
        raise ValueError("No data preprocessed successfully!")
    
    print(f"\n{'='*70}\nSuccessfully processed {len(successful_subjects)}/{len(file_pairs)} subjects")
    print(f"Individual subject files saved to: {output_path}")
    print(f"Global channel list: {global_channel_set}\n{'='*70}")
    
    print(f"\n{'='*70}\nCombining subjects into final dataset...\n{'='*70}")
    
    batch_size = 20
    all_statistics = {
        'total_windows': 0,
        'class_counts': defaultdict(int),
        'num_subjects': len(successful_subjects)
    }
    
    print("Gathering dataset statistics...")
    for idx in successful_subjects:
        subject_file = output_path / f'subject_{idx:04d}.npz'
        data = np.load(subject_file, allow_pickle=True)
        
        all_statistics['total_windows'] += len(data['y'])
        unique, counts = np.unique(data['y'], return_counts=True)
        for cls, cnt in zip(unique, counts):
            all_statistics['class_counts'][int(cls)] += int(cnt)
    
    print(f"\nDataset Statistics:")
    print(f"  Total subjects: {all_statistics['num_subjects']}")
    print(f"  Total windows: {all_statistics['total_windows']}")
    print(f"  Class distribution:")
    
    stage_names = {0: 'Wake', 1: 'N1', 2: 'N2', 3: 'N3', 4: 'REM'}
    for cls in range(5):
        cnt = all_statistics['class_counts'][cls]
        pct = 100.0 * cnt / max(1, all_statistics['total_windows'])
        print(f"    {stage_names[cls]}: {cnt} ({pct:.2f}%)")
    
    print(f"\nCombining data in batches of {batch_size} subjects...")
    
    X_batches = []
    y_batches = []
    subject_id_batches = []
    
    for batch_start in range(0, len(successful_subjects), batch_size):
        batch_end = min(batch_start + batch_size, len(successful_subjects))
        batch_subjects = successful_subjects[batch_start:batch_end]
        
        print(f"  Processing batch: subjects {batch_start+1}-{batch_end}/{len(successful_subjects)}")
        
        batch_X = []
        batch_y = []
        batch_ids = []
        
        for idx in batch_subjects:
            subject_file = output_path / f'subject_{idx:04d}.npz'
            data = np.load(subject_file, allow_pickle=True)
            
            batch_X.append(data['X'])
            batch_y.append(data['y'])
            batch_ids.extend([idx] * len(data['y']))
        
        if len(batch_X) > 0:
            X_batches.append(np.concatenate(batch_X, axis=0))
            y_batches.append(np.concatenate(batch_y, axis=0))
            subject_id_batches.append(np.array(batch_ids, dtype=int))
        
        del batch_X, batch_y, batch_ids
        gc.collect()
    
    print("Creating final combined dataset...")
    X_final = np.concatenate(X_batches, axis=0)
    y_final = np.concatenate(y_batches, axis=0)
    subject_ids_final = np.concatenate(subject_id_batches, axis=0)
    
    del X_batches, y_batches, subject_id_batches
    gc.collect()
    
    print(f"\nSaving final dataset to {final_output}...")
    np.savez_compressed(
        final_output,
        X=X_final,
        y=y_final,
        subject_ids=subject_ids_final,
        global_channel_names=np.array(global_channel_set, dtype=object),
        f_de=len(preprocessor.de_module.frequency_bands),
        target_distribution=loader.target_distribution,
        statistics=all_statistics
    )
    
    total_time = (datetime.now() - start_time).total_seconds()
    
    print(f"\n{'='*70}")
    print("✓ PREPROCESSING COMPLETED SUCCESSFULLY!")
    print(f"{'='*70}")
    print(f"Individual subjects saved to: {output_path}/")
    print(f"Combined dataset saved to: {final_output}")
    print(f"Final shapes: X={X_final.shape}, y={y_final.shape}")
    print(f"Fde (bands): {len(preprocessor.de_module.frequency_bands)}")
    print(f"Total processing time: {total_time/60:.2f} minutes")
    print(f"Average time per subject: {total_time/len(successful_subjects):.2f} seconds")
    print(f"{'='*70}\n")
    
    return X_final, y_final, subject_ids_final


# ----------------- Run as script -----------------
if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("SLEEP-EDF PREPROCESSING - RESEARCH-GRADE VERSION")
    print("=" * 70)
    
    # UPDATE this to your dataset directory
    DATA_DIR = "sleep-cassette"
    
    # Output paths
    OUTPUT_DIR = "preprocessed_subjects_research"
    FINAL_OUTPUT = "preprocessed_sleep_edf_research_final.npz"
    
    # GPU Configuration
    USE_GPU = True  # Set to False to force CPU processing
    
    # Save raw epochs flag (set to True only for debugging)
    SAVE_RAW_EPOCHS = False
    
    # Verify data directory exists
    if not os.path.exists(DATA_DIR):
        print(f"\n❌ ERROR: Data directory not found: {DATA_DIR}")
        print("Please update DATA_DIR to point to your Sleep-EDF dataset folder")
        exit(1)
    
    print(f"\nData directory: {DATA_DIR}")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Final output file: {FINAL_OUTPUT}")
    print(f"Save raw epochs: {SAVE_RAW_EPOCHS}")
    
    # Check GPU availability
    if USE_GPU:
        if GPU_AVAILABLE:
            print(f"\n{'='*70}")
            print("GPU INFORMATION")
            print(f"{'='*70}")
            try:
                gpu_device = cp.cuda.Device()
                mem_info = gpu_device.mem_info
                print(f"GPU Device ID: {gpu_device.id}")
                print(f"GPU Name: {gpu_device.compute_capability}")
                print(f"Total Memory: {mem_info[1] / 1e9:.2f} GB")
                print(f"Free Memory: {mem_info[0] / 1e9:.2f} GB")
                print(f"Used Memory: {(mem_info[1] - mem_info[0]) / 1e9:.2f} GB")
                print(f"{'='*70}")
            except Exception as e:
                print(f"Could not retrieve detailed GPU info: {e}")
        else:
            print("\n⚠ WARNING: GPU requested but CuPy not available")
            print("Install CuPy with one of these commands:")
            print("  For CUDA 11.x: pip install cupy-cuda11x")
            print("  For CUDA 12.x: pip install cupy-cuda12x")
            print("\nFalling back to CPU processing...")
            USE_GPU = False
    
    # Target distribution from ASTGSleep paper
    # (used only for reporting, not resampling)
    target_distribution = {
        0: 24.92,    # Wake
        1: 6.34,     # N1
        2: 39.62,    # N2
        3: 12.0985,  # N3
        4: 17.094    # REM
    }
    
    # Initialize components
    print("\n" + "=" * 70)
    print("INITIALIZING PREPROCESSING COMPONENTS")
    print("=" * 70)
    
    loader = SleepEDFDataLoader(
        data_dir=DATA_DIR,
        target_distribution=target_distribution
    )
    
    preprocessor = SleepEDFPreprocessor(
        target_fs=100,                    # 100 Hz sampling rate
        epoch_length=30,                  # 30-second epochs
        temporal_receptive_field=9,       # 9 context epochs (renamed)
        use_gpu=USE_GPU,                  # Enable/disable GPU
        save_raw_epochs=SAVE_RAW_EPOCHS   # Save raw epochs for debugging
    )
    
    print(f"✓ Loader initialized with target distribution:")
    for stage_id, pct in target_distribution.items():
        stage_names = {0: 'Wake', 1: 'N1', 2: 'N2', 3: 'N3', 4: 'REM'}
        print(f"  {stage_names[stage_id]}: {pct:.2f}%")
    
    print(f"✓ Preprocessor initialized:")
    print(f"  Target sampling rate: {preprocessor.target_fs} Hz")
    print(f"  Epoch length: {preprocessor.epoch_length} seconds")
    print(f"  Temporal receptive field: {preprocessor.temporal_receptive_field}")
    print(f"  DE frequency bands: {len(preprocessor.de_module.frequency_bands)}")
    print(f"  Frequency bands: {preprocessor.de_module.frequency_bands}")
    print(f"  GPU acceleration: {'ENABLED (DE only)' if USE_GPU and GPU_AVAILABLE else 'DISABLED'}")
    print(f"  Save raw epochs: {SAVE_RAW_EPOCHS}")
    
    # Start preprocessing
    try:
        print("\n" + "=" * 70)
        print("STARTING PREPROCESSING")
        print("=" * 70)
        
        X, y, subject_ids = prepare_sleep_edf_dataset_memory_efficient(
            data_dir=DATA_DIR,
            preprocessor=preprocessor,
            loader=loader,
            output_dir=OUTPUT_DIR,
            final_output=FINAL_OUTPUT
        )
        
        # Print final summary
        print("\n" + "=" * 70)
        print("FINAL DATASET SUMMARY")
        print("=" * 70)
        print(f"Features shape: {X.shape}")
        print(f"  - Windows: {X.shape[0]}")
        print(f"  - Channels: {X.shape[1]}")
        print(f"  - Context epochs: {X.shape[2]}")
        print(f"  - Frequency bands: {X.shape[3]}")
        print(f"\nLabels shape: {y.shape}")
        print(f"Unique subjects: {len(np.unique(subject_ids))}")
        
        print("\n" + "=" * 70)
        print("FINAL CLASS DISTRIBUTION")
        print("=" * 70)
        
        stage_names = {0: 'Wake', 1: 'N1', 2: 'N2', 3: 'N3', 4: 'REM'}
        unique, counts = np.unique(y, return_counts=True)
        total = len(y)
        
        for stage_id in range(5):
            if stage_id in unique:
                idx = np.where(unique == stage_id)[0][0]
                count = counts[idx]
                percentage = 100.0 * count / total
                print(f"{stage_names[stage_id]:6s}: {count:8d} windows ({percentage:6.2f}%)")
            else:
                print(f"{stage_names[stage_id]:6s}: {0:8d} windows ({0.0:6.2f}%)")
        
        print(f"\n{'Total':6s}: {total:8d} windows")
        
        # Verify output file
        if os.path.exists(FINAL_OUTPUT):
            file_size = os.path.getsize(FINAL_OUTPUT) / (1024 ** 3)  # GB
            print(f"\n✓ Output file created: {FINAL_OUTPUT}")
            print(f"  File size: {file_size:.2f} GB")
        
        # GPU memory cleanup
        if GPU_AVAILABLE and USE_GPU:
            print("\nCleaning up GPU memory...")
            cp.get_default_memory_pool().free_all_blocks()
            print("✓ GPU memory released")
        
        print("\n" + "=" * 70)
        print("✓ PREPROCESSING COMPLETED SUCCESSFULLY!")
        print("=" * 70)
        print("\nNext steps:")
        print("1. Load the preprocessed data using:")
        print(f"   data = np.load('{FINAL_OUTPUT}')")
        print("2. Access features with: data['X']")
        print("3. Access labels with: data['y']")
        print("4. Access subject IDs with: data['subject_ids']")
        print("5. Train your ASTGSleep model!")
        
        print("\n💡 Research-grade improvements applied:")
        print("  ✓ GPU acceleration for DE only (not filtering)")
        print("  ✓ Polyphase resampling (superior to FFT-based)")
        print("  ✓ EMG bandpass extended to 100 Hz")
        print("  ✓ Canonical non-overlapping frequency bands")
        print("  ✓ Epoch quality control with explicit validation")
        print("  ✓ Subject-wise normalization to prevent leakage")
        print("  ✓ Temporal receptive field concept alignment")
        
        print("=" * 70 + "\n")
    
    except KeyboardInterrupt:
        print("\n\n⚠ Preprocessing interrupted by user")
        print("Partial results may be saved in:", OUTPUT_DIR)
        if GPU_AVAILABLE and USE_GPU:
            print("Cleaning up GPU memory...")
            cp.get_default_memory_pool().free_all_blocks()
    
    except Exception as e:
        print("\n" + "=" * 70)
        print("❌ ERROR DURING PREPROCESSING")
        print("=" * 70)
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
        
        print("\nPlease check:")
        print("1. Data directory path is correct")
        print("2. PSG and Hypnogram files exist in the directory")
        print("3. Files follow Sleep-EDF naming convention")
        print("4. Sufficient disk space available")
        if USE_GPU:
            print("5. GPU has sufficient memory (try reducing batch size or disabling GPU)")
            print("6. CuPy is properly installed for your CUDA version")
        print("=" * 70 + "\n")
        
        if GPU_AVAILABLE and USE_GPU:
            print("Cleaning up GPU memory...")
            cp.get_default_memory_pool().free_all_blocks()