import librosa
import numpy as np
import torch
import torchaudio
import pandas as pd
from pathlib import Path

# Project root is 2 levels up from this file
PROJECT_ROOT = Path(__file__).parent.parent.parent
DEFAULT_ESC50_PATH = PROJECT_ROOT / "data" / "ESC-50"

# Enhanced preprocessing with proper ESC-50 handling
class ESC50Preprocessor:
    def __init__(self, 
                 sample_rate=16000,
                 n_mels=128,
                 n_fft=1024,
                 hop_length=512,
                 max_length=16000*5,  # Exactly 5 seconds
                 augment=False):
        
        self.sr = sample_rate
        self.n_mels = n_mels
        self.max_length = max_length
        self.augment = augment
        
        # Mel transform
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.sr,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            power=2.0
        )
        
        # Augmentation transforms
        if augment:
            self.time_mask = torchaudio.transforms.TimeMasking(time_mask_param=10)
            self.freq_mask = torchaudio.transforms.FrequencyMasking(freq_mask_param=8)
    
    def load_and_preprocess(self, filepath):
        """Load audio and convert to mel-spectrogram, \n
        Enhanced preprocessing with augmentation support"""
        # Load and basic preprocessing (same as before)
        filepath = Path(filepath)  # Ensure it's a Path object

        waveform, orig_sr = torchaudio.load(str(filepath))
        
        if orig_sr != self.sr:
            resampler = torchaudio.transforms.Resample(orig_sr, self.sr)
            waveform = resampler(waveform)
        
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        
        # ESC-50 clips are exactly 5 seconds, but ensure consistency
        if waveform.shape[1] != self.max_length:
            if waveform.shape[1] > self.max_length:
                waveform = waveform[:, :self.max_length]
            else:
                padding = self.max_length - waveform.shape[1]
                waveform = torch.nn.functional.pad(waveform, (0, padding))
        
        # Apply waveform augmentations if enabled
        if self.augment:
            waveform = self.apply_waveform_augmentation(waveform)
        
        # Convert to mel-spectrogram
        mel_spec = self.mel_transform(waveform)
        mel_spec = torch.log(mel_spec + 1e-8)
        
        # Apply spectrogram augmentations if enabled  
        if self.augment:
            mel_spec = self.apply_spectrogram_augmentation(mel_spec)
        
        # Transpose for sequence models
        mel_spec = mel_spec.squeeze(0).T
        
        return mel_spec
    
    def apply_waveform_augmentation(self, waveform):
        """Apply augmentations appropriate for environmental sounds"""
        # Small time shift (environmental sounds have structure)
        if torch.rand(1) < 0.3:
            shift = torch.randint(-int(0.05 * self.sr), int(0.05 * self.sr), (1,))
            waveform = torch.roll(waveform, shift.item(), dims=1)
        
        # Light noise addition
        if torch.rand(1) < 0.2:
            noise = torch.randn_like(waveform) * 0.005
            waveform = waveform + noise
        
        # Volume scaling
        if torch.rand(1) < 0.4:
            scale = torch.uniform(0.85, 1.15, (1,))
            waveform = waveform * scale
        
        return waveform
    
    def apply_spectrogram_augmentation(self, mel_spec):
        """Apply spectrogram-specific augmentations"""
        # SpecAugment-style masking (but lighter for environmental sounds)
        if torch.rand(1) < 0.3:
            mel_spec = self.time_mask(mel_spec)
        if torch.rand(1) < 0.3:
            mel_spec = self.freq_mask(mel_spec)
        
        return mel_spec

    def load_and_preprocess_tubelets(self, filepath):
        """Load audio and convert directly to tubelet format for V-JEPA2"""
        # Load regular mel-spectrogram
        mel_spec = self.load_and_preprocess(filepath)  # [time_steps, n_mels]
        
        # Convert to V-JEPA2 tubelet format
        # Reshape to video-like: [1, temporal_patches, freq_patches, patch_size]
        time_steps, n_mels = mel_spec.shape
        
        # Simple approach: treat each time step as a "frame"
        # Format: [channels=1, time, freq, context]
        tubelets = mel_spec.T.unsqueeze(0).unsqueeze(-1)  # [1, n_mels, time_steps, 1]
        
        return tubelets

def create_esc50_splits(esc50_path=None):
    """Create train/val/test splits from ESC-50 metadata"""
    if esc50_path is None:
        esc50_path = DEFAULT_ESC50_PATH
    else:
        esc50_path = Path(esc50_path)  # Convert to Path object

    meta_df = pd.read_csv(f"{esc50_path}/meta/esc50.csv")
    
    # ESC-50 uses 5-fold cross-validation, we'll use fold 5 as test
    train_df = meta_df[meta_df['fold'].isin([1, 2, 3])]
    val_df = meta_df[meta_df['fold'] == 4]
    test_df = meta_df[meta_df['fold'] == 5]
    
    return train_df, val_df, test_df
