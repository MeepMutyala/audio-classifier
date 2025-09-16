# src/datasets/esc50_dataset.py - THE ULTIMATE VERSION
import torch
from torch.utils.data import Dataset
import pandas as pd
from pathlib import Path
from src.preprocessing.audio_utils import ESC50Preprocessor

class ESC50Dataset(Dataset):
    """
    Unified ESC-50 dataset that works for ALL models with augmentation support
    - Supports Mamba, Liquid S4, and V-JEPA2 
    - Handles both sequence and tubelet output formats
    - Virtual augmentation expansion for training
    """
    
    def __init__(self, dataframe, esc50_path, preprocessor=None, 
                 model_type='sequence', augment=False, augment_factor=2):
        super().__init__()
        self.df = dataframe.reset_index(drop=True)
        self.esc50_path = Path(esc50_path)
        self.preprocessor = preprocessor or ESC50Preprocessor()
        self.model_type = model_type  # 'sequence' or 'tubelet'
        self.augment = augment
        self.augment_factor = augment_factor if augment else 1
        
        # Create class mapping  
        self.classes = sorted(self.df['category'].unique())
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        self.num_classes = len(self.classes)
        
        # Virtual expansion for augmentation
        if augment:
            self.virtual_length = len(self.df) * self.augment_factor
        else:
            self.virtual_length = len(self.df)
    
    def __len__(self):
        return self.virtual_length
    
    def __getitem__(self, idx):
        # Map virtual index to real index
        real_idx = idx % len(self.df)
        is_augmented = idx >= len(self.df)
        
        row = self.df.iloc[real_idx]
        audio_path = self.esc50_path / "audio" / row['filename']
        
        # Choose preprocessor based on augmentation
        if is_augmented:
            preprocessor = ESC50Preprocessor(augment=True)
        else:
            preprocessor = self.preprocessor
        
        # Process based on model type
        if self.model_type == 'tubelet':
            # For V-JEPA2: create tubelets
            if hasattr(preprocessor, 'load_and_preprocess_tubelets'):
                data = preprocessor.load_and_preprocess_tubelets(audio_path)
            else:
                # Fallback: convert mel-spec to tubelet format
                mel_spec = preprocessor.load_and_preprocess(audio_path)
                data = self._convert_to_tubelets(mel_spec)
        else:
            # For Mamba and Liquid S4: regular mel-spectrograms
            data = preprocessor.load_and_preprocess(audio_path)
        
        label = self.class_to_idx[row['category']]
        return data, torch.tensor(label, dtype=torch.long)
    
    def _convert_to_tubelets(self, mel_spec):
        """Convert mel-spectrogram to tubelet format for V-JEPA2"""
        # mel_spec: [time_steps, n_mels] 
        # Convert to video-like format: [1, T, H, W]
        # Where T=temporal_patches, H=freq_bins, W=context_width
        
        tubelet_size = 16  # Adjust based on V-JEPA2 config
        time_steps, n_mels = mel_spec.shape
        
        # Create overlapping temporal windows
        if time_steps < tubelet_size:
            # Pad if too short
            padding = tubelet_size - time_steps
            mel_spec = torch.nn.functional.pad(mel_spec, (0, 0, 0, padding))
            time_steps = tubelet_size
        
        # Reshape to tubelet format [1, T, H, W] 
        # For audio: T=time_patches, H=n_mels, W=1
        tubelets = mel_spec.unsqueeze(0).unsqueeze(-1)  # [1, time_steps, n_mels, 1]
        
        return tubelets
