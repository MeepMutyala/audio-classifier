"""
Shared utilities for all training notebooks
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import time
import os
import pandas as pd
from pathlib import Path
import librosa
import numpy as np
import torchaudio

# ESC-50 Dataset and Preprocessing
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
            self.time_mask = torchaudio.transforms.TimeMasking(time_mask_param=16)
            self.freq_mask = torchaudio.transforms.FrequencyMasking(freq_mask_param=8)
    
    def load_and_preprocess(self, filepath):
        """Load audio and convert to mel-spectrogram"""
        filepath = Path(filepath)
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
        """Apply waveform-level augmentations"""
        # Add small amount of noise
        noise = torch.randn_like(waveform) * 0.01
        waveform = waveform + noise
        
        # Random volume scaling
        volume_scale = torch.rand(1) * 0.2 + 0.9  # Scale between 0.9 and 1.1
        waveform = waveform * volume_scale
        
        return waveform
    
    def apply_spectrogram_augmentation(self, mel_spec):
        """Apply spectrogram-level augmentations"""
        # Time masking
        mel_spec = self.time_mask(mel_spec)
        
        # Frequency masking
        mel_spec = self.freq_mask(mel_spec)
        
        return mel_spec

class ESC50Dataset(torch.utils.data.Dataset):
    """ESC-50 dataset for audio classification"""
    
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
            data = self._convert_to_tubelets(preprocessor.load_and_preprocess(audio_path))
        else:
            # For Mamba and Liquid S4: regular mel-spectrograms
            data = preprocessor.load_and_preprocess(audio_path)
        
        label = self.class_to_idx[row['category']]
        return data, torch.tensor(label, dtype=torch.long)
    
    def _convert_to_tubelets(self, mel_spec):
        """Convert mel-spectrogram to tubelet format for V-JEPA2"""
        # mel_spec: [time_steps, n_mels]
        # Convert to [1, n_mels, time_steps, 1] for tubelet processing
        tubelets = mel_spec.T.unsqueeze(0).unsqueeze(-1)  # [1, n_mels, time_steps, 1]
        return tubelets

def create_esc50_splits(esc50_path):
    """Create train/val/test splits from ESC-50 metadata"""
    esc50_path = Path(esc50_path)
    meta_df = pd.read_csv(f"{esc50_path}/meta/esc50.csv")
    
    # ESC-50 uses 5-fold cross-validation, we'll use fold 5 as test
    train_df = meta_df[meta_df['fold'].isin([1, 2, 3])]
    val_df = meta_df[meta_df['fold'] == 4]
    test_df = meta_df[meta_df['fold'] == 5]
    
    return train_df, val_df, test_df

def create_dataloaders(esc50_path, model_type='sequence', batch_size=32, 
                      num_workers=4, augment=True, augment_factor=2):
    """Create train/val/test DataLoaders for any model type"""
    # Get data splits
    train_df, val_df, test_df = create_esc50_splits(esc50_path)
    
    # Create preprocessors
    train_preprocessor = ESC50Preprocessor(augment=augment)
    val_preprocessor = ESC50Preprocessor(augment=False)  # No augmentation for val/test
    
    # Create datasets
    train_dataset = ESC50Dataset(
        train_df, esc50_path, train_preprocessor, 
        model_type=model_type, augment=augment, augment_factor=augment_factor
    )
    
    val_dataset = ESC50Dataset(
        val_df, esc50_path, val_preprocessor,
        model_type=model_type, augment=False
    )
    
    test_dataset = ESC50Dataset(
        test_df, esc50_path, val_preprocessor,
        model_type=model_type, augment=False
    )
    
    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, 
        num_workers=num_workers, pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    
    return train_loader, val_loader, test_loader, train_dataset.num_classes

class AudioTrainer:
    def __init__(self, model, train_loader, val_loader, device='cuda'):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.criterion = nn.CrossEntropyLoss()
        
        # Print model info
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"Model loaded on {device}")
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        
    def train_epoch(self, optimizer):
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        pbar = tqdm(self.train_loader, desc="Training")
        for batch_idx, (data, target) in enumerate(pbar):
            try:
                data, target = data.to(self.device), target.to(self.device)
                
                optimizer.zero_grad()
                output = self.model(data)
                loss = self.criterion(output, target)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                _, predicted = output.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()
                
                # Update progress bar
                pbar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'Acc': f'{100.*correct/total:.2f}%'
                })
            except Exception as e:
                print(f"Error in training batch {batch_idx}: {e}")
                continue
        
        return total_loss / len(self.train_loader), 100. * correct / total
    
    def validate(self):
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc="Validation")
            for batch_idx, (data, target) in enumerate(pbar):
                try:
                    data, target = data.to(self.device), target.to(self.device)
                    output = self.model(data)
                    loss = self.criterion(output, target)
                    
                    total_loss += loss.item()
                    _, predicted = output.max(1)
                    total += target.size(0)
                    correct += predicted.eq(target).sum().item()
                    
                    pbar.set_postfix({
                        'Loss': f'{loss.item():.4f}',
                        'Acc': f'{100.*correct/total:.2f}%'
                    })
                except Exception as e:
                    print(f"Error in validation batch {batch_idx}: {e}")
                    continue
        
        return total_loss / len(self.val_loader), 100. * correct / total
    
    def train(self, epochs, lr=0.001, save_path=None):
        optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=10, factor=0.5)
        
        best_val_acc = 0
        patience_counter = 0
        early_stopping_patience = 20
        
        for epoch in range(epochs):
            print(f"\nEpoch {epoch+1}/{epochs}")
            
            # Train
            train_loss, train_acc = self.train_epoch(optimizer)
            
            # Validate
            val_loss, val_acc = self.validate()
            
            # Update learning rate
            scheduler.step(val_acc)
            
            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
            print(f"Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")
            
            # Save best model
            if val_acc > best_val_acc and save_path:
                best_val_acc = val_acc
                # Ensure directory exists
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                torch.save(self.model.state_dict(), save_path)
                print(f"New best model saved! Val Acc: {val_acc:.2f}%")
                patience_counter = 0
            else:
                patience_counter += 1
            
            # Early stopping
            if patience_counter >= early_stopping_patience:
                print(f"Early stopping triggered after {epoch+1} epochs")
                break
