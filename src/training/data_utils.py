# src/training/data_utils.py
import torch
from torch.utils.data import DataLoader
from src.datasets.esc50_dataset import ESC50Dataset
from src.preprocessing.audio_utils import create_esc50_splits, ESC50Preprocessor

def create_dataloaders(esc50_path, model_type='sequence', batch_size=32, 
                      num_workers=4, augment=True, augment_factor=2):
    """
    Create train/val/test DataLoaders for any model type
    
    Args:
        model_type: 'sequence' (Mamba/Liquid S4) or 'tubelet' (V-JEPA2)
        batch_size: Batch size for training
        num_workers: Number of data loading workers
        augment: Whether to use augmentation
        augment_factor: Augmentation multiplier
    """
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
