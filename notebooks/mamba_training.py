"""
Mamba Audio Classification Training Notebook
Copy this into Kaggle notebook cells
"""

# Cell 1: Install packages (based on the screenshot)
"""
# Uninstall existing packages to avoid conflicts
!pip uninstall -y torch torchvision torchaudio mamba-ssm causal-conv1d

# Install specific PyTorch version with CUDA 12.1 support
!pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu121

# Install Triton (required for Mamba)
!pip install triton

# Install causal-conv1d (dependency for mamba-ssm)
!pip install causal-conv1d>=1.4.0

# Install mamba-ssm with no build isolation
!pip install mamba-ssm --no-build-isolation

# Install other required packages
!pip install librosa pandas tqdm

print("✅ All packages installed successfully!")
"""

# Cell 2: Verify installation
"""
import torch
print(torch.__version__)

# Test selective_scan_cuda import (critical for Mamba)
import selective_scan_cuda
print(selective_scan_cuda)

print("✅ Mamba installation verified!")
"""

# Cell 3: Import libraries
"""
import torch
import torch.nn as nn
import sys
import os
from pathlib import Path
import pandas as pd

# Add the mamba src to Python path
mamba_src = Path('/kaggle/working/audio-classifier/external_models/mamba/src')
if mamba_src.exists() and str(mamba_src) not in sys.path:
    sys.path.insert(0, str(mamba_src))

# Import shared utilities
sys.path.append('/kaggle/working/audio-classifier/notebooks')
from shared_utils import create_dataloaders, AudioTrainer

# Import the Mamba model
from models.mamba_audio import MambaAudioClassifier

print("✅ All imports successful!")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA device: {torch.cuda.get_device_name(0)}")
"""

# Cell 4: Dataset setup
"""
# Setup dataset path
esc50_path = '/kaggle/working/audio-classifier/data/ESC-50'

# Check if dataset exists
if not Path(esc50_path).exists():
    print("❌ ESC-50 dataset not found! Please ensure the dataset is in the correct location.")
    print(f"Expected path: {esc50_path}")
else:
    print("✅ ESC-50 dataset found!")
    
    # Create data loaders
    train_loader, val_loader, test_loader, num_classes = create_dataloaders(
        esc50_path=esc50_path,
        model_type='sequence',  # Mamba uses sequence format
        batch_size=8,  # Smaller batch size for Mamba (memory intensive)
        num_workers=2,
        augment=True,
        augment_factor=2
    )
    
    print(f"✅ Data loaders created!")
    print(f"Number of classes: {num_classes}")
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")
"""

# Cell 5: Model setup
"""
# Model configuration
config = {
    'n_mels': 128,
    'num_classes': num_classes,
    'd_model': 512,  # Larger model for Mamba
    'n_layer': 12,   # More layers for Mamba
    'pool_method': 'mean'
}

# Create model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = MambaAudioClassifier(**config, device=device)

print(f"✅ Model created on {device}")
print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

# Test model with a sample batch
model.eval()
with torch.no_grad():
    sample_batch = next(iter(train_loader))
    sample_data, sample_target = sample_batch
    sample_data = sample_data.to(device)
    
    output = model(sample_data)
    print(f"✅ Model test successful!")
    print(f"Input shape: {sample_data.shape}")
    print(f"Output shape: {output.shape}")
"""

# Cell 6: Training
"""
# Training configuration
training_config = {
    'epochs': 50,
    'lr': 0.001,
    'save_path': '/kaggle/working/mamba_best.pth'
}

# Create trainer
trainer = AudioTrainer(model, train_loader, val_loader, device=device)

print("🚀 Starting Mamba training...")
print(f"Expected time: 3-4 hours")
print(f"GPU memory usage: ~12GB")
print("\n" + "="*50)

# Start training
trainer.train(
    epochs=training_config['epochs'],
    lr=training_config['lr'],
    save_path=training_config['save_path']
)

print("\n" + "="*50)
print("✅ Mamba training completed!")
print(f"Best model saved to: {training_config['save_path']}")
"""

# Cell 7: Testing
"""
# Load best model and test
if Path(training_config['save_path']).exists():
    model.load_state_dict(torch.load(training_config['save_path'], map_location=device))
    print("✅ Best model loaded for testing")
    
    # Test on test set
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
    
    test_accuracy = 100. * correct / total
    print(f"🎯 Test Accuracy: {test_accuracy:.2f}%")
else:
    print("❌ No saved model found for testing")
"""

# Cell 8: Results summary
"""
print("📊 Mamba Training Results Summary")
print("="*50)
print(f"Model: Mamba Audio Classifier")
print(f"Dataset: ESC-50")
print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
print(f"Training epochs: {training_config['epochs']}")
print(f"Learning rate: {training_config['lr']}")
print(f"Batch size: {train_loader.batch_size}")
if 'test_accuracy' in locals():
    print(f"Test accuracy: {test_accuracy:.2f}%")
print(f"Model saved to: {training_config['save_path']}")
print("\n✅ Training completed successfully!")
"""
