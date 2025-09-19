"""
Liquid S4 Audio Classification Training Notebook
Copy this into Kaggle notebook cells
"""

# Cell 1: Install packages
"""
# Install required packages
!pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
!pip install librosa pandas tqdm

print("✅ Packages installed successfully!")
"""

# Cell 2: Import libraries
"""
import torch
import torch.nn as nn
import sys
import os
from pathlib import Path
import pandas as pd

# Add the liquid-S4 src to Python path
liquid_s4_src = Path('/kaggle/working/audio-classifier/external_models/liquid-S4/src')
if liquid_s4_src.exists() and str(liquid_s4_src) not in sys.path:
    sys.path.insert(0, str(liquid_s4_src))

# Import shared utilities
sys.path.append('/kaggle/working/audio-classifier/notebooks')
from shared_utils import create_dataloaders, AudioTrainer

# Import the Liquid S4 model
from models.liquidS4_audio import LiquidS4AudioClassifier

print("✅ All imports successful!")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA device: {torch.cuda.get_device_name(0)}")
"""

# Cell 3: Dataset setup
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
        model_type='sequence',  # Liquid S4 uses sequence format
        batch_size=16,  # Smaller batch size for memory efficiency
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

# Cell 4: Model setup
"""
# Model configuration
config = {
    'n_mels': 128,
    'num_classes': num_classes,
    'd_model': 64,   # Standard for S4 models
    'n_layers': 6,   # S4 models are efficient with fewer layers
    'd_state': 64,   # Standard state dimension
    'l_max': None,   # Auto-adapt to sequence length
    'dropout': 0.1   # Prevents overfitting
}

# Create model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = LiquidS4AudioClassifier(**config, device=device)

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

# Cell 5: Training
"""
# Training configuration
training_config = {
    'epochs': 50,
    'lr': 0.001,
    'save_path': '/kaggle/working/liquid_s4_best.pth'
}

# Create trainer
trainer = AudioTrainer(model, train_loader, val_loader, device=device)

print("🚀 Starting Liquid S4 training...")
print(f"Expected time: 2-3 hours")
print(f"GPU memory usage: ~8GB")
print("\n" + "="*50)

# Start training
trainer.train(
    epochs=training_config['epochs'],
    lr=training_config['lr'],
    save_path=training_config['save_path']
)

print("\n" + "="*50)
print("✅ Liquid S4 training completed!")
print(f"Best model saved to: {training_config['save_path']}")
"""

# Cell 6: Testing
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

# Cell 7: Results summary
"""
print("📊 Liquid S4 Training Results Summary")
print("="*50)
print(f"Model: Liquid S4 Audio Classifier")
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
