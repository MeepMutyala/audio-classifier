# Create a test script to verify everything works
import torch
from src.models.mamba_audio import MambaAudioClassifier
from configs.model_configs import get_mamba_config

# Test model creation
config = get_mamba_config()
model = MambaAudioClassifier(**config)
print("✅ Model created successfully")

# Test forward pass
dummy_input = torch.randn(2, 157, 128)  # [batch, seq_len, n_mels]
output = model(dummy_input)
print(f"✅ Forward pass successful: {output.shape}")

# Test data loading
from src.training.data_utils import create_dataloaders

try:
    train_loader, val_loader, test_loader, num_classes = create_dataloaders(
        'data/ESC-50', model_type='sequence', batch_size=2
    )
    print("✅ Data loading successful")
    
    # Test one batch
    for data, target in train_loader:
        print(f"✅ Batch shape: {data.shape}, target shape: {target.shape}")
        break
except Exception as e:
    print(f"❌ Data loading failed: {e}")
