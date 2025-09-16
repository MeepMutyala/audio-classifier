# configs/model_configs.py
def get_mamba_config():
    return {
        'n_mels': 128,
        'num_classes': 50,
        'd_model': 512,  # SAFE: Good balance of capacity vs speed
        'n_layer': 12,   # SAFE: Sufficient depth for audio patterns
        'pool_method': 'mean'
    }

def get_liquid_s4_config():
    return {
        'n_mels': 128,
        'num_classes': 50,
        'd_model': 64,   # SAFE: Standard for S4 models
        'n_layers': 6,   # SAFE: S4 models are efficient with fewer layers
        'd_state': 64,   # SAFE: Standard state dimension
        'l_max': None,   # SAFE: Auto-adapt to sequence length
        'dropout': 0.1   # SAFE: Prevents overfitting
    }

def get_vjepa2_config():
    return {
        'num_classes': 50,
        'img_size': (128, 157),  # SAFE: Matches your mel-spectrogram dimensions
        'patch_size': 16,        # SAFE: Good for audio patches
        'num_frames': 16,        # SAFE: Captures temporal context
        'tubelet_size': 2,       # SAFE: Small tubelets for audio
        'embed_dim': 384,        # SAFE: Reduced from video-optimized 768
        'depth': 8,              # SAFE: Sufficient for audio classification
        'num_heads': 8           # SAFE: Matches embed_dim/48 ratio
    }

# Training hyperparameters
def get_training_config():
    return {
        'batch_size': 32,
        'learning_rate': 0.001,
        'epochs': 100,
        'weight_decay': 1e-4,
        'scheduler_patience': 10,
        'early_stopping_patience': 20
    }
