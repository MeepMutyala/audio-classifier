#!/usr/bin/env python3
"""
Cloud-optimized training script with automatic platform detection
and resource optimization for different cloud environments.
"""

import argparse
import torch
import os
import sys
from pathlib import Path

# Add project root to path and ensure vendored external models are importable
project_root = Path(__file__).parent.parent
# Ensure repository root is importable (for src.*)
sys.path.insert(0, str(project_root))

# Ensure external Liquid-S4 'src' directory is importable in all environments
external_src = project_root / 'external_models' / 'liquid-S4' / 'src'
if external_src.exists():
    sys.path.insert(0, str(external_src))


from src.models.mamba_audio import MambaAudioClassifier
from src.models.liquidS4_audio import LiquidS4AudioClassifier
from src.models.vjepa_audio import VJEPA2AudioClassifier
from src.training.data_utils import create_dataloaders
from src.training.trainer import AudioTrainer
from configs.model_configs import get_mamba_config, get_liquid_s4_config, get_vjepa2_config

def detect_cloud_environment():
    """Detect the current cloud environment and optimize settings accordingly."""
    env_info = {
        'platform': 'local',
        'gpu_memory': 0,
        'optimize_for_memory': False,
        'mixed_precision': False
    }
    
    # Check for common cloud environment variables
    if 'COLAB_GPU' in os.environ:
        env_info['platform'] = 'colab'
        env_info['gpu_memory'] = 12  # Typical Colab GPU memory
        env_info['optimize_for_memory'] = True
        env_info['mixed_precision'] = True
        print("Detected Google Colab environment")
        
    elif 'KAGGLE_KERNEL_RUN_TYPE' in os.environ:
        env_info['platform'] = 'kaggle'
        env_info['gpu_memory'] = 16  # Typical Kaggle GPU memory
        env_info['optimize_for_memory'] = True
        env_info['mixed_precision'] = True
        print("Detected Kaggle environment")
        
    elif 'SM_MODEL_DIR' in os.environ:
        env_info['platform'] = 'sagemaker'
        env_info['gpu_memory'] = 16  # Typical SageMaker GPU memory
        env_info['optimize_for_memory'] = False
        env_info['mixed_precision'] = True
        print("Detected AWS SageMaker environment")
        
    elif torch.cuda.is_available():
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        env_info['gpu_memory'] = gpu_memory
        if gpu_memory < 16:
            env_info['optimize_for_memory'] = True
            env_info['mixed_precision'] = True
        print(f"Detected local GPU with {gpu_memory:.1f}GB memory")
    
    return env_info

def get_optimized_config(model_name, env_info):
    """Get model configuration optimized for the current environment."""
    if model_name == 'mamba':
        config = get_mamba_config()
        if env_info['optimize_for_memory']:
            config['d_model'] = min(config['d_model'], 256)
            config['n_layer'] = min(config['n_layer'], 8)
    elif model_name == 'liquid_s4':
        config = get_liquid_s4_config()
        if env_info['optimize_for_memory']:
            config['d_model'] = min(config['d_model'], 32)
            config['n_layers'] = min(config['n_layers'], 4)
    elif model_name == 'vjepa2':
        config = get_vjepa2_config()
        if env_info['optimize_for_memory']:
            config['embed_dim'] = min(config['embed_dim'], 192)
            config['depth'] = min(config['depth'], 6)
    
    return config

def get_optimized_batch_size(model_name, env_info):
    """Get optimized batch size for the current environment."""
    base_batch_sizes = {
        'mamba': 32,
        'liquid_s4': 32,
        'vjepa2': 16
    }
    
    base_batch = base_batch_sizes[model_name]
    
    if env_info['platform'] == 'colab':
        return min(base_batch, 16)
    elif env_info['platform'] == 'kaggle':
        return min(base_batch, 24)
    elif env_info['gpu_memory'] < 8:
        return min(base_batch, 8)
    elif env_info['gpu_memory'] < 16:
        return min(base_batch, 16)
    
    return base_batch

def setup_mixed_precision():
    """Setup mixed precision training if supported."""
    if torch.cuda.is_available() and hasattr(torch.cuda, 'amp'):
        from torch.cuda.amp import GradScaler, autocast
        return True, GradScaler(), autocast
    return False, None, None

def main():
    parser = argparse.ArgumentParser(description='Cloud-optimized audio classification training')
    parser.add_argument('--model', choices=['mamba', 'liquid_s4', 'vjepa2'], required=True)
    parser.add_argument('--data_path', default='data/ESC-50')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--device', default='auto')
    parser.add_argument('--batch_size', type=int, default=None, help='Override auto-detected batch size')
    parser.add_argument('--mixed_precision', action='store_true', help='Enable mixed precision training')
    parser.add_argument('--save_frequency', type=int, default=10, help='Save checkpoint every N epochs')
    
    args = parser.parse_args()
    
    # Detect environment and optimize settings
    env_info = detect_cloud_environment()
    
    # Set device
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    
    print(f"Using device: {device}")
    print(f"Environment: {env_info['platform']}")
    print(f"GPU Memory: {env_info['gpu_memory']:.1f}GB")
    
    # Get optimized configuration
    config = get_optimized_config(args.model, env_info)
    batch_size = args.batch_size or get_optimized_batch_size(args.model, env_info)
    
    print(f"Model: {args.model}")
    print(f"Batch size: {batch_size}")
    print(f"Config: {config}")
    
    # Setup mixed precision if enabled
    use_amp, scaler, autocast_fn = setup_mixed_precision()
    if args.mixed_precision and use_amp:
        print("Mixed precision training enabled")
    else:
        use_amp = False
    
    # Create model
    if args.model == 'mamba':
        model_type = 'sequence'
        model = MambaAudioClassifier(**config, device=device)
    elif args.model == 'liquid_s4':
        model_type = 'sequence'
        model = LiquidS4AudioClassifier(**config, device=device)
    elif args.model == 'vjepa2':
        model_type = 'tubelet'
        model = VJEPA2AudioClassifier(**config)
    
    # Create data loaders
    train_loader, val_loader, test_loader, num_classes = create_dataloaders(
        args.data_path, 
        model_type=model_type, 
        batch_size=batch_size,
        num_workers=2 if env_info['platform'] in ['colab', 'kaggle'] else 4
    )
    
    # Create trainer
    trainer = AudioTrainer(model, train_loader, val_loader, device=device)
    
    # Setup checkpoint directory
    checkpoint_dir = Path('checkpoints')
    checkpoint_dir.mkdir(exist_ok=True)
    save_path = checkpoint_dir / f"{args.model}_best.pth"
    
    # Train
    print(f"\nStarting training for {args.epochs} epochs...")
    trainer.train(args.epochs, lr=args.lr, save_path=str(save_path))
    
    print(f"\nTraining completed! Best model saved to: {save_path}")
    
    # Print final statistics
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")
    print(f"Training environment: {env_info['platform']}")
    print(f"GPU memory used: {env_info['gpu_memory']:.1f}GB")

if __name__ == "__main__":
    main()
