# scripts/train.py
import argparse
import torch
from src.models.mamba_audio import MambaAudioClassifier
from src.models.liquidS4_audio import LiquidS4AudioClassifier
from src.models.vjepa_audio import VJEPA2AudioClassifier
from src.training.data_utils import create_dataloaders
from src.training.trainer import AudioTrainer
from configs.model_configs import get_mamba_config, get_liquid_s4_config, get_vjepa2_config
import os

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=['mamba', 'liquid_s4', 'vjepa2'], required=True)
    parser.add_argument('--data_path', default='data/ESC-50')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()
    
    # Determine model type and config
    if args.model == 'mamba':
        model_type = 'sequence'
        config = get_mamba_config()
        model = MambaAudioClassifier(**config)
    elif args.model == 'liquid_s4':
        model_type = 'sequence'
        config = get_liquid_s4_config()
        model = LiquidS4AudioClassifier(**config)
    elif args.model == 'vjepa2':
        model_type = 'tubelet'
        config = get_vjepa2_config()
        model = VJEPA2AudioClassifier(**config)
    
    # Create data loaders
    train_loader, val_loader, test_loader, num_classes = create_dataloaders(
        args.data_path, model_type=model_type, batch_size=args.batch_size
    )
    
    # Create trainer
    trainer = AudioTrainer(model, train_loader, val_loader, device=args.device)
    
    # Train
    os.makedirs('checkpoints', exist_ok=True)
    save_path = f"checkpoints/{args.model}_best.pth"
    trainer.train(args.epochs, lr=args.lr, save_path=save_path)

if __name__ == "__main__":
    main()
