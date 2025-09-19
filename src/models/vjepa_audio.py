import torch
import torch.nn as nn
import sys
import os
from pathlib import Path

# Add vjepa2 src to path and import directly
import os
current_dir = Path(__file__).resolve().parent
# Try multiple possible paths
possible_paths = [
    current_dir.parents[2] / 'external_models' / 'vjepa2' / 'src',  # From src/models/
    Path.cwd() / 'external_models' / 'vjepa2' / 'src',  # From project root
    Path('/kaggle/working/audio-classifier/external_models/vjepa2/src'),  # Kaggle specific
]

vjepa_src = None
for path in possible_paths:
    if path.is_dir():
        vjepa_src = path
        break

if vjepa_src is None:
    raise ImportError(f"Could not find vjepa2 src directory. Tried: {possible_paths}")

if str(vjepa_src) not in sys.path:
    sys.path.insert(0, str(vjepa_src))

# Import directly from the src structure
from models.vision_transformer import VisionTransformer
from models.attentive_pooler import AttentiveClassifier

class VJEPA2AudioClassifier(nn.Module):
    def __init__(self, 
                 num_classes=50,
                 img_size=(128, 157),  # (n_mels, time_frames)
                 patch_size=16,
                 num_frames=16,
                 tubelet_size=2,
                 embed_dim=384,
                 depth=8,
                 num_heads=8):
        super().__init__()
        
        # Create V-JEPA2 backbone
        self.backbone = VisionTransformer(
            img_size=img_size,
            patch_size=patch_size,
            num_frames=num_frames,
            tubelet_size=tubelet_size,
            in_chans=1,  # CHANGED: 1 channel for audio instead of 3 for RGB
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=4.0,
            qkv_bias=True,
            norm_layer=nn.LayerNorm,
        )
        
        # Create classification head
        self.classifier = AttentiveClassifier(
            embed_dim=embed_dim,
            num_heads=num_heads,
            depth=1,
            num_classes=num_classes,
        )
        
    def forward(self, x):
        # x: [batch, 1, T, H, W] where T=time, H=freq, W=context or 1
        features = self.backbone(x)  # [batch, num_patches, embed_dim]
        logits = self.classifier(features)  # [batch, num_classes]
        return logits
