import torch
import torch.nn as nn
import sys
import os

# Add liquid-S4 repository root to path
liquid_s4_root = os.path.join(os.path.dirname(__file__), '../../external_models/liquid-S4')
if liquid_s4_root not in sys.path:
    sys.path.insert(0, liquid_s4_root)

# Create ALL missing __init__.py files
def setup_liquid_s4_packages():
    """Create all missing __init__.py files in the liquid-S4 repository"""
    init_files = [
        'src/__init__.py',
        'src/models/__init__.py', 
        'src/tasks/__init__.py',
        'src/utils/__init__.py',
        'src/callbacks/__init__.py',
        'src/dataloaders/__init__.py',
        'src/models/sequence/__init__.py',
        'src/models/sequence/ss/__init__.py',
        'src/models/sequence/attention/__init__.py',
        'src/models/sequence/convs/__init__.py',
        'src/models/sequence/rnns/__init__.py',
        'src/models/sequence/ss/standalone/__init__.py',
        'src/models/nn/__init__.py',
        'src/models/baselines/__init__.py',
        'src/models/functional/__init__.py',
        'src/models/hippo/__init__.py',
        'src/models/s4/__init__.py'
    ]
    
    for init_file in init_files:
        file_path = os.path.join(liquid_s4_root, init_file)
        if not os.path.exists(file_path):
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            with open(file_path, 'w') as f:
                f.write('# Package init file\n')
            print(f"Created: {file_path}")

setup_liquid_s4_packages()

# Force reload of sys.modules to clear any cached imports
modules_to_remove = [key for key in sys.modules.keys() if key.startswith('src.')]
for module in modules_to_remove:
    del sys.modules[module]

# Now try importing with explicit module loading
import importlib.util
import importlib

def load_module_from_file(module_name, file_path):
    """Load a module from a file path and handle its dependencies"""
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    
    # Add the module to sys.modules before executing
    sys.modules[module_name] = module
    
    # Execute the module
    spec.loader.exec_module(module)
    return module

# Load dependencies first
try:
    # Load utils first (needed by other modules)
    utils_path = os.path.join(liquid_s4_root, 'src', 'utils', 'config.py')
    if os.path.exists(utils_path):
        load_module_from_file('src.utils.config', utils_path)
    
    # Load base modules
    base_path = os.path.join(liquid_s4_root, 'src', 'models', 'sequence', 'base.py')
    if os.path.exists(base_path):
        load_module_from_file('src.models.sequence.base', base_path)
    
    # Load block module
    block_path = os.path.join(liquid_s4_root, 'src', 'models', 'sequence', 'block.py')
    if os.path.exists(block_path):
        load_module_from_file('src.models.sequence.block', block_path)
    
    # Load components
    components_path = os.path.join(liquid_s4_root, 'src', 'models', 'nn', 'components.py')
    if os.path.exists(components_path):
        load_module_from_file('src.models.nn.components', components_path)
    
    # Now load the main modules
    sequence_model_path = os.path.join(liquid_s4_root, 'src', 'models', 'sequence', 'model.py')
    SequenceModel = load_module_from_file('src.models.sequence.model', sequence_model_path).SequenceModel
    
    s4_path = os.path.join(liquid_s4_root, 'src', 'models', 'sequence', 'ss', 's4.py')
    S4 = load_module_from_file('src.models.sequence.ss.s4', s4_path).S4
    
    decoders_path = os.path.join(liquid_s4_root, 'src', 'tasks', 'decoders.py')
    NDDecoder = load_module_from_file('src.tasks.decoders', decoders_path).NDDecoder
    
    print("✅ Successfully loaded liquid-S4 modules with explicit loading!")
    
except Exception as e:
    print(f"❌ Explicit loading failed: {e}")
    print("The liquid-S4 repository may have complex dependencies that can't be easily resolved.")
    print("At this point, we may need to either:")
    print("1. Use a different S4 implementation")
    print("2. Modify the liquid-S4 repository structure")
    print("3. Use the standalone S4 implementation")
    raise e

class LiquidS4AudioClassifier(nn.Module):
    """Audio classification wrapper for Liquid S4"""
    
    def __init__(self, 
                 n_mels=128, 
                 num_classes=50,
                 d_model=64,
                 n_layers=8,
                 d_state=64,
                 l_max=None,  # Will be set based on sequence length
                 dropout=0.0,
                 device=None,
                 dtype=None):
        super().__init__()
        
        # Create factory_kwargs for consistent device/dtype handling
        factory_kwargs = {"device": device, "dtype": dtype}
        
        # Input projection layer to convert mel-spectrogram features to d_model
        self.input_projection = nn.Linear(n_mels, d_model, **factory_kwargs)
        
        # Create S4 layer configuration
        s4_layer_config = {
            "_name_": "s4",
            "d_model": d_model,
            "d_state": d_state,
            "l_max": l_max,
            "channels": 1,
            "bidirectional": False,
            "activation": "gelu",
            "postact": "glu",
            "dropout": dropout,
            "mode": "nplr",
            "measure": "legs",
            "rank": 1,
            "dt_min": 0.001,
            "dt_max": 0.1,
            "lr": {
                "dt": 0.001,
                "A": 0.001,
                "B": 0.001
            },
            "n_ssm": 1,
            "liquid_kernel": None,  # Can be set to "polyb" or "kb" for liquid variants
            "liquid_degree": 2,
            "allcombs": True,
            "lcontract": None,
            "deterministic": False,
            "verbose": True
        }
        
        # Create the S4 backbone using SequenceModel
        self.backbone = SequenceModel(
            d_model=d_model,
            n_layers=n_layers,
            transposed=True,  # (B, H, L) format
            dropout=dropout,
            tie_dropout=False,
            prenorm=True,
            n_repeat=1,
            layer=[s4_layer_config],
            residual="R",  # Residual connection
            norm="layer",  # Layer normalization
            pool=None,  # No pooling between layers
            track_norms=True,
            dropinp=0.0,
        )
        
        # Create classification decoder (NDDecoder with pooling)
        self.classifier = NDDecoder(
            d_model=d_model,
            d_output=num_classes,
            mode="pool"  # Mean pooling over sequence length
        )
        
    def forward(self, x):
        """
        Forward pass for audio classification
        Args:
            x: [batch, seq_len, n_mels] - mel-spectrogram input
        Returns:
            logits: [batch, num_classes] - classification logits
        """
        # Project input to model dimension
        x = self.input_projection(x)  # [batch, seq_len, d_model]
        
        # Pass through S4 backbone
        features = self.backbone(x)  # [batch, seq_len, d_model]
        
        # Classify using NDDecoder (includes pooling)
        logits = self.classifier(features)  # [batch, num_classes]
        
        return logits

class LiquidS4AudioClassifierAdvanced(nn.Module):
    """Advanced Liquid S4 classifier with liquid kernel support"""
    
    def __init__(self, 
                 n_mels=128, 
                 num_classes=50,
                 d_model=64,
                 n_layers=8,
                 d_state=64,
                 l_max=None,
                 dropout=0.0,
                 liquid_kernel="polyb",  # "polyb" or "kb" for liquid variants
                 liquid_degree=2,
                 device=None,
                 dtype=None):
        super().__init__()
        
        factory_kwargs = {"device": device, "dtype": dtype}
        
        # Input projection
        self.input_projection = nn.Linear(n_mels, d_model, **factory_kwargs)
        
        # Advanced S4 layer with liquid kernel
        s4_layer_config = {
            "_name_": "s4",
            "d_model": d_model,
            "d_state": d_state,
            "l_max": l_max,
            "channels": 1,
            "bidirectional": False,
            "activation": "gelu",
            "postact": "glu",
            "dropout": dropout,
            "mode": "nplr",
            "measure": "legs",
            "rank": 1,
            "dt_min": 0.001,
            "dt_max": 0.1,
            "lr": {
                "dt": 0.001,
                "A": 0.001,
                "B": 0.001
            },
            "n_ssm": 1,
            "liquid_kernel": liquid_kernel,  # Enable liquid kernel
            "liquid_degree": liquid_degree,
            "allcombs": True,
            "lcontract": "tanh",  # LeCun or tanh contraction
            "deterministic": False,
            "verbose": True
        }
        
        # Create backbone with liquid S4
        self.backbone = SequenceModel(
            d_model=d_model,
            n_layers=n_layers,
            transposed=True,
            dropout=dropout,
            tie_dropout=False,
            prenorm=True,
            n_repeat=1,
            layer=[s4_layer_config],
            residual="R",
            norm="layer",
            pool=None,
            track_norms=True,
            dropinp=0.0,
        )
        
        # Classification head
        self.classifier = NDDecoder(
            d_model=d_model,
            d_output=num_classes,
            mode="pool"
        )
        
    def forward(self, x):
        x = self.input_projection(x)
        features = self.backbone(x)
        logits = self.classifier(features)
        return logits
