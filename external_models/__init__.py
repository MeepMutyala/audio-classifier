import sys
import os

# Add both submodule src directories to Python path
_current_dir = os.path.dirname(__file__)

# Add the src directories, not the root directories
_liquid_s4_src = os.path.join(_current_dir, 'liquid-S4', 'src')
_vjepa2_src = os.path.join(_current_dir, 'vjepa2', 'src') 
_mamba_src = os.path.join(_current_dir, 'mamba', 'src')

# Add to path if not already there  
for path in [_liquid_s4_src, _vjepa2_src, _mamba_src]:
    if os.path.exists(path) and path not in sys.path:
        sys.path.insert(0, path)
