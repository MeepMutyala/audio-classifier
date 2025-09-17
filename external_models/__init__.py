import sys, os
_current_dir = os.path.dirname(__file__)
for sub in ['liquid-S4', 'vjepa2']:
    abs_src = os.path.join(_current_dir, sub, 'src')   # add their src, not root
    if abs_src not in sys.path:
        sys.path.insert(0, abs_src)
