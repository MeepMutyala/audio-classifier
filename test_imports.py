#!/usr/bin/env python3
"""
Test script to verify that the model imports work correctly.
Run this in your Kaggle notebook to test the imports.
"""

import sys
from pathlib import Path

# Add the project root to path
project_root = Path.cwd()
sys.path.insert(0, str(project_root))

print("Testing imports...")

try:
    print("Testing VJEPA2 imports...")
    from src.models.vjepa_audio import VJEPA2AudioClassifier
    print("✅ VJEPA2AudioClassifier imported successfully!")
    
    # Test instantiation
    model = VJEPA2AudioClassifier(num_classes=50)
    print("✅ VJEPA2AudioClassifier instantiated successfully!")
    
except Exception as e:
    print(f"❌ VJEPA2 import failed: {e}")
    import traceback
    traceback.print_exc()

try:
    print("\nTesting LiquidS4 imports...")
    from src.models.liquidS4_audio import LiquidS4AudioClassifier
    print("✅ LiquidS4AudioClassifier imported successfully!")
    
    # Test instantiation
    model = LiquidS4AudioClassifier(num_classes=50)
    print("✅ LiquidS4AudioClassifier instantiated successfully!")
    
except Exception as e:
    print(f"❌ LiquidS4 import failed: {e}")
    import traceback
    traceback.print_exc()

print("\nImport test completed!")
