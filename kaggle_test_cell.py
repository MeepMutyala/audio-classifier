# Run this cell in your Kaggle notebook to test the imports

# First, make sure we're in the right directory
import os
print(f"Current working directory: {os.getcwd()}")

# Test the imports
print("Testing model imports...")

try:
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
    from src.models.liquidS4_audio import LiquidS4AudioClassifier
    print("✅ LiquidS4AudioClassifier imported successfully!")
    
    # Test instantiation
    model = LiquidS4AudioClassifier(num_classes=50)
    print("✅ LiquidS4AudioClassifier instantiated successfully!")
    
except Exception as e:
    print(f"❌ LiquidS4 import failed: {e}")
    import traceback
    traceback.print_exc()

print("Import test completed!")
