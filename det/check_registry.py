"""Script to check if models are properly registered in the mmdet registry."""

import sys
import os
from mmdet.registry import MODELS

# Import core components
import mmdet
import mmdet.models
from mmengine.registry import RUNNERS
from mmdet.models.detectors import FasterRCNN

# Try to access and print available model keys
print(f"Available model keys in MODELS registry:")
print(f"Total models registered: {len(MODELS.module_dict)}")
print(f"Sample keys: {list(MODELS.module_dict.keys())[:20]}")

# Check specifically for FasterRCNN
print("\nChecking for FasterRCNN in registry:")
if 'FasterRCNN' in MODELS.module_dict:
    print("✅ FasterRCNN is registered!")
else:
    print("❌ FasterRCNN is NOT registered!")
    # Check if class exists but not registered
    print(f"FasterRCNN class exists: {FasterRCNN is not None}")
    
    # Try direct registration (this is a hack, normally not needed)
    try:
        MODELS.register_module()(FasterRCNN)
        print("✅ Manually registered FasterRCNN")
    except Exception as e:
        print(f"❌ Failed to manually register: {e}")

# Print mmdet version
print(f"\nMMDetection version: {mmdet.__version__}")
print(f"MMDetection path: {mmdet.__file__}")

if __name__ == "__main__":
    sys.exit(0) 