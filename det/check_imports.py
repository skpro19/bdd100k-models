"""Script to check Python path and imports."""

import sys
import os
import inspect

# Print Python path
print("Python path:")
for path in sys.path:
    print(f"  - {path}")

# Try to import mmdet and check where it's imported from
import mmdet
print(f"\nmmdet version: {mmdet.__version__}")
print(f"mmdet location: {mmdet.__file__}")

# Check mmdet.models
import mmdet.models
print(f"\nmmdet.models location: {mmdet.models.__file__}")

# Check registry imports
from mmdet.registry import MODELS
print(f"\nMODELS registry module path: {inspect.getmodule(MODELS).__file__}")
print(f"MODELS registry keys count: {len(MODELS.module_dict)}")
print(f"Sample registry keys: {list(MODELS.module_dict.keys())[:10]}")

# Check for FasterRCNN specifically
from mmdet.models.detectors import FasterRCNN
print(f"\nFasterRCNN class: {FasterRCNN}")
print(f"FasterRCNN module: {inspect.getmodule(FasterRCNN).__file__}")

# Check if it's registered by name
print(f"\nFasterRCNN in registry: {'FasterRCNN' in MODELS.module_dict}")
if 'FasterRCNN' in MODELS.module_dict:
    print(f"Registry entry: {MODELS.module_dict['FasterRCNN']}")
else:
    # Check for similar names
    faster_rcnn_keys = [k for k in MODELS.module_dict.keys() if 'faster' in k.lower()]
    print(f"Similar keys in registry: {faster_rcnn_keys}") 