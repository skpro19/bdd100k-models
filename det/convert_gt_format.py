#!/usr/bin/env python3
"""
Script to convert ground truth BDD100K annotations to the format expected by the evaluation script.
This addresses the validation errors due to missing fields.
"""

import json
import os
import sys

def convert_gt_format(input_path, output_path):
    """
    Convert original ground truth to match the format expected by the evaluation script.
    
    Args:
        input_path: Path to original ground truth JSON
        output_path: Path to save the converted ground truth JSON
    """
    print(f"Loading ground truth from {input_path}...")
    with open(input_path, 'r') as f:
        gt_data = json.load(f)
    
    print(f"Loaded {len(gt_data)} annotations")
    
    # Create the new format structure
    frames = []
    
    for item in gt_data:
        # Create a new frame with required fields
        frame = {
            'name': item.get('name', ''),
            'url': item.get('url', ''),  # Required by the Frame model
            'videoName': item.get('videoName', ''),
            'intrinsics': item.get('intrinsics', {}),
            'extrinsics': item.get('extrinsics', {}),
            'attributes': item.get('attributes', {}),
            'timestamp': item.get('timestamp', 0),
            'frameIndex': item.get('frameIndex', 0),
            'size': item.get('size', {}),
            'labels': []
        }
        
        # Process labels
        for label in item.get('labels', []):
            new_label = {
                'id': label.get('id', ''),
                'index': label.get('index', 0),
                'manualShape': label.get('manualShape', False),
                'manualAttributes': label.get('manualAttributes', False),
                'score': label.get('score', 1.0),  # Add default score
                'attributes': label.get('attributes', {}),
                'category': label.get('category', ''),
                'box2d': label.get('box2d', None),
                'box3d': None,  # Required by the Label model
                'poly2d': None,  # Required by the Label model
                'rle': None,     # Required by the Label model
                'graph': None    # Required by the Label model
            }
            
            frame['labels'].append(new_label)
        
        frames.append(frame)
    
    # Create the final structure
    converted_data = {
        'frames': frames,
        'groups': [],
        'config': {}
    }
    
    print(f"Saving converted ground truth to {output_path}...")
    with open(output_path, 'w') as f:
        json.dump(converted_data, f)
    
    print(f"Converted {len(frames)} frames with a total of {sum(len(frame['labels']) for frame in frames)} labels")
    return True

def main():
    """Main function."""
    gt_path = "/media/skumar/External/bdd100k/data/bdd100k_labels_release/bdd100k/labels/bdd100k_labels_images_val.json"
    output_path = "/media/skumar/External/bdd100k/bdd100k-models/det/inference_output/faster_rcnn_r50_fpn_1x/gt_converted.json"
    
    if not os.path.exists(gt_path):
        print(f"Error: Ground truth file not found: {gt_path}")
        return 1
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    if convert_gt_format(gt_path, output_path):
        print("Conversion successful!")
        return 0
    else:
        print("Conversion failed!")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 