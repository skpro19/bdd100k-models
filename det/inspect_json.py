#!/usr/bin/env python3
"""
Script to inspect and compare prediction and ground truth formats.
"""

import json
import os
import sys

def inspect_gt_json(filepath):
    """Inspect ground truth JSON format."""
    with open(filepath, 'r') as f:
        gt = json.load(f)
    
    print(f"Ground truth: {len(gt)} annotations")
    if len(gt) > 0:
        print(f"Ground truth annotation keys: {gt[0].keys()}")
        labels = gt[0].get('labels', [])
        if labels:
            print(f"Ground truth label keys: {labels[0].keys()}")
            if 'box2d' in labels[0]:
                print(f"Ground truth box2d format: {labels[0]['box2d']}")

def inspect_pred_json(filepath):
    """Inspect prediction JSON format."""
    with open(filepath, 'r') as f:
        pred = json.load(f)
    
    print(f"Prediction main keys: {pred.keys()}")
    
    frames = pred.get('frames', [])
    print(f"Number of frames: {len(frames)}")
    
    if frames:
        print(f"Frame keys: {frames[0].keys()}")
        labels = frames[0].get('labels', [])
        print(f"Number of labels in first frame: {len(labels)}")
        
        if labels:
            print(f"Prediction label keys: {labels[0].keys()}")
            if 'box2d' in labels[0]:
                print(f"Prediction box2d format: {labels[0]['box2d']}")

def compare_formats(gt_filepath, pred_filepath):
    """Compare the GT and prediction formats."""
    # First inspect both files
    print("Ground Truth Format:")
    inspect_gt_json(gt_filepath)
    
    print("\nPrediction Format:")
    inspect_pred_json(pred_filepath)
    
    # Now compare key structures
    with open(gt_filepath, 'r') as f:
        gt = json.load(f)
    with open(pred_filepath, 'r') as f:
        pred = json.load(f)
    
    print("\nFormat Differences:")
    # GT is a list of annotations, pred has 'frames' list
    if isinstance(gt, list) and 'frames' in pred:
        print("- Ground truth is a list, predictions have 'frames' list")
    
    # Compare label structure
    if gt and gt[0].get('labels'):
        gt_label = gt[0]['labels'][0]
        if pred.get('frames') and pred['frames'][0].get('labels'):
            pred_label = pred['frames'][0]['labels'][0]
            
            gt_keys = set(gt_label.keys())
            pred_keys = set(pred_label.keys())
            
            print(f"- GT label keys not in pred: {gt_keys - pred_keys}")
            print(f"- Pred label keys not in GT: {pred_keys - gt_keys}")

def main():
    """Main function."""
    gt_path = "/media/skumar/External/bdd100k/data/bdd100k_labels_release/bdd100k/labels/bdd100k_labels_images_val.json"
    pred_path = "./inference_output/faster_rcnn_r50_fpn_1x/det.json"
    
    if not os.path.exists(gt_path):
        print(f"Error: Ground truth file not found: {gt_path}")
        return 1
    
    if not os.path.exists(pred_path):
        print(f"Error: Prediction file not found: {pred_path}")
        return 1
    
    compare_formats(gt_path, pred_path)
    return 0

if __name__ == "__main__":
    sys.exit(main()) 