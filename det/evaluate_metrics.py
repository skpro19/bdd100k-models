#!/usr/bin/env python
"""
Evaluate detection results using the scalabel library.
This script computes the standard detection metrics like mAP.
"""

import argparse
import json
import os
from pprint import pprint
from typing import Dict, List, Any

import numpy as np
from scalabel.label.io import load
from scalabel.eval.detect import evaluate_det
from scalabel.label.typing import Config


def parse_args() -> argparse.Namespace:
    """Parse command line arguments for evaluation."""
    parser = argparse.ArgumentParser(description="Evaluate detection results")
    parser.add_argument(
        "--pred-json", 
        type=str, 
        default="./inference_output/faster_rcnn_r50_fpn_1x_sample/det.json",
        help="Path to the prediction JSON file"
    )
    parser.add_argument(
        "--gt-json", 
        type=str, 
        default="/media/skumar/External/bdd100k/data/bdd100k_labels_release/bdd100k/labels/bdd100k_labels_images_val.json",
        help="Path to the ground truth JSON file"
    )
    parser.add_argument(
        "--output-dir", 
        type=str, 
        default=None,
        help="Directory to save evaluation results (default: same as pred-json directory)"
    )
    parser.add_argument(
        "--iou-thr", 
        type=float, 
        default=0.5,
        help="IoU threshold for matching predictions with ground truth"
    )
    return parser.parse_args()


def main() -> None:
    """Main function to evaluate detection results."""
    args = parse_args()
    
    # Set output directory
    if args.output_dir is None:
        args.output_dir = os.path.dirname(args.pred_json)
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load predictions and ground truth
    print(f"Loading predictions from: {args.pred_json}")
    pred_frames = load(args.pred_json).frames
    print(f"Loaded {len(pred_frames)} prediction frames")
    
    print(f"Loading ground truth from: {args.gt_json}")
    gt_frames = load(args.gt_json).frames
    print(f"Loaded {len(gt_frames)} ground truth frames")
    
    # Filter ground truth to match the prediction frames
    pred_names = {frame.name for frame in pred_frames}
    filtered_gt_frames = [frame for frame in gt_frames if frame.name in pred_names]
    print(f"Filtered to {len(filtered_gt_frames)} ground truth frames matching predictions")
    
    # Set up config with category names
    # This matches the BDD100K 10 detection classes in order
    categories = [
        "pedestrian", "rider", "car", "truck", "bus", 
        "train", "motorcycle", "bicycle", "traffic light", "traffic sign"
    ]
    config = Config(categories=categories)
    
    # Evaluate
    print(f"Evaluating detection results with IoU threshold: {args.iou_thr}")
    result = evaluate_det(filtered_gt_frames, pred_frames, config)
    
    # Print results
    print("\n===== Detection Evaluation Results =====")
    print(f"mAP: {result.AP:.4f}")
    
    # Print per-category results
    print("\nPer-category results:")
    for i, category in enumerate(categories):
        if i < len(result.AP_cat):
            print(f"{category:15s}: AP = {result.AP_cat[i]:.4f}")
    
    # Save detailed results to JSON
    output_file = os.path.join(args.output_dir, "eval_results.json")
    with open(output_file, "w") as f:
        result_dict = result.model_dump()
        json.dump(result_dict, f, indent=2)
    print(f"\nDetailed results saved to: {output_file}")


if __name__ == "__main__":
    main() 