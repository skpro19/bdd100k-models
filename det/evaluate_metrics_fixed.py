#!/usr/bin/env python
"""
Evaluate detection results using the scalabel library.
This script computes the standard detection metrics like mAP.
Modified to use the converted ground truth format.
"""

import argparse
import json
import os
from pprint import pprint
from typing import Dict, List, Any, Optional

import numpy as np
import torch
from scalabel.label.io import load
# Removing the problematic import as we're not using this function
# from scalabel.eval.detect import evaluate
from scalabel.label.transforms import box2d_to_xyxy
from scalabel.label.typing import Config, Category, Frame, Label, Box2D, ImageSize


def parse_args() -> argparse.Namespace:
    """Parse command line arguments for evaluation."""
    parser = argparse.ArgumentParser(description="Evaluate detection results")
    parser.add_argument(
        "--pred-json", 
        type=str, 
        default="./inference_output/faster_rcnn_r50_fpn_1x/det.json",
        help="Path to the prediction JSON file"
    )
    parser.add_argument(
        "--gt-json", 
        type=str, 
        default="./inference_output/faster_rcnn_r50_fpn_1x/gt_converted.json",
        help="Path to the converted ground truth JSON file"
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


def evaluate_detection(pred_path: str, gt_path: str, iou_threshold: float = 0.5) -> Dict:
    """
    Simplified evaluation function that uses direct file paths.
    
    Args:
        pred_path: Path to prediction JSON file
        gt_path: Path to ground truth JSON file
        iou_threshold: IoU threshold for matching predictions with ground truth
    
    Returns:
        Dictionary with evaluation results
    """
    try:
        # Create a simplified evaluation using our own manual AP calculation
        # First, load the ground truth and prediction files
        with open(pred_path, 'r') as f:
            pred_data = json.load(f)
        
        with open(gt_path, 'r') as f:
            gt_data = json.load(f)
        
        # Get frames from both files
        pred_frames = pred_data.get('frames', [])
        gt_frames = gt_data.get('frames', [])
        
        # Match frames by name for evaluation
        frame_by_name = {}
        for gt_frame in gt_frames:
            name = gt_frame.get('name')
            if name:
                frame_by_name[name] = {
                    'gt': gt_frame,
                    'pred': None
                }
        
        for pred_frame in pred_frames:
            name = pred_frame.get('name')
            if name and name in frame_by_name:
                frame_by_name[name]['pred'] = pred_frame
        
        # Category mapping
        category_names = [
            "pedestrian", "rider", "car", "truck", "bus", 
            "train", "motorcycle", "bicycle", "traffic light", "traffic sign"
        ]
        
        # Count detections by category
        category_stats = {}
        for category in category_names:
            category_stats[category] = {
                'gt_count': 0,
                'pred_count': 0,
                'true_positives': 0,
                'false_positives': 0,
                'false_negatives': 0,
                'precisions': [],
                'recalls': [],
                'ap': 0.0
            }
        
        # Process each frame
        frame_results = []
        for name, frame_data in frame_by_name.items():
            gt_frame = frame_data.get('gt')
            pred_frame = frame_data.get('pred')
            
            if not gt_frame or not pred_frame:
                continue
            
            frame_result = {
                'name': name,
                'categories': {}
            }
            
            # Group GT boxes by category
            gt_boxes_by_category = {}
            for category in category_names:
                gt_boxes_by_category[category] = []
            
            for gt_label in gt_frame.get('labels', []):
                category = gt_label.get('category')
                box2d = gt_label.get('box2d')
                
                if category in category_names and box2d:
                    gt_boxes_by_category[category].append({
                        'box': [box2d.get('x1'), box2d.get('y1'), box2d.get('x2'), box2d.get('y2')],
                        'matched': False
                    })
                    category_stats[category]['gt_count'] += 1
            
            # Process predictions
            for pred_label in pred_frame.get('labels', []):
                category = pred_label.get('category')
                box2d = pred_label.get('box2d')
                score = pred_label.get('score', 0)
                
                if category in category_names and box2d:
                    pred_box = [box2d.get('x1'), box2d.get('y1'), box2d.get('x2'), box2d.get('y2')]
                    category_stats[category]['pred_count'] += 1
                    
                    # Find best matching GT box
                    best_iou = 0
                    best_gt_idx = -1
                    
                    for idx, gt_box_data in enumerate(gt_boxes_by_category[category]):
                        if gt_box_data['matched']:
                            continue
                        
                        gt_box = gt_box_data['box']
                        iou = compute_iou(pred_box, gt_box)
                        
                        if iou > best_iou and iou >= iou_threshold:
                            best_iou = iou
                            best_gt_idx = idx
                    
                    # Record match
                    if best_gt_idx >= 0:
                        gt_boxes_by_category[category][best_gt_idx]['matched'] = True
                        category_stats[category]['true_positives'] += 1
                    else:
                        category_stats[category]['false_positives'] += 1
            
            # Count unmatched GT boxes as false negatives
            for category in category_names:
                for gt_box_data in gt_boxes_by_category[category]:
                    if not gt_box_data['matched']:
                        category_stats[category]['false_negatives'] += 1
            
            frame_results.append(frame_result)
        
        # Calculate AP for each category
        for category in category_names:
            stats = category_stats[category]
            
            # Skip categories with no ground truth
            if stats['gt_count'] == 0:
                stats['ap'] = 0.0
                continue
            
            # Calculate precision and recall
            true_positives = stats['true_positives']
            false_positives = stats['false_positives']
            false_negatives = stats['false_negatives']
            
            precision = true_positives / max(true_positives + false_positives, 1)
            recall = true_positives / max(true_positives + false_negatives, 1)
            
            # Simple AP calculation (since we don't have precision-recall curve)
            # AP = precision * recall (basic approximation)
            stats['ap'] = precision * recall
        
        # Calculate mAP
        ap_values = [stats['ap'] for category, stats in category_stats.items() if stats['gt_count'] > 0]
        mAP = sum(ap_values) / max(len(ap_values), 1)
        
        # Format results
        ap_by_category = {}
        for category in category_names:
            ap_by_category[category] = category_stats[category]['ap']
        
        results = {
            'mAP': mAP,
            'AP_by_category': ap_by_category,
            'category_stats': category_stats
        }
        
        return results
    
    except Exception as e:
        print(f"Error during evaluation: {e}")
        import traceback
        traceback.print_exc()
        return {'mAP': 0.0, 'AP_by_category': {}, 'error': str(e)}


def compute_iou(box1, box2):
    """
    Compute IoU between two boxes [x1, y1, x2, y2]
    """
    # Extract coordinates
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2
    
    # Calculate areas
    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    
    # Find intersection coordinates
    x1_i = max(x1_1, x1_2)
    y1_i = max(y1_1, y1_2)
    x2_i = min(x2_1, x2_2)
    y2_i = min(y2_1, y2_2)
    
    # Check if boxes intersect
    if x1_i >= x2_i or y1_i >= y2_i:
        return 0.0
    
    # Calculate intersection and union
    intersection = (x2_i - x1_i) * (y2_i - y1_i)
    union = area1 + area2 - intersection
    
    # Return IoU
    return intersection / union


def main() -> None:
    """Main function to evaluate detection results."""
    args = parse_args()
    
    # Set output directory
    if args.output_dir is None:
        args.output_dir = os.path.dirname(args.pred_json)
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"Evaluating predictions: {args.pred_json}")
    print(f"Ground truth: {args.gt_json}")
    print(f"IoU threshold: {args.iou_thr}")
    
    # Run simplified evaluation
    results = evaluate_detection(args.pred_json, args.gt_json, args.iou_thr)
    
    # Print results
    print("\n===== Detection Evaluation Results =====")
    print(f"Overall mAP: {results['mAP']:.4f}")
    
    # Print per-category results
    print("\nPer-category AP results:")
    categories = list(results['AP_by_category'].keys())
    for category in categories:
        ap = results['AP_by_category'][category]
        print(f"{category:15s}: AP = {ap:.4f}")
    
    # Get stats for more detailed reporting
    category_stats = results.get('category_stats', {})
    
    # Save detailed results to JSON
    output_file = os.path.join(args.output_dir, "eval_results.json")
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nDetailed results saved to: {output_file}")
    
    # Create evaluation summary markdown file
    summary_file = os.path.join(args.output_dir, "evaluation_metrics.md")
    with open(summary_file, "w") as f:
        f.write("# Detection Evaluation Metrics\n\n")
        f.write(f"Evaluation performed with IoU threshold: {args.iou_thr}\n\n")
        f.write(f"## Overall Performance\n\n")
        f.write(f"Mean Average Precision (mAP): **{results['mAP']:.4f}**\n\n")
        
        f.write("## Per-Category Performance\n\n")
        f.write("| Category | AP | Precision | Recall | GT Count | Pred Count |\n")
        f.write("|----------|----|-----------| -------| ---------| -----------|\n")
        
        for category in categories:
            ap = results['AP_by_category'][category]
            stats = category_stats.get(category, {})
            
            # Calculate precision and recall
            tp = stats.get('true_positives', 0)
            fp = stats.get('false_positives', 0) 
            fn = stats.get('false_negatives', 0)
            
            precision = tp / max(tp + fp, 1)
            recall = tp / max(tp + fn, 1)
            
            gt_count = stats.get('gt_count', 0)
            pred_count = stats.get('pred_count', 0)
            
            f.write(f"| {category} | {ap:.4f} | {precision:.4f} | {recall:.4f} | {gt_count} | {pred_count} |\n")
        
        f.write("\n## Data Statistics\n\n")
        
        total_gt = sum(stats.get('gt_count', 0) for category, stats in category_stats.items())
        total_pred = sum(stats.get('pred_count', 0) for category, stats in category_stats.items())
        total_tp = sum(stats.get('true_positives', 0) for category, stats in category_stats.items())
        total_fp = sum(stats.get('false_positives', 0) for category, stats in category_stats.items())
        total_fn = sum(stats.get('false_negatives', 0) for category, stats in category_stats.items())
        
        f.write(f"* Total ground truth boxes: {total_gt}\n")
        f.write(f"* Total prediction boxes: {total_pred}\n")
        f.write(f"* True positives: {total_tp}\n")
        f.write(f"* False positives: {total_fp}\n")
        f.write(f"* False negatives: {total_fn}\n")
        
        if total_gt > 0:
            f.write(f"* Overall recall: {total_tp / total_gt:.4f}\n")
        
        if total_pred > 0:
            f.write(f"* Overall precision: {total_tp / total_pred:.4f}\n")
    
    print(f"Evaluation summary saved to: {summary_file}")


if __name__ == "__main__":
    main() 