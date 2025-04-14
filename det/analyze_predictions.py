#!/usr/bin/env python
"""
Analyze detection results from the inference output.
This script computes basic statistics about the predictions.
"""

import argparse
import json
import os
from collections import Counter, defaultdict
import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    """Parse command line arguments for the analysis."""
    parser = argparse.ArgumentParser(description="Analyze detection results")
    parser.add_argument(
        "--json-path", 
        type=str, 
        default="./inference_output/faster_rcnn_r50_fpn_1x_sample/det.json",
        help="Path to the prediction JSON file"
    )
    parser.add_argument(
        "--output-dir", 
        type=str, 
        default=None,
        help="Directory to save analysis results (default: same as json-path directory)"
    )
    parser.add_argument(
        "--score-thr", 
        type=float, 
        default=0.3,
        help="Score threshold for including predictions in the analysis"
    )
    return parser.parse_args()


def main() -> None:
    """Main function to analyze prediction results."""
    args = parse_args()
    
    # Set output directory
    if args.output_dir is None:
        args.output_dir = os.path.dirname(args.json_path)
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load predictions
    print(f"Loading predictions from: {args.json_path}")
    with open(args.json_path, 'r') as f:
        predictions = json.load(f)
    
    frames = predictions.get('frames', [])
    print(f"Loaded predictions for {len(frames)} images")
    
    # Initialize counters
    class_counts = Counter()
    class_scores = defaultdict(list)
    boxes_per_image = []
    score_distribution = defaultdict(int)  # Bin scores into ranges
    
    # Process predictions
    for frame in frames:
        labels = frame.get('labels', [])
        # Filter by threshold
        valid_labels = [label for label in labels if label.get('score', 0) >= args.score_thr]
        boxes_per_image.append(len(valid_labels))
        
        for label in valid_labels:
            category = label.get('category', 'unknown')
            score = label.get('score', 0)
            
            # Update counters
            class_counts[category] += 1
            class_scores[category].append(score)
            
            # Bin score for distribution analysis
            score_bin = int(score * 10) / 10  # Round to nearest 0.1
            score_distribution[score_bin] += 1
    
    # Compute statistics
    total_detections = sum(class_counts.values())
    avg_boxes_per_image = np.mean(boxes_per_image) if boxes_per_image else 0
    median_boxes_per_image = np.median(boxes_per_image) if boxes_per_image else 0
    max_boxes_per_image = max(boxes_per_image) if boxes_per_image else 0
    
    # Calculate average scores by class
    avg_scores_by_class = {
        cls: np.mean(scores) for cls, scores in class_scores.items() if scores
    }
    
    # Print summary statistics
    print("\n===== Detection Analysis Results =====")
    print(f"Total images: {len(frames)}")
    print(f"Total detections: {total_detections} (with score >= {args.score_thr})")
    print(f"Average detections per image: {avg_boxes_per_image:.2f}")
    print(f"Median detections per image: {median_boxes_per_image:.2f}")
    print(f"Maximum detections per image: {max_boxes_per_image}")
    
    # Print per-class statistics
    print("\nPer-class statistics:")
    class_names = sorted(class_counts.keys())
    for cls in class_names:
        count = class_counts[cls]
        percentage = 100 * count / total_detections if total_detections else 0
        avg_score = avg_scores_by_class.get(cls, 0)
        print(f"{cls:15s}: {count:5d} detections ({percentage:5.1f}%), avg score: {avg_score:.4f}")
    
    # Create visualization directory
    vis_dir = os.path.join(args.output_dir, "analysis_visualizations")
    os.makedirs(vis_dir, exist_ok=True)
    
    # Plot class distribution
    plt.figure(figsize=(12, 6))
    bars = plt.bar(class_names, [class_counts[cls] for cls in class_names])
    plt.xticks(rotation=45, ha='right')
    plt.title(f'Object Detection Count by Class (score >= {args.score_thr})')
    plt.xlabel('Class')
    plt.ylabel('Number of Detections')
    plt.tight_layout()
    
    # Add count values on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{height:.0f}',
                ha='center', va='bottom', rotation=0)
    
    plt.savefig(os.path.join(vis_dir, 'class_distribution.png'))
    print(f"\nSaved class distribution plot to: {os.path.join(vis_dir, 'class_distribution.png')}")
    
    # Plot average confidence score by class
    plt.figure(figsize=(12, 6))
    bars = plt.bar(class_names, [avg_scores_by_class.get(cls, 0) for cls in class_names])
    plt.xticks(rotation=45, ha='right')
    plt.title(f'Average Confidence Score by Class (score >= {args.score_thr})')
    plt.xlabel('Class')
    plt.ylabel('Average Score')
    plt.ylim(0, 1.0)
    plt.tight_layout()
    
    # Add score values on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.3f}',
                ha='center', va='bottom', rotation=0)
    
    plt.savefig(os.path.join(vis_dir, 'avg_score_by_class.png'))
    print(f"Saved average score plot to: {os.path.join(vis_dir, 'avg_score_by_class.png')}")
    
    # Plot score distribution
    score_bins = sorted(score_distribution.keys())
    plt.figure(figsize=(10, 6))
    bars = plt.bar([str(round(b, 1)) for b in score_bins], [score_distribution[b] for b in score_bins])
    plt.title(f'Distribution of Confidence Scores (score >= {args.score_thr})')
    plt.xlabel('Confidence Score')
    plt.ylabel('Number of Detections')
    plt.tight_layout()
    
    # Add count values on top of bars
    for bar in bars:
        height = bar.get_height()
        if height > 0:
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{height:.0f}',
                    ha='center', va='bottom', rotation=0)
    
    plt.savefig(os.path.join(vis_dir, 'score_distribution.png'))
    print(f"Saved score distribution plot to: {os.path.join(vis_dir, 'score_distribution.png')}")
    
    # Plot detections per image histogram
    plt.figure(figsize=(10, 6))
    plt.hist(boxes_per_image, bins=range(0, max(boxes_per_image) + 2), alpha=0.7, rwidth=0.85)
    plt.title(f'Detections per Image (score >= {args.score_thr})')
    plt.xlabel('Number of Detections')
    plt.ylabel('Number of Images')
    plt.xticks(range(0, max(boxes_per_image) + 2, max(1, (max(boxes_per_image) + 2) // 20)))
    plt.grid(axis='y', alpha=0.75)
    plt.tight_layout()
    plt.savefig(os.path.join(vis_dir, 'detections_per_image.png'))
    print(f"Saved detections per image plot to: {os.path.join(vis_dir, 'detections_per_image.png')}")
    
    # Create performance report
    report = {
        "total_images": len(frames),
        "total_detections": total_detections,
        "avg_detections_per_image": avg_boxes_per_image,
        "median_detections_per_image": median_boxes_per_image,
        "max_detections_per_image": max_boxes_per_image,
        "per_class_statistics": {
            cls: {
                "count": class_counts[cls],
                "percentage": 100 * class_counts[cls] / total_detections if total_detections else 0,
                "avg_score": avg_scores_by_class.get(cls, 0)
            } for cls in class_names
        },
        "score_distribution": {str(k): v for k, v in score_distribution.items()}
    }
    
    # Save report to JSON
    report_file = os.path.join(args.output_dir, "performance_report.json")
    with open(report_file, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nSaved performance report to: {report_file}")


if __name__ == "__main__":
    main() 