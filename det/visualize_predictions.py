import os
import json
import cv2
from scalabel.label.io import load
import argparse

def visualize():
    # --- Configuration ---
    parser = argparse.ArgumentParser(description='Visualize predictions from a det.json file.')
    parser.add_argument('--json-path', type=str, default='./inference_output/faster_rcnn_r50_fpn_1x_sample/det.json',
                        help='Path to the input prediction JSON file (det.json).')
    parser.add_argument('--image-dir', type=str, default='/media/skumar/External/bdd100k/data/bdd100k_images_100k/bdd100k/images/100k/val/',
                        help='Path to the directory containing the original validation images.')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Directory to save visualized images. Defaults to a subfolder within the JSON file\'s directory.')
    parser.add_argument('--limit', type=int, default=None,
                        help='Limit the number of images to visualize.')
    parser.add_argument('--score-thr', type=float, default=0.3,
                        help='Score threshold for displaying detections.')

    args = parser.parse_args()

    pred_json_path = args.json_path
    image_dir = args.image_dir
    output_dir = args.output_dir
    limit = args.limit
    score_thr = args.score_thr

    if not output_dir:
        output_dir = os.path.join(os.path.dirname(pred_json_path), 'visualizations_from_json')

    print(f"Input JSON:     {pred_json_path}")
    print(f"Image Directory: {image_dir}")
    print(f"Output Directory: {output_dir}")
    print(f"Score Threshold: {score_thr}")
    if limit:
        print(f"Visualization Limit: {limit}")

    os.makedirs(output_dir, exist_ok=True)

    # Define colors for different classes (optional, but helpful)
    # Simple color cycling for now
    colors = [
        (0, 0, 255), (0, 255, 0), (255, 0, 0), (0, 255, 255),
        (255, 0, 255), (255, 255, 0), (128, 0, 255), (0, 128, 255),
        (255, 128, 0), (128, 255, 0)
    ]
    category_colors = {}
    color_index = 0

    # --- Load predictions ---
    if not os.path.exists(pred_json_path):
        print(f"Error: Prediction JSON file not found at {pred_json_path}")
        return

    print(f"Loading predictions from: {pred_json_path}")
    try:
        predictions = load(pred_json_path)
        if not hasattr(predictions, 'frames') or not predictions.frames:
            print("Warning: No frames found in the prediction file or file format incorrect.")
            return
        print(f"Loaded predictions for {len(predictions.frames)} images.")
    except Exception as e:
        print(f"Error loading JSON file: {e}")
        return

    # --- Process each frame (image) ---
    processed_count = 0
    for frame in predictions.frames:
        if limit is not None and processed_count >= limit:
            print(f"Reached visualization limit of {limit}.")
            break

        image_name = frame.name
        if not image_name:
            print("Warning: Frame found with no name, skipping.")
            continue

        image_path = os.path.join(image_dir, image_name)
        output_path = os.path.join(output_dir, image_name)

        if not os.path.exists(image_path):
            print(f"Warning: Image not found at {image_path}, skipping {image_name}.")
            continue

        # Load image
        image = cv2.imread(image_path)
        if image is None:
            print(f"Warning: Could not read image {image_path}, skipping {image_name}.")
            continue

        # --- Draw bounding boxes from labels ---
        if frame.labels:
            for label in frame.labels:
                # Check score threshold
                score = label.score if label.score is not None else -1.0
                if score < score_thr:
                    continue

                if label.box2d:
                    # Scalabel Box2D stores x1, y1, x2, y2
                    x1 = int(label.box2d.x1)
                    y1 = int(label.box2d.y1)
                    x2 = int(label.box2d.x2)
                    y2 = int(label.box2d.y2)

                    category = label.category if label.category else "N/A"

                    # Get color for category
                    if category not in category_colors:
                        category_colors[category] = colors[color_index % len(colors)]
                        color_index += 1
                    color = category_colors[category]

                    # Draw rectangle
                    cv2.rectangle(image, (x1, y1), (x2, y2), color, 2) # thickness 2

                    # Prepare label text
                    label_text = f"{category}: {score:.2f}"

                    # Put text above the rectangle
                    text_size, _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                    text_w, text_h = text_size
                    # Ensure background doesn't go off-screen (top)
                    bg_y1 = max(y1 - text_h - 4, 0)
                    bg_y2 = max(y1, text_h + 4) # Ensure text fits if box is near top
                    text_y = max(y1 - 5, text_h) # Position text correctly

                    cv2.rectangle(image, (x1, bg_y1), (x1 + text_w, bg_y2), color, -1) # Background for text
                    cv2.putText(image, label_text, (x1, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1) # White text

        # --- Save the visualized image ---
        cv2.imwrite(output_path, image)
        processed_count += 1
        if processed_count % 20 == 0: # Print progress every 20 images
             print(f"  Processed {processed_count}/{len(predictions.frames)} images...")

    print(f"Finished visualization. {processed_count} images saved to: {output_dir}")

if __name__ == "__main__":
    visualize() 