#!/bin/bash

# Define the configuration file for the Faster R-CNN model (R-50-FPN 1x)
# This config assumes the BDD100K dataset is set up as expected by the mmdetection framework.
# The specific validation dataset used will be determined by the settings within this config file.
CONFIG_FILE="./configs/det/faster_rcnn_r50_fpn_1x_det_bdd100k.py"

# Define the output directory for the inference results
OUTPUT_DIR="./inference_output/faster_rcnn_r50_fpn_1x"

# Create the output directory if it doesn't exist
mkdir -p $OUTPUT_DIR

# Run the inference using test.py
# The script will automatically download the corresponding weights specified in the config
# (usually via the 'load_from' field) if they don't exist locally.
echo "Running inference with config: $CONFIG_FILE"
echo "Saving formatted results to: $OUTPUT_DIR"

python ./test.py ${CONFIG_FILE} \
    --format-only \
    --format-dir ${OUTPUT_DIR}

echo "Inference finished. Results saved in $OUTPUT_DIR" 