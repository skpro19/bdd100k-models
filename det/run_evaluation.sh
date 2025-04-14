#!/bin/bash

# Define the configuration file for the Faster R-CNN model (R-50-FPN 1x)
CONFIG_FILE="./configs/det/faster_rcnn_r50_fpn_1x_det_bdd100k.py"

# Define the output directory for the evaluation results (logs, metrics)
EVAL_OUTPUT_DIR="./evaluation_output/faster_rcnn_r50_fpn_1x"

# Create the output directory if it doesn't exist
mkdir -p $EVAL_OUTPUT_DIR

# Run the evaluation using test.py
# This will run inference on the entire validation set defined in the config
# and then compute the evaluation metrics (e.g., mAP).
# The script will automatically download the corresponding weights specified
# in the config if they don't exist locally.
echo "Running evaluation with config: $CONFIG_FILE"
echo "Saving logs and potential results to: $EVAL_OUTPUT_DIR"

python ./test.py ${CONFIG_FILE} \
    --work-dir ${EVAL_OUTPUT_DIR}

echo "Evaluation finished. Check the output above and files in $EVAL_OUTPUT_DIR for results." 