#!/bin/bash

# Define the configuration file for the Faster R-CNN model (R-50-FPN 1x)
CONFIG_FILE="./configs/det/faster_rcnn_r50_fpn_1x_det_bdd100k.py"

# Define the output directory for the SAMPLE evaluation results (logs, metrics)
EVAL_OUTPUT_DIR="./evaluation_output/faster_rcnn_r50_fpn_1x_sample"

# Define the number of samples to process
MAX_SAMPLES=100

# Create the output directory if it doesn't exist
mkdir -p $EVAL_OUTPUT_DIR

# Run the evaluation using test.py on a SAMPLE of the validation set
# This will run inference on MAX_SAMPLES images and then compute metrics.
echo "Running evaluation with config: $CONFIG_FILE on $MAX_SAMPLES samples"
echo "Saving logs and potential results to: $EVAL_OUTPUT_DIR"

python ./test.py ${CONFIG_FILE} \
    --work-dir ${EVAL_OUTPUT_DIR} \
    --max-samples ${MAX_SAMPLES} \
    --show-dir ${EVAL_OUTPUT_DIR}/visualizations_sample # Satisfy assertion and maybe trigger eval

echo "Sample evaluation finished. Check the output above and files in $EVAL_OUTPUT_DIR for results." 