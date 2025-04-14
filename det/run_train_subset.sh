#!/bin/bash

# Script to run train.py for 1 epoch on a subset of BDD100k training data.

# Get the directory where the script is located
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

# Change to the script's directory so relative paths work correctly
cd "$SCRIPT_DIR" || exit 1

# Define the configuration file for the Faster R-CNN model
# This config file contains the base model, dataset (train/val), and schedule settings.
# CONFIG_FILE="./configs/det/faster_rcnn_r50_fpn_1x_det_bdd100k.py"
CONFIG_FILE="./configs/det/minimal_faster_rcnn.py"

# Define the working directory for this specific training run
# Logs, checkpoints, etc., will be saved here.
WORK_DIR="./work_dirs/minimal_faster_rcnn_subset_1epoch"

# Create the working directory if it doesn't exist (train.py also does this, but good practice)
mkdir -p $WORK_DIR

# Run the training script
# train.py will load the CONFIG_FILE, modify it to use a subset (first 200 samples)
# and run for only 1 epoch, saving results to WORK_DIR.
echo "Starting subset training (1 epoch) with config: $CONFIG_FILE"
echo "Results will be saved in: $WORK_DIR"

python ./train.py ${CONFIG_FILE} --work-dir ${WORK_DIR}

# Check the exit code of the python script
exit_code=$?
if [ $exit_code -eq 0 ]; then
    echo "Subset training finished successfully."
else
    echo "Subset training encountered an error (Exit Code: $exit_code)."
fi

exit $exit_code 