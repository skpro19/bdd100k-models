#!/bin/bash

# Script to run train.py on a subset of BDD100k training data with configurable epochs

# Get the directory where the script is located
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

# Change to the script's directory so relative paths work correctly
cd "$SCRIPT_DIR" || exit 1

# Define the configuration file for the Faster R-CNN model
CONFIG_FILE="./configs/det/minimal_faster_rcnn.py"

# Number of epochs (default: 1)
EPOCHS=${1:-1}

# Define the working directory for this specific training run
WORK_DIR="./work_dirs/minimal_faster_rcnn_subset_${EPOCHS}epoch"

# Create the working directory if it doesn't exist
mkdir -p $WORK_DIR

# Run the training script
echo "Starting subset training (${EPOCHS} epoch(s)) with config: $CONFIG_FILE"
echo "Results will be saved in: $WORK_DIR"

python ./train.py ${CONFIG_FILE} --work-dir ${WORK_DIR} --epochs ${EPOCHS}

# Check the exit code
exit_code=$?
if [ $exit_code -eq 0 ]; then
    echo "Training finished successfully."
else
    echo "Training encountered an error (Exit Code: $exit_code)."
fi

exit $exit_code
