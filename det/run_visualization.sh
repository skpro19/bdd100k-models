#!/bin/bash

# Script to run the visualize_predictions.py script

# Default arguments can be overridden by passing them to this script
# Example: ./run_visualization.sh --limit 10 --score-thr 0.5

echo "Running visualization script..."

python visualize_predictions.py "$@"

exit_code=$?

if [ $exit_code -eq 0 ]; then
    echo "Visualization script finished successfully."
else
    echo "Visualization script encountered an error (Exit Code: $exit_code)."
fi

exit $exit_code 