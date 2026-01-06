#!/bin/bash
set -e

echo "Starting Heart Risk Prediction Application..."

# Run data preprocessing if needed
if [ ! -f "data/processed/train.csv" ]; then
    echo "Running data preprocessing..."
    python src/data_preprocessing.py
fi

# Start Gradio application
echo "Starting Gradio interface..."
python app/app_gradio.py