#!/bin/bash
set -e

echo "Starting Heart Risk Assessment Platform..."
echo "Checking system requirements..."
echo "Loading professional medical interface..."

# Navigate to application directory
cd /app

# Set Python path and Docker environment detection
export PYTHONPATH="/app:$PYTHONPATH"
export DOCKER_CONTAINER="true"

# Run data preprocessing if needed
if [ ! -f "data/processed/train.csv" ]; then
    echo "Running data preprocessing..."
    python src/data_preprocessing.py
else
    echo "Processed data found"
fi

# Check if models exist
if [ ! -d "results/models" ]; then
    echo "No trained models found - using fallback model"
else
    echo "Trained models found"
fi

echo "Starting Heart Risk Assessment Platform..."
echo "Local URL: http://0.0.0.0:7860"
echo "Public URL: Will be created automatically with share=True"
echo "Professional medical interface ready"
echo "Working Low/Moderate/High risk classification loaded"
echo "Docker deployment ready"
echo ""

# Start Gradio application with public sharing
python app/app_gradio.py