#!/bin/bash
set -e

echo "🫀 Starting Heart Risk Prediction Application..."
echo "📋 Checking system requirements..."

# Navigate to application directory
cd /app

# Set Python path
export PYTHONPATH="/app:$PYTHONPATH"

# Run data preprocessing if needed
if [ ! -f "data/processed/train.csv" ]; then
    echo "⚙️  Running data preprocessing..."
    python src/data_preprocessing.py
else
    echo "✅ Processed data found"
fi

# Check if models exist
if [ ! -d "results/models" ]; then
    echo "⚠️  No trained models found - using fallback model"
else
    echo "✅ Trained models found"
fi

echo "🚀 Starting Professional Heart Disease Risk Prediction App..."
echo "📱 Local URL: http://0.0.0.0:7860"
echo "🌐 Public URL: Will be generated automatically with share=True"
echo "🐳 Docker deployment ready"
echo ""

# Start Gradio application with public sharing
python app/app_gradio.py