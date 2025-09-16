#!/bin/bash

# Cloud setup script for audio classification training
# This script sets up the environment on various cloud platforms

set -e

echo "Setting up audio classification training environment..."

# Detect platform
if [ -n "$COLAB_GPU" ]; then
    PLATFORM="colab"
    echo "Detected Google Colab environment"
elif [ -n "$KAGGLE_KERNEL_RUN_TYPE" ]; then
    PLATFORM="kaggle"
    echo "Detected Kaggle environment"
elif [ -n "$SM_MODEL_DIR" ]; then
    PLATFORM="sagemaker"
    echo "Detected AWS SageMaker environment"
else
    PLATFORM="local"
    echo "Detected local environment"
fi

# Install system dependencies
echo "Installing system dependencies..."
if command -v apt-get &> /dev/null; then
    sudo apt-get update
    sudo apt-get install -y ffmpeg libsndfile1
elif command -v yum &> /dev/null; then
    sudo yum update -y
    sudo yum install -y ffmpeg libsndfile
fi

# Install Python dependencies
echo "Installing Python dependencies..."
pip install -r requirements-cloud.txt

# Download ESC-50 dataset if not present
if [ ! -d "data/ESC-50" ]; then
    echo "Downloading ESC-50 dataset..."
    mkdir -p data
    cd data
    wget -q https://github.com/karolpiczak/ESC-50/archive/master.zip
    unzip -q master.zip
    mv ESC-50-master ESC-50
    rm master.zip
    cd ..
    echo "ESC-50 dataset downloaded successfully!"
else
    echo "ESC-50 dataset already exists!"
fi

# Initialize git submodules
echo "Initializing git submodules..."
git submodule update --init --recursive

# Create necessary directories
echo "Creating directories..."
mkdir -p checkpoints
mkdir -p results
mkdir -p logs

# Set up Weights & Biases (optional)
if [ -n "$WANDB_API_KEY" ]; then
    echo "Setting up Weights & Biases..."
    wandb login
else
    echo "WANDB_API_KEY not found. Skipping W&B setup."
    echo "To enable experiment tracking, set WANDB_API_KEY environment variable."
fi

# Platform-specific optimizations
case $PLATFORM in
    "colab")
        echo "Applying Colab optimizations..."
        # Enable memory growth for TensorFlow (if used)
        export TF_FORCE_GPU_ALLOW_GROWTH=true
        # Set optimal number of workers
        export OMP_NUM_THREADS=2
        ;;
    "kaggle")
        echo "Applying Kaggle optimizations..."
        # Kaggle-specific settings
        export OMP_NUM_THREADS=4
        ;;
    "sagemaker")
        echo "Applying SageMaker optimizations..."
        # SageMaker-specific settings
        export OMP_NUM_THREADS=8
        ;;
    "local")
        echo "Local environment detected. No special optimizations applied."
        ;;
esac

echo "Setup completed successfully!"
echo ""
echo "To start training, run:"
echo "  python scripts/cloud_train.py --model mamba"
echo "  python scripts/cloud_train.py --model liquid_s4"
echo "  python scripts/cloud_train.py --model vjepa2"
echo ""
echo "For help, run: python scripts/cloud_train.py --help"
