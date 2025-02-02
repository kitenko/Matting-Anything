#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

echo "Installing system dependencies..."

# Update package list and install necessary system libraries
apt-get update && apt-get install -y \
    libgl1 

echo "Upgrading pip to the latest version..."
# Upgrade pip to ensure compatibility with the latest packages
pip install --upgrade pip

echo "Installing primary Python dependencies..."

rm -rf /usr/local/lib/python3.10/dist-packages/cv2
rm -rf /usr/local/lib/python3.10/dist-packages/opencv*

# Install primary Python dependencies with specified versions
pip install \
    opencv-contrib-python==4.10.0.84 \
    pytorch-lightning==2.4.0 \
    optuna==3.6.1 \
    jsonargparse==4.33.2 \
    datasets==3.1.0 \
    cylimiter==0.4.2 \
    transformers==4.46.2 \
    python-dotenv==1.0.1 \
    albumentations==1.4.21 \
    toml==0.10.2 \
    easydict==1.13 \
    scikit-image==0.25.0 \
    tensorboardX==2.6.2.2 \
    imgaug==0.4.0 \
    supervision==0.25.1 \
    addict==2.4.0 \
    yapf==0.43.0 \
    timm==1.0.14 \
    hydra-core==1.3.2 \
    iopath==0.1.10 \
    gradio==5.14.0 \
    decord==0.6.0 \
    # moviepy==2.1.2 \


echo "Installing jsonargparse with signatures support..."
# Upgrade jsonargparse with signatures support
pip install -U 'jsonargparse[signatures]>=4.27.7'

echo "Installing development tools..."

# Install development tools with specified versions
pip install \
    pylint==3.3.1 \
    black==24.8.0 \
    pre-commit==3.8.0 \
    mypy==1.11.2 \
    types-pyyaml==6.0.12.20240917 \
    pytest-mock==3.14.0

echo "Setting up pre-commit hooks..."

# Check if we are in a Git repository before installing pre-commit hooks
if git rev-parse --is-inside-work-tree > /dev/null 2>&1; then
    pre-commit install
    echo "Pre-commit hooks set up successfully."
else
    echo "Not in a Git repository. Skipping pre-commit hook setup."
fi

echo "All dependencies have been successfully installed."
