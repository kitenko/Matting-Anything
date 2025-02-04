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
    opencv-python==4.11.0.86 \
    gradio==5.14.0 \
    easydict==1.13 \
    scikit-image==0.25.1 \
    tensorboardX==2.6.2.2 \
    hydra-core==1.3.2 \
    iopath==0.1.10 \
    toml==0.10.2 \
    imgaug==0.4.0 \
    albumentations==2.0.3 \
    transformers==4.48.2 \
    addict==2.4.0 \
    yapf==0.43.0 \
    timm==1.0.14 \
    av==14.1.0 \
    decord==0.6.0 \



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
