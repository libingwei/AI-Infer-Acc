#!/bin/bash
set -e

echo "INFO: Starting environment setup for CUDA 12.4 and a compatible TensorRT."

# Step 1: Purge any existing CUDA and TensorRT installations to ensure a clean slate.
echo "INFO: Purging existing CUDA installations..."
sudo apt-get -y purge "cuda-*" "libcudnn8*" "libnvidi*" "tensorrt*"
sudo apt-get -y autoremove
sudo apt-get -y clean

# Step 2: Install CUDA Toolkit 12.4, which is compatible with the Colab driver.
echo "INFO: Installing CUDA Toolkit 12.4..."
# Add NVIDIA's repository keys and setup the repo
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
# We need to remove the temp file to avoid issues with the next wget
rm cuda-keyring_1.1-1_all.deb
sudo apt-get update

# Install the specific CUDA toolkit version
sudo apt-get -y install cuda-toolkit-12-4

# Step 3: Find and install the correct TensorRT version for CUDA 12.4.
echo "INFO: Installing TensorRT for CUDA 12.4..."

# Add the TensorRT repository
wget https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu2204/x86_64/nvidia-machine-learning-repo-ubuntu2204_1.0.0-1_amd64.deb
sudo dpkg -i nvidia-machine-learning-repo-ubuntu2204_1.0.0-1_amd64.deb
rm nvidia-machine-learning-repo-ubuntu2204_1.0.0-1_amd64.deb
sudo apt-get update

# Define the target TensorRT version that is compatible with CUDA 12.4
# We use a wildcard to match all required packages to this version.
# Note: You can find available versions with `apt-cache madison libnvinfer-dev`
TRT_VERSION="10.0.1-1+cuda12.4"

echo "INFO: Pinning installation to TensorRT version ${TRT_VERSION}"

# Install all necessary TensorRT packages, pinning them to the specific version
sudo apt-get install -y \
    libnvinfer-dev=${TRT_VERSION} \
    libnvinfer-plugin-dev=${TRT_VERSION} \
    libnvonnxparsers-dev=${TRT_VERSION} \
    libnvinfer-samples=${TRT_VERSION} \
    python3-libnvinfer=${TRT_VERSION}

echo "INFO: Successfully installed version-pinned TensorRT."


# Step 4: Verify the installation.
echo "INFO: Verifying installations..."
echo "NVCC Version:"
nvcc --version

echo "TensorRT (dpkg) Version:"
dpkg -l | grep libnvinfer

echo "INFO: Environment setup complete."