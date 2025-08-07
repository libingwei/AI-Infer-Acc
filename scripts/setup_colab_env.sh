#!/bin/bash
set -e

echo "INFO: Starting environment setup for CUDA 12.4 and a compatible TensorRT."

# Step 1: Purge any existing CUDA and TensorRT installations to ensure a clean slate.
echo "INFO: Purging existing CUDA installations..."
sudo apt-get -y purge "cuda-*" "libcudnn8*" "libnvidi*" "tensorrt*" &>/dev/null
sudo apt-get -y autoremove &>/dev/null
sudo apt-get -y clean

# Step 2: Install CUDA Toolkit 12.4, which is compatible with the Colab driver.
echo "INFO: Installing CUDA Toolkit 12.4..."
wget -q https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
rm cuda-keyring_1.1-1_all.deb
sudo apt-get update
sudo apt-get -y install cuda-toolkit-12-4

# Step 3: Dynamically find and install the correct TensorRT version for CUDA 12.4.
echo "INFO: Searching for a compatible TensorRT version for CUDA 12.4..."

# Search for available libnvinfer-dev packages and filter for ones built for cuda12.4
# Then extract the version string of the first match.
TRT_VERSION_STRING=$(apt-cache madison libnvinfer-dev | grep 'cuda12.4' | head -n 1 | awk '{print $3}')

if [ -z "$TRT_VERSION_STRING" ]; then
    echo "ERROR: Could not find a compatible TensorRT version for CUDA 12.4."
    echo "Please check the NVIDIA repositories or your network connection."
    exit 1
fi

echo "INFO: Found compatible TensorRT version. Pinning installation to ${TRT_VERSION_STRING}"

# Install all necessary TensorRT packages, pinning them to the dynamically found version.
# The main package to target is `libnvinfer-dev`. It will pull other dependencies.
sudo apt-get install -y --allow-downgrades \
    libnvinfer-dev=${TRT_VERSION_STRING} \
    libnvonnxparsers-dev=${TRT_VERSION_STRING} \
    libnvinfer-samples=${TRT_VERSION_STRING} \
    python3-libnvinfer=${TRT_VERSION_STRING}

echo "INFO: Successfully installed version-pinned TensorRT."


# Step 4: Verify the installation.
echo "INFO: Verifying installations..."
echo "NVCC Version:"
nvcc --version

echo "TensorRT (dpkg) Version:"
dpkg -l | grep libnvinfer

echo "INFO: Environment setup complete."