#!/bin/bash
set -e
export DEBIAN_FRONTEND=noninteractive

echo "INFO: System driver supports CUDA 12.4. Aligning environment to this version."

# Step 1: Clean apt cache and update package lists.
echo "INFO: Cleaning apt cache and updating package lists..."
sudo apt-get clean
sudo apt-get update

# Step 2: Ensure the NVIDIA repository keyring is installed.
echo "INFO: Setting up CUDA repository keyring..."
wget -q https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
rm cuda-keyring_1.1-1_all.deb
sudo apt-get update

# Step 3: Install CUDA 12.4 runtime libraries to match the driver.
echo "INFO: Installing CUDA 12.4 runtime libraries..."
sudo apt-get -y install \
    cuda-libraries-12-4 \
    cuda-libraries-dev-12-4

# Step 4: Forcefully purge any existing TensorRT installations to prevent conflicts.
echo "INFO: Purging any potentially conflicting TensorRT packages..."
sudo apt-get -y purge "libnvinfer*" "tensorrt*" &>/dev/null
sudo apt-get -y autoremove &>/dev/null

# Step 5: Dynamically and strictly find the correct TensorRT version for CUDA 12.4.
echo "INFO: Searching for a compatible TensorRT version specifically for CUDA 12.4..."
TRT_VERSION_STRING=$(apt-cache madison libnvinfer-dev | grep 'cuda12\.4' | head -n 1 | awk '{print $3}')

if [ -z "$TRT_VERSION_STRING" ]; then
    echo "ERROR: Could not find a compatible TensorRT version for CUDA 12.4 in the apt repositories."
    exit 1
fi

echo "INFO: Found compatible TensorRT version for CUDA 12.4. Pinning installation to ${TRT_VERSION_STRING}"

# Step 6: Install all necessary TensorRT packages and ALL their dependencies, explicitly pinning every single package version.
# This is the final, most robust solution to prevent apt from choosing incompatible dependency versions.
echo "INFO: Installing TensorRT and all its dependencies with strict version pinning..."
sudo apt-get install -y --allow-downgrades \
    libnvinfer-dev=${TRT_VERSION_STRING} \
    libnvinfer10=${TRT_VERSION_STRING} \
    libnvinfer-headers-dev=${TRT_VERSION_STRING} \
    libnvinfer-headers-plugin-dev=${TRT_VERSION_STRING} \
    libnvonnxparsers-dev=${TRT_VERSION_STRING} \
    libnvonnxparsers10=${TRT_VERSION_STRING} \
    libnvinfer-plugin-dev=${TRT_VERSION_STRING} \
    libnvinfer-plugin10=${TRT_VERSION_STRING} \
    python3-libnvinfer=${TRT_VERSION_STRING} \
    libnvinfer-samples=${TRT_VERSION_STRING} \
    libnvinfer-lean-dev=${TRT_VERSION_STRING} \
    libnvinfer-lean10=${TRT_VERSION_STRING} \
    libnvinfer-dispatch-dev=${TRT_VERSION_STRING} \
    libnvinfer-dispatch10=${TRT_VERSION_STRING} \
    libnvinfer-vc-plugin-dev=${TRT_VERSION_STRING} \
    libnvinfer-vc-plugin10=${TRT_VERSION_STRING}

echo "INFO: Successfully installed CUDA 12.4 libraries and version-pinned TensorRT."

# Step 7: Verify the installation.
echo "INFO: Verifying installations..."
echo "NVIDIA Driver (from nvidia-smi):"
nvidia-smi | grep "CUDA Version"
echo "NVCC Version (may still show a different version, but runtime is now aligned):"
nvcc --version
echo "TensorRT (dpkg) Version:"
dpkg -l | grep libnvinfer

echo "INFO: Environment setup complete. The runtime environment is now aligned with the CUDA 12.4 driver."