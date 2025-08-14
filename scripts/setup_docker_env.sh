#!/bin/bash
set -e
export DEBIAN_FRONTEND=noninteractive

echo "INFO: Setting up Docker development environment with CUDA 12.4, TensorRT, and SSH service"

# Step 1: Update system and install basic dependencies
echo "INFO: Updating system and installing basic dependencies..."
apt-get update
apt-get install -y --no-install-recommends \
    wget curl gnupg2 software-properties-common \
    build-essential cmake git vim \
    openssh-server sudo net-tools

# Step 2: Clean up any existing CUDA installations
echo "INFO: Cleaning up existing CUDA installations..."
apt-get remove --purge -y 'cuda*' 'nvidia*' || true
apt-get autoremove -y || true

# Step 3: Set up NVIDIA/CUDA repository
echo "INFO: Setting up NVIDIA repository..."
wget -q https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
dpkg -i cuda-keyring_1.1-1_all.deb
rm cuda-keyring_1.1-1_all.deb
apt-get update

# Step 4: Install complete CUDA 12.4 toolkit
echo "INFO: Installing CUDA 12.4 complete toolkit..."
apt-get install -y \
    cuda-toolkit-12-4 \
    cuda-runtime-12-4 \
    cuda-libraries-12-4 \
    cuda-libraries-dev-12-4 \
    cuda-compiler-12-4 \
    cuda-nvcc-12-4 \
    cuda-cudart-dev-12-4

# Step 5: Install TensorRT
echo "INFO: Installing TensorRT for CUDA 12.4..."
# Remove any existing TensorRT
apt-get purge -y "libnvinfer*" "tensorrt*" || true

# Find compatible TensorRT version
TRT_VERSION_STRING=$(apt-cache madison libnvinfer-dev | grep 'cuda12\.4' | head -n 1 | awk '{print $3}')

if [ -z "$TRT_VERSION_STRING" ]; then
    echo "WARNING: Could not find TensorRT for CUDA 12.4. Installing latest available version."
    apt-get install -y \
        libnvinfer-dev \
        libnvonnxparsers-dev \
        libnvinfer-plugin-dev \
        python3-libnvinfer
else
    echo "INFO: Installing TensorRT version: ${TRT_VERSION_STRING}"
    apt-get install -y --allow-downgrades \
        libnvinfer-dev=${TRT_VERSION_STRING} \
        libnvinfer10=${TRT_VERSION_STRING} \
        libnvonnxparsers-dev=${TRT_VERSION_STRING} \
        libnvonnxparsers10=${TRT_VERSION_STRING} \
        libnvinfer-plugin-dev=${TRT_VERSION_STRING} \
        libnvinfer-plugin10=${TRT_VERSION_STRING} \
        python3-libnvinfer=${TRT_VERSION_STRING}
fi

# Step 6: Install OpenCV
echo "INFO: Installing OpenCV..."
apt-get install -y \
    libopencv-dev \
    python3-opencv

# Step 7: Set up CUDA environment
echo "INFO: Setting up CUDA environment..."
export CUDA_HOME=/usr/local/cuda-12.4
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# Create symlink
ln -sf /usr/local/cuda-12.4 /usr/local/cuda

# Add to system-wide environment
cat >> /etc/environment << EOF
CUDA_HOME=/usr/local/cuda-12.4
PATH=/usr/local/cuda-12.4/bin:\$PATH
LD_LIBRARY_PATH=/usr/local/cuda-12.4/lib64:\$LD_LIBRARY_PATH
EOF

# Add to bash profile
cat >> /root/.bashrc << EOF
export CUDA_HOME=/usr/local/cuda-12.4
export PATH=\$CUDA_HOME/bin:\$PATH
export LD_LIBRARY_PATH=\$CUDA_HOME/lib64:\$LD_LIBRARY_PATH
EOF

# Step 8: Configure SSH service
echo "INFO: Configuring SSH service..."
mkdir -p /var/run/sshd

# Configure SSH to allow root login and password authentication
sed -i 's/#PermitRootLogin prohibit-password/PermitRootLogin yes/' /etc/ssh/sshd_config
sed -i 's/#PasswordAuthentication yes/PasswordAuthentication yes/' /etc/ssh/sshd_config
sed -i 's/#PubkeyAuthentication yes/PubkeyAuthentication yes/' /etc/ssh/sshd_config
sed 's@session\s*required\s*pam_loginuid.so@session optional pam_loginuid.so@g' -i /etc/pam.d/sshd

# Generate SSH host keys
ssh-keygen -A

# Set default root password (should be changed via environment variable)
echo "root:dockerdev" | chpasswd

# Step 9: Create startup script
cat > /usr/local/bin/start-services.sh << 'EOF'
#!/bin/bash

# Set root password from environment variable if provided
if [ -n "$ROOT_PASSWORD" ]; then
    echo "Setting root password from environment variable"
    echo "root:$ROOT_PASSWORD" | chpasswd
else
    echo "WARNING: Using default password. Set ROOT_PASSWORD environment variable for security."
fi

# Start SSH service
echo "Starting SSH server..."
/usr/sbin/sshd -D &

# Keep container running
echo "Docker development environment ready!"
echo "SSH server is running on port 22"
echo "Connect with: ssh root@<container_ip>"
echo "Default password: dockerdev (change with ROOT_PASSWORD env var)"

# Wait for services
wait
EOF

chmod +x /usr/local/bin/start-services.sh

# Step 10: Clean up
echo "INFO: Cleaning up..."
apt-get clean
rm -rf /var/lib/apt/lists/*

# Step 11: Verify installation
echo "INFO: Verifying installation..."
echo "CUDA Version:"
/usr/local/cuda/bin/nvcc --version || echo "nvcc not found"
echo "TensorRT Version:"
dpkg -l | grep libnvinfer | head -5
echo "OpenCV Version:"
pkg-config --modversion opencv4 2>/dev/null || echo "OpenCV pkg-config not available"

echo "INFO: Docker environment setup complete!"
echo "INFO: Run 'source /root/.bashrc' to reload environment variables"
echo "INFO: Start services with '/usr/local/bin/start-services.sh'"
