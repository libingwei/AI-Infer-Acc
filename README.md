# AI-Infer-Acceleration
AI推理加速计划

## 项目介绍

本项目旨在通过优化AI模型的推理过程，提高推理速度和效率。我们将从以下几个方面进行优化：

1. 模型量化
2. 模型剪枝
3. 模型压缩
4. 模型并行
5. 模型加速 
6. 模型部署
7. 模型评估

## 构建项目 (Building the Project)

本项目使用 CMake 进行构建管理。请确保您的环境中已安装 NVIDIA CUDA Toolkit 和 TensorRT。
> 在Google Colab中构建安装TensorRT. 请确保在Google Colab中运行以下命令，以确保环境配置正确。
```bash
bash scripts/setup_colab_env.sh
```

### 1. 准备环境与依赖

首先，安装项目所需的 Python 依赖，用于生成 ONNX 模型。

```bash
pip install -r requirements.txt
```

### 2. 生成 ONNX 模型

运行脚本来生成用于后续步骤的 ONNX 模型文件。脚本会自动处理预训练权重的下载和缓存。

```bash
python scripts/generate_onnx_model.py
```

### 3. 编译 C++ 程序

我们推荐使用 CMake 的“不切换目录”构建方式。所有命令都在项目根目录下执行。

```bash
# 1. 配置项目 (首次构建或 CMakeLists.txt 变更后执行)
cmake -S . -B build

# 2. 执行构建 (使用4个核心并行编译以加快速度)
cmake --build build -- -j4
```

编译成功后，所有可执行文件将位于 `bin/` 目录下。

### 4. 运行与性能测试

#### 引擎转换

使用 `onnx_to_trt` 程序将 ONNX 模型转换为不同精度的 TensorRT 引擎。

```bash
# 用法: ./bin/onnx_to_trt <输入ONNX路径> <输出文件名前缀> <精度>

# 生成 FP32 引擎 (models/resnet18.trt)
./bin/onnx_to_trt models/resnet18.onnx models/resnet18 fp32

# 生成 FP16 引擎 (models/resnet18_fp16.trt)
./bin/onnx_to_trt models/resnet18.onnx models/resnet18 fp16
```

#### 推理与基准测试

使用 `trt_inference` 程序加载引擎进行推理，并进行性能测试。

```bash
# 用法: ./bin/trt_inference <引擎路径> [批处理大小]

# 测试 FP32 引擎, batch size = 32
./bin/trt_inference models/resnet18.trt 32

# 测试 FP16 引擎, batch size = 32
./bin/trt_inference models/resnet18_fp16.trt 32
```

## 性能基准测试 (Performance Benchmark)

我们在 Google Colab 的 NVIDIA T4 GPU (15GB 显存) 上对 ResNet18 模型进行了基准测试。所有延迟数据均为 100 次推理运行的平均值。

| 精度模式 | Batch Size | 平均延迟 (ms) | 吞吐量 (FPS) | 性能提升 (vs FP32) |
| :--- | :---: | :---: | :---: | :---: |
| FP32 | 1 | 2.26 ms | 441 FPS | - |
| **FP16** | **1** | **1.01 ms** | **994 FPS** | **2.25x** |
| FP32 | 32 | 25.76 ms | 1,242 FPS | - |
| **FP16** | **32** | **8.02 ms** | **3,987 FPS** | **3.21x** |

### 结论

1.  **FP16 加速效果显著**: 在 T4 GPU 上，使用 FP16 精度可带来 **2-3 倍** 的性能提升。在批处理大小为 32 时，加速效果尤为明显，吞吐量提升超过 3.2 倍。
2.  **批处理 (Batching) 是提升吞吐量的关键**: 对于 FP16 模型，将批处理大小从 1 增加到 32，总吞吐量（FPS）**提升了 4 倍**。这证明了在推理时应尽可能使用更大的批处理大小，以充分利用 GPU 的并行计算能力。
