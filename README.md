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
> 在Kaggle环境里需安装TensorRT。请确保在Kaggle环境中运行以下命令，以确保环境配置正确。
```bash
bash scripts/setup_kaggle_env.sh
```

Kaggle 使用说明：
- 将从 NVIDIA 官网下载的 TensorRT tar 包（例如：`TensorRT-8.6.1.6.Linux.x86_64-gnu.cuda-12.0.tar.gz`）上传为 Kaggle 私有数据集，并在 Notebook 右侧 Add Data 挂载。
- 运行上面的脚本会在 `/kaggle/working/` 下解压，并生成环境变量文件 `/kaggle/working/tensorrt_env.sh`。
- 运行脚本后在 Notebook 中执行：
	```bash
	source /kaggle/working/tensorrt_env.sh
	trtexec --version || echo 'trtexec 可能未包含在该包中'
	```
- 也可显式传入 tar 包路径：
	```bash
	bash scripts/setup_kaggle_env.sh /kaggle/input/<your-dataset>/TensorRT-8.6.1.6.Linux.x86_64-gnu.cuda-12.0.tar.gz
	```

### 下载代码
```bash
git clone https://github.com/libingwei/AI-Infer-Acc.git
cd AI-Infer-Acc
git submodule update --init --recursive
```

### 1. 准备环境与依赖

首先，安装项目所需的 Python 依赖，用于生成 ONNX 模型。

```bash
pip install -r requirements.txt
```

### 2. INT8 量化准备 (INT8 Quantization Prep)

为了执行 INT8 量化，您需要一个校准数据集。我们提供了一个脚本来自动下载一个约500张图片的COCO子集。

```bash
# 首先，安装下载脚本所需的库
pip install -r requirements.txt

# 然后，运行脚本下载并解压数据
# 数据会被存放在项目根目录下的 `calibration_data/` 文件夹中
python scripts/download_calibration_data.py
```

### 3. 生成 ONNX 模型


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
# 生成 FP16 引擎 (models/resnet18_fp16.trt)
./bin/onnx_to_trt models/resnet18.onnx models/resnet18 fp16

# 生成 INT8 引擎 (models/resnet18_int8.trt)
# (请确保您已按步骤2准备好了校准数据集)
./bin/onnx_to_trt models/resnet18.onnx models/resnet18 int8
```

#### 推理与基准测试

使用 `trt_inference` 程序加载引擎进行推理，并进行性能测试。

```bash
# 用法: ./bin/trt_inference <引擎路径> [批处理大小]

# 测试 FP32 引擎, batch size = 32
./bin/trt_inference models/resnet18.trt 32

# 测试 FP16 引擎, batch size = 32
./bin/trt_inference models/resnet18_fp16.trt 32

# 测试 INT8 引擎, batch size = 32
./bin/trt_inference models/resnet18_int8.trt 32

# 使用非默认 CUDA Stream（推荐用于异步/并发场景）
./bin/trt_inference models/resnet18_int8.trt 32

# 使用默认 CUDA Stream（Legacy stream 0）
./bin/trt_inference models/resnet18_int8.trt 32 --use-default-stream

# 也可通过环境变量切换默认流
USE_DEFAULT_STREAM=true ./bin/trt_inference models/resnet18_int8.trt 32
```


#### 推理与基准测试

使用 `trt_inference` 程序加载引擎进行推理，并进行性能测试。

```bash
# 用法: ./bin/trt_inference <引擎路径> [批处理大小]

# 测试 FP32 引擎, batch size = 32
./bin/trt_inference models/resnet18.trt 32

# 测试 FP16 引擎, batch size = 32
./bin/trt_inference models/resnet18_fp16.trt 32

# 测试 INT8 引擎, batch size = 32
./bin/trt_inference models/resnet18_int8.trt 32
```

我们在 Google Colab 的 NVIDIA T4 GPU (15GB 显存) 上对 ResNet18 模型进行了基准测试。所有延迟数据均为 100 次推理运行的平均值。


| 精度模式 | Batch Size | 平均延迟 (ms) | 吞吐量 (FPS) | 性能提升 (vs FP32) |
| :------- | :----------: | :----------: | :----------: | :----------------: |
| FP32     | 1            | 2.26 ms      | 441 FPS      | -                  |
| **FP16** | **1**        | **1.01 ms**  | **994 FPS**  | **2.25x**          |
| **INT8** | **1**        | **0.81 ms**  | **1,240 FPS** | **2.81x**          |
| FP32     | 32           | 25.76 ms     | 1,242 FPS    | -                  |
| **FP16** | **32**       | **8.53 ms**  | **3,753 FPS** | **3.02x**          |
| **INT8** | **32**       | **4.78 ms**  | **6,701 FPS** | **5.39x**          |


### 结论

1.  **FP16 加速效果显著**: 在 T4 GPU 上，使用 FP16 精度可带来 **2-3 倍** 的性能提升。在批处理大小为 32 时，加速效果尤为明显，吞吐量提升约 **3.0 倍**。
2.  **批处理 (Batching) 是提升吞吐量的关键**: 对于 FP16 模型，将批处理大小从 1 增加到 32，总吞吐量（FPS）**提升了 4 倍**。这证明了在推理时应尽可能使用更大的批处理大小，以充分利用 GPU 的并行计算能力。

### 非默认流 vs 默认流 对比（Batch=32，NVIDIA T4）

在单模型单流的合成基准中，非默认流与默认流的差异不大；但非默认流更利于与 H2D/D2H 拷贝以及其他核（或其他模型流）重叠，从而在真实并发/流水线场景中提升整体吞吐。


| 精度 | 默认流 延迟 (ms) | 默认流 吞吐 (FPS) | 非默认流 延迟 (ms) | 非默认流 吞吐 (FPS) | Δ吞吐 vs 默认 |
| :---: | :---: | :---: | :---: | :---: | :---: |
| FP32 | 25.76 | 1,242 | 25.78 | 1,241 | -0.1% |
| FP16 | 8.02 | 3,987 | 8.53 | 3,753 | -5.9% |
| INT8 | 4.51 | 7,097 | 4.78 | 6,701 | -5.6% |


注：上表“默认流”数据来自旧版默认流实现；“非默认流”数据来自新增的自建非默认 CUDA stream，并使用 CUDA 事件计时。

## 精度一致性对比（无标签一致性度量）

当没有标签集时，我们提供了一个一致性评估工具 `trt_compare`，用于对比 FP32 与 FP16/INT8 的结果差异，指标包括：
- Top-1 一致性（两引擎预测的 argmax 类别是否一致）
- Top-5 一致性（Top-5 类别集合是否有交集）
- 平均余弦相似度（输出向量层面）
- 平均 L2 距离（输出向量层面）

使用示例：

```bash
# 用法: ./bin/trt_compare <baseline_engine_fp32> <test_engine_fp16_or_int8> <images_dir> [max_images]

# 与 FP16 对比（在 calibration_data 上取前 200 张）
./bin/trt_compare models/resnet18.trt models/resnet18_fp16.trt calibration_data 200

# 与 INT8 对比
./bin/trt_compare models/resnet18.trt models/resnet18_int8.trt calibration_data 200
```

你可以把输出结果整理成如下表格（示例占位，填入你的实测数字）：

| 基线 vs 测试 | Top-1 一致性 | Top-5 一致性 | Avg Cosine | Avg L2 |
| :-- | :--: | :--: | :--: | :--: |
| FP32 vs FP16 | 99.8% | 100% | 0.999 | 0.012 |
| FP32 vs INT8 | 99.4% | 99.9% | 0.996 | 0.053 |

说明：
- 这是一种“无标签”的一致性评估，反映量化或低精度推理对输出分布的一致性影响。
- 若你有 ImageNet 验证集及标签，建议计算真正的 Top-1/Top-5 精度对比，以获得更贴近应用的结论。

## 使用 ImageNet 进行有标签评估（可选）

我们不提供 ImageNet 数据集的下载。若你拥有官方的验证集与 devkit 压缩包，可按以下步骤准备：

1) 将以下文件放到 `datasets/` 目录下（自行获取）：
- ILSVRC2012_img_val.tar
- ILSVRC2012_devkit_t12.tar.gz

2) 运行准备脚本，解包并生成标签 CSV：

```bash
python scripts/prepare_imagenet_val.py
```

脚本会在项目根目录生成：
- `imagenet_val/val/`：验证集图片
- `imagenet_val_labels.csv`：`filename,labelIndex`（0..999）

3) 运行有标签评估（计算真实 Acc@1/Acc@5 与量化损失）：

```bash
# FP32 vs FP16（前 1000 张）
./bin/trt_compare models/resnet18.trt models/resnet18_fp16.trt imagenet_val/val 1000 --labels imagenet_val_labels.csv

# FP32 vs INT8
./bin/trt_compare models/resnet18.trt models/resnet18_int8.trt imagenet_val/val 1000 --labels imagenet_val_labels.csv
```

将输出整理为如下表格（示例占位）：

| 对比 | Acc@1 基线 | Acc@1 测试 | ΔAcc@1 (pp) | Acc@5 基线 | Acc@5 测试 | ΔAcc@5 (pp) |
| :-- | :--: | :--: | :--: | :--: | :--: | :--: |
| FP32 vs FP16 | 69.5% | 69.4% | -0.1 | 88.8% | 88.8% | 0.0 |
| FP32 vs INT8 | 69.5% | 69.0% | -0.5 | 88.8% | 88.5% | -0.3 |

建议：为保证公平对比，请保持与训练/导出时一致的预处理（当前示例使用 resize 到 224x224，归一化到 [0,1]，未使用 mean/std 标准化）。

## 运行选项速查表

以下为三大可执行程序常用参数与环境变量速查，便于快速检索与组合使用。

### onnx_to_trt（ONNX → TensorRT 引擎）

- 位置参数：
	- <onnx_path>：输入 ONNX 路径
	- <out_prefix>：输出文件名前缀（会生成 <prefix>.trt 等）
	- <precision>：fp32 | fp16 | int8
	- [calib_dir]（仅 INT8）：可选的校准图片目录（优先级最高）
- 环境变量：
	- CALIB_DATA_DIR：INT8 校准目录（当未在命令行第4参提供时作为回退）
	- IMAGENET_CENTER_CROP=1：校准预处理启用“短边缩放到256 + 中心裁剪”
	- IMAGENET_NORM=1：校准预处理启用 ImageNet mean/std 标准化（先缩放到[0,1]再标准化）
- 说明：
	- 自动创建动态输入的优化配置档（min=1, opt=1, max=32）；若缺省分辨率无法解析，会回退到 224x224。
	- INT8 会自动读写校准缓存，无需重复标定（当缓存命中时）。

示例：

```bash
CALIB_DATA_DIR=calibration_data \
IMAGENET_CENTER_CROP=1 IMAGENET_NORM=1 \
./bin/onnx_to_trt models/resnet18.onnx models/resnet18 int8
```

### trt_inference（推理/基准测试）

- 位置参数：
	- <engine_path>：TensorRT 引擎路径
	- [batch_size]：批大小
- 可选参数：
	- --use-default-stream：使用默认 CUDA Stream（stream 0）
	- --pinned：使用固定页锁定内存（cudaHostAlloc）进行 H2D/D2H 拷贝
	- --pipeline：启用双缓冲 + 复制/计算双流流水线以争取拷贝/计算重叠
- 环境变量：
	- USE_DEFAULT_STREAM=1|true：等效于 --use-default-stream
	- USE_PINNED=1|true：等效于 --pinned

示例：

```bash
USE_PINNED=1 ./bin/trt_inference models/resnet18_int8.trt 32 --pipeline
```

### trt_compare（一致性与精度对比）

- 位置参数：
	- <baseline_engine_fp32>：基线（通常为 FP32）
	- <test_engine_fp16_or_int8>：待测引擎（FP16/INT8）
	- <images_dir>：图片目录
	- [max_images]：最多评估图片数
- 可选参数：
	- --labels <csv_path>：提供标签 CSV 开启 Acc@1/Acc@5 评估（格式：filename,labelIndex）
	- --center-crop：启用短边缩放到256 + 中心裁剪
	- --imagenet-norm：启用 ImageNet mean/std 标准化（先缩放到[0,1]再标准化）
- 环境变量：
	- LABELS_CSV：等效于 --labels
- 输出指标：
	- 无标签：Top-1/Top-5 一致性、平均余弦相似度、平均 L2 距离
	- 有标签：Acc@1/Acc@5（基线/测试与差值）

示例：

```bash
./bin/trt_compare models/resnet18.trt models/resnet18_int8.trt calibration_data 200 \
	--center-crop --imagenet-norm

LABELS_CSV=imagenet_val_labels.csv \
./bin/trt_compare models/resnet18.trt models/resnet18_fp16.trt imagenet_val/val 1000
```
