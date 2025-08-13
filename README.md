# AI-Infer-Acceleration
AI 推理加速计划

## 项目介绍

本项目旨在通过优化AI模型的推理过程，提高推理速度和效率。我们将从以下几个方面进行优化：

1. 模型量化
2. 模型剪枝
3. 模型压缩
4. 模型并行
5. 模型加速 
6. 模型部署
7. 模型评估

> 常见问题与故障排查请见：[`docs/FAQ.md`](docs/FAQ.md)

> 专注方向：YOLOv8 加速与部署

本仓库包含一个专注于 YOLOv8 的加速子项目，覆盖从 ONNX 导出、TensorRT 引擎转换（FP32/FP16/INT8，动态 H×W）、到推理与可视化、以及插件化后处理的完整链路。建议直接参考：

- `projects/trt-yolov8-accelerator/README.md`

子项目提供：
- 导出脚本 `scripts/export_yolov8_onnx.py`（支持 `--device/--nms/--half/--simplify`，并打印 I/O 形状）
- 转换工具 `onnx_to_trt_yolo`（支持 `--fp32/--fp16/--int8`、INT8 标定、动态 H×W、占位 `--decode-plugin`）
- 推理工具 `yolo_trt_infer`（支持 `--decode cpu|plugin`、`--has-nms`、`--class-agnostic`、`--topk`）
- 共享工具库：letterbox 预处理、CPU 解码与 NMS 等

## 构建与环境 (Build & Environment)

### 下载代码
```bash
git clone https://github.com/libingwei/AI-Infer-Acc.git
cd AI-Infer-Acc
git submodule update --init --recursive
```

本项目使用 CMake 进行构建管理。请确保您的环境中已安装 NVIDIA CUDA Toolkit 和 TensorRT。
> 在 Google Colab 中构建安装 TensorRT：
```bash
bash scripts/setup_colab_env.sh
```
> 在 Kaggle 环境里安装 TensorRT：
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


### 1) 准备 Python 依赖

首先，安装项目所需的 Python 依赖，用于生成 ONNX 模型。

```bash
pip install -r requirements.txt
```

### 2) INT8 量化准备 (INT8 Quantization Prep)

为了执行 INT8 量化，您需要一个校准数据集。我们提供了一个脚本来自动下载一个约500张图片的COCO子集。

```bash
# 首先，安装下载脚本所需的库
pip install -r requirements.txt

# 然后，运行脚本下载并解压数据
# 数据会被存放在项目根目录下的 `calibration_data/` 文件夹中
python scripts/download_calibration_data.py
```

可选：标定图片收集选项
- CALIB_RECURSIVE=1 启用递归收集子目录下的图片（默认关闭，仅扫描目录根）
- 支持的扩展名：.jpg/.jpeg/.png（大小写均可）

示例：
```bash
# 递归使用自备标定集（ImageNet 风格推荐用于分类），并与后续推理预处理对齐
CALIB_RECURSIVE=1 IMAGENET_CENTER_CROP=1 IMAGENET_NORM=1 \
./bin/onnx_to_trt models/resnet18.onnx models/resnet18 int8
```

### 3) 生成 ONNX 模型


运行脚本来生成用于后续步骤的 ONNX 模型文件。脚本会自动处理预训练权重的下载和缓存。

```bash
python scripts/generate_onnx_model.py
```

### 4) 编译 C++ 程序

我们推荐使用 CMake 的“不切换目录”构建方式。所有命令都在项目根目录下执行。

```bash
# 1. 配置项目 (首次构建或 CMakeLists.txt 变更后执行)
cmake -S . -B build

# 2. 执行构建 (使用4个核心并行编译以加快速度)
cmake --build build -- -j4
```

编译成功后，所有可执行文件将位于 `bin/` 目录下。

提示：YOLOv8 子项目可单独启用插件目标（实验）：

```bash
cmake -S . -B build -DYOLO_BUILD_PLUGINS=ON
cmake --build build -j
```

提示：构建过程会将 TensorRT/cuDNN 等运行库复制并“就地”放入 `bin/`，并写出 `bin/env.sh` 用于设置运行期 `LD_LIBRARY_PATH`（某些场景下，TensorRT 内部 dlopen 会忽略 rpath，需显式导出）。运行前可：

```bash
source bin/env.sh
```

## 运行与性能测试

### 引擎转换（ONNX → TensorRT）

使用 `onnx_to_trt` 程序将 ONNX 模型转换为不同精度的 TensorRT 引擎。

```bash
# 用法: ./bin/onnx_to_trt <输入ONNX路径> <输出文件名前缀> <精度>

# 生成 FP32 引擎 (models/resnet18.trt)
./bin/onnx_to_trt models/resnet18.onnx models/resnet18 fp32

# 生成 FP16 引擎 (models/resnet18_fp16.trt)
./bin/onnx_to_trt models/resnet18.onnx models/resnet18 fp16

# 生成 INT8 引擎 (models/resnet18_int8.trt)
# (请确保您已按步骤2准备好了校准数据集)
./bin/onnx_to_trt models/resnet18.onnx models/resnet18 int8
```

### 推理与基准测试（单引擎吞吐/延迟）

使用 `trt_inference` 程序加载引擎进行推理与计时。

```bash
# 用法: ./bin/trt_inference <引擎路径> [批处理大小]

# 测试 FP32 引擎, batch size = 32
./bin/trt_inference models/resnet18.trt 32

# 测试 FP16 引擎, batch size = 32
./bin/trt_inference models/resnet18_fp16.trt 32

# 测试 INT8 引擎, batch size = 32
./bin/trt_inference models/resnet18_int8.trt 32

# 使用默认 CUDA Stream（legacy stream 0）
./bin/trt_inference models/resnet18_int8.trt 32 --use-default-stream

# 使用非默认 CUDA Stream（独立 stream，便于复制/计算重叠）
./bin/trt_inference models/resnet18_int8.trt 32

# 也可通过环境变量切换默认流
USE_DEFAULT_STREAM=true ./bin/trt_inference models/resnet18_int8.trt 32
```


#### Kaggle GPU T4×2 + TensorRT 10.4（单卡，Batch=32，最新）

环境：Tesla T4（2 张，单卡测），Driver 560.35.03，CUDA Runtime 12.6，nvcc 12.5，TensorRT 10.4。结果为 100 次迭代均值。

| 精度模式 | Batch Size | 平均延迟 (ms) | 吞吐量 (FPS) | 性能提升 (vs FP32) |
| :------- | :--------: | :-----------: | :----------: | :----------------: |
| FP32     | 32         | 25.3633       | 1,261.66     | -                  |
| FP16     | 32         | 8.6169        | 3,713.64     | 2.94x              |
| INT8     | 32         | 4.2185        | 7,585.72     | 6.01x              |


### 结论（性能）

- FP16/INT8 在 T4 上带来显著加速，Batch=32 时 FP16≈3.0x、INT8≈6.0x 相比 FP32。
- Batching 是吞吐提升关键；结合独立 CUDA stream 与流水线可进一步提升真实场景并发性能。

### 默认流 vs 非默认流（Batch=32，NVIDIA T4，INT8）

- 单模型单流下两者差异很小；但非默认流利于与 H2D/D2H 以及其他核重叠，真实业务更有优势。

| 模式 | 平均延迟 (ms) | 吞吐量 (FPS) | 备注 |
| :--: | :-----------: | :----------: | :-- |
| 非默认流 | 4.2185 | 7,585.72 | 推荐 |
| 默认流 | 4.2276 | 7,569.31 | TRT 提示：默认流可能因内部同步影响性能 |

## 精度评估

### 一致性对比（无标签）

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

Kaggle（calibration_data 前 200 张）实测：

| 基线 vs 测试 | Top-1 一致性 | Top-5 一致性 | Avg Cosine | Avg L2 |
| :-- | :--: | :--: | :--: | :--: |
| FP32 vs FP16 | 98% | 100% | 0.999991 | 0.327497 |
| FP32 vs INT8 | 89% | 100% | 0.997317 | 5.4276 |

说明：
- 这是一种“无标签”的一致性评估，反映量化或低精度推理对输出分布的一致性影响。
- 若你有 ImageNet 验证集及标签，建议计算真正的 Top-1/Top-5 精度对比，以获得更贴近应用的结论。

### 使用 ImageNet 进行有标签评估（可选）

我们不提供 ImageNet 数据集的下载。若你拥有官方的验证集与 devkit 压缩包，可按以下步骤准备：

1) 将以下文件放到 `datasets/` 目录下（自行获取）：
- ILSVRC2012_img_val.tar
- ILSVRC2012_devkit_t12.tar.gz

2) 运行准备脚本，解包并生成标签 CSV（会生成 `imagenet_classes.tsv`；默认不重组目录，保持扁平的 `val/*.JPEG` 结构）：

```bash
python scripts/prepare_imagenet_val.py

# 如需要按 WNID 重组为子目录，可添加：
# python scripts/prepare_imagenet_val.py --reorg

# 可选：先离线导出类别映射（建议）
# 这会把 torchvision 自带的 imagenet_class_index.json 复制到 assets/ 下，供脚本优先使用
python scripts/export_imagenet_class_index.py
```

脚本会在项目根目录生成：
- `imagenet_val/val/`：验证集图片（默认扁平 `val/*.JPEG`，如使用 `--reorg` 则为 `val/<wnid>/xxx.JPEG`）
- `imagenet_val_labels.csv`：`filename,labelIndex`（0..999）
- `imagenet_classes.tsv`：`index\twnid\tname`（类名便于可读输出）
（若 `assets/imagenet_class_index.json` 存在，将被优先使用，以保证无网环境下一致的标签映射。）

3) 运行有标签评估（计算真实 Acc@1/Acc@5 与量化损失）：

```bash
# FP32 vs FP16（前 1000 张）
./bin/trt_compare models/resnet18.trt models/resnet18_fp16.trt imagenet_val/val 1000 \
	--labels imagenet_val_labels.csv --center-crop --imagenet-norm \
	--class-names imagenet_classes.tsv --inspect 5

# FP32 vs INT8
./bin/trt_compare models/resnet18.trt models/resnet18_int8.trt imagenet_val/val 1000 \
	--labels imagenet_val_labels.csv --center-crop --imagenet-norm \
	--class-names imagenet_classes.tsv --inspect 5
```

Kaggle（前 1000 张 ImageNet 验证集，启用 `--center-crop --imagenet-norm`）最新结果：

| 对比 | Acc@1 基线 | Acc@1 测试 | ΔAcc@1 (pp) | Acc@5 基线 | Acc@5 测试 | ΔAcc@5 (pp) |
| :-- | :--: | :--: | :--: | :--: | :--: | :--: |
| FP32 vs FP16 | 69.0% | 68.9% | -0.1 | 88.8% | 88.8% | 0.0 |
| FP32 vs INT8 | 69.0% | 53.1% | -15.9 | 88.8% | 77.5% | -11.3 |

注意：
- 预处理需与训练/导出对齐：短边 256 + 中心裁剪 224×224；在启用 `--imagenet-norm` 时我们会自动 BGR→RGB、缩放到 [0,1] 并用 ImageNet mean/std 归一化。
- `trt_compare` 支持 `.jpg/.jpeg/.JPEG/.png` 并递归遍历子目录（例如 `val/<wnid>/...`）。

可选：若已重组为子目录、但需要恢复为扁平目录（macOS zsh）：

```bash
# 将所有子目录里的图片移动到 val/ 根目录
for d in val/*(/); do mv "$d"/* val/; done
# 删除空子目录
rmdir val/*(/)
```

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
	- 支持配置动态分辨率优化档：
		- --hw-min HxW、--hw-opt HxW、--hw-max HxW（例如 320x320 / 640x640 / 1280x1280）
		- 未显式指定时，将使用缺省档位；个别模型无法解析输入尺寸时会回退到 224x224。
	- INT8 会自动读写校准缓存，无需重复标定（当缓存命中时）。

示例：

```bash
# INT8：按 ImageNet 预处理进行标定
CALIB_DATA_DIR=calibration_data \
IMAGENET_CENTER_CROP=1 IMAGENET_NORM=1 \
./bin/onnx_to_trt models/resnet18.onnx models/resnet18 int8

# 动态分辨率（分类模型示例）
./bin/onnx_to_trt models/resnet18.onnx models/resnet18_fp16 fp16 \
  --hw-min 224x224 --hw-opt 224x224 --hw-max 1024x1024
```

### trt_inference（推理/基准测试）

- 位置参数：
	- <engine_path>：TensorRT 引擎路径
	- [batch_size]：批大小
- 可选参数：
	- --use-default-stream：使用默认 CUDA Stream（stream 0）
	- --pinned：使用固定页锁定内存（cudaHostAlloc）进行 H2D/D2H 拷贝
	- --pipeline：启用双缓冲 + 复制/计算双流流水线以争取拷贝/计算重叠
	- --hw HxW：当引擎为动态分辨率时，设置运行期输入高度与宽度（如 640x640）
- 环境变量：
	- USE_DEFAULT_STREAM=1|true：等效于 --use-default-stream
	- USE_PINNED=1|true：等效于 --pinned

示例：

```bash
USE_PINNED=1 ./bin/trt_inference models/resnet18_int8.trt 32 --pipeline

# 在动态分辨率引擎上指定 640x640 运行（例如检测/分割场景）
./bin/trt_inference models/resnet18_fp16.trt 8 --hw 640x640
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
	- --class-names <tsv>：加载 `index\twnid\tname` 的类别表用于可读输出
	- --inspect <N>：打印前 N 张图片的 Top-1/Top-5 可读检查
- 环境变量：
	- LABELS_CSV：等效于 --labels
- 输出指标：
	- 无标签：Top-1/Top-5 一致性、平均余弦相似度、平均 L2 距离
	- 有标签：Acc@1/Acc@5（基线/测试与差值）

注：会递归遍历子目录并收集 `.jpg/.jpeg/.JPEG/.png` 图片，兼容两种目录结构：
- 扁平：`imagenet_val/val/*.JPEG`
- 按 WNID 子目录：`imagenet_val/val/<wnid>/*.JPEG`

示例：

```bash
./bin/trt_compare models/resnet18.trt models/resnet18_int8.trt calibration_data 200 \
	--center-crop --imagenet-norm

LABELS_CSV=imagenet_val_labels.csv \
./bin/trt_compare models/resnet18.trt models/resnet18_fp16.trt imagenet_val/val 1000 \
	--class-names imagenet_classes.tsv --inspect 5 --center-crop --imagenet-norm 
```
