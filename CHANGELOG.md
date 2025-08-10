# 变更记录（本地改动与使用说明）

日期：2025-08-10

## 总览
本次提交围绕三条主线进行了增强：
- INT8 标定/评估的“正确性基线”统一（加入 ImageNet 常用预处理选项）。
- 推理执行的性能工程（非默认流、pinned host 内存、双缓冲流水线重叠拷贝与计算）。
- 精度/一致性评估与 ImageNet 有标签评估链路（工具与脚本）。

## 主要改动

### 1) INT8 校准器：对齐 ImageNet 预处理
文件：`src/int8_calibrator.h`, `src/int8_calibrator.cpp`
- 新增环境变量开关：
  - `IMAGENET_CENTER_CROP=1`：短边缩放至 256，中心裁剪到 WxH（典型 224x224）。
  - `IMAGENET_NORM=1`：在 [0,1] 缩放后按 mean/std 归一化（mean=[0.485,0.456,0.406]，std=[0.229,0.224,0.225]）。
- 目的：与评估工具预处理口径保持一致，得到可信的 INT8 精度损失评估。

使用建议：
- 生成 INT8 引擎前设置上述环境变量，之后运行 `onnx_to_trt` 完成引擎构建。

---

### 2) 推理程序：流/内存/流水线
文件：`src/trt_inference.cpp`
- 新增命令行与环境开关：
  - `--use-default-stream` 或 `USE_DEFAULT_STREAM=1`：使用默认流（否则默认创建非默认流）。
  - `--pinned` 或 `USE_PINNED=1`：启用 pinned host 内存，加速 H2D/D2H。
  - `--pipeline`：启用双缓冲流水线，使用 `copyStream` 与 `computeStream` 两个非默认流重叠拷贝与计算。
- 计时：使用 CUDA 事件在相应流上计时，避免全局同步误差。
- 另外：自动发现输入/输出张量名，避免硬编码 `input`/`output`。

示例：
```bash
# 非默认流 + pinned + 双缓冲流水线（推荐对比性能）
./bin/trt_inference models/resnet18_int8.trt 32 --pinned --pipeline

# 默认流（用于对照）
./bin/trt_inference models/resnet18_int8.trt 32 --use-default-stream
```

---

### 3) 一致性/精度评估工具
文件：`src/trt_compare.cpp`, `CMakeLists.txt`
- 无标签一致性指标（已有）：Top-1/Top-5 一致性、平均余弦相似度、平均 L2 距离。
- 新增有标签评估能力：
  - 参数 `--labels <labels_csv>` 或环境变量 `LABELS_CSV`，计算 FP32 与（FP16/INT8）各自的 Acc@1/Acc@5，并输出精度差（百分点）。
- 新增预处理开关（与校准器对齐）：
  - `--center-crop`：短边 256，中心裁剪到 224x224（或当前 WxH）。
  - `--imagenet-norm`：按 ImageNet mean/std 归一化（在 [0,1] 缩放后）。

示例：
```bash
# 无标签一致性（在 calibration_data 上选取 200 张）
./bin/trt_compare models/resnet18.trt models/resnet18_int8.trt calibration_data 200 --center-crop --imagenet-norm

# 有标签评估（需先准备 ImageNet 验证集与标签 CSV）
./bin/trt_compare models/resnet18.trt models/resnet18_fp16.trt imagenet_val/val 1000 --labels imagenet_val_labels.csv --center-crop --imagenet-norm
```

---

### 4) ImageNet 验证集准备脚本
文件：`scripts/prepare_imagenet_val.py`
- 期望你已将官方包放入 `datasets/`：`ILSVRC2012_img_val.tar`, `ILSVRC2012_devkit_t12.tar.gz`。
- 作用：解包 devkit 与验证集，生成 `imagenet_val/val/` 与 `imagenet_val_labels.csv`（`filename,labelIndex`）。
- 便于 `trt_compare` 的有标签评估。

示例：
```bash
python scripts/prepare_imagenet_val.py
# 然后：
./bin/trt_compare models/resnet18.trt models/resnet18_int8.trt imagenet_val/val 1000 --labels imagenet_val_labels.csv --center-crop --imagenet-norm
```

---

### 5) 构建系统（CMake）
文件：`CMakeLists.txt`
- 新增目标：`trt_compare`（链接 CUDA、TensorRT、OpenCV）。
- 其他：确保 `onnx_to_trt` 目标链接 OpenCV；自动寻找 CUDA/legacy CUDA；指定 TensorRT 库路径。

---

### 6) 文档（README）
- 新增/更新：
  - 非默认流 vs 默认流 对比表（Batch=32）。
  - 精度一致性评估章节（无标签）与 ImageNet 有标签评估流程。
  - 运行示例中加入如何切换流模式与环境变量开关。

---

### 7) 其他既有增强（方便查阅）
- `scripts/download_calibration_data.py`：使用 COCO 官方地址与镜像、扁平化到 `calibration_data/`、支持 `CALIB_MAX_IMAGES` 限制。
- `src/onnx_to_trt.cpp`：
  - 支持校准目录优先级：CLI 第4参 > 环境变量 `CALIB_DATA_DIR` > 默认 `calibration_data`。
  - 使用真实输入张量名与通用维度获取；动态尺寸回退 224x224 用于校准预处理。
  - 优化档：MIN/OPT/MAX = 1/1/32；工作空间上限 1GB。
  - 校准器生命周期管理，避免构建时释放导致崩溃。

## 兼容性与注意事项
- OpenCV BGR vs RGB：若导出/训练链路严格使用 RGB，请确保均按 RGB 均值/方差进行归一化；如需，我可补充 BGR/RGB 切换选项。
- 预处理口径：建议在校准与评估均开启 `center-crop` 与 `imagenet-norm`，与常见 ImageNet 基线对齐；否则请在 README/报告中明确口径。
- 流水线示例侧重结构清晰：实际生产可扩展多 stage、多流与真实数据源，以提升重叠比例。

## 快速指令速查
```bash
# 1) 生成 INT8 引擎（按 ImageNet 预处理标定）
export IMAGENET_CENTER_CROP=1
export IMAGENET_NORM=1
./bin/onnx_to_trt models/resnet18.onnx models/resnet18 int8

# 2) 评估一致性/精度
./bin/trt_compare models/resnet18.trt models/resnet18_int8.trt calibration_data 200 --center-crop --imagenet-norm
./bin/trt_compare models/resnet18.trt models/resnet18_int8.trt imagenet_val/val 1000 --labels imagenet_val_labels.csv --center-crop --imagenet-norm

# 3) 性能测试（非默认流 + pinned + 双缓冲流水线）
./bin/trt_inference models/resnet18_int8.trt 32 --pinned --pipeline
```

---

如需把这些开关汇总到 README 的“运行选项速查表”，我可以继续补充文档段落与示例。
