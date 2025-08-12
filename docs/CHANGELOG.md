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
  - `--pipeline`：启用双缓冲流水线，使用 `copyStream` 与 `computeStream`` 两个非默认流重叠拷贝与计算。
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

日期：2025-08-13

## 总览
本次围绕“评测文档易用性、离线可复现能力与数据准备体验”进行了整理：
- README 结构与内容统一，补充最新 Kaggle 实测、目录兼容说明与可读抽检。
- 离线 ImageNet 类目映射落地（随仓库提供 assets/ 与导出脚本）。
- ImageNet 准备脚本默认不再重组目录，更贴合常见数据布局。
- FAQ/故障排查草稿上线，覆盖 Kaggle/TRT/预处理与标签对齐等常见问题。

## 主要改动

### 1) README 整理与最新评测
文件：`README.md`
- 统一“构建与环境”与“运行与性能测试”结构；强调构建期将 TRT/cuDNN 真实 .so 复制到 `bin/` 与运行前 `source bin/env.sh`（TRT 内部 dlopen 可能忽略 rpath）。
- 更新 Kaggle 实测（T4 单卡，TensorRT 10.4，Batch=32）结果与说明；区分默认流/非默认流，补充性能与示例命令。
- 精度评估部分加入 `--class-names imagenet_classes.tsv` 与 `--inspect` 的可读抽检示例；强调 `--center-crop --imagenet-norm` 预处理；说明递归扫描与“扁平/按 WNID 子目录”两种目录结构均可用，并提供两种结构的示例命令。

### 2) ImageNet 验证集准备脚本默认不重组目录
文件：`scripts/prepare_imagenet_val.py`
- 默认保持扁平结构 `imagenet_val/val/*.JPEG`，仅在显式 `--reorg` 时按 WNID 重组为 `val/<wnid>/...`。
- 优先从 `assets/imagenet_class_index.json` 读取类目映射；回退到 torchvision 包内 JSON；继续解析 devkit `meta.mat` 获取 ILSVRC2012_ID→WNID。
- 产物：`imagenet_val_labels.csv` 与 `imagenet_classes.tsv`；支持 `.JPEG/.jpeg`。

### 3) 离线类目映射与导出脚本
文件：`assets/imagenet_class_index.json`, `assets/README.md`, `scripts/export_imagenet_class_index.py`
- 仓库内置标准 `imagenet_class_index.json` 以保证无网环境下一致标签顺序；脚本可在本地（具备 torch/torchvision）环境中重新导出到 `assets/`。

### 4) trt_compare 增强（配合文档）
文件：`apps/trt_compare/src/trt_compare.cpp`
- 参数：`--class-names <tsv>`、`--inspect <N>`；递归收集 `.jpg/.jpeg/.JPEG/.png`；便于人读验证与兼容子目录/扁平目录。

### 5) FAQ/故障排查草稿
文件：`docs/FAQ.md`
- 覆盖：Kaggle TRT 软链/丢库问题、env.sh 与 LD_LIBRARY_PATH、ImageNet 预处理/标签映射、val 目录结构兼容、devkit 无 map_clsloc.txt 的处理、离线映射、Kaggle 快速自检、NvInfer.h 找不到等。
- 在 README 顶部加入链接入口。

## 注意事项
- 现已默认不重组目录；若此前使用旧版脚本重组为 WNID 子目录，可参考 README/FAQ 的一键还原扁平命令（macOS zsh）。
- 如评测 Acc 异常，优先检查是否启用 `--center-crop --imagenet-norm` 且是否使用仓库内的 `assets/imagenet_class_index.json`；若在 Kaggle 运行，还需确认 `source bin/env.sh` 已生效。
