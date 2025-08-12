# 项目 TODO 清单（Roadmap）

> 面向 AI 模型推理加速与部署岗位，按“已完成 / 待办（优先级）/ 亮眼项目”分组管理。

## 🧭 仓库策略（已决策）
- 亮眼项目采用“独立仓库”，当前仓库仅保留链接与结果汇总；暂不使用 git submodule。
- 如需共享基建代码，优先考虑后续以 git subtree 同步，避免 submodule 的使用复杂度。

## 🚀 新仓库初始化任务清单
- [x] 新仓库创建（trt-yolov8-accelerator，路径：`projects/trt-yolov8-accelerator/`），Apache-2.0 许可证
- [x] 目录搭建：src/include、scripts、docker（plugins/benchmarks/triton/models 后续补齐）
- [x] 最小可运行骨架：
  - [x] CMake 工程与 `onnx_to_trt_yolo`、`yolo_trt_infer`
  - [x] YOLOv8 ONNX 导出脚本 `projects/trt-yolov8-accelerator/scripts/export_yolov8_onnx.py`
  - [x] Dockerfile（`projects/trt-yolov8-accelerator/docker/Dockerfile`）
- [x] README 首版（`projects/trt-yolov8-accelerator/README.md`）与草稿（`docs/NEW_REPO_README_draft.md`）
- [ ] CI（可选）：构建与样例运行校验

## ✅ 已完成（可复现）
- [x] PyTorch → ONNX 导出（ResNet18）：`scripts/generate_onnx_model.py`
- [x] ONNX → TensorRT 引擎构建（FP32/FP16/INT8）：`apps/onnx_to_trt/src/onnx_to_trt.cpp`
- [x] INT8 标定实现与数据准备：`libs/trt_utils/include/trt_utils/int8_calibrator.h`、`libs/trt_utils/src/int8_calibrator.cpp`，`scripts/download_calibration_data.py`
- [x] C++ 推理与性能基准：`apps/trt_inference/src/trt_inference.cpp`（非默认流、Pinned 内存、双流流水线）
- [x] 一致性/精度评估工具：`apps/trt_compare/src/trt_compare.cpp`（无标签一致性 + 有标签 Acc@1/Acc@5）
- [x] CMake 目标配置：`CMakeLists.txt`（onnx_to_trt / trt_inference / trt_compare / simple_trt_test）
- [x] Kaggle/Colab 环境脚本：`scripts/setup_kaggle_env.sh`、`scripts/setup_colab_env.sh`
- [x] 运行库就地打包与环境：构建阶段复制 TensorRT/cuDNN 到 `bin/`，生成 `bin/env.sh`
- [x] ImageNet 验证集准备：`scripts/prepare_imagenet_val.py`（支持 devkit、子集、labels CSV、classes TSV、assets 优先）
- [x] 离线类别映射：`assets/imagenet_class_index.json` 与导出脚本 `scripts/export_imagenet_class_index.py`
- [x] trt_compare 增强：递归扫描、`.jpeg/.JPEG` 支持、`--class-names`、`--inspect` 可读输出
- [x] README 更新：Kaggle T4×2（TRT 10.4）最新性能与精度对齐说明

## 🟡 待办（通用能力补齐）
- [ ] 动态尺寸完善（中优）
  - [ ] 在 `onnx_to_trt` 中为输入添加多档动态 H/W 的 Optimization Profiles（如 min=1x3x320x320、opt=1x3x640x640、max=1x3x1280x1280）
  - [ ] 在 `trt_inference` 中根据实际输入设置 shape 并验证 batch×H×W 的变更
  - [ ] 在 `trt_compare` 中新增 `--hw <HxW>` 或自动从图像尺寸推断并 resize/crop 对齐
- [ ] trtexec 熟练度与基线（中优）
  - [ ] 使用 trtexec 复刻 ONNX→TRT（FP32/FP16/INT8）转换，开启层级 profiler 与详细日志
  - [ ] 保存命令与输出到 `benchmarks/trtexec_logs/`
- [ ] Nsight Systems/Compute 性能剖析（高优）
  - [ ] 对 `trt_inference` 的 FP32/FP16/INT8 运行做时间线与 kernel 分析，量化 H2D/D2H 与 compute 重叠率
  - [ ] 输出报告与瓶颈项，并提出/验证一条优化（如更大 batch、更多 stream、Pinned 内存使能等）
- [ ] Docker 化（中优）
  - [ ] 提供 `Dockerfile` 与构建脚本，包含 CUDA、TensorRT、OpenCV、构建依赖
  - [ ] 提供 `docker run`/`compose` 示例，确保 `bin/` 可执行程序能在容器内运行
- [ ] Pybind11 封装（中优）
  - [ ] 将引擎加载与推理包装为 Python 模块，支持 numpy/tensor 输入输出
  - [ ] 简单单测（pytest）覆盖加载、推理、异常路径
- [ ] 文档打磨（中优）
  - [ ] `docs/NEW_REPO_README_draft.md` 补充实测截图与表格
  - [ ] 主 README 增加一页式“常见问题/故障排查”

## 🔵 第三、四周亮眼项目：基于 TensorRT 的自动驾驶目标检测与推理加速（未做）
- [ ] 数据与场景（高优）
  - [ ] 选择 KITTI/nuScenes 或从仿真平台导出视频流
  - [ ] 确定评测 protocol（分辨率、NMS 阈值、置信度阈值、mAP 计算设置）
- [ ] 模型与引擎（高优）
  - [ ] 将 YOLOv8 导出为动态输入 ONNX，并用 onnxruntime 验证正确性
  - [ ] 构建 FP16 与 INT8 的 TensorRT 引擎，生成/缓存校准表
- [ ] 自定义插件（核心亮点，高优）
  - [ ] 基于 `IPluginV2DynamicExt` 实现后处理（Decode/NMS）插件，支持动态 H/W
  - [ ] 在构建流程中替换 ONNX 中相应节点或在解析后手动添加插件层
  - [ ] 为插件添加最小单测与性能对比（vs host 后处理）
- [ ] 性能剖析与优化（高优）
  - [ ] 记录 FP32/FP16/INT8 在不同 batch、分辨率下的 Latency/FPS
  - [ ] 对关键 kernel 进行 Nsight Compute 分析，提出优化并复测
  - [ ] 绘制图表（Latency/Throughput/mAP vs 精度/分辨率/批大小）
- [ ] 服务化与部署（中高）
  - [ ] 使用 Triton Inference Server 部署，配置模型仓库与 `config.pbtxt`（动态 shape/批处理/并发）
  - [ ] 提供客户端脚本（Python/C++）进行吞吐与延迟测试
- [ ] 实时 Demo 与复现（中高）
  - [ ] C++ 实时检测 Demo：读取视频/相机，渲染框与类别
  - [ ] 提供一键复现脚本与必要的下载/准备脚本
- [ ] 文档与发布（高优）
  - [ ] 在 `README.md` 增加项目背景、使用指南、性能图表、精度对比与常见问题
  - [ ] 生成公开可读的结果表与图（`docs/` 或 `reports/`）

## 🟣 可选加分项
- [ ] 扩展至 EfficientNet / ViT 分类与 DeepLab/U-Net 分割模型的优化对比
- [ ] ONNX Runtime / OpenVINO CPU 端基线与对比
- [ ] TVM 编译探索与小型对比实验

---
提示：可将以上 TODO 拆解为仓库 issues（按“数据/模型/插件/服务化/文档”模块）并设定里程碑，便于跟踪执行进度。
