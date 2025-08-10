
# AI模型推理加速工程师能力提升与面试准备计划

> 目标：在四周内，将您在大型C++系统工程中的深厚背景，与AI模型推理加速的核心技能相结合，打造出具有强竞争力的技术画像，成功获得心仪的Offer。

---

## 1. 核心优势与能力差距分析

### 1.1. 我的优势
- **强大的C++功底**: 具备在复杂系统中（如Xsim）进行高性能C++开发的能力，这是推理岗位的核心与基石。
- **性能导向思维**: 拥有从系统层面（Grading, DTC）分析和评估性能的经验，能快速将此思维迁移到模型优化上。
- **复杂系统驾驭能力**: 有在自动驾驶这一尖端领域处理复杂软件架构的经验，学习曲线陡峭，适应能力强。

### 1.2. 需要弥补的差距
当前经验主要集中在模型运行的“后半段”（仿真、评估），需要快速补齐“前半段”的技能，即**如何让AI模型本身运行得更快、更省资源**。

---

## 2. 必备核心技能清单 (Checklist)

### 2.1. 推理引擎技术 (Inference Engine Technology)
- [ ] **NVIDIA TensorRT (精通)**
  - [ ] **`trtexec` 工具**: 熟练使用进行性能分析（Profiling）、层耗时分析和快速验证。
  - [x] **ONNX 工作流**: 掌握 PyTorch/TensorFlow -> ONNX 的转换，并能解决常见算子不兼容问题。（已完成：PyTorch ResNet18 -> ONNX，见 `scripts/generate_onnx_model.py`；ONNX -> TensorRT 引擎，见 `apps/onnx_to_trt/src/onnx_to_trt.cpp`）
  - [ ] **量化 (Quantization)**:
    - [x] **FP16**: 理解原理并熟练应用。（已完成：`onnx_to_trt` 支持 FP16；`trt_inference` 与 `test_record.md` 有性能对比数据）
  - [x] **INT8 (PTQ)**: 掌握后训练量化完整流程，能独立实现 `IInt8Calibrator` 接口，并评估精度损失。（已完成：实现 `IInt8EntropyCalibrator2` 于 `libs/trt_utils/include/trt_utils/int8_calibrator.h` 和 `libs/trt_utils/src/int8_calibrator.cpp`，校准数据下载脚本 `scripts/download_calibration_data.py`，一致性/精度评估程序 `apps/trt_compare/src/trt_compare.cpp`）
    - [ ] **(了解) QAT**: 了解量化感知训练的基本思想。
  - [ ] **自定义插件 (Custom Plugin)**: 掌握 `IPluginV2` 系列接口，能为TensorRT不支持的算子编写自定义插件。
  - [ ] **动态尺寸 (Dynamic Shapes)**: 理解并能处理输入尺寸可变的模型。（基础已覆盖：已配置动态 batch 的 Optimization Profile；H/W 动态待完善）

- [ ] **其他推理框架 (了解)**
  - [ ] **ONNX Runtime**: 了解其基本用法和CPU端优化选项。
  - [ ] **OpenVINO**: (若目标公司涉足Intel硬件) 了解其基本工作流。
  - [ ] **TVM**: (加分项) 了解其作为深度学习编译器的基本原理。

### 2.2. 底层编程能力 (Low-Level Programming)
- [ ] **CUDA 编程**:
  - [ ] **基础**: 能编写简单CUDA Kernel，理解`thread`, `block`, `grid`。
  - [x] **核心概念**: 理解CUDA Stream（用于并行化）和 Pinned Memory（用于加速数据传输）。（已完成：`apps/trt_inference/src/trt_inference.cpp` 使用非默认/默认流、Pinned memory、双流流水线）
- [ ] **Python/C++ 交互**:
  - [ ] **Pybind11**: 熟练掌握，能够封装C++推理库给Python调用。

### 2.3. AI模型知识 (AI Model Knowledge)
- [ ] **经典模型架构**:
  - [ ] **CV**: ResNet, EfficientNet, YOLO系列, Vision Transformer (ViT)。（部分覆盖：已完成 ResNet18 实战，其余待补）
  - [ ] **分析**: 能大致分析出模型的主要计算瓶颈（计算密集 vs. 访存密集）。

### 2.4. 工具链 (Toolchain)
- [ ] **性能剖析**: **NVIDIA Nsight Systems / Nsight Compute**，必须熟练使用。
- [ ] **容器化**: **Docker**，能编写Dockerfile构建包含驱动、CUDA、TensorRT的镜像。
- [ ] **模型服务化**: (加分项) 了解 **Triton Inference Server** 的基本部署流程。

---

## 3. 四周快速提升行动计划

### 第一周：打通核心流程，建立正反馈
- **目标**: 成功跑通一个标准模型的 `PyTorch -> ONNX -> TensorRT` 完整流程。
- **任务**:
  1.  **环境搭建**: 安装CUDA、cuDNN、TensorRT，并跑通官方`sampleOnnxMNIST`示例。（未做：已以自研示例验证 TensorRT 环境与编译链）
  2.  **模型导出**: 使用PyTorch加载预训练的`ResNet50`，将其导出为ONNX格式。（已完成但使用 ResNet18：见 `scripts/generate_onnx_model.py`）
  3.  **引擎转换**: 使用`trtexec`将ONNX转为FP32和FP16的TensorRT引擎。（已完成但使用自研 `onnx_to_trt` 完成 FP32/FP16/INT8 引擎构建）
  4.  **性能对比**: 记录并对比FP32和FP16模式下的延迟（Latency）和吞吐量（Throughput）。（已完成：见 `test_record.md` 与 `README.md` 的数据）
- **产出物**: 一份详细的笔记，记录了所有命令、步骤和性能对比数据。

### 第二周：攻坚核心技术——量化与C++部署
- **目标**: 掌握INT8量化，并完成C++端到端推理部署。
- **任务**:
  1.  **INT8量化**:（已完成）
      - 准备一个小的校准数据集（如ImageNet验证集中的100-500张图片）。
      - 学习并实现TensorRT的`IInt8Calibrator`接口，生成INT8量化引擎。
      - 对比FP32, FP16, INT8三种模式的性能，并（可选）测试INT8模型的精度损失。
  2.  **C++部署**:（已完成）
      - 编写一个CMake C++项目。
      - 实现功能：加载`.engine`文件，读取图片并预处理，执行推理，打印结果。
- **产出物**:
  - 一个可运行的C++项目，用于加载TensorRT引擎并执行推理。
  - 更新的性能对比笔记，加入INT8的数据。

### 第三、四周：打造亮眼项目，形成简历亮点
- **目标**: 完成一个与自动驾驶背景强相关、覆盖推理加速核心能力的开源项目。（未做）
- **项目名称**: **基于 TensorRT 的自动驾驶目标检测与推理加速**
- **任务分解**:
  1.  **数据集与场景设计**
    - 选用 KITTI/nuScenes 等自动驾驶数据集，或从仿真平台导出视频流。
    - 统一预处理与评测 protocol，便于公平对比（分辨率、后处理阈值等）。
  2.  **模型转换与插件开发**
    - 将 YOLOv8 转 ONNX 并构建 TensorRT 引擎（FP32/FP16/INT8）。
    - 为后处理（NMS/Decode）实现 `IPluginV2DynamicExt` 插件，支持动态输入尺寸。
  3.  **性能分析与优化**
    - 使用 Nsight Systems/Compute 定位瓶颈（H2D/D2H、kernel、并发）。
    - 对比不同精度下的 Latency/Throughput 与 mAP，绘制图表。
  4.  **服务化与部署**
    - 使用 Triton Inference Server 部署，引入动态 shape、批处理策略。
    - 提供实时 Demo：从仿真/录像流读取并渲染检测结果。
  5.  **文档与可视化**
    - 完整 README：环境、步骤、性能图表、精度对比、常见问题。
    - 提供 Docker 镜像/脚本，包含 CUDA/TensorRT/依赖，支持一键复现。
- **产出物**:
  - GitHub 开源项目（代码 + README + 图表）。
  - TensorRT 插件源码与单元测试。
  - Triton 部署配置（模型仓库结构与 config）。
  - 可运行 Demo（视频/相机输入）。
  - Dockerfile/镜像与复现实验脚本。

---

## 4. 推荐学习资源
- **官方文档**:
  - [NVIDIA TensorRT Documentation](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html)
  - [NVIDIA CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html)
- **GitHub Repos**:
  - [TensorRT Official Samples](https://github.com/NVIDIA/TensorRT/tree/main/samples)
  - [YOLOv8 Official Repo](https://github.com/ultralytics/ultralytics)
  - [TensorRT_Pro (中文社区优秀资源)](https://github.com/shouxieai/tensorrt_pro)
- **教程与文章**:
  - 在知乎、CSDN、Medium上搜索“TensorRT INT8量化”、“TensorRT 自定义插件”等关键词，有大量优质的中文实践文章。

---
