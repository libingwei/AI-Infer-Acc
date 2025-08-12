# 基于 TensorRT 的自动驾驶目标检测与推理加速（项目草稿）

> 这是独立开源仓库（建议名：trt-yolov8-accelerator）的 README 草稿，用于展示亮点、复现步骤与基准结果。

## 项目亮点
- YOLOv8 → ONNX → TensorRT（FP32/FP16/INT8），支持动态 batch/分辨率
- 自定义 TensorRT 插件（`IPluginV2DynamicExt`）实现后处理（Decode/NMS），端到端 GPU 推理
- Nsight Systems/Compute 性能剖析：H2D/D2H 与 kernel 并发、瓶颈定位与优化
- Triton Inference Server 部署：动态 shape/批处理/并发，含客户端压测
- Docker 一键环境：CUDA/TensorRT/依赖齐备，跑通即得结果

## 快速开始（草稿示例）
```bash
# 1) 构建镜像
# docker build -t trt-yolov8-accelerator:latest -f docker/Dockerfile .

# 2) 生成/拉取 ONNX
# python scripts/export_yolov8_onnx.py --weights yolov8n.pt --dynamic

# 3) 构建 TensorRT 引擎
# ./bin/onnx_to_trt_yolo models/yolov8n.onnx models/yolov8n fp16 --dynamic --int8 --calib calib_data/

# 4) 运行基准
# ./bin/yolo_trt_infer models/yolov8n_fp16.trt --source samples/highway.mp4 --batch 16
```

注：本仓库中已存在一个示例项目骨架：`projects/trt-yolov8-accelerator/`，包含 CMake、导出脚本与 Dockerfile，可参考其 README。

## 目录结构（预期）
- src/, include/: 引擎加载、预处理、推理、后处理
- plugins/: NMS/Decode 插件实现与单测
- scripts/: 导出 ONNX、评测、可视化、数据准备
- benchmarks/: 机器/配置、Latency/FPS/mAP 结果与图表
- docker/: Dockerfile 与 compose（CUDA/TensorRT/构建链）
- triton/models/: 模型仓库与 config.pbtxt
- docs/: 报告、图表、设计说明

## 基准与图表（占位）
- T4 / 3080 / A10 / Orin：FP32/FP16/INT8 Latency/FPS 曲线
- 不同分辨率与 batch 的性能面
- 插件 vs host 后处理对比
- Triton 并发与延迟分布

## 许可证
- 预期使用 Apache-2.0
