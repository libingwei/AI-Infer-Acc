# FAQ / 故障排查（草稿）

> 常见环境与评测问题的快速定位与解决方案。

## 1. Kaggle 上 TensorRT 库软链损坏/找不到 .so

症状：
- 构建/运行时报找不到 nvinfer/nvonnxparser 等 .so，或 `NvInfer.h 找不到`。
- Kaggle 数据集挂载的 TensorRT 目录中存在 0 字节或断开的软链接。

建议与解法：
- 使用仓库自带的构建逻辑即可：CMake 会挑选真实的“带版本号”的 .so，复制到 `bin/`，并在 `bin/` 下再生成 `.so.MAJOR` 与 `.so` 的“实体拷贝”（而非软链）。
- 构建后执行：`source bin/env.sh`。该脚本会把 `bin/` 置于 `LD_LIBRARY_PATH` 前面，避免内部 dlopen 找不到依赖。
- 若仍失败，检查 `bin/` 是否存在如 `libnvinfer.so.10.x.x`、`libnvinfer.so.10`、`libnvinfer.so` 三者的实体文件；以及 `libcudnn_*` 是否到位（TensorRT>=9 依赖 cuDNN v9）。

## 2. 运行时报 dlopen 错误，但已设置 RPATH

原因：
- 某些场景下，TensorRT 内部的 dlopen 不遵循可执行文件的 rpath 设置。

解决：
- 运行前显式 `source bin/env.sh`，确保 `LD_LIBRARY_PATH` 指向 `bin/`，让 runtime 能优先找到复制到本地的依赖库。

## 3. ImageNet 评测精度异常（Acc 接近 0%）

常见原因：
- 预处理与训练不一致（未中心裁剪、未 RGB、未做 mean/std 归一化）。
- 标签映射顺序不匹配（WNID → index 未对齐模型类别顺序）。

检查与解决：
- 运行 `trt_compare` 时开启：`--center-crop --imagenet-norm`。在 `--imagenet-norm` 下，预处理会自动做 BGR→RGB、[0,1] 缩放和 ImageNet mean/std 归一化。
- 确保存在 `assets/imagenet_class_index.json`（已随仓库提供标准版本），`prepare_imagenet_val.py` 会优先使用它，以生成与 torchvision 类别顺序一致的 `imagenet_val_labels.csv` 与 `imagenet_classes.tsv`。
- 使用 `--class-names imagenet_classes.tsv --inspect 5` 做人工抽检，确认类别名与预测是否合理。

## 4. val/ 目录结构：扁平 vs 按 WNID 子目录

现状：
- `prepare_imagenet_val.py` 默认不重组目录，输出扁平结构：`imagenet_val/val/*.JPEG`。
- 若需要可加 `--reorg` 将图片移动到 `imagenet_val/val/<wnid>/`。
- `trt_compare` 会递归扫描，兼容两种结构（并支持 .jpg/.jpeg/.JPEG/.png）。

还原为扁平结构（macOS zsh）：
```bash
for d in val/*(/); do mv "$d"/* val/; done
rmdir val/*(/)
```

## 5. ImageNet devkit 缺 map_clsloc.txt，如何生成标签？

- 我们不依赖 map_clsloc.txt；脚本会解析 devkit 的 `meta.mat` 获取 ILSVRC2012_ID→WNID 映射。
- 若存在 `assets/imagenet_class_index.json`，则用 WNID→index（0..999）生成 `imagenet_val_labels.csv`；
- 同时写出 `imagenet_classes.tsv`（`index\twnid\tname`），便于可读输出。

## 6. 我想完全离线准备 ImageNet 映射

- 本仓库已内置 `assets/imagenet_class_index.json` 标准版本，可直接使用。
- 如需从本机环境导出，可运行 `python scripts/export_imagenet_class_index.py`（需要已安装 torchvision/torch）。

## 7. Kaggle 环境如何快速检查 TRT 版本与可用性？

- 运行 `bash scripts/setup_kaggle_env.sh` 解压并写出 `/kaggle/working/tensorrt_env.sh`；
- 随后 `source /kaggle/working/tensorrt_env.sh` 并尝试 `trtexec --version`（注意，有些包不含 trtexec）。

## 8. 编译时报找不到 NvInfer.h

- 说明头文件未被正确指向到 TensorRT 解压路径。优先使用我们的环境脚本并在 CMake 配置时确保 `TensorRT_ROOT` 等变量可被发现。
- 通常通过 `scripts/setup_kaggle_env.sh` 或在本机设置好 TRT 安装根目录即可解决。

---

若你的问题不在以上列表，请提交 issue 并附上：机器/驱动/CUDA/TRT 版本、是否已 `source bin/env.sh`、运行目录结构截图与关键日志片段，便于排查。
