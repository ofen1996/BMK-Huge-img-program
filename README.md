# BMK Huge Image Program

面向超大尺寸 BioMarker 芯片图像的阵列定位、几何配准与多通道合并原型。项目通过 ONNX/YOLO 关键点检测、规则点阵模板和 libvips 图像处理，支持在内存受限场景中处理高分辨率芯片图像。

## 核心能力

- 使用 `model/*.onnx` 检测芯片关键点
- 构建标准芯片圆点阵列并与原图匹配
- 支持 `huge` 芯片模式及大面积阵列参数
- 通过 libvips 合并多通道大图
- 以 `setting/*.ini` 固化扫描仪、图块和芯片规格

## 项目结构

| 路径 | 说明 |
| --- | --- |
| `match_imgs.py` | 主配准逻辑、标准点阵生成与图像匹配 |
| `merge_channels_pics_by_libvips.py` | 多通道图像合并 |
| `need/KpDetectByYolo.py` | ONNX/YOLO 关键点检测与 NMS |
| `need/BmTiffLib.py` | TIFF/大图辅助操作 |
| `model/` | 关键点检测模型 |
| `setting/` | S1000–S3000/huge 等参数预设 |

## 环境准备

建议使用 Python 3.9+，并安装：

```bash
pip install numpy opencv-python tifffile pillow pyvips onnxruntime torch scikit-learn
```

需要额外安装系统级 libvips。Windows 用户请将 libvips 的 `bin` 目录加入 PATH；部分脚本中保留了开发机路径，应改为本机路径或由环境变量统一配置。

## 使用建议

1. 选择最接近采集条件的 `setting/*.ini`，确认 `base_mode`、图像尺寸与扫描仪参数。
2. 确保 `model/` 下的 ONNX 模型与代码保持相对路径。
3. 在小图或裁剪样本上验证关键点检测与阵列匹配。
4. 再运行配准/合并脚本处理完整数据，并将输出写入独立目录。

## 注意事项

本仓库当前是面向特定设备与内部数据格式的工程原型，尚未提供稳定的通用 CLI 或自动化测试。超大图像会占用大量内存、磁盘和处理时间；生产使用前应固定配置、记录软件版本，并做结果抽检。

项目未声明开源许可证。模型、数据和配置仅应在获授权范围内使用。
