# SHARP PLY Generator 3D 高斯模型生成器

基于 Apple [SHARP](https://github.com/apple/ml-sharp) 模型，从单张照片生成 3D Gaussian Splatting PLY 文件。

> 生成的 PLY 文件可用 [spatial-photo-viewer](https://github.com/baotianyi/spatial-photo-viewer) 进行预览，支持陀螺仪视差效果。

## 功能

- 单张图片 → 3D 高斯点云 PLY 文件
- 批量处理，自动遍历 `input/` 文件夹
- 支持 CUDA / Apple MPS / CPU 推理
- 可调焦距和分辨率参数

## 环境要求

- Python 3.10+
- PyTorch（CUDA 或 MPS）
- [SHARP](https://github.com/apple/ml-sharp) 模型代码及权重

## 使用方法

### 1. 准备目录结构

```
sharp-ply-generator/
├── run.py
├── model/
│   └── sharp_model.pt    # SHARP 模型权重
├── input/
│   ├── photo1.jpg        # 输入图片
│   └── photo2.png
├── output/               # 自动创建，输出 PLY 文件
└── sharp/                # SHARP 模型代码
```

### 2. 放置 SHARP 模型

将 [SHARP](https://github.com/apple/ml-sharp) 仓库中的 `sharp/` 目录和模型权重 `.pt` 文件分别放入项目根目录和 `model/` 文件夹。

### 3. 运行

```bash
python run.py
```

脚本会：
1. 自动检测 `model/` 中的 `.pt` 权重文件
2. 遍历 `input/` 中的所有图片（jpg/png/webp/heic 等）
3. 预处理（最长边缩放到 2048px）
4. SHARP 模型推理生成 3D 高斯点云
5. 输出 PLY 文件到 `output/`

### 参数配置

在 `run.py` 顶部可调整：

```python
FOCAL_LENGTH_MM = 50.0      # 焦距（毫米）
LONGEST_SIDE    = 2048       # 输入图片最长边缩放目标
INTERNAL_SHAPE  = (1536, 1536)  # SHARP 内部推理分辨率（固定）
```

## 输出示例

```
模型: sharp_model.pt
设备: mps
焦距: 50.0mm | 最长边: 2048px
图片数: 3
--------------------------------------------------
[1/3] photo1.jpg
  -> photo1.ply | 63.0MB | 4.2s
[2/3] photo2.png
  -> photo2.ply | 63.0MB | 3.8s
...
完成: 3 成功, 0 失败 | 总耗时 12.1s
```

## 相关项目

- [spatial-photo-viewer](https://github.com/baotianyi/spatial-photo-viewer) — 生成的 PLY 文件的 Web 查看器，支持陀螺仪视差效果

## License

[MIT](LICENSE)
