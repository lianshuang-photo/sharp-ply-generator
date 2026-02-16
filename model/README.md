# Model

从 Apple SHARP 官方仓库获取模型代码和权重：

```bash
# 克隆 SHARP 仓库（包含 sharp/ Python 包）
git clone https://github.com/apple/ml-sharp.git temp && cp -r temp/sharp ../sharp && rm -rf temp

# 下载模型权重
curl -O https://docs-assets.developer.apple.com/ml-research/models/sharp/sharp_model.pt
```

详情见 https://github.com/apple/ml-sharp
