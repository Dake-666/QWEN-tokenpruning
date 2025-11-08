# Qwen-Image-Edit 使用指南

本项目用于使用 Qwen-Image-Edit 模型进行图像编辑。

## 安装依赖

首先安装所需依赖：

```bash
pip install -r requirements.txt
```

或者直接安装 diffusers（从 GitHub）：

```bash
pip install git+https://github.com/huggingface/diffusers
pip install torch pillow transformers accelerate
```

## 使用方法

1. 准备输入图片，命名为 `input.png` 并放在当前目录
2. 运行示例脚本：

```bash
python example.py
```

## 模型信息

- 模型名称: Qwen/Qwen-Image-Edit
- 模型页面: https://huggingface.co/Qwen/Qwen-Image-Edit
- 许可证: Apache 2.0

## 注意事项

- 需要 CUDA 支持的 GPU 才能运行
- 首次运行时会自动下载模型，请确保网络连接正常
- 确保输入图片名为 `input.png` 或修改代码中的图片路径

## 故障排除

### 错误：AttributeError: 'dict' object has no attribute 'to_dict'

**解决方案**：

```bash
# 方法1：运行自动修复（推荐）
python 0_环境检查和修复.py

# 方法2：手动升级 transformers
pip install --upgrade transformers>=4.48.0

# 然后重新运行
python 2_load_and_inference_lightning.py
```

详细说明请查看：`快速修复_transformers错误.md`


