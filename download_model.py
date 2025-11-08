"""
下载 Qwen-Image-Edit 模型
首次运行时会自动从 Hugging Face 下载模型文件
"""
import os
from diffusers import QwenImageEditPipeline

print("开始下载 Qwen-Image-Edit 模型...")
print("这可能需要一些时间，请确保网络连接正常...")

# 下载模型（首次运行时会自动下载）
pipeline = QwenImageEditPipeline.from_pretrained("Qwen/Qwen-Image-Edit")

print("\n模型下载完成！")
print("模型已保存到:", os.path.expanduser("~/.cache/huggingface/hub"))


