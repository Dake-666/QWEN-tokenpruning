"""
Qwen-Image-Edit 使用示例
参考: https://huggingface.co/Qwen/Qwen-Image-Edit
"""
import os
from PIL import Image
import torch

from diffusers import QwenImageEditPipeline

# 加载模型管道
pipeline = QwenImageEditPipeline.from_pretrained("Qwen/Qwen-Image-Edit")
print("pipeline 加载完成")

# 设置精度和设备
pipeline.to(torch.bfloat16)
pipeline.to("cuda")
pipeline.set_progress_bar_config(disable=None)

# 加载输入图片（需要将图片放在当前目录）
image = Image.open("./input.png").convert("RGB")

# 编辑提示词
prompt = "Change the rabbit's color to purple, with a flash light background."

# 设置输入参数
inputs = {
    "image": image,
    "prompt": prompt,
    "generator": torch.manual_seed(0),
    "true_cfg_scale": 4.0,
    "negative_prompt": " ",
    "num_inference_steps": 50,
}

# 执行图像编辑
with torch.inference_mode():
    output = pipeline(**inputs)
    output_image = output.images[0]
    output_image.save("output_image_edit.png")
    print("图片已保存至", os.path.abspath("output_image_edit.png"))


