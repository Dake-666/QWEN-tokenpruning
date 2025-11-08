"""
Qwen-Image-Edit 使用示例
参考: https://huggingface.co/Qwen/Qwen-Image-Edit
"""
import os
from datetime import datetime
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

# 创建输出目录
output_dir = "outputs"
os.makedirs(output_dir, exist_ok=True)

# 生成带时间戳的输出文件名
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_filename = f"output_edit_50steps_{timestamp}.png"
output_path = os.path.join(output_dir, output_filename)

# 执行图像编辑
print("开始推理（50步）...")
with torch.inference_mode():
    output = pipeline(**inputs)
    output_image = output.images[0]
    
    # 保存带时间戳的版本
    output_image.save(output_path)
    print(f"图片已保存至: {os.path.abspath(output_path)}")
    
    # 保存最新版本
    latest_path = os.path.join(output_dir, "latest_output_50steps.png")
    output_image.save(latest_path)
    print(f"最新版本: {os.path.abspath(latest_path)}")


