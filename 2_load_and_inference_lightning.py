"""
步骤2: 加载 Lightning LoRA 并运行推理
按照官方 Hugging Face 指引实现
"""
import torch
import math
from PIL import Image
from diffusers import QwenImageEditPipeline, FlowMatchEulerDiscreteScheduler

def setup_lightning_pipeline():
    """
    按照官方指引设置 Lightning Pipeline
    """
    print("=" * 60)
    print("设置 Qwen-Image-Edit Lightning Pipeline")
    print("=" * 60)
    
    # 步骤1: 配置调度器（按照官方配置）
    print("\n1. 配置 FlowMatchEulerDiscreteScheduler...")
    scheduler_config = {
        "base_image_seq_len": 256,
        "base_shift": math.log(3),  # 官方推荐配置
        "invert_sigmas": False,
        "max_image_seq_len": 8192,
        "max_shift": math.log(3),
        "num_train_timesteps": 1000,
        "shift": 1.0,
        "shift_terminal": None,
        "stochastic_sampling": False,
        "time_shift_type": "exponential",
        "use_beta_sigmas": False,
        "use_dynamic_shifting": True,  # 重要：启用动态 shifting
        "use_exponential_sigmas": False,
        "use_karras_sigmas": False,
    }
    scheduler = FlowMatchEulerDiscreteScheduler.from_config(scheduler_config)
    
    # 步骤2: 加载基础 Qwen-Image-Edit 模型
    print("\n2. 加载基础模型: Qwen/Qwen-Image-Edit...")
    pipe = QwenImageEditPipeline.from_pretrained(
        "Qwen/Qwen-Image-Edit",
        scheduler=scheduler,  # 使用配置的调度器
        torch_dtype=torch.bfloat16
    )
    
    # 步骤3: 加载 Lightning LoRA 权重
    print("\n3. 加载 Lightning LoRA 权重...")
    try:
        pipe.load_lora_weights(
            "lightx2v/Qwen-Image-Lightning",
            weight_name="Qwen-Image-Edit-2509/Qwen-Image-Edit-2509-Lightning-4steps-V1.0-bf16.safetensors"
        )
        print("   ✅ LoRA 权重加载成功")
    except Exception as e:
        print(f"   ❌ LoRA 加载失败: {e}")
        print("   尝试从本地加载...")
        pipe.load_lora_weights(
            "./models/lightning_lora/Qwen-Image-Edit-2509",
            weight_name="Qwen-Image-Edit-2509-Lightning-4steps-V1.0-bf16.safetensors"
        )
    
    # 步骤4: 移动到 GPU
    print("\n4. 移动到 CUDA...")
    pipe.to("cuda")
    
    print("\n✅ Pipeline 设置完成！")
    return pipe

def run_lightning_inference(
    pipe,
    image_path="input.png",
    prompt="Change the rabbit's color to purple, with a flash light background.",
    output_path="output_lightning_4steps.png"
):
    """
    使用 Lightning 运行推理
    """
    print("\n" + "=" * 60)
    print("运行 Lightning 推理")
    print("=" * 60)
    
    # 加载输入图像
    print(f"\n1. 加载输入图像: {image_path}")
    image = Image.open(image_path).convert("RGB")
    print(f"   图像尺寸: {image.size}")
    
    # 准备推理参数（按照官方示例）
    print("\n2. 准备推理参数...")
    inference_params = {
        "image": image,
        "prompt": prompt,
        "negative_prompt": " ",  # 空字符串
        "num_inference_steps": 4,  # ⭐ Lightning 4步
        "true_cfg_scale": 1.0,  # ⭐ 官方推荐使用 1.0（而非 4.0）
        "generator": torch.manual_seed(0),
    }
    
    print(f"   - Prompt: {prompt}")
    print(f"   - 推理步数: {inference_params['num_inference_steps']}")
    print(f"   - CFG Scale: {inference_params['true_cfg_scale']}")
    
    # 执行推理
    print("\n3. 执行推理...")
    with torch.inference_mode():
        output = pipe(**inference_params)
        output_image = output.images[0]
    
    # 保存结果
    print(f"\n4. 保存结果: {output_path}")
    output_image.save(output_path)
    
    print("\n✅ 推理完成！")
    return output_image

def main():
    """
    主流程
    """
    # 设置 Pipeline
    pipe = setup_lightning_pipeline()
    
    # 运行推理
    run_lightning_inference(
        pipe,
        image_path="input.png",
        prompt="Change the rabbit's color to purple, with a flash light background.",
        output_path="output_lightning_4steps.png"
    )

if __name__ == "__main__":
    main()

