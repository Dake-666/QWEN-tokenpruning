"""
步骤1: 下载 Qwen-Image-Edit Lightning LoRA 权重
官方仓库: https://huggingface.co/lightx2v/Qwen-Image-Lightning
"""
from huggingface_hub import hf_hub_download
import os

def download_lightning_lora():
    """
    下载 Lightning LoRA 权重
    """
    repo_id = "lightx2v/Qwen-Image-Lightning"
    
    # 可选的权重文件（根据需要选择）
    weight_files = {
        "4steps": "Qwen-Image-Edit-2509/Qwen-Image-Edit-2509-Lightning-4steps-V1.0-bf16.safetensors",
        "8steps": "Qwen-Image-Lightning-8steps-V1.0.safetensors",
    }
    
    local_dir = "./models/lightning_lora"
    os.makedirs(local_dir, exist_ok=True)
    
    print("=" * 60)
    print("开始下载 Qwen-Image-Edit Lightning LoRA 权重")
    print("=" * 60)
    
    # 下载 4步版本（用于 Edit 任务）
    print("\n下载 4步版本（Edit-2509）...")
    try:
        model_path = hf_hub_download(
            repo_id=repo_id,
            filename=weight_files["4steps"],
            local_dir=local_dir,
            local_dir_use_symlinks=False
        )
        print(f"✅ 下载成功: {model_path}")
    except Exception as e:
        print(f"❌ 下载失败: {e}")
    
    print("\n下载完成！LoRA 权重保存在: {}".format(local_dir))
    print("\n注意: Lightning 是 LoRA 权重，需要加载到基础模型上")

if __name__ == "__main__":
    download_lightning_lora()

