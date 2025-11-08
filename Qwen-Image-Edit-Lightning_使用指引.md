# Qwen-Image-Edit-2509-Lightning 模型使用指引

## 📋 模型信息

- **模型名称**: Qwen-Image-Edit-2509-Lightning-4steps-V1.0-bf16
- **模型路径**: `lightx2v/Qwen-Image-Lightning/Qwen-Image-Edit-2509/`
- **模型文件**: `Qwen-Image-Edit-2509-Lightning-4steps-V1.0-bf16.safetensors`
- **特点**: 知识蒸馏模型，仅需 **4 步推理**（原版需 50 步）
- **精度**: bfloat16

---

## 🔽 一、模型下载

### 方法1：使用 Hugging Face CLI 下载

```bash
# 安装 huggingface_hub
pip install huggingface_hub

# 下载模型文件
huggingface-cli download lightx2v/Qwen-Image-Lightning \
    Qwen-Image-Edit-2509/Qwen-Image-Edit-2509-Lightning-4steps-V1.0-bf16.safetensors \
    --local-dir ./models/lightning \
    --local-dir-use-symlinks False
```

### 方法2：使用 Python 脚本下载

创建 `download_lightning_model.py`：

```python
from huggingface_hub import hf_hub_download
import os

# 配置路径
repo_id = "lightx2v/Qwen-Image-Lightning"
filename = "Qwen-Image-Edit-2509/Qwen-Image-Edit-2509-Lightning-4steps-V1.0-bf16.safetensors"
local_dir = "./models/lightning"

# 创建目录
os.makedirs(local_dir, exist_ok=True)

# 下载模型
print(f"正在下载 {filename}...")
model_path = hf_hub_download(
    repo_id=repo_id,
    filename=filename,
    local_dir=local_dir,
    local_dir_use_symlinks=False
)

print(f"模型已下载到: {model_path}")
```

运行：
```bash
python download_lightning_model.py
```

### 方法3：手动下载

1. 访问：https://huggingface.co/lightx2v/Qwen-Image-Lightning/tree/main/Qwen-Image-Edit-2509
2. 下载 `Qwen-Image-Edit-2509-Lightning-4steps-V1.0-bf16.safetensors`
3. 保存到：`./models/lightning/Qwen-Image-Edit-2509-Lightning-4steps-V1.0-bf16.safetensors`

---

## 🚀 二、使用方法

### 方法1：直接加载 Lightning 模型（推荐）

Lightning 模型通常是**完整的 transformer 权重**，可以直接替换原模型的 transformer 部分。

**步骤1：创建加载脚本**

创建 `load_lightning_model.py`：

```python
import torch
from diffusers import QwenImageEditPipeline
from safetensors.torch import load_file

# 1. 加载原始 Qwen-Image-Edit Pipeline（包含 VAE, Text Encoder 等）
print("加载基础 Pipeline...")
pipeline = QwenImageEditPipeline.from_pretrained(
    "Qwen/Qwen-Image-Edit",
    torch_dtype=torch.bfloat16
)

# 2. 加载 Lightning Transformer 权重
lightning_model_path = "./models/lightning/Qwen-Image-Edit-2509-Lightning-4steps-V1.0-bf16.safetensors"
print(f"加载 Lightning 模型: {lightning_model_path}")

# 加载 safetensors 文件
lightning_state_dict = load_file(lightning_model_path)

# 3. 将 Lightning 权重加载到 Transformer
# 注意：确保权重键名匹配
pipeline.transformer.load_state_dict(lightning_state_dict, strict=False)

# 4. 移动到 GPU
pipeline.to("cuda")

print("Lightning 模型加载完成！")
print(f"Transformer 设备: {next(pipeline.transformer.parameters()).device}")
print(f"Transformer 精度: {next(pipeline.transformer.parameters()).dtype}")

# 保存完整 Pipeline（可选）
# pipeline.save_pretrained("./models/qwen-image-edit-lightning")
```

**步骤2：运行推理（4 步）**

创建 `inference_lightning.py`：

```python
import torch
from PIL import Image
from diffusers import QwenImageEditPipeline
from safetensors.torch import load_file

# 加载 Pipeline（使用上面的方法）
pipeline = QwenImageEditPipeline.from_pretrained(
    "Qwen/Qwen-Image-Edit",
    torch_dtype=torch.bfloat16
)

# 加载 Lightning 权重
lightning_state_dict = load_file(
    "./models/lightning/Qwen-Image-Edit-2509-Lightning-4steps-V1.0-bf16.safetensors"
)
pipeline.transformer.load_state_dict(lightning_state_dict, strict=False)
pipeline.to("cuda")

# 准备输入
image = Image.open("input.png").convert("RGB")
prompt = "Change the rabbit's color to purple, with a flash light background."

# ⭐ 关键：Lightning 模型只需 4 步推理
inputs = {
    "image": image,
    "prompt": prompt,
    "generator": torch.manual_seed(0),
    "true_cfg_scale": 4.0,
    "negative_prompt": " ",  # 空字符串也可以
    "num_inference_steps": 4,  # ⭐ Lightning: 4 步（原版 50 步）
}

# 推理
print("开始推理（4 步）...")
with torch.inference_mode():
    output = pipeline(**inputs)
    output_image = output.images[0]
    output_image.save("output_lightning_4steps.png")
    print("推理完成！输出保存至: output_lightning_4steps.png")
```

---

### 方法2：检查权重结构并手动映射

如果权重键名不匹配，需要手动映射：

创建 `check_and_load_lightning.py`：

```python
import torch
from safetensors.torch import load_file
from diffusers import QwenImageEditPipeline

# 加载原始 pipeline
pipeline = QwenImageEditPipeline.from_pretrained(
    "Qwen/Qwen-Image-Edit",
    torch_dtype=torch.bfloat16
)

# 加载 Lightning 权重
lightning_path = "./models/lightning/Qwen-Image-Edit-2509-Lightning-4steps-V1.0-bf16.safetensors"
lightning_state_dict = load_file(lightning_path)

# 检查权重键名
print("=== Lightning 模型键名（前10个）===")
for i, key in enumerate(list(lightning_state_dict.keys())[:10]):
    print(f"{i+1}. {key}")

print("\n=== Transformer 原始键名（前10个）===")
transformer_state_dict = pipeline.transformer.state_dict()
for i, key in enumerate(list(transformer_state_dict.keys())[:10]):
    print(f"{i+1}. {key}")

# 尝试匹配键名
print("\n=== 尝试加载 ===")
try:
    pipeline.transformer.load_state_dict(lightning_state_dict, strict=False)
    print("✅ 加载成功（使用 strict=False）")
except Exception as e:
    print(f"❌ 加载失败: {e}")
    print("\n需要手动映射键名...")
```

---

## 🔄 三、LoRA 合并（如果适用）

### 判断是否为 LoRA 权重

检查权重键名是否包含 `lora_A`、`lora_B` 或 `alpha`：

```python
from safetensors.torch import load_file

lightning_state_dict = load_file(
    "./models/lightning/Qwen-Image-Edit-2509-Lightning-4steps-V1.0-bf16.safetensors"
)

# 检查是否为 LoRA
is_lora = any("lora_A" in key or "lora_B" in key for key in lightning_state_dict.keys())

if is_lora:
    print("这是 LoRA 权重，需要合并到基础模型")
    # 使用 pipeline 的 load_lora_weights 方法
else:
    print("这是完整模型权重，可以直接加载")
```

### 如果是 LoRA，使用以下方法合并

**方法A：运行时加载（不合并）**

```python
from diffusers import QwenImageEditPipeline
import torch

pipeline = QwenImageEditPipeline.from_pretrained(
    "Qwen/Qwen-Image-Edit",
    torch_dtype=torch.bfloat16
)

# 加载 LoRA（运行时）
pipeline.load_lora_weights(
    "./models/lightning",
    weight_name="Qwen-Image-Edit-2509-Lightning-4steps-V1.0-bf16.safetensors"
)

# 设置 LoRA scale（可选）
# pipeline.fuse_lora(lora_scale=1.0)  # 融合 LoRA 到权重中

pipeline.to("cuda")
```

**方法B：合并 LoRA 到权重（永久）**

```python
from diffusers import QwenImageEditPipeline
import torch

pipeline = QwenImageEditPipeline.from_pretrained(
    "Qwen/Qwen-Image-Edit",
    torch_dtype=torch.bfloat16
)

# 加载 LoRA
pipeline.load_lora_weights(
    "./models/lightning",
    weight_name="Qwen-Image-Edit-2509-Lightning-4steps-V1.0-bf16.safetensors"
)

# ⭐ 融合 LoRA 到基础权重（永久合并）
pipeline.fuse_lora(lora_scale=1.0)

# 保存合并后的模型
pipeline.save_pretrained("./models/qwen-image-edit-lightning-merged")

# 之后可以直接加载合并后的模型
# pipeline = QwenImageEditPipeline.from_pretrained(
#     "./models/qwen-image-edit-lightning-merged",
#     torch_dtype=torch.bfloat16
# )
```

---

## 🔧 四、替换原模型文件

### 方案1：替换 Transformer 权重文件

如果 Lightning 模型是完整的 transformer 权重：

```bash
# 1. 找到原模型的 transformer 权重位置
# 通常在：~/.cache/huggingface/hub/models--Qwen--Qwen-Image-Edit/snapshots/.../transformer/

# 2. 备份原文件
cp transformer/model.safetensors transformer/model.safetensors.backup

# 3. 替换为 Lightning 模型
cp ./models/lightning/Qwen-Image-Edit-2509-Lightning-4steps-V1.0-bf16.safetensors \
   transformer/model.safetensors

# 4. 使用原 pipeline 加载（会自动使用新权重）
```

### 方案2：创建新的模型目录

```bash
# 1. 复制整个模型目录
cp -r ~/.cache/huggingface/hub/models--Qwen--Qwen-Image-Edit/snapshots/<latest> \
      ./models/qwen-image-edit-lightning

# 2. 替换 transformer 权重
cp ./models/lightning/Qwen-Image-Edit-2509-Lightning-4steps-V1.0-bf16.safetensors \
   ./models/qwen-image-edit-lightning/transformer/model.safetensors

# 3. 从新目录加载
# pipeline = QwenImageEditPipeline.from_pretrained(
#     "./models/qwen-image-edit-lightning",
#     torch_dtype=torch.bfloat16
# )
```

---

## 📝 五、完整使用示例

创建 `lightning_inference_complete.py`：

```python
"""
Qwen-Image-Edit Lightning 模型完整推理示例
"""
import torch
from PIL import Image
from diffusers import QwenImageEditPipeline
from safetensors.torch import load_file
import os

def load_lightning_pipeline(
    base_model="Qwen/Qwen-Image-Edit",
    lightning_model_path="./models/lightning/Qwen-Image-Edit-2509-Lightning-4steps-V1.0-bf16.safetensors",
    device="cuda",
    dtype=torch.bfloat16
):
    """
    加载 Lightning 模型 Pipeline
    
    Args:
        base_model: 基础模型路径（Hugging Face ID 或本地路径）
        lightning_model_path: Lightning 模型权重路径
        device: 设备
        dtype: 数据类型
    """
    print(f"1. 加载基础 Pipeline: {base_model}")
    pipeline = QwenImageEditPipeline.from_pretrained(
        base_model,
        torch_dtype=dtype
    )
    
    if os.path.exists(lightning_model_path):
        print(f"2. 加载 Lightning 模型: {lightning_model_path}")
        lightning_state_dict = load_file(lightning_model_path)
        
        # 尝试加载权重
        try:
            pipeline.transformer.load_state_dict(lightning_state_dict, strict=True)
            print("   ✅ 精确匹配加载成功")
        except:
            print("   ⚠️ 精确匹配失败，尝试宽松加载...")
            missing, unexpected = pipeline.transformer.load_state_dict(
                lightning_state_dict, strict=False
            )
            if missing:
                print(f"   ⚠️ 缺失键: {missing[:5]}...")
            if unexpected:
                print(f"   ⚠️ 额外键: {unexpected[:5]}...")
    else:
        print(f"   ⚠️ Lightning 模型文件不存在: {lightning_model_path}")
        print("   使用原始模型（50步推理）")
    
    pipeline.to(device)
    return pipeline

def run_inference(
    pipeline,
    image_path="input.png",
    prompt="Change the rabbit's color to purple, with a flash light background.",
    output_path="output_lightning.png",
    num_steps=4,  # Lightning: 4步
    true_cfg_scale=4.0,
    negative_prompt=" ",
):
    """
    运行推理
    """
    # 加载图像
    image = Image.open(image_path).convert("RGB")
    
    # 准备输入
    inputs = {
        "image": image,
        "prompt": prompt,
        "generator": torch.manual_seed(0),
        "true_cfg_scale": true_cfg_scale,
        "negative_prompt": negative_prompt,
        "num_inference_steps": num_steps,  # ⭐ Lightning: 4步
    }
    
    # 推理
    print(f"\n3. 开始推理（{num_steps}步）...")
    with torch.inference_mode():
        output = pipeline(**inputs)
        output_image = output.images[0]
        output_image.save(output_path)
        print(f"   ✅ 推理完成！输出: {output_path}")
    
    return output_image

if __name__ == "__main__":
    # 加载 Pipeline
    pipeline = load_lightning_pipeline()
    
    # 运行推理
    run_inference(
        pipeline,
        image_path="input.png",
        prompt="Change the rabbit's color to purple, with a flash light background.",
        output_path="output_lightning_4steps.png",
        num_steps=4,  # Lightning 模型
    )
    
    print("\n✅ 完成！")
```

---

## ⚙️ 六、关键参数说明

### Lightning 模型专用参数

| 参数 | 原版值 | Lightning 值 | 说明 |
|------|--------|--------------|------|
| `num_inference_steps` | 50 | **4** | ⭐ 推理步数 |
| `true_cfg_scale` | 4.0 | 4.0 | CFG 强度（可保持） |
| `torch_dtype` | bfloat16 | bfloat16 | 精度（匹配模型） |

### 注意事项

1. **推理步数**：Lightning 模型设计为 4 步，使用更多步数可能不会提升质量
2. **模型兼容性**：确保 Lightning 模型与基础模型版本匹配（2509 版本）
3. **权重格式**：如果是 safetensors 格式，需要使用 `load_file` 加载
4. **设备内存**：Lightning 模型推理更快，但模型大小可能相同

---

## 🐛 七、常见问题排查

### 问题1：权重键名不匹配

**症状**：`load_state_dict` 报错，提示键名不匹配

**解决方案**：
```python
# 使用 strict=False
pipeline.transformer.load_state_dict(lightning_state_dict, strict=False)

# 或手动映射键名
def map_keys(old_dict, key_mapping):
    new_dict = {}
    for old_key, new_key in key_mapping.items():
        if old_key in old_dict:
            new_dict[new_key] = old_dict[old_key]
    return new_dict
```

### 问题2：模型结构不匹配

**症状**：权重形状不匹配

**解决方案**：
- 确认 Lightning 模型是否与基础模型版本匹配
- 检查模型配置文件（config.json）

### 问题3：推理结果不理想

**解决方案**：
- 确认使用 `num_inference_steps=4`
- 检查 `true_cfg_scale` 参数
- 尝试不同的 `negative_prompt`

---

## 📚 八、参考资源

1. **模型仓库**: https://huggingface.co/lightx2v/Qwen-Image-Lightning
2. **原版模型**: https://huggingface.co/Qwen/Qwen-Image-Edit
3. **Diffusers 文档**: https://huggingface.co/docs/diffusers
4. **Lightning 论文**: Knowledge Distillation for Fast Diffusion Models

---

## ✅ 总结

1. **下载模型**：使用 Hugging Face CLI 或 Python 脚本
2. **加载方式**：
   - 如果是完整权重：直接 `load_state_dict`
   - 如果是 LoRA：使用 `load_lora_weights` + `fuse_lora`
3. **推理参数**：`num_inference_steps=4`（关键）
4. **替换文件**：可以替换原模型的 transformer 权重文件

**关键代码**：
```python
# 加载 Lightning 权重
lightning_state_dict = load_file("lightning_model.safetensors")
pipeline.transformer.load_state_dict(lightning_state_dict, strict=False)

# 4步推理
output = pipeline(image, prompt, num_inference_steps=4)
```

