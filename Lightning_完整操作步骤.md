# Qwen-Image-Edit Lightning 完整操作步骤

## 🎯 关键发现

根据官方 Hugging Face 文档：
- **Lightning 是 LoRA 权重**，不是完整模型
- 需要先加载基础模型 `Qwen/Qwen-Image-Edit`
- 然后使用 `load_lora_weights()` 加载 Lightning LoRA
- 使用 `fuse_lora()` 可以将 LoRA 永久融合到权重中

---

## 📚 官方文档

- **Hugging Face**: https://huggingface.co/lightx2v/Qwen-Image-Lightning
- **基础模型**: https://huggingface.co/Qwen/Qwen-Image-Edit
- **参考仓库**: https://github.com/huggingface/diffusers

---

## 🔧 准备工作

### 1. 安装依赖

```bash
# 安装最新版 diffusers（从 GitHub）
pip install git+https://github.com/huggingface/diffusers

# 安装其他依赖
pip install torch transformers accelerate pillow safetensors
```

### 2. 确认环境

```bash
python -c "import diffusers; print(f'diffusers 版本: {diffusers.__version__}')"
python -c "import torch; print(f'PyTorch 版本: {torch.__version__}')"
python -c "import torch; print(f'CUDA 可用: {torch.cuda.is_available()}')"
```

---

## 📥 步骤1: 下载 Lightning LoRA 权重

### 方法A：运行下载脚本

```bash
python 1_download_lightning_lora.py
```

### 方法B：手动下载命令

```bash
# 使用 Hugging Face CLI
huggingface-cli download lightx2v/Qwen-Image-Lightning \
    Qwen-Image-Edit-2509/Qwen-Image-Edit-2509-Lightning-4steps-V1.0-bf16.safetensors \
    --local-dir ./models/lightning_lora
```

### 方法C：在线加载（推荐）

脚本会自动从 Hugging Face 下载，无需手动操作。

---

## 🚀 步骤2: 加载并运行推理

### 完整流程（按照官方指引）

运行准备好的脚本：

```bash
python 2_load_and_inference_lightning.py
```

### 代码说明

脚本执行以下步骤：

#### 1. 配置调度器（关键配置）

```python
scheduler_config = {
    "base_image_seq_len": 256,
    "base_shift": math.log(3),  # ⭐ 官方推荐
    "use_dynamic_shifting": True,  # ⭐ 重要：启用动态 shifting
    # ... 其他配置
}
scheduler = FlowMatchEulerDiscreteScheduler.from_config(scheduler_config)
```

**关键参数解释**：
- `base_shift`: `math.log(3)` ≈ 1.099（官方推荐）
- `use_dynamic_shifting`: `True`（必须启用，用于 Lightning）

#### 2. 加载基础模型

```python
pipe = QwenImageEditPipeline.from_pretrained(
    "Qwen/Qwen-Image-Edit",
    scheduler=scheduler,  # ⭐ 使用配置的调度器
    torch_dtype=torch.bfloat16
)
```

#### 3. 加载 Lightning LoRA

```python
pipe.load_lora_weights(
    "lightx2v/Qwen-Image-Lightning",
    weight_name="Qwen-Image-Edit-2509/Qwen-Image-Edit-2509-Lightning-4steps-V1.0-bf16.safetensors"
)
```

**自动下载**：首次运行会自动从 Hugging Face 下载 LoRA 权重。

#### 4. 运行推理

```python
output = pipe(
    image=input_image,
    prompt="Change the rabbit's color to purple",
    negative_prompt=" ",
    num_inference_steps=4,  # ⭐ Lightning: 4步
    true_cfg_scale=1.0,  # ⭐ 官方推荐 1.0（注意：不是 4.0）
    generator=torch.manual_seed(0),
)
```

**关键参数**：
- `num_inference_steps=4`: Lightning 4步推理
- `true_cfg_scale=1.0`: 官方推荐（而非原版的 4.0）

---

## 🔀 步骤3: 融合 LoRA（可选）

如果希望永久保存 Lightning 权重，避免每次加载 LoRA：

```bash
python 3_merge_lora_to_weights.py
```

### 融合流程

```python
# 1. 加载基础模型 + LoRA
pipe = QwenImageEditPipeline.from_pretrained("Qwen/Qwen-Image-Edit")
pipe.load_lora_weights("lightx2v/Qwen-Image-Lightning", ...)

# 2. 融合 LoRA 到权重
pipe.fuse_lora(lora_scale=1.0)

# 3. 保存融合后的模型
pipe.save_pretrained("./models/qwen-image-edit-lightning-merged")

# 4. 之后直接加载融合后的模型（无需再加载 LoRA）
pipe = QwenImageEditPipeline.from_pretrained("./models/qwen-image-edit-lightning-merged")
```

---

## 📊 与原版对比

| 特性 | 原版 Qwen-Image-Edit | Lightning 版本 |
|------|---------------------|---------------|
| **推理步数** | 50 步 | **4 步** ⚡ |
| **CFG Scale** | 4.0 | **1.0** |
| **调度器配置** | 默认 | **自定义** (dynamic shifting) |
| **模型类型** | 完整模型 | **LoRA 权重** |
| **加载方式** | `from_pretrained` | `load_lora_weights` |
| **推理速度** | 慢 (~50秒) | **快 (~4秒)** ⚡ |

---

## 🎛️ 完整参数说明

### 调度器参数（FlowMatchEulerDiscreteScheduler）

```python
{
    "base_image_seq_len": 256,        # 基础图像序列长度
    "base_shift": math.log(3),        # ⭐ 基础偏移（1.099）
    "max_image_seq_len": 8192,        # 最大图像序列长度
    "max_shift": math.log(3),         # 最大偏移
    "use_dynamic_shifting": True,     # ⭐ 动态偏移（必须启用）
    "time_shift_type": "exponential", # 时间偏移类型
    "num_train_timesteps": 1000,      # 训练时间步数
}
```

### 推理参数

```python
{
    "image": PIL.Image,               # 输入图像
    "prompt": str,                    # 编辑指令
    "negative_prompt": " ",           # 负面提示词（空字符串）
    "num_inference_steps": 4,         # ⭐ Lightning: 4步
    "true_cfg_scale": 1.0,            # ⭐ CFG 强度: 1.0
    "generator": torch.Generator,     # 随机数生成器
}
```

---

## ⚠️ 重要注意事项

### 1. Lightning 是 LoRA，不是完整模型

- ❌ **错误**: 直接 `load_state_dict(lightning_weights)`
- ✅ **正确**: 使用 `load_lora_weights()`

### 2. CFG Scale 使用 1.0

- Lightning 模型训练时优化了 CFG
- 官方推荐 `true_cfg_scale=1.0`（而非 4.0）

### 3. 必须配置调度器

- 必须使用自定义调度器配置
- `base_shift=math.log(3)` 和 `use_dynamic_shifting=True` 是关键

### 4. 推理步数固定为 4

- Lightning-4steps 设计为 4 步推理
- 使用更多步数不会提升质量

---

## 🔍 验证步骤

### 检查 LoRA 是否加载成功

```python
# 检查 LoRA 适配器
print("已加载的 LoRA 适配器:", pipe.get_list_adapters())

# 检查 transformer 的 LoRA 层
for name, module in pipe.transformer.named_modules():
    if "lora" in name.lower():
        print(f"LoRA 层: {name}")
```

### 检查推理速度

```python
import time

start = time.time()
output = pipe(image, prompt, num_inference_steps=4)
elapsed = time.time() - start

print(f"推理时间: {elapsed:.2f} 秒")
# 预期: ~4-8秒（4步），而原版 ~40-50秒（50步）
```

---

## 📝 完整工作流示例

```bash
# 1. 准备环境
pip install git+https://github.com/huggingface/diffusers
pip install torch transformers pillow

# 2. 准备输入图像
# 将图像命名为 input.png 放在当前目录

# 3. 运行推理（自动下载 LoRA）
python 2_load_and_inference_lightning.py

# 4. （可选）融合 LoRA 并保存
python 3_merge_lora_to_weights.py

# 5. 检查输出
# 输出图像: output_lightning_4steps.png
```

---

## 🐛 常见问题

### Q1: LoRA 加载失败？

**A**: 确保网络连接正常，或手动下载 LoRA 权重文件。

### Q2: 推理结果不理想？

**A**: 
- 确认 `true_cfg_scale=1.0`（不是 4.0）
- 确认调度器配置正确
- 确认 `num_inference_steps=4`

### Q3: 速度没有提升？

**A**: 
- 确认 LoRA 已成功加载
- 确认使用 4 步推理（而非 50 步）
- 检查是否使用了 GPU

---

## 📂 文件结构

```
F:\Diffusers\
├── 1_download_lightning_lora.py          # 下载脚本
├── 2_load_and_inference_lightning.py     # 推理脚本（主要）
├── 3_merge_lora_to_weights.py            # 融合脚本（可选）
├── Lightning_完整操作步骤.md             # 本文档
├── input.png                             # 输入图像（需准备）
└── models/
    ├── lightning_lora/                   # LoRA 权重（自动下载）
    └── qwen-image-edit-lightning-merged/ # 融合后模型（可选）
```

---

## ✅ 总结

**核心流程**（3步）：

1. **配置调度器**：`base_shift=math.log(3)`, `use_dynamic_shifting=True`
2. **加载 LoRA**：`pipe.load_lora_weights(...)`
3. **4步推理**：`num_inference_steps=4`, `true_cfg_scale=1.0`

**关键代码**：

```python
# 完整流程
from diffusers import QwenImageEditPipeline, FlowMatchEulerDiscreteScheduler
import torch, math

scheduler = FlowMatchEulerDiscreteScheduler.from_config({
    "base_shift": math.log(3), "use_dynamic_shifting": True, ...
})

pipe = QwenImageEditPipeline.from_pretrained(
    "Qwen/Qwen-Image-Edit", scheduler=scheduler, torch_dtype=torch.bfloat16
)

pipe.load_lora_weights(
    "lightx2v/Qwen-Image-Lightning",
    weight_name="Qwen-Image-Edit-2509/Qwen-Image-Edit-2509-Lightning-4steps-V1.0-bf16.safetensors"
)

pipe.to("cuda")

output = pipe(image, prompt, num_inference_steps=4, true_cfg_scale=1.0)
```

---

**按照此步骤操作即可完全按照官方 Hugging Face 指引使用 Lightning 模型！**

