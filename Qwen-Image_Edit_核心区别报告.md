# Qwen-Image-Edit 核心区别报告（按重要性排序）

## 🎯 执行摘要

Qwen-Image-Edit 的核心创新是**双路径架构**：图像同时送入 Qwen2.5-VL（语义控制）和 VAE（外观控制），实现精确的图像编辑。

---

## ⭐ 一、最重要：双路径输入架构

### 1.1 整体架构对比

#### 原版 Qwen-Image
```
输入: 文本 Prompt
  ↓
Tokenizer
  ↓
Text Encoder (仅处理文本)
  ↓
Prompt Embeddings
  ↓
去噪循环生成图像
```

#### Edit 版本（双路径）
```
输入图像 ──┬──→ Qwen2.5-VL ──→ 多模态 Embeddings ─┐
           │                                    │
           └──→ VAE Encoder ──→ Image Latents ──┼──→ 去噪循环
                                                │
文本 Prompt ───→ Processor ───→ Text Encoder ──┘
```

### 1.2 代码实现：双路径的核心

#### 路径1：Qwen2.5-VL（语义理解）

```python
# 位置: pipelines/qwenimage/pipeline_qwenimage_edit.py:226-271
def _get_qwen_prompt_embeds(
    self,
    prompt: Union[str, List[str]] = None,
    image: Optional[torch.Tensor] = None,  # ⭐ 关键：接收图像
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
):
    txt = [template.format(e) for e in prompt]
    
    # ⭐ 核心：processor 同时处理文本和图像
    model_inputs = self.processor(
        text=txt,
        images=image,  # 图像输入
        padding=True,
        return_tensors="pt",
    ).to(device)
    
    # ⭐ 核心：text_encoder 接收多模态输入
    outputs = self.text_encoder(
        input_ids=model_inputs.input_ids,
        attention_mask=model_inputs.attention_mask,
        pixel_values=model_inputs.pixel_values,      # 图像像素
        image_grid_thw=model_inputs.image_grid_thw,  # 图像布局
        output_hidden_states=True,
    )
    
    return prompt_embeds, encoder_attention_mask
```

**关键点**：
- ✅ `processor` 将图像和文本打包为多模态输入
- ✅ `text_encoder` 同时处理文本 tokens 和图像像素
- ✅ 输出的 embeddings 包含**图像语义理解**

#### 路径2：VAE Encoder（视觉外观）

```python
# 位置: pipelines/qwenimage/pipeline_qwenimage_edit.py:395-416
def _encode_vae_image(self, image: torch.Tensor, generator: torch.Generator):
    # ⭐ 关键：VAE 编码图像为 latent
    image_latents = retrieve_latents(
        self.vae.encode(image), 
        generator=generator, 
        sample_mode="argmax"  # 确定性编码
    )
    
    # 归一化处理
    latents_mean = torch.tensor(self.vae.config.latents_mean)
    latents_std = torch.tensor(self.vae.config.latents_std)
    image_latents = (image_latents - latents_mean) / latents_std
    
    return image_latents
```

**关键点**：
- ✅ VAE 将图像编码为潜在空间表示
- ✅ `argmax` 模式确保编码稳定性
- ✅ 提供**视觉外观参考**

### 1.3 双路径流程图

```
┌─────────────────────────────────────────────────────────┐
│                    编辑任务开始                          │
└─────────────────────────────────────────────────────────┘
                        ↓
        ┌───────────────┴───────────────┐
        │                                │
┌───────▼────────┐            ┌──────────▼────────┐
│  输入图像       │            │  文本编辑指令     │
└───────┬────────┘            └──────────┬───────┘
        │                                │
        ├─────────────┐                  │
        │             │                  │
        ▼             ▼                  ▼
┌─────────────┐  ┌──────────┐    ┌──────────────┐
│ VAE Encoder │  │Processor │    │  Text Token  │
│             │  │          │    │              │
│ 编码视觉特征│  │ 多模态处理│    │  文本token   │
└──────┬──────┘  └────┬─────┘    └──────┬───────┘
       │              │                  │
       │              ▼                  │
       │       ┌──────────────┐         │
       │       │  Text Encoder │         │
       │       │ (Qwen2.5-VL)  │         │
       │       │               │         │
       │       │ 处理文本+图像 │         │
       │       └───────┬───────┘         │
       │               │                │
       │               ▼                │
       │       ┌──────────────┐        │
       │       │ 多模态Embeddings│       │
       │       │ (语义理解)    │        │
       │       └───────┬───────┘        │
       │               │                │
       └───────┬───────┴────────────────┘
               │
               ▼
      ┌──────────────────┐
      │  准备Latents     │
      │                  │
      │ Image Latents   │ ← 来自VAE
      │ (视觉参考)       │
      └────────┬─────────┘
               │
               ▼
      ┌──────────────────┐
      │   去噪循环       │
      │                  │
      │ [去噪latents +   │
      │  图像latents]    │
      └────────┬─────────┘
               │
               ▼
      ┌──────────────────┐
      │ Transformer      │
      │                  │
      │ 接收:            │
      │ • 拼接的latents  │
      │ • 多模态embeddings│
      └────────┬─────────┘
               │
               ▼
        编辑后的图像
```

---

## 🔥 二、去噪循环中的 Latent 拼接策略

### 2.1 核心差异对比

#### 原版：仅使用去噪 latents
```python
# 位置: pipelines/qwenimage/pipeline_qwenimage.py
for i, t in enumerate(timesteps):
    latent_model_input = latents  # 只有去噪的latents
    
    noise_pred = self.transformer(
        hidden_states=latent_model_input,
        encoder_hidden_states=prompt_embeds,
        ...
    )
```

#### Edit 版本：拼接原始图像 latents
```python
# 位置: pipelines/qwenimage/pipeline_qwenimage_edit.py:810-828
for i, t in enumerate(timesteps):
    latent_model_input = latents  # 当前去噪状态
    
    # ⭐ 关键：拼接原始图像 latents
    if image_latents is not None:
        latent_model_input = torch.cat([latents, image_latents], dim=1)
        #           ↑ 当前去噪       ↑ 原始图像特征
        #         [B, seq_len1, C] + [B, seq_len2, C]
        #         = [B, seq_len1+seq_len2, C]
    
    # Transformer 同时看到去噪状态和原始图像
    noise_pred = self.transformer(
        hidden_states=latent_model_input,  # 拼接后的输入
        encoder_hidden_states=prompt_embeds,  # 包含图像理解
        ...
    )
    
    # ⭐ 只取前部分（对应去噪部分）
    noise_pred = noise_pred[:, : latents.size(1)]
```

### 2.2 拼接策略流程图

```
去噪步骤 t
    ↓
┌─────────────────┐
│ 当前 Latents    │ ← 正在去噪的部分
│ [B, L1, C]      │
└───────┬─────────┘
        │
        │ torch.cat([:, dim=1])
        │
        ├─────────────────┐
        │                 │
┌───────▼─────────┐ ┌─────▼──────────┐
│ 原始图像 Latents│ │ 当前去噪 Latents│
│ [B, L2, C]      │ │ [B, L1, C]      │
└───────┬─────────┘ └───────┬─────────┘
        │                   │
        └───────┬───────────┘
                │
                ▼
    ┌───────────────────────┐
    │ Latent Model Input    │
    │ [B, L1+L2, C]         │
    │                        │
    │ [去噪部分 | 原图部分]  │
    └───────────┬────────────┘
                │
                ▼
    ┌───────────────────────┐
    │   Transformer         │
    │                        │
    │ 输入:                  │
    │ • 拼接的 latents       │
    │ • 多模态 embeddings    │
    │                        │
    │ 输出:                  │
    │ • 噪声预测 [B, L1+L2]  │
    └───────────┬────────────┘
                │
                ▼
    ┌───────────────────────┐
    │ 截取前 L1 部分        │
    │ noise_pred[:, :L1]    │
    │ (只用于去噪更新)       │
    └───────────────────────┘
```

**优势**：
- ✅ Transformer 同时访问去噪状态和原始图像
- ✅ 实现精确的区域控制（哪些改、哪些不变）
- ✅ 保持视觉一致性

### 2.3 原始图像 Latents 的冻结机制

**问题**：既然原始图像 latents 不更新，计算过程中是否做了冻结处理？

**答案分析**：

#### 1. 全局 `@torch.no_grad()` 装饰器

```python
# 位置: pipelines/qwenimage/pipeline_qwenimage_edit.py:546
@torch.no_grad()  # ⭐ 整个推理过程在无梯度模式下
@replace_example_docstring(EXAMPLE_DOC_STRING)
def __call__(self, ...):
    # 所有计算都在 no_grad 模式下
    ...
```

**影响**：整个 `__call__` 方法在无梯度模式下运行，理论上**所有张量都不计算梯度**。

#### 2. 原始图像 Latents 的生成

```python
# 位置: pipelines/qwenimage/pipeline_qwenimage_edit.py:395-416
def _encode_vae_image(self, image: torch.Tensor, generator: torch.Generator):
    # VAE 编码（在 no_grad 上下文中）
    image_latents = retrieve_latents(
        self.vae.encode(image), 
        generator=generator, 
        sample_mode="argmax"  # 确定性编码
    )
    # 归一化后返回
    return image_latents  # ⭐ 这些 latents 在 no_grad 模式下生成
```

#### 3. 去噪循环中的使用

```python
# 位置: pipelines/qwenimage/pipeline_qwenimage_edit.py:810-828
for i, t in enumerate(timesteps):
    latent_model_input = latents
    
    # ⭐ 直接拼接，没有显式的 .detach()
    if image_latents is not None:
        latent_model_input = torch.cat([latents, image_latents], dim=1)
    
    # Transformer 调用（在 no_grad 模式下）
    noise_pred = self.transformer(
        hidden_states=latent_model_input,
        ...
    )
    
    # ⭐ 只取前部分（去噪部分）用于更新
    noise_pred = noise_pred[:, : latents.size(1)]
    
    # 更新 latents（只更新去噪部分）
    latents = self.scheduler.step(noise_pred, t, latents, ...)[0]
    # ⭐ image_latents 不会在这里更新
```

### 2.4 冻结机制总结

| 机制 | 实现方式 | 说明 |
|------|---------|------|
| **全局无梯度** | `@torch.no_grad()` | 整个推理过程不计算梯度 |
| **隐式冻结** | `image_latents` 生成后固定 | 只参与前向传播，不参与更新 |
| **显式冻结** | ❌ **未实现** | 代码中没有 `.detach()` 或 `.requires_grad_(False)` |

**为什么不需要显式冻结？**

1. ✅ **推理模式**：`@torch.no_grad()` 使整个流程不计算梯度
2. ✅ **更新机制**：只有 `latents` 通过 `scheduler.step()` 更新，`image_latents` 不参与更新
3. ✅ **计算效率**：在 `no_grad` 模式下，PyTorch 自动优化，不会为不变张量分配梯度缓存

**潜在优化（如果需要在训练模式下）**：

```python
# 如果将来需要支持训练模式，可以考虑显式冻结
if image_latents is not None:
    # 选项1: detach
    image_latents = image_latents.detach()
    
    # 选项2: requires_grad=False
    image_latents.requires_grad_(False)
    
    latent_model_input = torch.cat([latents, image_latents], dim=1)
```

**流程图**：

```
┌─────────────────────────────────┐
│ @torch.no_grad()               │ ← 全局无梯度模式
│ (整个推理过程)                  │
└────────────┬────────────────────┘
             │
             ▼
┌─────────────────────────────────┐
│ prepare_latents()               │
│                                 │
│ • image_latents ← VAE编码       │ ← 生成后固定
│ • latents ← 随机噪声            │ ← 会更新
└────────────┬────────────────────┘
             │
             ▼
┌─────────────────────────────────┐
│ 去噪循环 (50步)                 │
│                                 │
│ for each timestep:              │
│   latent_model_input =           │
│      cat([latents, image_latents])│
│                                 │
│   noise_pred = transformer(...) │ ← 前向传播
│                                 │
│   noise_pred = noise_pred[:, :L1]│ ← 只取去噪部分
│                                 │
│   latents = scheduler.step(...)  │ ← 只更新 latents
│   # image_latents 保持不变      │ ← ✅ 隐式冻结
└─────────────────────────────────┘
```

**结论**：
- ✅ 当前实现通过 `@torch.no_grad()` 实现隐式冻结
- ✅ `image_latents` 只参与前向传播，不参与更新
- ✅ 在推理模式下，这种设计是高效且安全的
- ⚠️ 如果需要训练模式，建议显式添加 `.detach()` 或 `.requires_grad_(False)`

---

## 📝 三、多模态 Prompt 编码差异

### 3.1 编码流程对比

#### 原版：纯文本编码
```python
# 位置: pipelines/qwenimage/pipeline_qwenimage.py:188-224
def _get_qwen_prompt_embeds(self, prompt, device, dtype):
    # 1. 文本模板
    txt = [template.format(e) for e in prompt]
    
    # 2. 仅文本 tokenization
    txt_tokens = self.tokenizer(txt, ...)
    
    # 3. 仅文本编码
    encoder_hidden_states = self.text_encoder(
        input_ids=txt_tokens.input_ids,
        attention_mask=txt_tokens.attention_mask,
        # ❌ 无图像输入
    )
```

#### Edit 版本：多模态编码
```python
# 位置: pipelines/qwenimage/pipeline_qwenimage_edit.py:226-271
def _get_qwen_prompt_embeds(self, prompt, image, device, dtype):
    txt = [template.format(e) for e in prompt]
    
    # ⭐ 关键：processor 处理多模态
    model_inputs = self.processor(
        text=txt,
        images=image,  # ✅ 图像输入
        ...
    )
    
    # ⭐ 关键：text_encoder 接收图像
    outputs = self.text_encoder(
        input_ids=model_inputs.input_ids,
        attention_mask=model_inputs.attention_mask,
        pixel_values=model_inputs.pixel_values,      # ✅ 图像
        image_grid_thw=model_inputs.image_grid_thw,  # ✅ 布局
    )
```

### 3.2 Prompt 模板差异

#### 原版模板（描述图像）
```python
prompt_template_encode = (
    "<|im_start|>system\n"
    "Describe the image by detailing the color, shape, size, "
    "texture, quantity, text, spatial relationships of the objects "
    "and background:<|im_end|>\n"
    "<|im_start|>user\n{}<|im_end|>\n"
    "<|im_start|>assistant\n"
)
# 功能：描述图像内容
```

#### Edit 版本模板（理解编辑指令）
```python
prompt_template_encode = (
    "<|im_start|>system\n"
    "Describe the key features of the input image (color, shape, "
    "size, texture, objects, background), then explain how the "
    "user's text instruction should alter or modify the image. "
    "Generate a new image that meets the user's requirements while "
    "maintaining consistency with the original input where appropriate."
    "<|im_end|>\n"
    "<|im_start|>user\n"
    "<|vision_start|><|image_pad|><|vision_end|>{}<|im_end|>\n"
    "<|im_start|>assistant\n"
)
# 功能：理解编辑指令并生成新图像
# ⭐ 包含图像占位符: <|vision_start|><|image_pad|><|vision_end|>
```

**关键差异**：
- ✅ Edit 模板包含图像占位符
- ✅ 引导模型理解"如何修改"而非"描述什么"
- ✅ 强调"保持一致性"

### 3.3 编码流程对比图

```
原版 Qwen-Image:
┌─────────┐
│ 文本    │
└────┬────┘
     │
     ▼
┌─────────┐
│Tokenizer│
└────┬────┘
     │
     ▼
┌─────────────┐
│Text Encoder │
│ (仅文本)    │
└────┬────────┘
     │
     ▼
┌─────────────┐
│Text Embeddings│
└─────────────┘

Edit 版本:
┌─────────┐  ┌─────────┐
│ 文本    │  │ 图像    │
└────┬────┘  └────┬────┘
     │            │
     └─────┬──────┘
           │
           ▼
    ┌──────────────┐
    │  Processor   │
    │ (多模态打包) │
    └──────┬───────┘
           │
           ▼
    ┌──────────────┐
    │Text Encoder  │
    │(Qwen2.5-VL)  │
    │              │
    │ 同时处理:     │
    │ • 文本tokens │
    │ • 图像像素   │
    └──────┬───────┘
           │
           ▼
    ┌──────────────┐
    │多模态Embeddings│
    │ (文本+图像语义) │
    └──────────────┘
```

---

## 🎨 四、图像预处理与 VAE 编码

### 4.1 Edit 版本独有的图像编码方法

```python
# 位置: pipelines/qwenimage/pipeline_qwenimage_edit.py:395-416
def _encode_vae_image(self, image: torch.Tensor, generator: torch.Generator):
    # ⭐ VAE 编码图像
    image_latents = retrieve_latents(
        self.vae.encode(image), 
        generator=generator, 
        sample_mode="argmax"  # 确定性编码
    )
    
    # 归一化到 VAE 潜在空间
    latents_mean = torch.tensor(self.vae.config.latents_mean)
    latents_std = torch.tensor(self.vae.config.latents_std)
    image_latents = (image_latents - latents_mean) / latents_std
    
    return image_latents
```

**调用位置**：
```python
# prepare_latents 中
if image is not None:
    image_latents = self._encode_vae_image(image=image, generator=generator)
    image_latents = self._pack_latents(image_latents, ...)
```

### 4.2 VAE 编码流程图

```
输入图像 [B, C, H, W]
    ↓
┌─────────────┐
│VAE Encoder │
│  (压缩编码) │
└──────┬──────┘
       │
       ▼
┌─────────────┐
│ Image Latents│
│ [B, Z, H', W']│
│ (压缩特征)   │
└──────┬──────┘
       │
       ▼
┌─────────────┐
│  归一化     │
│ (减去均值)  │
│ (除以标准差)│
└──────┬──────┘
       │
       ▼
┌─────────────┐
│ Pack Latents│
│ (打包为序列)│
│ [B, L, C]   │
└──────┬──────┘
       │
       ▼
   存储为 image_latents
   用于去噪循环拼接
```

---

## 🔧 五、初始化差异

### 5.1 新增 Processor 组件

#### 原版初始化
```python
# pipelines/qwenimage/pipeline_qwenimage.py:154-170
def __init__(
    self,
    scheduler,
    vae,
    text_encoder,
    tokenizer,          # ⚠️ 无 processor
    transformer,
):
    self.register_modules(
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        transformer=transformer,
        scheduler=scheduler,
    )
```

#### Edit 版本初始化
```python
# pipelines/qwenimage/pipeline_qwenimage_edit.py:187-205
def __init__(
    self,
    scheduler,
    vae,
    text_encoder,
    tokenizer,
    processor,          # ⭐ 新增
    transformer,
):
    self.register_modules(
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        processor=processor,  # ⭐ 注册 processor
        transformer=transformer,
        scheduler=scheduler,
    )
```

**Processor 作用**：
- ✅ 将文本和图像打包为多模态输入
- ✅ 生成 `pixel_values` 和 `image_grid_thw`
- ✅ 是双路径架构的关键组件

---

## 📊 六、数据流对比总结

### 6.1 原版 Qwen-Image 完整流程

```
文本 Prompt
    ↓
┌──────────────┐
│ 1. Tokenizer │ → 文本 tokens
└──────┬───────┘
       │
       ▼
┌──────────────┐
│ 2. Text Encoder│ → 文本 embeddings
└──────┬───────┘
       │
       ▼
┌──────────────┐
│ 3. 生成随机  │ → 初始 latents [B, L, C]
│    噪声      │
└──────┬───────┘
       │
       ▼
┌──────────────┐
│ 4. 去噪循环  │
│    50步      │
│              │
│ Transformer: │
│ • latents    │
│ • embeddings │
└──────┬───────┘
       │
       ▼
┌──────────────┐
│ 5. VAE Decode│ → 生成图像
└──────────────┘
```

### 6.2 Edit 版本完整流程

```
输入图像 + 文本指令
    ↓
    ├─────────────────────┐
    │                     │
    ▼                     ▼
┌──────────┐         ┌──────────┐
│Processor │         │VAE Encode│
│ 多模态打包│         │视觉编码   │
└────┬─────┘         └────┬─────┘
     │                    │
     ▼                    │
┌──────────┐              │
│Text Encode│             │
│(Qwen2.5-VL)│            │
│ 语义理解  │             │
└────┬─────┘              │
     │                    │
     ▼                    ▼
┌──────────┐         ┌──────────┐
│多模态Embed│         │Image Latents│
└────┬─────┘         └────┬─────┘
     │                    │
     └────────┬───────────┘
              │
              ▼
     ┌──────────────┐
     │准备 Latents  │
     │              │
     │• 随机噪声    │
     │• 图像latents │
     └──────┬───────┘
            │
            ▼
     ┌──────────────┐
     │ 去噪循环50步 │
     │              │
     │ 每步:        │
     │ • 拼接latents│
     │ • Transformer│
     └──────┬───────┘
            │
            ▼
     ┌──────────────┐
     │ VAE Decode   │ → 编辑后图像
     └──────────────┘
```

---

## 🎯 七、关键代码位置索引

| 功能 | 文件位置 | 行号 |
|------|---------|------|
| **双路径核心：多模态编码** | `pipeline_qwenimage_edit.py` | 226-271 |
| **VAE 图像编码** | `pipeline_qwenimage_edit.py` | 395-416 |
| **去噪循环：Latent拼接** | `pipeline_qwenimage_edit.py` | 810-828 |
| **Prompt模板定义** | `pipeline_qwenimage_edit.py` | 213-214 |
| **Processor注册** | `pipeline_qwenimage_edit.py` | 187-205 |

---

## ✅ 总结

**Qwen-Image-Edit 的三个核心创新**：

1. ⭐ **双路径输入**：图像同时进入 Qwen2.5-VL（语义）和 VAE（外观）
2. ⭐ **Latent拼接**：去噪时拼接原始图像和当前状态
3. ⭐ **多模态编码**：文本+图像统一编码，理解编辑意图

这三个创新协同工作，实现了精确的图像编辑能力。

---

## 🎯 八、CFG (Classifier-Free Guidance) 实现详解

### 8.1 CFG 原理

CFG 通过**对比条件预测和无条件预测**来增强生成质量：

**核心公式**：
```
noise_pred_final = neg_noise_pred + scale * (noise_pred - neg_noise_pred)
```

其中：
- `noise_pred`：条件预测（使用 positive prompt）
- `neg_noise_pred`：无条件预测（使用 negative prompt）
- `scale`：CFG 强度（通常为 4.0）

**意义**：放大条件与无条件预测的差异，使生成更符合 prompt。

### 8.2 代码实现步骤

#### 步骤1：检查 CFG 是否启用

```python
# 位置: pipelines/qwenimage/pipeline_qwenimage_edit.py:705-718
has_neg_prompt = negative_prompt is not None or (
    negative_prompt_embeds is not None and negative_prompt_embeds_mask is not None
)

# 检查条件
if true_cfg_scale > 1 and not has_neg_prompt:
    logger.warning("CFG scale > 1 but no negative_prompt provided")
elif true_cfg_scale <= 1 and has_neg_prompt:
    logger.warning("negative_prompt provided but CFG scale <= 1")

# ⭐ 决定是否启用 CFG
do_true_cfg = true_cfg_scale > 1 and has_neg_prompt
```

**关键条件**：
- ✅ `true_cfg_scale > 1`（默认 4.0）
- ✅ 提供了 `negative_prompt`（即使为空字符串 " " 也可以）

#### 步骤2：编码条件和非条件 Prompt

```python
# 位置: pipelines/qwenimage/pipeline_qwenimage_edit.py:719-737
# 编码条件 prompt（positive）
prompt_embeds, prompt_embeds_mask = self.encode_prompt(
    image=prompt_image,  # ⭐ Edit 版本：同时编码图像
    prompt=prompt,
    ...
)

# 如果启用 CFG，编码非条件 prompt（negative）
if do_true_cfg:
    negative_prompt_embeds, negative_prompt_embeds_mask = self.encode_prompt(
        image=prompt_image,  # ⭐ 注意：使用相同的图像！
        prompt=negative_prompt,
        ...
    )
```

**关键点**：
- ✅ 条件和非条件 prompt 使用**相同的输入图像**
- ✅ 区别在于文本指令（positive vs negative）

#### 步骤3：去噪循环中的双重预测

```python
# 位置: pipelines/qwenimage/pipeline_qwenimage_edit.py:815-848
for i, t in enumerate(timesteps):
    # 准备输入（包含原始图像 latents）
    latent_model_input = torch.cat([latents, image_latents], dim=1)
    
    # ⭐ 第一次前向传播：条件预测
    with self.transformer.cache_context("cond"):
        noise_pred = self.transformer(
            hidden_states=latent_model_input,
            timestep=timestep / 1000,
            encoder_hidden_states=prompt_embeds,  # 条件 embeddings
            encoder_hidden_states_mask=prompt_embeds_mask,
            ...
        )[0]
        noise_pred = noise_pred[:, : latents.size(1)]
    
    # ⭐ 第二次前向传播：非条件预测（如果启用 CFG）
    if do_true_cfg:
        with self.transformer.cache_context("uncond"):
            neg_noise_pred = self.transformer(
                hidden_states=latent_model_input,  # ⭐ 相同的 latents
                timestep=timestep / 1000,           # ⭐ 相同的时间步
                encoder_hidden_states=negative_prompt_embeds,  # 非条件 embeddings
                encoder_hidden_states_mask=negative_prompt_embeds_mask,
                ...
            )[0]
        neg_noise_pred = neg_noise_pred[:, : latents.size(1)]
```

**关键点**：
- ✅ 使用相同的 `latent_model_input`（相同的去噪状态和原始图像）
- ✅ 使用相同的 `timestep`
- ✅ 区别在于 `encoder_hidden_states`（条件 vs 非条件 embeddings）

### 8.3 Cache Context 机制

**问题**：为什么使用 `cache_context("cond")` 和 `cache_context("uncond")`？

**答案**：这是 Transformer 的**缓存优化机制**，用于：
- ✅ 区分条件和非条件的计算缓存
- ✅ 避免 KV cache 混淆
- ✅ 提高计算效率（可以复用部分计算结果）

```python
with self.transformer.cache_context("cond"):
    # 条件预测（缓存标记为 "cond"）
    noise_pred = self.transformer(...)

with self.transformer.cache_context("uncond"):
    # 非条件预测（缓存标记为 "uncond"）
    neg_noise_pred = self.transformer(...)
```

### 8.4 CFG 合并公式

```python
# 位置: pipelines/qwenimage/pipeline_qwenimage_edit.py:844
# ⭐ CFG 核心公式
comb_pred = neg_noise_pred + true_cfg_scale * (noise_pred - neg_noise_pred)

# 展开：
# comb_pred = neg_noise_pred + scale * noise_pred - scale * neg_noise_pred
# comb_pred = (1 - scale) * neg_noise_pred + scale * noise_pred
```

**公式解析**：
- 当 `scale = 1`：`comb_pred = noise_pred`（无条件）
- 当 `scale > 1`：放大 `(noise_pred - neg_noise_pred)` 的差异
- 当 `scale = 4.0`（默认）：强条件引导

**数学意义**：
```
噪声预测 = 无条件预测 + 4.0 * (条件预测 - 无条件预测)
        = -3 * 无条件预测 + 4 * 条件预测
```

这表示：**朝着条件预测方向移动，偏离无条件预测**。

### 8.5 归一化处理（Normalization）

```python
# 位置: pipelines/qwenimage/pipeline_qwenimage_edit.py:846-848
# ⭐ 归一化处理：保持预测的尺度一致性
cond_norm = torch.norm(noise_pred, dim=-1, keepdim=True)
noise_norm = torch.norm(comb_pred, dim=-1, keepdim=True)
noise_pred = comb_pred * (cond_norm / noise_norm)
```

**目的**：
- ✅ 保持 `comb_pred` 与 `noise_pred` 的**范数一致**
- ✅ 防止 CFG 导致预测尺度过大
- ✅ 提高数值稳定性

**计算方式**：
- 计算条件预测的 L2 范数：`||noise_pred||`
- 计算合并预测的 L2 范数：`||comb_pred||`
- 缩放合并预测：`comb_pred * (||noise_pred|| / ||comb_pred||)`

### 8.6 完整 CFG 流程图

```
去噪步骤 t
    ↓
┌─────────────────────────────────────┐
│ 准备 Latent Model Input            │
│ • latents (当前去噪状态)            │
│ • image_latents (原始图像)          │
│ → latent_model_input               │
└──────────────┬──────────────────────┘
               │
               ├──────────────────────────┐
               │                          │
       ┌───────▼────────┐        ┌───────▼────────┐
       │ 条件预测路径    │        │ 非条件预测路径  │
       │                │        │                │
       │ cache_context  │        │ cache_context  │
       │ ("cond")       │        │ ("uncond")     │
       │                │        │                │
       │ Transformer:    │        │ Transformer:   │
       │ • prompt_embeds │        │ • neg_embeds   │
       │ • latent_input  │        │ • latent_input │
       │ • timestep      │        │ • timestep     │
       │                │        │                │
       └───────┬────────┘        └───────┬────────┘
               │                          │
               ▼                          ▼
       ┌──────────────┐           ┌──────────────┐
       │ noise_pred   │           │ neg_noise_pred│
       │ (条件预测)   │           │ (非条件预测) │
       └───────┬──────┘           └───────┬──────┘
               │                          │
               └──────────┬───────────────┘
                          │
                          ▼
              ┌───────────────────────┐
              │ CFG 合并公式          │
              │                       │
              │ comb_pred =           │
              │   neg_pred +          │
              │   scale *             │
              │   (pred - neg_pred)   │
              └───────────┬───────────┘
                          │
                          ▼
              ┌───────────────────────┐
              │ 归一化处理            │
              │                       │
              │ norm_cond = ||pred||  │
              │ norm_comb = ||comb||  │
              │ final = comb *        │
              │        (norm_cond /   │
              │         norm_comb)   │
              └───────────┬───────────┘
                          │
                          ▼
              ┌───────────────────────┐
              │ Scheduler Step        │
              │                       │
              │ latents =             │
              │   scheduler.step(     │
              │     final_pred, ...)  │
              └───────────────────────┘
```

### 8.7 CFG 关键代码位置总结

| 功能 | 代码位置 | 关键代码 |
|------|---------|---------|
| **CFG 启用检查** | `pipeline_qwenimage_edit.py:705-718` | `do_true_cfg = true_cfg_scale > 1 and has_neg_prompt` |
| **非条件编码** | `pipeline_qwenimage_edit.py:728-737` | `encode_prompt(image, negative_prompt, ...)` |
| **条件预测** | `pipeline_qwenimage_edit.py:816-828` | `transformer(..., prompt_embeds)` |
| **非条件预测** | `pipeline_qwenimage_edit.py:831-843` | `transformer(..., negative_prompt_embeds)` |
| **CFG 合并** | `pipeline_qwenimage_edit.py:844` | `comb_pred = neg_pred + scale * (pred - neg_pred)` |
| **归一化** | `pipeline_qwenimage_edit.py:846-848` | `comb_pred * (cond_norm / comb_norm)` |

### 8.8 CFG 在 Edit 版本的特殊性

**与原版对比**：

| 特性 | 原版 Qwen-Image | Edit 版本 |
|------|----------------|-----------|
| **图像输入** | ❌ 无 | ✅ **有**（同时用于条件和非条件） |
| **Prompt 编码** | 仅文本 | ✅ **文本+图像**（多模态） |
| **CFG 公式** | ✅ 相同 | ✅ 相同 |
| **归一化** | ✅ 相同 | ✅ 相同 |
| **Cache 机制** | ✅ 相同 | ✅ 相同 |

**Edit 版本的创新**：
- ✅ 条件和非条件的 prompt embeddings 都包含**图像理解**
- ✅ 这意味着 CFG 是在"理解图像"的基础上进行引导
- ✅ 更精确地控制编辑方向和强度

**示例**：
```python
# 条件 prompt（positive）
prompt = "Change the rabbit's color to purple"
→ 编码为：图像语义 + "紫色兔子"的指令

# 非条件 prompt（negative）
negative_prompt = " "
→ 编码为：图像语义 + 空指令（理解为"保持原样"）

# CFG 作用：放大"变为紫色"与"保持原样"的差异
```

### 8.9 CFG Scale 参数影响

| Scale 值 | 效果 | 公式解析 |
|----------|------|---------|
| `1.0` | 无 CFG | `comb_pred = noise_pred` |
| `2.0` | 弱引导 | `comb_pred = -1 * neg_pred + 2 * pred` |
| `4.0` (默认) | 强引导 | `comb_pred = -3 * neg_pred + 4 * pred` |
| `7.5+` | 过度引导 | 可能导致不自然的结果 |

**建议值**：`true_cfg_scale = 4.0`（默认），根据任务调整。

---

**总结**：Qwen-Image-Edit 的 CFG 实现采用标准 CFG 公式，但创新在于**条件和非条件都包含多模态（图像+文本）理解**，使引导更精确、更符合编辑意图。

