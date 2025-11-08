# Qwen-Image-Edit 与原版 Qwen-Image 详细对比报告

## 📋 执行摘要

Qwen-Image-Edit 是在 Qwen-Image 基础上专门为图像编辑任务设计的变体。核心创新在于**双路径输入架构**：将输入图像同时送入 Qwen2.5-VL（视觉-语言模型）和 VAE 编码器，实现语义编辑与视觉外观控制的有机结合。

---

## 🔍 一、架构层面差异

### 1.1 模型组件对比

| 组件 | Qwen-Image (原版) | Qwen-Image-Edit |
|------|------------------|-----------------|
| **Text Encoder** | Qwen2_5_VLForConditionalGeneration | ✅ 相同 |
| **Tokenizer** | Qwen2Tokenizer | ✅ 相同 |
| **Processor** | ❌ 无 | ✅ **Qwen2VLProcessor** (新增) |
| **Transformer** | QwenImageTransformer2DModel | ✅ 相同 |
| **VAE** | AutoencoderKLQwenImage | ✅ 相同 |
| **Scheduler** | FlowMatchEulerDiscreteScheduler | ✅ 相同 |

**关键发现**：Edit 版本新增了 `processor` 组件，用于处理多模态输入（文本+图像）。

---

### 1.2 初始化参数差异

#### 原版 Qwen-Image
```python
def __init__(
    self,
    scheduler: FlowMatchEulerDiscreteScheduler,
    vae: AutoencoderKLQwenImage,
    text_encoder: Qwen2_5_VLForConditionalGeneration,
    tokenizer: Qwen2Tokenizer,
    transformer: QwenImageTransformer2DModel,
):
```

#### Edit 版本
```python
def __init__(
    self,
    scheduler: FlowMatchEulerDiscreteScheduler,
    vae: AutoencoderKLQwenImage,
    text_encoder: Qwen2_5_VLForConditionalGeneration,
    tokenizer: Qwen2Tokenizer,
    processor: Qwen2VLProcessor,  # ⭐ 新增
    transformer: QwenImageTransformer2DModel,
):
```

**代码位置**：
- 原版：```154:160:pipelines/qwenimage/pipeline_qwenimage.py```
- Edit版：```187:195:pipelines/qwenimage/pipeline_qwenimage_edit.py```

---

## 📝 二、Prompt 模板与文本编码差异

### 2.1 Prompt 模板对比

#### 原版 Qwen-Image 模板
```python
prompt_template_encode = (
    "<|im_start|>system\n"
    "Describe the image by detailing the color, shape, size, texture, quantity, "
    "text, spatial relationships of the objects and background:<|im_end|>\n"
    "<|im_start|>user\n{}<|im_end|>\n"
    "<|im_start|>assistant\n"
)
prompt_template_encode_start_idx = 34  # 丢弃前34个token
```

**功能**：引导模型**描述**图像内容，用于文本到图像生成任务。

#### Edit 版本模板
```python
prompt_template_encode = (
    "<|im_start|>system\n"
    "Describe the key features of the input image (color, shape, size, texture, objects, background), "
    "then explain how the user's text instruction should alter or modify the image. "
    "Generate a new image that meets the user's requirements while maintaining consistency "
    "with the original input where appropriate.<|im_end|>\n"
    "<|im_start|>user\n"
    "<|vision_start|><|image_pad|><|vision_end|>{}<|im_end|>\n"
    "<|im_start|>assistant\n"
)
prompt_template_encode_start_idx = 64  # 丢弃前64个token（系统提示更长）
```

**功能**：
1. 首先理解输入图像的**关键特征**
2. 然后解释用户的**编辑指令**如何修改图像
3. 生成符合要求的新图像，同时保持与原图的一致性

**代码位置**：
- 原版：```176:177:pipelines/qwenimage/pipeline_qwenimage.py```
- Edit版：```213:214:pipelines/qwenimage/pipeline_qwenimage_edit.py```

---

### 2.2 文本编码方法差异

#### 原版：纯文本编码（`_get_qwen_prompt_embeds`）

```python
def _get_qwen_prompt_embeds(
    self,
    prompt: Union[str, List[str]] = None,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
):
    # 1. 只处理文本
    txt = [template.format(e) for e in prompt]
    
    # 2. 使用 tokenizer 处理文本
    txt_tokens = self.tokenizer(
        txt, max_length=self.tokenizer_max_length + drop_idx,
        padding=True, truncation=True, return_tensors="pt"
    ).to(device)
    
    # 3. 仅使用文本输入调用 text_encoder
    encoder_hidden_states = self.text_encoder(
        input_ids=txt_tokens.input_ids,
        attention_mask=txt_tokens.attention_mask,
        output_hidden_states=True,
    )
```

**代码位置**：```188:224:pipelines/qwenimage/pipeline_qwenimage.py```

#### Edit 版本：多模态编码（双路径核心实现）

```python
def _get_qwen_prompt_embeds(
    self,
    prompt: Union[str, List[str]] = None,
    image: Optional[torch.Tensor] = None,  # ⭐ 新增图像输入
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
):
    # 1. 准备文本模板（包含图像占位符）
    txt = [template.format(e) for e in prompt]
    
    # 2. ⭐ 使用 processor 处理多模态输入（文本+图像）
    model_inputs = self.processor(
        text=txt,
        images=image,  # 图像同时输入
        padding=True,
        return_tensors="pt",
    ).to(device)
    
    # 3. ⭐ 调用 text_encoder 同时处理文本和图像
    outputs = self.text_encoder(
        input_ids=model_inputs.input_ids,
        attention_mask=model_inputs.attention_mask,
        pixel_values=model_inputs.pixel_values,      # ⭐ 图像像素值
        image_grid_thw=model_inputs.image_grid_thw,  # ⭐ 图像网格信息
        output_hidden_states=True,
    )
```

**代码位置**：```226:271:pipelines/qwenimage/pipeline_qwenimage_edit.py```

**关键差异**：
1. ✅ Edit 版本接受 `image` 参数
2. ✅ 使用 `processor` 同时处理文本和图像
3. ✅ `text_encoder` 接收 `pixel_values` 和 `image_grid_thw`（图像特征）

---

## 🖼️ 三、双路径图像处理架构

### 3.1 图像输入流程对比

#### 原版 Qwen-Image：无图像输入
```
用户输入
  └─> 纯文本 Prompt
      └─> Tokenizer
          └─> Text Encoder (仅文本)
              └─> Prompt Embeddings
```

#### Edit 版本：双路径处理

**路径1：语义理解路径（Qwen2.5-VL）**
```
输入图像
  └─> Qwen2VLProcessor (多模态处理)
      └─> Text Encoder (Qwen2.5-VL)
          ├─ pixel_values (图像特征)
          ├─ image_grid_thw (图像布局)
          └─ input_ids (文本token)
          └─> 多模态 Embeddings (语义理解)
```

**路径2：视觉外观路径（VAE）**
```
输入图像
  └─> VaeImageProcessor (图像预处理)
      └─> VAE Encoder
          └─> Image Latents (视觉特征)
```

**代码实现**：

1. **路径1 - Qwen2.5-VL（语义控制）**
   ```python
   # encode_prompt 中调用
   prompt_embeds, prompt_embeds_mask = self.encode_prompt(
       image=prompt_image,  # ⭐ 图像送入VL模型
       prompt=prompt,
       ...
   )
   ```
   位置：```718:727:pipelines/qwenimage/pipeline_qwenimage_edit.py```

2. **路径2 - VAE Encoder（外观控制）**
   ```python
   # prepare_latents 中调用
   image_latents = self._encode_vae_image(image=image, generator=generator)
   ```
   位置：```395:416:pipelines/qwenimage/pipeline_qwenimage_edit.py```

---

### 3.2 VAE 编码方法（Edit 版本特有）

Edit 版本新增 `_encode_vae_image` 方法：

```python
def _encode_vae_image(self, image: torch.Tensor, generator: torch.Generator):
    # 1. VAE 编码图像
    image_latents = retrieve_latents(
        self.vae.encode(image), 
        generator=generator, 
        sample_mode="argmax"  # 使用 argmax 而非采样
    )
    
    # 2. 归一化处理（与 VAE 配置一致）
    latents_mean = torch.tensor(self.vae.config.latents_mean)
    latents_std = torch.tensor(self.vae.config.latents_std)
    image_latents = (image_latents - latents_mean) / latents_std
    
    return image_latents
```

**特点**：
- 使用 `sample_mode="argmax"` 而非随机采样，确保编码稳定性
- 进行标准化处理，匹配 VAE 的潜在空间分布

**代码位置**：```395:416:pipelines/qwenimage/pipeline_qwenimage_edit.py```

---

## 🔄 四、去噪循环中的差异

### 4.1 Latent 准备阶段

#### 原版 Qwen-Image
```python
def prepare_latents(
    self,
    image=None,  # 无图像输入
    batch_size,
    num_channels_latents,
    height, width,
    dtype, device, generator,
    latents=None,
):
    # 仅生成随机噪声
    if latents is None:
        latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        latents = self._pack_latents(latents, ...)
    
    return latents, None  # ⭐ 无 image_latents
```

#### Edit 版本
```python
def prepare_latents(
    self,
    image,  # ⭐ 接收图像输入
    batch_size,
    num_channels_latents,
    height, width,
    dtype, device, generator,
    latents=None,
):
    # 1. 如果提供了图像，编码为 latents
    if image is not None:
        image_latents = self._encode_vae_image(image=image, generator=generator)
        image_latents = self._pack_latents(image_latents, ...)
    
    # 2. 准备去噪的初始 latents
    if latents is None:
        latents = randn_tensor(shape, generator=generator, ...)
        latents = self._pack_latents(latents, ...)
    
    return latents, image_latents  # ⭐ 返回两者
```

**代码位置**：
- 原版：```399:420:pipelines/qwenimage/pipeline_qwenimage.py```
- Edit版：```471:524:pipelines/qwenimage/pipeline_qwenimage_edit.py```

---

### 4.2 去噪循环中的 Latent 连接

#### 原版 Qwen-Image
```python
for i, t in enumerate(timesteps):
    # 直接使用 latents 作为输入
    latent_model_input = latents
    
    # 调用 Transformer
    noise_pred = self.transformer(
        hidden_states=latent_model_input,
        encoder_hidden_states=prompt_embeds,
        ...
    )
```

#### Edit 版本
```python
for i, t in enumerate(timesteps):
    # ⭐ 连接当前去噪 latents 与原始图像 latents
    latent_model_input = latents
    if image_latents is not None:
        latent_model_input = torch.cat([latents, image_latents], dim=1)
        #             ↑ 当前去噪状态    ↑ 原始图像特征
    
    # 调用 Transformer（接收拼接后的输入）
    noise_pred = self.transformer(
        hidden_states=latent_model_input,  # ⭐ 包含原始图像信息
        encoder_hidden_states=prompt_embeds,  # ⭐ 包含多模态语义
        ...
    )
    
    # 只取前部分（latents对应部分）作为预测
    noise_pred = noise_pred[:, : latents.size(1)]
```

**代码位置**：```810:828:pipelines/qwenimage/pipeline_qwenimage_edit.py```

**技术要点**：
1. **Concatenation 策略**：`[当前去噪latents | 原始图像latents]`
2. **双向信息流**：
   - 原始图像 latents → 提供视觉外观参考
   - 文本+图像 embeddings → 提供语义编辑指令
3. **输出截取**：Transformer 输出只取前 `latents.size(1)` 部分，对应去噪部分

---

## 🎯 五、功能层面差异总结

### 5.1 输入处理差异

| 特性 | 原版 Qwen-Image | Edit 版本 |
|------|----------------|-----------|
| **文本输入** | ✅ 必需 | ✅ 必需 |
| **图像输入** | ❌ 不支持 | ✅ **必需**（编辑目标图像） |
| **Processor** | ❌ 无 | ✅ Qwen2VLProcessor |
| **多模态编码** | ❌ 仅文本 | ✅ **文本+图像** |

### 5.2 编码阶段差异

| 阶段 | 原版 Qwen-Image | Edit 版本 |
|-----|----------------|----------|
| **文本编码** | Tokenizer → Text Encoder (纯文本) | Processor → Text Encoder (**多模态**) |
| **图像编码路径1** | ❌ 无 | ✅ **Qwen2.5-VL** (语义理解) |
| **图像编码路径2** | ❌ 无 | ✅ **VAE Encoder** (视觉外观) |
| **Prompt Embeddings** | 仅文本语义 | 文本语义 + 图像语义 |

### 5.3 去噪循环差异

| 特性 | 原版 Qwen-Image | Edit 版本 |
|-----|----------------|----------|
| **初始 Latents** | 随机噪声 | 随机噪声 + **图像 latents** |
| **Transformer 输入** | 仅去噪 latents | **拼接** [去噪 latents, 图像 latents] |
| **条件控制** | 文本 embeddings | **文本+图像** embeddings |

---

## 💡 六、技术优势分析

### 6.1 双路径架构的优势

#### 语义理解路径（Qwen2.5-VL）
- ✅ **理解图像内容**：识别对象、场景、关系
- ✅ **理解编辑意图**：将文本指令映射到视觉修改
- ✅ **保持语义一致性**：在编辑时保留原图的语义结构

#### 视觉外观路径（VAE）
- ✅ **保持视觉细节**：保留纹理、颜色、光照
- ✅ **精确区域控制**：在需要保持不变的区域提供参考
- ✅ **外观一致性**：确保编辑后的图像与原图视觉风格一致

### 6.2 Edit 版本的创新点

1. **双路径并行处理**
   - 语义路径：理解"要做什么"
   - 外观路径：参考"怎么做"

2. **Latent 拼接策略**
   - 在去噪过程中持续注入原始图像信息
   - 允许 Transformer 同时访问编辑状态和参考状态

3. **多模态条件融合**
   - 文本指令与图像理解在同一个 embedding 空间中融合
   - 实现了语义和视觉的统一控制

---

## 📊 七、代码行数对比

| 文件 | 原版 Qwen-Image | Edit 版本 | 差异 |
|------|----------------|-----------|------|
| **总行数** | ~772 行 | ~900 行 | +128 行 |
| **初始化** | ~40 行 | ~45 行 | +5 行 (processor) |
| **文本编码** | ~38 行 | ~46 行 | +8 行 (多模态) |
| **Latent 准备** | ~22 行 | ~54 行 | +32 行 (VAE编码) |
| **去噪循环** | ~70 行 | ~75 行 | +5 行 (latent拼接) |

---

## 🔬 八、详细代码对照表

### 8.1 关键方法对比

| 方法 | 原版 | Edit 版本 | 差异说明 |
|------|------|----------|---------|
| `__init__` | 无 processor | ✅ 有 processor | 支持多模态输入 |
| `_get_qwen_prompt_embeds` | `(prompt, device)` | `(prompt, **image**, device)` | ⭐ 新增图像参数 |
| `encode_prompt` | `(prompt, device, ...)` | `(prompt, **image**, device, ...)` | ⭐ 新增图像参数 |
| `prepare_latents` | 仅生成随机噪声 | ✅ **编码图像 + 生成噪声** | ⭐ 新增 VAE 编码 |
| `_encode_vae_image` | ❌ 无 | ✅ 新增方法 | ⭐ Edit 版本特有 |
| 去噪循环中的 `latent_model_input` | `latents` | `cat([latents, image_latents])` | ⭐ 拼接操作 |

### 8.2 Prompt 模板对比

| 特性 | 原版 | Edit 版本 |
|------|------|----------|
| **系统提示长度** | 34 tokens | 64 tokens |
| **是否包含图像占位符** | ❌ | ✅ `<|vision_start|><|image_pad|><|vision_end|>` |
| **核心任务** | 描述图像 | **理解编辑指令并生成新图像** |

---

## 🎓 九、设计理念对比

### 9.1 原版 Qwen-Image 设计理念

- **目标**：从文本生成全新图像
- **策略**：纯文本条件生成
- **优势**：生成速度快，适合创意创作

### 9.2 Edit 版本设计理念

- **目标**：在现有图像基础上精确编辑
- **策略**：**双路径条件控制**（语义+外观）
- **优势**：
  1. ✅ 保持原图一致性
  2. ✅ 理解编辑上下文
  3. ✅ 精确控制编辑区域
  4. ✅ 支持语义编辑和外观编辑

---

## 📝 十、使用场景对比

### 原版 Qwen-Image 适用场景
- 🎨 从零开始生成图像
- 🖼️ 创意设计
- 📝 文本到图像转换

### Edit 版本适用场景
- ✏️ 图像编辑（颜色、对象、背景等）
- 🔄 风格转换
- 📋 文本精确编辑
- 🎯 局部修改（添加/删除/替换对象）
- 🌈 语义编辑（视角变化、风格转换）

---

## 🔚 十一、总结

Qwen-Image-Edit 的核心创新在于**双路径架构**：

1. **语义路径（Qwen2.5-VL）**：图像 + 文本 → 多模态理解 → 编辑指令理解
2. **外观路径（VAE）**：图像 → 视觉特征编码 → 外观参考

通过这两条路径的协同工作，Edit 版本能够在保持原图一致性的同时，实现精确的语义和外观编辑。

**关键代码差异位置**：
- 文本编码：```226:271:pipelines/qwenimage/pipeline_qwenimage_edit.py```
- VAE 编码：```395:416:pipelines/qwenimage/pipeline_qwenimage_edit.py```
- 去噪循环：```810:828:pipelines/qwenimage/pipeline_qwenimage_edit.py```

---

**报告生成时间**：基于 diffusers 代码库分析  
**版本**：diffusers 0.36.0.dev0


