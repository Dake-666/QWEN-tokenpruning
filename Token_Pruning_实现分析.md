# Token Pruning 在 Qwen-Image-Edit 中的实现分析

## 🎯 目标方案

**策略**：
- 步骤 1, 3：完整计算所有 tokens
- 步骤 2：重用步骤 1 的 image tokens hidden states
- 步骤 4：重用步骤 3 的 image tokens hidden states

**预期加速**：约 40-50%（跳过 2/4 步骤的 image tokens 计算）

---

## 🔍 关键技术点分析

### 1. Token 的定义和位置

#### 在 Pipeline 层面

```python
# pipelines/qwenimage/pipeline_qwenimage_edit.py:810-812
latent_model_input = latents  # [B, L1, C]
if image_latents is not None:
    latent_model_input = torch.cat([latents, image_latents], dim=1)
    # 形状: [B, L1+L2, C]
    # L1: 去噪 tokens 数量
    # L2: 图像 tokens 数量
```

#### 在 Transformer 层面

```python
# models/transformers/transformer_qwenimage.py:618
hidden_states = self.img_in(hidden_states)  # [B, L1+L2, inner_dim]

# 经过 60 层 blocks
for block in self.transformer_blocks:
    encoder_hidden_states, hidden_states = block(
        hidden_states=hidden_states,  # [B, L1+L2, inner_dim]
        ...
    )
```

**关键**：需要在每个 block 中分离 L1 和 L2 部分。

---

## 🔬 实现方案设计

### 方案 A：在 TransformerBlock 层面修改（推荐）

#### 修改位置

```python
# models/transformers/transformer_qwenimage.py:411-476
class QwenImageTransformerBlock(nn.Module):
    def forward(
        self,
        hidden_states: torch.Tensor,  # [B, L1+L2, D]
        encoder_hidden_states: torch.Tensor,
        encoder_hidden_states_mask: torch.Tensor,
        temb: torch.Tensor,
        image_rotary_emb,
        joint_attention_kwargs,
    ):
```

#### 修改策略

```python
def forward_with_pruning(self, hidden_states, ...):
    # 1. 检查是否需要 pruning
    if should_prune and cached_image_hidden is not None:
        L_denoise = denoise_token_length
        
        # 2. 分离 tokens
        denoise_hidden = hidden_states[:, :L_denoise]  # 去噪部分
        image_hidden = hidden_states[:, L_denoise:]     # 图像部分
        
        # 3. 只对去噪部分做完整计算
        # 图像部分使用缓存参与计算（提供 K, V）
        
        # 4. 计算时的策略
        # 图像 tokens:
        #   - Q: 不计算（反正不用）
        #   - K, V: 使用缓存的 hidden state 计算（供去噪 query）
        #   - MLP: 不计算（反正最后不用）
        
        # 去噪 tokens:
        #   - Q, K, V: 正常计算
        #   - MLP: 正常计算
```

---

### 方案 B：在注意力层面修改（最优但复杂）

#### 修改位置

```python
# models/transformers/transformer_qwenimage.py:250-360
class QwenDoubleStreamAttnProcessor2_0:
    def __call__(self, attn, hidden_states, encoder_hidden_states, ...):
```

#### 核心逻辑

```python
def attention_with_pruning(attn, hidden_states, ...):
    L_denoise = pruning_context.denoise_token_length
    
    # 分离去噪和图像 tokens
    denoise_hidden = hidden_states[:, :L_denoise]
    image_hidden = hidden_states[:, L_denoise:]  # 这是缓存的
    
    # === QKV 投影 ===
    # 去噪 tokens: 完整投影
    denoise_q = attn.to_q(denoise_hidden)
    denoise_k = attn.to_k(denoise_hidden)
    denoise_v = attn.to_v(denoise_hidden)
    
    # 图像 tokens: 只投影 K, V（不投影 Q，因为不需要查询）
    image_k = attn.to_k(image_hidden)  # ⭐ 使用缓存的 hidden state
    image_v = attn.to_v(image_hidden)  # ⭐ 使用缓存的 hidden state
    # 不计算 image_q（节省）
    
    # 文本 tokens: 正常处理
    txt_q = attn.to_add_q(encoder_hidden_states)
    txt_k = attn.to_add_k(encoder_hidden_states)
    txt_v = attn.to_add_v(encoder_hidden_states)
    
    # === 拼接并计算注意力 ===
    joint_query = torch.cat([txt_q, denoise_q], dim=1)  # 不包含 image_q
    joint_key = torch.cat([txt_k, image_k, denoise_k], dim=1)
    joint_value = torch.cat([txt_v, image_v, denoise_v], dim=1)
    
    # 计算注意力（image tokens 提供 K,V 但不主动查询）
    attention_output = dispatch_attention_fn(joint_query, joint_key, joint_value, ...)
    
    # === 分离输出 ===
    txt_output = attention_output[:, :L_txt]
    denoise_output = attention_output[:, L_txt:]
    image_output = 使用缓存  # 不更新
    
    # === MLP 处理 ===
    # 只对去噪 tokens 计算 MLP
    denoise_mlp_output = self.img_mlp(denoise_modulated)
    # 图像 tokens 跳过 MLP
```

---

## ⚠️ 实现挑战

### 挑战 1：索引管理的复杂性

```python
# 需要在整个 forward pass 中维护索引
L_denoise = ?  # 如何传递到每个 block？
L_image = ?

# 方案：通过 attention_kwargs 传递
attention_kwargs = {
    "denoise_token_length": L_denoise,
    "enable_pruning": True,
    "cached_hidden": cached_states
}
```

---

### 挑战 2：缓存的时机和位置

```python
# 问题：在哪里缓存？

# 选项 A：在 block 输出处缓存
# 每个 block 都需要缓存 → 60 层 × 2 个缓存点 = 120 个缓存

# 选项 B：只在最后一层缓存
# 只缓存最终的 hidden states
# 但每个 block 都需要它，可能导致信息不匹配
```

**我的建议**：在每个 block 都缓存（虽然内存开销大，但正确性高）

---

### 挑战 3：注意力计算的不对称性

```python
# 问题：image tokens 不生成 Q，但参与 K,V
# 这会导致 attention mask 的不对称

# 原始：
joint_query = [txt_q, img_q, denoise_q]  # 形状: [B, L_txt+L_img+L_denoise, H, D]
joint_key = [txt_k, img_k, denoise_k]    # 形状: [B, L_txt+L_img+L_denoise, H, D]

# Pruning 后：
joint_query = [txt_q, denoise_q]         # 形状: [B, L_txt+L_denoise, H, D]  ⚠️ 缩短了
joint_key = [txt_k, img_k, denoise_k]    # 形状: [B, L_txt+L_img+L_denoise, H, D]

# 注意力矩阵形状不匹配！
# Q: [B, H, L_txt+L_denoise, D]
# K: [B, H, L_txt+L_img+L_denoise, D]
# QK^T: [B, H, L_txt+L_denoise, L_txt+L_img+L_denoise]  ← 这是可以的！
```

**好消息**：注意力可以处理不对称的 Q 和 K！

---

## 💡 简化实现方案

考虑到复杂性，我建议分两阶段：

### 阶段 1：简化版（只跳过 MLP）⭐ 先实现这个

```python
def forward_pruning_v1(self, hidden_states, ...):
    """
    只跳过 image tokens 的 MLP，保留注意力计算
    """
    L_denoise = denoise_token_length
    
    # === 注意力：正常计算（所有 tokens）===
    img_attn_output, txt_attn_output = self.attn(...)
    hidden_states = hidden_states + img_gate1 * img_attn_output
    
    # === MLP：只计算去噪 tokens ===
    if should_prune:
        # 只对去噪部分计算 MLP
        denoise_hidden = hidden_states[:, :L_denoise]
        denoise_normed = self.img_norm2(denoise_hidden)
        denoise_modulated, gate = self._modulate(denoise_normed, img_mod2)
        denoise_mlp = self.img_mlp(denoise_modulated)
        denoise_hidden = denoise_hidden + gate * denoise_mlp
        
        # 图像部分：重用缓存（跳过 MLP）
        image_hidden = cached_image_hidden[:, L_denoise:]
        
        # 合并
        hidden_states = torch.cat([denoise_hidden, image_hidden], dim=1)
    else:
        # 正常计算
        img_normed2 = self.img_norm2(hidden_states)
        img_modulated2, img_gate2 = self._modulate(img_normed2, img_mod2)
        img_mlp_output = self.img_mlp(img_modulated2)
        hidden_states = hidden_states + img_gate2 * img_mlp_output
    
    return encoder_hidden_states, hidden_states
```

**优势**：
- ✅ 实现简单
- ✅ 注意力完整（保证信息流动）
- ✅ 只优化 MLP（约 30% 加速）

---

### 阶段 2：完整版（注意力 + MLP）

如果阶段 1 效果好，再实现完整版。

---

## 📊 预期效果分析

### 计算量分解（每个 Block）

| 组件 | 去噪 tokens | 图像 tokens | 总计 |
|------|-----------|-----------|------|
| **Q 投影** | L1 × D² | L2 × D² | (L1+L2) × D² |
| **K 投影** | L1 × D² | L2 × D² | (L1+L2) × D² |
| **V 投影** | L1 × D² | L2 × D² | (L1+L2) × D² |
| **注意力** | - | - | O(L² × D) |
| **MLP** | L1 × 4D² | L2 × 4D² | (L1+L2) × 4D² |

### 简化版加速（只跳过 MLP）

- 跳过步骤 2, 4 的 image MLP
- 节省：2 × 60层 × L2 × 4D² × 2步 = **约 30-40%**

### 完整版加速（跳过 Q + MLP）

- 跳过步骤 2, 4 的 image Q 投影和 MLP
- 节省：2 × 60层 × L2 × (D² + 4D²) × 2步 = **约 40-50%**

---

## ⚠️ 我仍然担心的问题

### 1. **Image Tokens 的角色演化**

在 Edit 任务中，image tokens 不仅是静态参考：

```python
# 步骤 1: 高噪声水平
去噪 tokens: [大量噪声]
图像 tokens: 提供"这是什么物体"的语义引导

# 步骤 2-3: 中等噪声
去噪 tokens: [部分清晰]
图像 tokens: 提供"细节纹理"的外观引导

# 步骤 4: 低噪声
去噪 tokens: [基本清晰]
图像 tokens: 提供"精确对齐"的参考
```

**问题**：不同步骤需要 image tokens 的不同"视角"。冻结可能导致引导不匹配。

---

### 2. **双流架构的耦合性**

```python
# QwenImageTransformerBlock 中
# 图像流和文本流是耦合的
encoder_hidden_states, hidden_states = block(...)

# 文本流会影响图像流，图像流会影响文本流
# 冻结图像 tokens 可能破坏这种平衡
```

---

### 3. **实验验证的必要性**

建议对比三个版本：

```python
# A. Baseline（无 pruning）
python 5_lightning_with_token_pruning.py --no-pruning

# B. 简化版（只跳过 MLP）
python 5_lightning_with_token_pruning.py --pruning-mode mlp

# C. 完整版（跳过 Q + MLP）
python 5_lightning_with_token_pruning.py --pruning-mode full
```

分别测量：
- 推理时间
- PSNR / SSIM（与无 pruning 对比）
- 视觉质量

---

## 🎯 我的实现建议

### 第一步：最小可行实现（MVP）

创建一个简单的测试版本：

```python
# 伪代码
class PrunableQwenImageEditPipeline:
    def __call__(self, ...):
        for i, t in enumerate(timesteps):
            if i in [0, 2]:  # 步骤 1, 3
                # 完整计算
                output = transformer(latent_model_input, ...)
                # 缓存 image tokens 的输出
                cache[i] = output[:, denoise_length:]
            else:  # 步骤 2, 4
                # 使用缓存
                denoise_input = latent_model_input[:, :denoise_length]
                cached_image = cache[i-1]
                
                # 只计算去噪部分（但这需要修改 transformer 内部）
                output_denoise = transformer(
                    denoise_input,
                    cached_image_for_attention=cached_image,
                    ...
                )
```

**问题**：这需要 transformer 支持分离计算，目前不支持。

---

### 第二步：Monkey Patch Transformer

创建一个 wrapper 来拦截和修改计算：

```python
def create_pruning_wrapper(original_block, denoise_len, cache_dict):
    """
    创建带 pruning 的 block wrapper
    """
    def wrapped_forward(hidden_states, step_idx, ...):
        if step_idx in [1, 3]:  # 需要 prune
            # 分离
            denoise_h = hidden_states[:, :denoise_len]
            image_h = cache_dict[step_idx - 1]  # 使用上一步缓存
            
            # ⚠️ 这里有个问题：
            # 原始 forward 期望完整的 hidden_states
            # 我们需要"欺骗"它，让它以为在处理完整输入
            # 但实际上 image 部分是缓存的
            
            # 策略：构造一个假的完整输入
            fake_full_hidden = torch.cat([denoise_h, image_h], dim=1)
            
            # 调用原始 forward
            output = original_block(fake_full_hidden, ...)
            
            # 只取去噪部分的输出
            output_denoise = output[:, :denoise_len]
            # 图像部分继续使用缓存
            output_final = torch.cat([output_denoise, image_h], dim=1)
            
            return output_final
        else:
            # 完整计算
            output = original_block(hidden_states, ...)
            # 缓存 image 部分
            cache_dict[step_idx] = output[:, denoise_len:].clone()
            return output
    
    return wrapped_forward
```

---

## 🚧 实现的核心难点

### 难点 1：在 Block 内部区分 denoise 和 image tokens

**问题**：Block 不知道输入的哪部分是 denoise，哪部分是 image。

**解决方案**：通过 `attention_kwargs` 传递元数据

```python
attention_kwargs = {
    "denoise_token_length": L_denoise,
    "current_step": i,
    "enable_pruning": True,
    "image_cache": cache_dict
}
```

---

### 难点 2：60 层 Block 的缓存管理

**问题**：每一层的输出都不同，如何缓存？

**方案 A**：缓存每一层的 image tokens
```python
layer_caches = {
    0: [layer0_image_hidden, layer1_image_hidden, ..., layer59_image_hidden],
    2: [layer0_image_hidden, layer1_image_hidden, ..., layer59_image_hidden],
}
```
内存开销：60层 × 2缓存 × L2 × D × 4bytes ≈ 几百MB

**方案 B**：只缓存输入和输出
- 问题：中间层的信息不匹配

**建议**：使用方案 A，内存开销可接受

---

## 📝 完整实现路线图

### 第 1 阶段：准备工作

1. ✅ 创建 TokenPruningContext 类
2. ✅ 创建自定义 Pipeline 类
3. ⏳ 修改 Transformer Block forward

### 第 2 阶段：核心实现

1. 修改 `QwenImageTransformerBlock.forward`
2. 添加缓存管理逻辑
3. 在 pipeline 的去噪循环中集成

### 第 3 阶段：测试验证

1. 测试推理速度
2. 测试输出质量
3. 对比实验

---

## 🤔 我的建议（请确认）

鉴于实现复杂度，我建议：

**方案 A：完整但正确的实现**（我开始实现了）
- 需要深度修改 Transformer 内部
- 实现复杂，但效果可控
- 预计需要 500-800 行代码

**方案 B：简化版先验证**
- 只在 pipeline 层面做 token 分离
- 使用简单的 monkey patch
- 快速验证想法，然后再优化

**您希望我继续完整实现，还是先做一个简化版快速测试？** 

另外，我担心的是：**在 4 步推理中，每步的作用都很关键，pruning 2 步可能影响较大**。建议先实现能对比的版本，测量质量损失。
