# Token Pruning 完整实现 - 使用说明

## 🎯 实现目标

在 Qwen-Image-Edit Lightning (4步) 基础上，进一步通过 Token Pruning 提升推理速度。

**Pruning 策略**:
- **步骤 1**: 完整计算，缓存所有层的 image tokens hidden states
- **步骤 2**: 重用步骤 1 的缓存 ⚡
- **步骤 3**: 完整计算，缓存所有层的 image tokens hidden states  
- **步骤 4**: 重用步骤 3 的缓存 ⚡

**理论加速**: 40-50% (跳过 2/4 步骤的 image tokens 计算)

---

## 📁 文件说明

| 文件 | 说明 |
|------|------|
| `pruning_modules.py` | 核心模块：PrunableTransformerBlock, PrunableAttention, TokenPruningCache |
| `pruning_pipeline_full.py` | 自定义 Pipeline 类，管理 pruning 状态 |
| `run_with_token_pruning.py` | 主执行脚本，支持命令行参数 |
| `Token_Pruning_实现分析.md` | 技术分析文档 |
| `Token_Pruning_使用说明.md` | 本文档 |

---

## 🚀 使用方法

### 基础用法

```bash
# 启用 Token Pruning（默认）
python run_with_token_pruning.py -i input.png -p "Make it purple"

# 输出: outputs_pruning/output_pruning_TIMESTAMP.png
```

### 对比实验（重要）

```bash
# 1. 运行基线（无 Pruning）
python run_with_token_pruning.py \
    -i input.png \
    -p "Your editing prompt" \
    --no-pruning

# 2. 运行 Pruning 版本
python run_with_token_pruning.py \
    -i input.png \
    -p "Your editing prompt"

# 3. 对比结果
ls -lh outputs_pruning/latest_baseline.png
ls -lh outputs_pruning/latest_pruning.png

# 4. 查看推理时间（在输出中）
```

---

## 📊 命令行参数

| 参数 | 短参数 | 默认值 | 说明 |
|------|--------|--------|------|
| `--input` | `-i` | `input.png` | 输入图片路径 |
| `--prompt` | `-p` | (默认 prompt) | 编辑指令 |
| `--output_dir` | `-o` | `outputs_pruning` | 输出目录 |
| `--steps` | `-s` | `4` | 推理步数 |
| `--cfg` | `-c` | `1.0` | CFG Scale |
| `--no-pruning` | - | `False` | 禁用 pruning（对比用） |

---

## 🔬 技术细节

### Pruning 实现位置

1. **PrunableQwenDoubleStreamAttnProcessor** (`pruning_modules.py`)
   - 在注意力计算中跳过 image tokens 的 Q 投影
   - image tokens 仍提供 K, V 供查询

2. **PrunableQwenImageTransformerBlock** (`pruning_modules.py`)
   - 在 MLP 计算中跳过 image tokens
   - 使用缓存的 hidden states

3. **TokenPruningQwenImageEditPipeline** (`pruning_pipeline_full.py`)
   - 管理去噪循环中的 pruning 状态
   - 记录和传递 token 长度信息

### 缓存策略

```python
# 步骤 1 (i=0): 
#   - 完整计算所有 60 层
#   - 缓存每层的 image tokens hidden states

# 步骤 2 (i=1):
#   - 去噪 tokens: 正常计算
#   - 图像 tokens: 使用步骤 1 的缓存（60 层）
#   - 节省: image tokens 的 Q投影 + MLP

# 步骤 3 (i=2):
#   - 完整计算所有 60 层
#   - 缓存每层的 image tokens hidden states

# 步骤 4 (i=3):
#   - 去噪 tokens: 正常计算
#   - 图像 tokens: 使用步骤 3 的缓存（60 层）
#   - 节省: image tokens 的 Q投影 + MLP
```

---

## ⚠️ 重要注意事项

### 1. 内存使用

Token Pruning 需要缓存：
- 60 层 × 2 个缓存点（步骤 1, 3）
- 每个缓存: `[B, L_image, D]`
- 估计内存：约 500MB - 1GB（取决于图像尺寸）

### 2. 质量影响

Pruning 可能影响输出质量，建议：
- 始终与基线对比
- 使用 PSNR / SSIM 量化评估
- 视觉检查编辑效果

### 3. 首次运行

第一次运行会：
- 下载基础模型（~20GB）
- 下载 Lightning LoRA（~2GB）
- 应用 pruning 补丁（几秒钟）

---

## 🧪 实验示例

### 实验 1: 速度对比

```bash
# 基线
time python run_with_token_pruning.py -p "Make purple" --no-pruning

# Pruning
time python run_with_token_pruning.py -p "Make purple"

# 对比推理时间
```

### 实验 2: 质量对比

```bash
# 生成多个样本
for i in {1..5}; do
    python run_with_token_pruning.py -p "Add rainbow" --no-pruning
    python run_with_token_pruning.py -p "Add rainbow"
done

# 视觉对比 outputs_pruning/ 中的结果
```

### 实验 3: 不同编辑任务

```bash
# 颜色修改
python run_with_token_pruning.py -p "Change color to blue"

# 对象添加
python run_with_token_pruning.py -p "Add a hat"

# 风格转换
python run_with_token_pruning.py -p "Transform to anime style"

# 背景修改
python run_with_token_pruning.py -p "Change background to sunset"
```

---

## 📈 预期结果

### 推理速度

| 模式 | 推理时间 | 加速比 |
|------|---------|--------|
| Baseline (无 Pruning) | ~5-8 秒 | 1.0x |
| Token Pruning | **~3-5 秒** | **1.5-2x** ⚡ |

### 质量评估

建议使用以下指标：
- PSNR: > 35 dB（较好）
- SSIM: > 0.95（较好）
- 视觉检查：编辑效果是否符合预期

---

## 🐛 故障排除

### 错误 1: 导入失败

```bash
ModuleNotFoundError: No module named 'pruning_modules'
```

**解决**: 确保在项目根目录运行：
```bash
cd ~/efs/cy/EDIT/QWEN-tokenpruning
python run_with_token_pruning.py -i input.png -p "Your prompt"
```

### 错误 2: CUDA OOM

```bash
# 启用内存优化
# 在脚本中添加（未来版本）:
pipe.enable_model_cpu_offload()
pipe.vae.enable_tiling()
```

### 错误 3: 输出质量下降明显

这可能是 pruning 策略需要调整：
- 尝试减少 pruning 步骤（只 prune 步骤 2 或 4）
- 调整 CFG scale

---

## 📝 下一步开发

1. ✅ 基础实现（当前版本）
2. ⏳ 添加性能分析工具
3. ⏳ 添加质量评估工具
4. ⏳ 优化缓存策略
5. ⏳ 支持更多 pruning 策略

---

## 🎓 技术参考

- 论文: CAT (Cache-Assisted Token Pruning)
- 基础模型: Qwen-Image-Edit
- 加速模型: Qwen-Image-Lightning
- 框架: Hugging Face Diffusers

---

**开始实验**: `python run_with_token_pruning.py -i input.png -p "Your prompt"`

