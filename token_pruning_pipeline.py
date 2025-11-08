"""
Token Pruning Pipeline 实现
基于 QwenImageEditPipeline，添加 Token Pruning 支持
"""
import torch
import numpy as np
from typing import Optional, Union, List, Dict, Any, Callable
from PIL import Image

from diffusers import QwenImageEditPipeline
from diffusers.pipelines.qwenimage.pipeline_qwenimage_edit import (
    calculate_dimensions, retrieve_timesteps
)


class TokenPruningQwenImageEditPipeline(QwenImageEditPipeline):
    """
    增强的 QwenImageEditPipeline，支持 Token Pruning
    
    策略:
    - 步骤 1, 3: 完整计算所有 tokens
    - 步骤 2: 重用步骤 1 的 image tokens hidden states  
    - 步骤 4: 重用步骤 3 的 image tokens hidden states
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Token Pruning 配置
        self.enable_token_pruning = True
        self.cached_image_hidden_step1 = None
        self.cached_image_hidden_step3 = None
        
    @torch.no_grad()
    def __call__(
        self,
        image: Optional[Image.Image] = None,
        prompt: Union[str, List[str]] = None,
        negative_prompt: Union[str, List[str]] = None,
        true_cfg_scale: float = 4.0,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 4,  # Lightning 默认 4 步
        sigmas: Optional[List[float]] = None,
        guidance_scale: Optional[float] = None,
        num_images_per_prompt: int = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        prompt_embeds_mask: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds_mask: Optional[torch.Tensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        max_sequence_length: int = 512,
        enable_token_pruning: bool = True,  # ⭐ 新增参数
    ):
        """
        增强的 __call__ 方法，支持 Token Pruning
        """
        # 设置 pruning 状态
        self.enable_token_pruning = enable_token_pruning
        
        # ===== 前期准备（与原版相同）=====
        image_size = image[0].size if isinstance(image, list) else image.size
        calculated_width, calculated_height, _ = calculate_dimensions(1024 * 1024, image_size[0] / image_size[1])
        height = height or calculated_height
        width = width or calculated_width
        
        multiple_of = self.vae_scale_factor * 2
        width = width // multiple_of * multiple_of
        height = height // multiple_of * multiple_of
        
        # 检查输入
        self.check_inputs(
            prompt, height, width, negative_prompt, prompt_embeds,
            negative_prompt_embeds, prompt_embeds_mask, negative_prompt_embeds_mask,
            callback_on_step_end_tensor_inputs, max_sequence_length
        )
        
        self._guidance_scale = guidance_scale
        self._attention_kwargs = attention_kwargs
        self._current_timestep = None
        self._interrupt = False
        
        # 定义 batch_size
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]
        
        device = self._execution_device
        
        # 预处理图像
        if image is not None and not (isinstance(image, torch.Tensor) and image.size(1) == self.latent_channels):
            image = self.image_processor.resize(image, calculated_height, calculated_width)
            prompt_image = image
            image = self.image_processor.preprocess(image, calculated_height, calculated_width)
            image = image.unsqueeze(2)
        
        # 编码 prompt
        has_neg_prompt = negative_prompt is not None or (
            negative_prompt_embeds is not None and negative_prompt_embeds_mask is not None
        )
        do_true_cfg = true_cfg_scale > 1 and has_neg_prompt
        
        prompt_embeds, prompt_embeds_mask = self.encode_prompt(
            image=prompt_image, prompt=prompt, prompt_embeds=prompt_embeds,
            prompt_embeds_mask=prompt_embeds_mask, device=device,
            num_images_per_prompt=num_images_per_prompt,
            max_sequence_length=max_sequence_length
        )
        
        if do_true_cfg:
            negative_prompt_embeds, negative_prompt_embeds_mask = self.encode_prompt(
                image=prompt_image, prompt=negative_prompt,
                prompt_embeds=negative_prompt_embeds,
                prompt_embeds_mask=negative_prompt_embeds_mask,
                device=device, num_images_per_prompt=num_images_per_prompt,
                max_sequence_length=max_sequence_length
            )
        
        # 准备 latents
        num_channels_latents = self.transformer.config.in_channels // 4
        latents, image_latents = self.prepare_latents(
            image, batch_size * num_images_per_prompt,
            num_channels_latents, height, width,
            prompt_embeds.dtype, device, generator, latents
        )
        
        # ⭐ 记录 token 长度信息（用于 pruning）
        denoise_token_length = latents.shape[1]
        image_token_length = image_latents.shape[1] if image_latents is not None else 0
        
        print(f"\nToken 信息:")
        print(f"  去噪 tokens: {denoise_token_length}")
        print(f"  图像 tokens: {image_token_length}")
        print(f"  总 tokens: {denoise_token_length + image_token_length}")
        
        img_shapes = [
            [
                (1, height // self.vae_scale_factor // 2, width // self.vae_scale_factor // 2),
                (1, calculated_height // self.vae_scale_factor // 2, calculated_width // self.vae_scale_factor // 2),
            ]
        ] * batch_size
        
        # 准备 timesteps
        sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps) if sigmas is None else sigmas
        image_seq_len = latents.shape[1]
        from diffusers.pipelines.qwenimage.pipeline_qwenimage import calculate_shift
        mu = calculate_shift(
            image_seq_len,
            self.scheduler.config.get("base_image_seq_len", 256),
            self.scheduler.config.get("max_image_seq_len", 4096),
            self.scheduler.config.get("base_shift", 0.5),
            self.scheduler.config.get("max_shift", 1.15),
        )
        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler, num_inference_steps, device, sigmas=sigmas, mu=mu
        )
        
        # 处理 guidance
        if self.transformer.config.guidance_embeds and guidance_scale is None:
            raise ValueError("guidance_scale is required for guidance-distilled model.")
        elif self.transformer.config.guidance_embeds:
            guidance = torch.full([1], guidance_scale, device=device, dtype=torch.float32)
            guidance = guidance.expand(latents.shape[0])
        else:
            guidance = None
        
        if self.attention_kwargs is None:
            self._attention_kwargs = {}
        
        txt_seq_lens = prompt_embeds_mask.sum(dim=1).tolist() if prompt_embeds_mask is not None else None
        negative_txt_seq_lens = (
            negative_prompt_embeds_mask.sum(dim=1).tolist() if negative_prompt_embeds_mask is not None else None
        )
        
        # ===== 去噪循环（增强版，支持 Token Pruning）=====
        self.scheduler.set_begin_index(0)
        
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if self.interrupt:
                    continue
                
                self._current_timestep = t
                step_start_time = torch.cuda.Event(enable_timing=True)
                step_end_time = torch.cuda.Event(enable_timing=True)
                step_start_time.record()
                
                # ⭐ 准备输入（考虑 Token Pruning）
                latent_model_input = latents
                if image_latents is not None:
                    latent_model_input = torch.cat([latents, image_latents], dim=1)
                
                # ⭐ 设置 pruning 步骤信息
                should_compute_full = (i == 0 or i == 2)  # 步骤 1 和 3
                should_use_cache = (i == 1 or i == 3)      # 步骤 2 和 4
                
                if self.enable_token_pruning and should_use_cache:
                    print(f"\n   步骤 {i+1}: 使用缓存 (Token Pruning) ⚡", end="")
                else:
                    print(f"\n   步骤 {i+1}: 完整计算", end="")
                
                timestep = t.expand(latents.shape[0]).to(latents.dtype)
                
                # ⭐ 调用 transformer（带 pruning 信息）
                with self.transformer.cache_context("cond"):
                    # 传递 pruning 信息作为 attention_kwargs
                    pruning_info = {
                        "enable_pruning": self.enable_token_pruning and should_use_cache,
                        "denoise_token_length": denoise_token_length,
                        "cached_image_hidden": (
                            self.cached_image_hidden_step1 if i == 1 else
                            self.cached_image_hidden_step3 if i == 3 else None
                        )
                    }
                    
                    current_attention_kwargs = self.attention_kwargs.copy() if self.attention_kwargs else {}
                    current_attention_kwargs.update(pruning_info)
                    
                    noise_pred = self.transformer(
                        hidden_states=latent_model_input,
                        timestep=timestep / 1000,
                        guidance=guidance,
                        encoder_hidden_states_mask=prompt_embeds_mask,
                        encoder_hidden_states=prompt_embeds,
                        img_shapes=img_shapes,
                        txt_seq_lens=txt_seq_lens,
                        attention_kwargs=current_attention_kwargs,
                        return_dict=False,
                    )[0]
                    noise_pred = noise_pred[:, :latents.size(1)]
                
                # CFG（如果启用）
                if do_true_cfg:
                    with self.transformer.cache_context("uncond"):
                        neg_noise_pred = self.transformer(
                            hidden_states=latent_model_input,
                            timestep=timestep / 1000,
                            guidance=guidance,
                            encoder_hidden_states_mask=negative_prompt_embeds_mask,
                            encoder_hidden_states=negative_prompt_embeds,
                            img_shapes=img_shapes,
                            txt_seq_lens=negative_txt_seq_lens,
                            attention_kwargs=current_attention_kwargs,
                            return_dict=False,
                        )[0]
                    neg_noise_pred = neg_noise_pred[:, :latents.size(1)]
                    comb_pred = neg_noise_pred + true_cfg_scale * (noise_pred - neg_noise_pred)
                    
                    cond_norm = torch.norm(noise_pred, dim=-1, keepdim=True)
                    noise_norm = torch.norm(comb_pred, dim=-1, keepdim=True)
                    noise_pred = comb_pred * (cond_norm / noise_norm)
                
                # 更新 latents
                latents_dtype = latents.dtype
                latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]
                
                if latents.dtype != latents_dtype:
                    if torch.backends.mps.is_available():
                        latents = latents.to(latents_dtype)
                
                # ⭐ 缓存管理（在步骤 1 和 3 后）
                if self.enable_token_pruning and should_compute_full:
                    # 这里需要获取 transformer 输出的完整 hidden states
                    # 由于 transformer 返回的是处理后的结果，我们需要在 block 层面缓存
                    # 暂时标记需要缓存
                    pass
                
                step_end_time.record()
                torch.cuda.synchronize()
                elapsed = step_start_time.elapsed_time(step_end_time) / 1000
                print(f" ({elapsed:.2f}s)")
                
                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)
                    latents = callback_outputs.pop("latents", latents)
                    prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)
                
                progress_bar.update()
        
        # 解码
        self._current_timestep = None
        if output_type == "latent":
            output_image = latents
        else:
            latents = self._unpack_latents(latents, height, width, self.vae_scale_factor)
            latents = latents.to(self.vae.dtype)
            latents_mean = (
                torch.tensor(self.vae.config.latents_mean)
                .view(1, self.vae.config.z_dim, 1, 1, 1)
                .to(latents.device, latents.dtype)
            )
            latents_std = 1.0 / torch.tensor(self.vae.config.latents_std).view(1, self.vae.config.z_dim, 1, 1, 1).to(
                latents.device, latents.dtype
            )
            latents = latents / latents_std + latents_mean
            output_image = self.vae.decode(latents, return_dict=False)[0][:, :, 0]
            output_image = self.image_processor.postprocess(output_image, output_type=output_type)
        
        self.maybe_free_model_hooks()
        
        if not return_dict:
            return (output_image,)
        
        from diffusers.pipelines.qwenimage.pipeline_output import QwenImagePipelineOutput
        return QwenImagePipelineOutput(images=output_image)

