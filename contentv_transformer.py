## Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
## 
## Licensed under the Apache License, Version 2.0 (the "License");
## you may not use this file except in compliance with the License.
## You may obtain a copy of the License at
## 
##     http://www.apache.org/licenses/LICENSE-2.0
## 
## Unless required by applicable law or agreed to in writing, software
## distributed under the License is distributed on an "AS IS" BASIS,
## WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
## See the License for the specific language governing permissions and
## limitations under the License.

## Modified from https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/transformers/transformer_sd3.py

from diffusers import SD3Transformer2DModel
from diffusers.models.embeddings import get_3d_sincos_pos_embed
from diffusers.models.modeling_outputs import Transformer2DModelOutput
import torch
from einops import rearrange
from typing import Any, Dict, Optional, Union
from diffusers.utils import USE_PEFT_BACKEND, logging, scale_lora_layers, unscale_lora_layers
import os
USE_ASCEND_NPU = int(os.getenv('USE_ASCEND_NPU', '0'))
if USE_ASCEND_NPU:
    import torch_npu
    import math


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


class SD3PatchEmbed(torch.nn.Module):

    def __init__(self, patch_size, in_channels, embed_dim):
        super().__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.proj = torch.nn.Conv2d(
            in_channels, embed_dim, 
            kernel_size=(patch_size, patch_size), 
            stride=(patch_size, patch_size), 
            bias=True)

    def forward(self, x):
        b, _, t, h, w = x.shape
        x = rearrange(x, "b c t h w -> (b t) c h w")
        x = self.proj(x)
        x = rearrange(x, "(b t) c h w -> b (t h w) c", b=b)

        pos_embed = get_3d_sincos_pos_embed(
            embed_dim=self.embed_dim, 
            spatial_size=(w//self.patch_size, h//self.patch_size), 
            temporal_size=t,
            spatial_interpolation_scale=0.5,
            temporal_interpolation_scale=1.0,
            output_type="pt",
        )
        pos_embed = pos_embed.flatten(0, 1).unsqueeze(0).to(device=x.device, dtype=x.dtype)
        return x + pos_embed


class SD3AttnProcessorNPU:

    def __call__(
        self,
        attn,
        hidden_states,
        encoder_hidden_states,
        attention_mask,
    ):
        residual = hidden_states
        batch_size = hidden_states.shape[0]

        ## Latent projections
        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        ## QK-Norm
        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        ## Text projections
        encoder_hidden_states_query_proj = attn.add_q_proj(encoder_hidden_states)
        encoder_hidden_states_key_proj = attn.add_k_proj(encoder_hidden_states)
        encoder_hidden_states_value_proj = attn.add_v_proj(encoder_hidden_states)

        encoder_hidden_states_query_proj = encoder_hidden_states_query_proj.view(
            batch_size, -1, attn.heads, head_dim
        ).transpose(1, 2)
        encoder_hidden_states_key_proj = encoder_hidden_states_key_proj.view(
            batch_size, -1, attn.heads, head_dim
        ).transpose(1, 2)
        encoder_hidden_states_value_proj = encoder_hidden_states_value_proj.view(
            batch_size, -1, attn.heads, head_dim
        ).transpose(1, 2)

        ## QK-Norm
        if attn.norm_added_q is not None:
            encoder_hidden_states_query_proj = attn.norm_added_q(encoder_hidden_states_query_proj)
        if attn.norm_added_k is not None:
            encoder_hidden_states_key_proj = attn.norm_added_k(encoder_hidden_states_key_proj)

        ## Merge QKV
        query = torch.cat([query, encoder_hidden_states_query_proj], dim=2)
        key = torch.cat([key, encoder_hidden_states_key_proj], dim=2)
        value = torch.cat([value, encoder_hidden_states_value_proj], dim=2)

        ## SDPA
        if not USE_ASCEND_NPU:
            hidden_states = torch.nn.functional.scaled_dot_product_attention(query, key, value, dropout_p=0.0, is_causal=False)
        else:
            hidden_states = torch_npu.npu_fusion_attention(
                query, key, value,
                head_num=attn.heads,
                input_layout="BNSD",
                pse=None,
                atten_mask=attention_mask,
                scale=1.0 / math.sqrt(head_dim),
                pre_tockens=65536,
                next_tockens=65536,
                keep_prob=1.0,
                sync=False,
                inner_precise=0,
            )[0]

        ## Unmerge QKV
        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)
        hidden_states, encoder_hidden_states = (
            hidden_states[:, : residual.shape[1]].contiguous(),
            hidden_states[:, residual.shape[1] :].contiguous(),
        )

        # Output projection
        if not attn.context_pre_only:
            encoder_hidden_states = attn.to_add_out(encoder_hidden_states)
        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)

        return hidden_states, encoder_hidden_states


class SD3Transformer3DModel(SD3Transformer2DModel):

    def __init__(
        self,
        in_channels = 16,
        out_channels = 16,
        patch_size = 2,
        num_layers = 38,
        num_attention_heads = 38,
        attention_head_dim = 64,
    ):
        self.inner_dim = num_attention_heads * attention_head_dim
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            patch_size=patch_size,
            num_layers = num_layers,
            attention_head_dim = attention_head_dim,
            num_attention_heads = num_attention_heads,
            caption_projection_dim = self.inner_dim,
            pos_embed_max_size = 192,
            qk_norm='rms_norm',
        )
        self.pos_embed = SD3PatchEmbed(
            patch_size=self.config.patch_size,
            in_channels=self.config.in_channels,
            embed_dim=self.inner_dim,
        )
        if USE_ASCEND_NPU:
            self.set_attn_processor(SD3AttnProcessorNPU())

    def unpatchify(self, x, frame, height, width):
        _, s, c = x.shape
        p = self.config.patch_size
        t, h, w = frame, height // p, width // p
        x = rearrange(x, "b (t h w) (p q c) -> b c t (h p) (w q)", h=h, w=w, p=p, q=p)
        return x

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor = None,
        pooled_projections: torch.Tensor = None,
        timestep: torch.LongTensor = None,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
        return_dict: bool = True,
    ) -> Union[torch.Tensor, Transformer2DModelOutput]:

        if joint_attention_kwargs is not None:
            joint_attention_kwargs = joint_attention_kwargs.copy()
            lora_scale = joint_attention_kwargs.pop("scale", 1.0)
        else:
            lora_scale = 1.0

        if USE_PEFT_BACKEND:
            # weight the lora layers by setting `lora_scale` for each PEFT layer
            scale_lora_layers(self, lora_scale)
        else:
            if joint_attention_kwargs is not None and joint_attention_kwargs.get("scale", None) is not None:
                logger.warning(
                    "Passing `scale` via `joint_attention_kwargs` when not using the PEFT backend is ineffective."
                )

        ## Patchify and add position embedding
        _, _, frame, height, width = hidden_states.shape
        hidden_states = self.pos_embed(hidden_states)

        ## Add timestep and text embedding
        temb = self.time_text_embed(timestep, pooled_projections)
        encoder_hidden_states = self.context_embedder(encoder_hidden_states)

        ## Transformer blocks
        for block in self.transformer_blocks:
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                encoder_hidden_states, hidden_states = self._gradient_checkpointing_func(
                    block,
                    hidden_states,
                    encoder_hidden_states,
                    temb,
                    joint_attention_kwargs,
                )
            else:
                encoder_hidden_states, hidden_states = block(
                    hidden_states=hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    temb=temb,
                    joint_attention_kwargs=joint_attention_kwargs,
                )

        # Unpatchify
        hidden_states = self.norm_out(hidden_states, temb)
        hidden_states = self.proj_out(hidden_states)
        output = self.unpatchify(hidden_states, frame, height, width)

        if USE_PEFT_BACKEND:
            # remove `lora_scale` from each PEFT layer
            unscale_lora_layers(self, lora_scale)

        if not return_dict:
            return (output,)

        return Transformer2DModelOutput(sample=output)
