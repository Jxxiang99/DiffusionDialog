from collections import defaultdict
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union, Dict, Any

import torch
import torch.utils.checkpoint
from torch import nn
import pdb
import os
import math

from transformers.activations import ACT2FN
from transformers.models.bart.modeling_bart import (
    _expand_mask,
    BartAttention,
    BartPretrainedModel,
    shift_tokens_right,
    BartDecoderLayer,
    _make_causal_mask
)
from model.bart_diffusion.configuration_BartDiffusion import BartDiffusionConfig
from model.bart_diffusion.diffusion.nn import timestep_embedding

class SiLU(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

class LatentTransformer(BartPretrainedModel):
    def __init__(self, config: BartDiffusionConfig, time_channel=128):
        super().__init__(config)
        self.layers = nn.ModuleList([BartDecoderLayer(config) for _ in range(config.latent_transformer_layers)])
        # time_channel = config.d_model
        self.time_embed_dim = time_channel
        self.time_embed = nn.Sequential(
            nn.Linear(time_channel, self.time_embed_dim * 4),
            SiLU(),
            nn.Linear(self.time_embed_dim * 4, config.d_model * config.decoder_latent_size),
        )
        self.dropout = config.dropout
        self.latent_size = config.decoder_latent_size
        self.embed_latents = nn.Embedding(config.decoder_latent_size, config.d_model, padding_idx=None)
        self.input_proj1 = nn.Linear(config.d_model, config.decoder_ffn_dim)
        self.input_proj2 = nn.Linear(config.decoder_ffn_dim, config.d_model)
        self.activation_fn = ACT2FN[config.activation_function]
        self.activation_dropout = config.activation_dropout
        self.final_layer_norm = nn.LayerNorm(config.d_model)
        self.post_init()
    
    def init_from_bart(self, pretrained_model_path):
        bart_state_dict = torch.load(os.path.join(pretrained_model_path, "pytorch_model.bin"))
        missing_para = []
        for n, p in self.named_parameters():
            decoder_name = 'model.decoder.' + n
            if decoder_name in bart_state_dict:
                p.data.copy_(bart_state_dict[decoder_name].data)
            else:
                missing_para.append(decoder_name)
        # for n, p in bart_state_dict.items():
        #     print(n)
    
    def prepare_decoder_input_ids_from_labels(self, labels: torch.Tensor):
        return shift_tokens_right(labels, self.config.pad_token_id, self.config.decoder_start_token_id)
    
    def _prepare_decoder_attention_mask(self, attention_mask, input_shape, inputs_embeds, past_key_values_length):
        # create causal mask
        # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
        combined_attention_mask = None
        if input_shape[-1] > 1:
            combined_attention_mask = _make_causal_mask(
                input_shape, inputs_embeds.dtype, past_key_values_length=past_key_values_length
            ).to(inputs_embeds.device)

        if attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            expanded_attn_mask = _expand_mask(attention_mask, inputs_embeds.dtype, tgt_len=input_shape[-1])
            combined_attention_mask = (
                expanded_attn_mask if combined_attention_mask is None else expanded_attn_mask + combined_attention_mask
            )
        return combined_attention_mask
    
    def input_projct(self, latent):
        residual = latent
        latent = self.activation_fn(self.input_proj1(latent))
        latent = nn.functional.dropout(latent, p=self.activation_dropout, training=self.training)
        latent = self.input_proj2(latent)
        latent = nn.functional.dropout(latent, p=self.dropout, training=self.training)
        latent = self.final_layer_norm(latent + residual)
        return latent
    
    def forward(self, x, time, encoder_hidden_state, encoder_attention_mask, encoder_latent):
        time = timestep_embedding(time, self.time_embed_dim)
        time_embed = self.time_embed(time).view(x.shape)

        latent_ids = torch.arange(self.latent_size, dtype=torch.long, device=x.device)
        latent_ids = latent_ids[None, :].expand(x.shape[0], -1)
        latent_embeds = self.embed_latents(latent_ids)
        
        if encoder_latent is None:
            encoder_latent_size = 0
        else:
            encoder_latent_size = encoder_latent.shape[1]

        latent_self_attention_mask = torch.ones((x.shape[0], self.latent_size + encoder_latent_size), dtype=encoder_attention_mask.dtype, device=encoder_attention_mask.device)
        latent_self_attention_mask = _expand_mask(latent_self_attention_mask, x.dtype)
        latent_cross_attention_mask = _expand_mask(encoder_attention_mask, x.dtype, tgt_len=self.latent_size + encoder_latent_size)
        x = self.input_projct(x)
        latent_embed = x + time_embed + latent_embeds
        # latent_embed = self.layernorm_embedding(latent_embed)
        #[bsz, latent_size, embed_dim]
        hidden_state = latent_embed
        if encoder_latent is not None:
            hidden_state = torch.cat([encoder_latent, hidden_state], dim=1)
        for idx, layer in enumerate(self.layers):
            hidden_state = layer(
                hidden_states=hidden_state,
                attention_mask=latent_self_attention_mask,
                encoder_hidden_states=encoder_hidden_state,
                encoder_attention_mask=latent_cross_attention_mask,
                use_cache=False
            )[0]
        hidden_state = hidden_state[:,-self.latent_size:,:]
        return hidden_state