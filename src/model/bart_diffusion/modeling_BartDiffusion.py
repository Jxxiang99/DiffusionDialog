from collections import defaultdict
import copy
from dataclasses import dataclass
import imp
import math
import random
from re import L
import warnings
from typing import List, Optional, Tuple, Union, Dict, Any

import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
import os
import pdb

from transformers.activations import ACT2FN
from transformers.modeling_outputs import (
    BaseModelOutput,
    BaseModelOutputWithPastAndCrossAttentions,
    CausalLMOutputWithCrossAttentions,
    Seq2SeqLMOutput,
    Seq2SeqModelOutput,
    ModelOutput,
    Seq2SeqQuestionAnsweringModelOutput,
    Seq2SeqSequenceClassifierOutput,
)
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import (
    add_code_sample_docstrings,
    add_end_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
    replace_return_docstrings,
)
from transformers.models.bart.modeling_bart import (
    _expand_mask,
    _make_causal_mask,
    BartAttention,
    BartDecoder,
    BartEncoderLayer,
    BartLearnedPositionalEmbedding,
    BartPretrainedModel,
    BartClassificationHead,
    shift_tokens_right,
)
from model.bart_diffusion.configuration_BartDiffusion import BartDiffusionConfig
from model.bart_diffusion.diffusion.gaussian_diffusion import mean_flat
import torch.nn.functional as F
from model.bart_diffusion.modeling_clip import LatentCLIP
from model.bart_diffusion.LatentClassifier import LatentClassifier
from model.bart_diffusion.LatentTransformer import LatentTransformer
from model.bart_diffusion.diffusion.resample import AdaptiveSampler, LossSecondMomentResampler


logger = logging.get_logger(__name__)

@dataclass
class BartDiffusionOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    masked_loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    masked_logits: torch.FloatTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    decoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    decoder_attentions: Optional[Tuple[torch.FloatTensor]] = None
    cross_attentions: Optional[Tuple[torch.FloatTensor]] = None
    encoder_last_hidden_state: Optional[torch.FloatTensor] = None
    encoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    encoder_attentions: Optional[Tuple[torch.FloatTensor]] = None

@dataclass
class BartDiffusionModelOutput(ModelOutput):
    last_hidden_state: torch.FloatTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    decoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    decoder_attentions: Optional[Tuple[torch.FloatTensor]] = None
    cross_attentions: Optional[Tuple[torch.FloatTensor]] = None
    encoder_last_hidden_state: Optional[torch.FloatTensor] = None
    encoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    encoder_attentions: Optional[Tuple[torch.FloatTensor]] = None

@dataclass
class BartDiffusionEncoderOutput(ModelOutput):
    last_hidden_state: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    latent_states: torch.FloatTensor = None

def mean_flat(tensor):
    """
    Take the mean over all non-batch dimensions.
    """
    return tensor.mean(dim=list(range(1, len(tensor.shape))))

def cosin_sim(a, b):
    a = a / (torch.linalg.norm(a, dim=-1, keepdim=True) + 1e-12)
    b = b / (torch.linalg.norm(b, dim=-1, keepdim=True) + 1e-12)
    consim = a @ b.t()
    return consim
    
class BartLatentAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        is_decoder: bool = False,
        bias: bool = True,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads

        if (self.head_dim * num_heads) != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim}"
                f" and `num_heads`: {num_heads})."
            )
        self.scaling = self.head_dim**-0.5
        self.is_decoder = is_decoder

        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(
        self,
        hidden_states: torch.Tensor,
        key_value_states: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        layer_head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        decoder_latent: torch.Tensor = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """Input shape: Batch x Time x Channel"""

        # if key_value_states are provided this layer is used as a cross-attention layer
        # for the decoder
        is_cross_attention = key_value_states is not None

        bsz, tgt_len, _ = hidden_states.size()

        # get query proj
        query_states = self.q_proj(hidden_states) * self.scaling
        # get key, value proj

        if is_cross_attention and past_key_value is not None:
            # reuse k,v, cross_attentions
            key_states = past_key_value[0]
            value_states = past_key_value[1]
        elif is_cross_attention:
            # cross_attentions
            key_states = self._shape(self.k_proj(key_value_states), -1, bsz)
            value_states = self._shape(self.v_proj(key_value_states), -1, bsz)
        elif past_key_value is not None:
            # reuse k, v, self_attention
            key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
            value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)
        else:
            # self_attention
            key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
            value_states = self._shape(self.v_proj(hidden_states), -1, bsz)

        if self.is_decoder:
            # if cross_attention save Tuple(torch.Tensor, torch.Tensor) of all cross attention key/value_states.
            # Further calls to cross_attention layer can then reuse all cross-attention
            # key/value_states (first "if" case)
            # if uni-directional self-attention (decoder) save Tuple(torch.Tensor, torch.Tensor) of
            # all previous decoder key/value_states. Further calls to uni-directional self-attention
            # can concat previous decoder key/value_states to current projected key/value_states (third "elif" case)
            # if encoder bi-directional self-attention `past_key_value` is always `None`
            past_key_value = (key_states, value_states)

        proj_shape = (bsz * self.num_heads, -1, self.head_dim)
        query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
        if decoder_latent is not None:
            latent_key, latent_value = decoder_latent.chunk(chunks=2, dim=-1)
            key_states = torch.cat([latent_key, key_states], dim=2)
            value_states = torch.cat([latent_value, value_states], dim=2)
        key_states = key_states.view(*proj_shape)
        value_states = value_states.view(*proj_shape)

        src_len = key_states.size(1)
        attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))

        if attn_weights.size() != (bsz * self.num_heads, tgt_len, src_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz * self.num_heads, tgt_len, src_len)}, but is"
                f" {attn_weights.size()}"
            )

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, tgt_len, src_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, tgt_len, src_len)}, but is {attention_mask.size()}"
                )
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        attn_weights = nn.functional.softmax(attn_weights, dim=-1)

        if layer_head_mask is not None:
            if layer_head_mask.size() != (self.num_heads,):
                raise ValueError(
                    f"Head mask for a single layer should be of size {(self.num_heads,)}, but is"
                    f" {layer_head_mask.size()}"
                )
            attn_weights = layer_head_mask.view(1, -1, 1, 1) * attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        if output_attentions:
            # this operation is a bit awkward, but it's required to
            # make sure that attn_weights keeps its gradient.
            # In order to do so, attn_weights have to be reshaped
            # twice and have to be reused in the following
            attn_weights_reshaped = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights_reshaped.view(bsz * self.num_heads, tgt_len, src_len)
        else:
            attn_weights_reshaped = None

        attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)

        attn_output = torch.bmm(attn_probs, value_states)

        if attn_output.size() != (bsz * self.num_heads, tgt_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, tgt_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
        attn_output = attn_output.transpose(1, 2)

        # Use the `embed_dim` from the config (stored in the class) rather than `hidden_state` because `attn_output` can be
        # partitioned aross GPUs when using tensor-parallelism.
        attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)

        attn_output = self.out_proj(attn_output)

        return attn_output, attn_weights_reshaped, past_key_value

class BartDiffusionDecoderLayer(nn.Module):
    def __init__(self, config: BartDiffusionConfig) -> None:
        super().__init__()
        self.embed_dim = config.d_model

        self.self_attn = BartLatentAttention(
            embed_dim=self.embed_dim,
            num_heads=config.decoder_attention_heads,
            dropout=config.attention_dropout,
            is_decoder=True,
        )
        self.dropout = config.dropout
        self.activation_fn = ACT2FN[config.activation_function]
        self.activation_dropout = config.activation_dropout

        self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        self.encoder_attn = BartAttention(
            self.embed_dim,
            config.decoder_attention_heads,
            dropout=config.attention_dropout,
            is_decoder=True,
        )
        self.encoder_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        self.fc1 = nn.Linear(self.embed_dim, config.decoder_ffn_dim)
        self.fc2 = nn.Linear(config.decoder_ffn_dim, self.embed_dim)
        self.final_layer_norm = nn.LayerNorm(self.embed_dim)

        self.latent_K = nn.Linear(config.d_model, config.d_model, bias=True)
        self.latent_V = nn.Linear(config.d_model, config.d_model, bias=True)
        self.num_head = config.decoder_attention_heads

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        layer_head_mask: Optional[torch.Tensor] = None,
        cross_attn_layer_head_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = True,
        decoder_latent:Optional[torch.Tensor] = None,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`): attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            encoder_hidden_states (`torch.FloatTensor`):
                cross attention input to the layer of shape `(batch, seq_len, embed_dim)`
            encoder_attention_mask (`torch.FloatTensor`): encoder attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            layer_head_mask (`torch.FloatTensor`): mask for attention heads in a given layer of size
                `(encoder_attention_heads,)`.
            cross_attn_layer_head_mask (`torch.FloatTensor`): mask for cross-attention heads in a given layer of
                size `(decoder_attention_heads,)`.
            past_key_value (`Tuple(torch.FloatTensor)`): cached past key and value projection states
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
        """
        residual = hidden_states
        # Self Attention
        # decoder uni-directional self-attention cached key/values tuple is at positions 1,2
        self_attn_past_key_value = past_key_value[:2] if past_key_value is not None else None
        # add present self-attn cache to positions 1,2 of present_key_value tuple
        if decoder_latent is not None:
            bsz, latent_size = decoder_latent.shape[:2]
            latent_key = self.latent_K(decoder_latent)
            latent_value = self.latent_V(decoder_latent)
            latent_key = latent_key.view(bsz, latent_size, self.num_head, -1).transpose(1, 2).contiguous()
            latent_value = latent_value.view(bsz, latent_size, self.num_head, -1).transpose(1, 2).contiguous()
            decoder_latent = torch.cat([latent_key, latent_value], dim=-1)

        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            past_key_value=self_attn_past_key_value,
            attention_mask=attention_mask,
            layer_head_mask=layer_head_mask,
            output_attentions=output_attentions,
            decoder_latent=decoder_latent,
        )
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)

        # # Cross-Attention Block
        cross_attn_present_key_value = None
        cross_attn_weights = None
        if encoder_hidden_states is not None:
            residual = hidden_states

            # cross_attn cached key/values tuple is at positions 3,4 of present_key_value tuple
            cross_attn_past_key_value = past_key_value[-2:] if past_key_value is not None else None
            if cross_attn_past_key_value == (None, None):
                cross_attn_past_key_value = None
            hidden_states, cross_attn_weights, cross_attn_present_key_value = self.encoder_attn(
                hidden_states=hidden_states,
                key_value_states=encoder_hidden_states,
                attention_mask=encoder_attention_mask,
                layer_head_mask=cross_attn_layer_head_mask,
                past_key_value=cross_attn_past_key_value,
                output_attentions=output_attentions,
            )
            hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
            hidden_states = residual + hidden_states
            hidden_states = self.encoder_attn_layer_norm(hidden_states)

            # add cross-attn to positions 3,4 of present_key_value tuple
            present_key_value = present_key_value + cross_attn_present_key_value

        # Fully Connected
        residual = hidden_states
        hidden_states = self.activation_fn(self.fc1(hidden_states))
        hidden_states = nn.functional.dropout(hidden_states, p=self.activation_dropout, training=self.training)
        hidden_states = self.fc2(hidden_states)
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states
        hidden_states = self.final_layer_norm(hidden_states)

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights, cross_attn_weights)

        if use_cache:
            outputs += (present_key_value,)

        return outputs

class BartDiffusionDecoder(BartPretrainedModel):
    """
    Transformer decoder consisting of *config.decoder_layers* layers. Each layer is a [`BartDecoderLayer`]

    Args:
        config: BartConfig
        embed_tokens (nn.Embedding): output embedding
    """

    def __init__(self, config: BartDiffusionConfig, embed_tokens: Optional[nn.Embedding] = None):
        super().__init__(config)
        self.dropout = config.dropout
        self.layerdrop = config.decoder_layerdrop
        self.padding_idx = config.pad_token_id
        self.max_target_positions = config.max_position_embeddings
        self.embed_scale = math.sqrt(config.d_model) if config.scale_embedding else 1.0
        self.num_head = config.decoder_attention_heads
        self.embed_dim = config.d_model

        if embed_tokens is not None:
            self.embed_tokens = embed_tokens
        else:
            self.embed_tokens = nn.Embedding(config.vocab_size, config.d_model, self.padding_idx)

        self.embed_positions = BartLearnedPositionalEmbedding(
            config.max_position_embeddings,
            config.d_model,
        )
        self.layers = nn.ModuleList([BartDiffusionDecoderLayer(config) for _ in range(config.decoder_layers)])
        self.layernorm_embedding = nn.LayerNorm(config.d_model)

        self.gradient_checkpointing = False
        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

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

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        decoder_latent:Optional[torch.Tensor] = None,
    ) -> Union[Tuple, BaseModelOutputWithPastAndCrossAttentions]:

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both decoder_input_ids and decoder_inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either decoder_input_ids or decoder_inputs_embeds")

        # past_key_values_length
        past_key_values_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids) * self.embed_scale

        # embed positions
        positions = self.embed_positions(input_shape, past_key_values_length)
        hidden_states = inputs_embeds + positions

        if decoder_latent is None:
            attention_mask = self._prepare_decoder_attention_mask(
                attention_mask, input_shape, inputs_embeds, past_key_values_length
            )
        else:
            attention_mask = self._prepare_decoder_attention_mask(
                attention_mask, input_shape, inputs_embeds, past_key_values_length + decoder_latent.shape[1]
            )

        # expand encoder attention mask
        if encoder_hidden_states is not None and encoder_attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            encoder_attention_mask = _expand_mask(encoder_attention_mask, inputs_embeds.dtype, tgt_len=input_shape[-1])

        hidden_states = self.layernorm_embedding(hidden_states)
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        all_cross_attentions = () if (output_attentions and encoder_hidden_states is not None) else None
        next_decoder_cache = () if use_cache else None

        # check if head_mask/cross_attn_head_mask has a correct number of layers specified if desired
        for attn_mask, mask_name in zip([head_mask, cross_attn_head_mask], ["head_mask", "cross_attn_head_mask"]):
            if attn_mask is not None:
                if attn_mask.size()[0] != (len(self.layers)):
                    raise ValueError(
                        f"The `{mask_name}` should be specified for {len(self.layers)} layers, but it is for"
                        f" {head_mask.size()[0]}."
                    )

        for idx, decoder_layer in enumerate(self.layers):
            # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            dropout_probability = random.uniform(0, 1)
            if self.training and (dropout_probability < self.layerdrop):
                continue

            past_key_value = past_key_values[idx] if past_key_values is not None else None

            if self.gradient_checkpointing and self.training:

                if use_cache:
                    logger.warning(
                        "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                    )
                    use_cache = False

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        # None for past_key_value
                        return module(*inputs, output_attentions, use_cache)

                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(decoder_layer),
                    hidden_states,
                    attention_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    head_mask[idx] if head_mask is not None else None,
                    cross_attn_head_mask[idx] if cross_attn_head_mask is not None else None,
                    None,
                )
            else:

                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_attention_mask,
                    layer_head_mask=(head_mask[idx] if head_mask is not None else None),
                    cross_attn_layer_head_mask=(
                        cross_attn_head_mask[idx] if cross_attn_head_mask is not None else None
                    ),
                    past_key_value=past_key_value,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    decoder_latent=decoder_latent,
                )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache += (layer_outputs[3 if output_attentions else 1],)

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

                if encoder_hidden_states is not None:
                    all_cross_attentions += (layer_outputs[2],)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None
        if not return_dict:
            return tuple(
                v
                for v in [hidden_states, next_cache, all_hidden_states, all_self_attns, all_cross_attentions]
                if v is not None
            )
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
            cross_attentions=all_cross_attentions,
        )

class BartDiffusionEncoder(BartPretrainedModel):
    """
    Transformer encoder consisting of *config.encoder_layers* self attention layers. Each layer is a
    [`BartEncoderLayer`].

    Args:
        config: BartConfig
        embed_tokens (nn.Embedding): output embedding
    """

    def __init__(self, config: BartDiffusionConfig, embed_tokens: Optional[nn.Embedding] = None):
        super().__init__(config)

        self.dropout = config.dropout
        self.layerdrop = config.encoder_layerdrop

        embed_dim = config.d_model
        self.padding_idx = config.pad_token_id
        self.max_source_positions = config.max_position_embeddings
        self.embed_scale = math.sqrt(embed_dim) if config.scale_embedding else 1.0
        self.embed_dim = config.d_model
        self.with_clip_loss = config.loss_dict['with_clip_loss']

        if embed_tokens is not None:
            self.embed_tokens = embed_tokens
        else:
            self.embed_tokens = nn.Embedding(config.vocab_size, embed_dim, self.padding_idx)

        self.embed_positions = BartLearnedPositionalEmbedding(
            config.max_position_embeddings,
            embed_dim,
        )
        self.layers = nn.ModuleList([BartEncoderLayer(config) for _ in range(config.encoder_layers)])
        self.layernorm_embedding = nn.LayerNorm(embed_dim)
        self.role_embedding = nn.Embedding(config.max_role_embedding, embed_dim, padding_idx=None)
        self.turn_embedding = nn.Embedding(config.max_turn_embedding, embed_dim, padding_idx=None)

        self.gradient_checkpointing = False
        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value
    
    def re_parameterize(self, mu, log_var):
        std = log_var.mul(.5).exp()
        eps = torch.zeros_like(std).normal_()
        return mu + torch.mul(eps, std)

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        turn_ids: torch.LongTensor = None,
        role_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        latent_embed = None
    ) -> Union[Tuple, BaseModelOutput]:

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids) * self.embed_scale

        embed_pos = self.embed_positions(input_shape)
        embed_role = self.role_embedding(role_ids)
        embed_turn = self.turn_embedding(turn_ids)
#  + embed_role + embed_turn
        hidden_states = inputs_embeds + embed_pos + embed_role + embed_turn

        if latent_embed is not None:
            hidden_states = torch.cat([latent_embed, hidden_states], dim=1)
            masks = torch.ones(latent_embed.shape[:2], dtype=attention_mask.dtype, device=attention_mask.device)
            attention_mask = torch.cat([masks, attention_mask], dim=1)

        hidden_states = self.layernorm_embedding(hidden_states)
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)

        # expand attention_mask
        if attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            attention_mask = _expand_mask(attention_mask, inputs_embeds.dtype)
            if latent_embed is not None:
                attention_mask[:,:,latent_embed.shape[1]:,:latent_embed.shape[1]] = torch.finfo(inputs_embeds.dtype).min
            # #让encoder latent只看到当前问句
            # if self.latent_size > 0:
            #     #[bsz, 1, seq_len]
            #     turn_mask = turn_ids[:,None,None,:].repeat(1, 1, self.latent_size, 1)
            #     attention_mask[:,:,:self.latent_size,self.latent_size:][turn_mask != 2] = torch.finfo(inputs_embeds.dtype).min



        encoder_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None

        # check if head_mask has a correct number of layers specified if desired
        if head_mask is not None:
            if head_mask.size()[0] != (len(self.layers)):
                raise ValueError(
                    f"The head_mask should be specified for {len(self.layers)} layers, but it is for"
                    f" {head_mask.size()[0]}."
                )

        for idx, encoder_layer in enumerate(self.layers):
            if output_hidden_states:
                encoder_states = encoder_states + (hidden_states,)
            # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
            dropout_probability = random.uniform(0, 1)
            if self.training and (dropout_probability < self.layerdrop):  # skip the layer
                layer_outputs = (None, None)
            else:
                if self.gradient_checkpointing and self.training:

                    def create_custom_forward(module):
                        def custom_forward(*inputs):
                            return module(*inputs, output_attentions)

                        return custom_forward

                    layer_outputs = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(encoder_layer),
                        hidden_states,
                        attention_mask,
                        (head_mask[idx] if head_mask is not None else None),
                    )
                else:
                    layer_outputs = encoder_layer(
                        hidden_states,
                        attention_mask,
                        layer_head_mask=(head_mask[idx] if head_mask is not None else None),
                        output_attentions=output_attentions,
                    )

                hidden_states = layer_outputs[0]

            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)

        if output_hidden_states:
            encoder_states = encoder_states + (hidden_states,)

        if latent_embed is None:
            latent_states = None
        else:
            latent_states = hidden_states[:,:latent_embed.shape[1],:]
            hidden_states = hidden_states[:,latent_embed.shape[1]:,:]

        if not return_dict:
            return tuple(v for v in [hidden_states, encoder_states, all_attentions] if v is not None)
        
        return BartDiffusionEncoderOutput(
            last_hidden_state=hidden_states, 
            hidden_states=encoder_states, 
            attentions=all_attentions,
            latent_states=latent_states,
        )

class BartDiffusionLatentLayer(nn.Module):
    def __init__(self, config: BartDiffusionConfig) -> None:
        super().__init__()
        self.embed_dim = config.d_model

        self.self_attn = BartAttention(
            embed_dim=self.embed_dim,
            num_heads=config.decoder_attention_heads,
            dropout=config.attention_dropout,
            is_decoder=True,
        )
        self.dropout = config.dropout
        self.activation_fn = ACT2FN[config.activation_function]
        self.activation_dropout = config.activation_dropout

        self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        self.encoder_attn = BartAttention(
            self.embed_dim,
            config.decoder_attention_heads,
            dropout=config.attention_dropout,
            is_decoder=True,
        )
        self.encoder_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        self.fc1 = nn.Linear(self.embed_dim, config.decoder_ffn_dim)
        self.fc2 = nn.Linear(config.decoder_ffn_dim, self.embed_dim)
        self.final_layer_norm = nn.LayerNorm(self.embed_dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:

        # # Cross-Attention Block
        residual = hidden_states
        hidden_states, _, _ = self.encoder_attn(
            hidden_states=hidden_states,
            key_value_states=encoder_hidden_states,
        )
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states
        hidden_states = self.encoder_attn_layer_norm(hidden_states)

        residual = hidden_states
        hidden_states, _, _ = self.self_attn(
            hidden_states=hidden_states,
        )
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)

        # Fully Connected
        residual = hidden_states
        hidden_states = self.activation_fn(self.fc1(hidden_states))
        hidden_states = nn.functional.dropout(hidden_states, p=self.activation_dropout, training=self.training)
        hidden_states = self.fc2(hidden_states)
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states
        hidden_states = self.final_layer_norm(hidden_states)

        outputs = (hidden_states,)

        return outputs



class BartDiffusionModel(BartPretrainedModel):
    def __init__(self, config: BartDiffusionConfig):
        super().__init__(config)
        padding_idx, vocab_size = config.pad_token_id, config.vocab_size
        self.embed_dim = config.d_model
        self.shared = nn.Embedding(vocab_size, config.d_model, padding_idx)
        self.encoder = BartDiffusionEncoder(config, self.shared)
        self.decoder = BartDiffusionDecoder(config, self.shared)
        self.decoder_layers = config.decoder_layers
        self.with_clip_loss = config.loss_dict['with_clip_loss']
        self.with_rc_loss = config.loss_dict['with_rc_loss']
        self.with_gold_rc_loss = config.loss_dict['with_gold_rc_loss']
        self.with_sim_loss = config.loss_dict['with_sim_loss']
        self.with_classifier_loss = config.loss_dict['with_classifier_loss']
        self.with_noise_loss = config.loss_dict['with_noise_loss']
        self.with_bow_loss = config.loss_dict['with_bow_loss']
        self.with_diffusion_rc_loss = config.loss_dict['with_diffusion_rc_loss']
        self.with_tT_loss = config.loss_dict['with_tT_loss']
        if self.with_classifier_loss:
            self.latent_classifier = LatentClassifier(config)
        if self.with_clip_loss:
            self.clip = LatentCLIP(config)
        if self.with_bow_loss:
            self.bow_fn = nn.Linear(config.d_model, config.d_model)
        self.LatentTransformer = LatentTransformer(config)
        self.encoder_latent_size = config.encoder_latent_size
        self.decoder_latent_size = config.decoder_latent_size
        if self.encoder_latent_size > 0:
            self.embed_encoder_latent = nn.Embedding(self.encoder_latent_size, config.d_model, padding_idx=None)
        self.embed_decoder_latent = nn.Embedding(self.decoder_latent_size, config.d_model, padding_idx=None)
        self.activation_fn = ACT2FN[config.activation_function]
        self.activation_dropout = config.activation_dropout
        self.dropout = config.dropout
        self.final_layer_norm = nn.LayerNorm(config.d_model)
        self.latent_fn1 = nn.Linear(config.d_model, config.decoder_ffn_dim)
        self.latent_fn2 = nn.Linear(config.decoder_ffn_dim, config.d_model)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.shared

    def set_input_embeddings(self, value):
        self.shared = value
        self.encoder.embed_tokens = self.shared
        self.decoder.embed_tokens = self.shared

    def get_encoder(self):
        return self.encoder

    def get_decoder(self):
        return self.decoder
    
    def latent_proj(self, latent):
        residual = latent
        latent = self.activation_fn(self.latent_fn1(latent))
        latent = nn.functional.dropout(latent, p=self.activation_dropout, training=self.training)
        latent = self.latent_fn2(latent)
        latent = nn.functional.dropout(latent, p=self.dropout, training=self.training)
        latent = self.final_layer_norm(latent + residual)
        return latent
    
    def get_input_latent(self, decoder_latent):
        bsz, latent_size = decoder_latent.shape[:2]
        # latent = torch.zeros(bsz, decoder_latent.shape[-1]).to(decoder_latent.device)
        # for i in range(bsz):
        #     latent[i] = decoder_latent[i][-i-1]
        latent = decoder_latent[:,-self.model_latent_size:,:]
        # latent = latent.unsqueeze(1)
        return latent
    
    def get_sim_loss(self, latent):
        latent_1 = latent.unsqueeze(0)
        latent_2 = latent.unsqueeze(1)
        sim_loss = ((latent_1 - latent_2) ** 2).mean()
        return -1.0 * sim_loss
    
    def get_bow_loss(self, latent, labels, decoder_attention_mask, t_weights):
        latent_featrue = self.bow_fn(latent)
        shared = self.shared.weight.detach()
        weights = decoder_attention_mask.sum(-1).to(latent_featrue.dtype)
        _shape = decoder_attention_mask.shape[:2]
        latent_logits = F.linear(latent_featrue, shared)
        latent_softmax = nn.functional.softmax(latent_logits, dim=1)
        latent_logits = (latent_softmax * latent_logits).sum(1)
        # latent_logits = latent_logits.mean(1)
        latent_logits = latent_logits.repeat(labels.shape[1], 1)
        loss_fc = nn.CrossEntropyLoss(reduction='none')
        bow_loss = loss_fc(latent_logits, labels.transpose(1, 0).contiguous().view(-1))
        bow_loss = bow_loss.view(_shape[1], _shape[0]).transpose(1, 0)
        bow_loss = bow_loss.sum(-1) / weights * t_weights
        bow_loss = bow_loss.mean()
        return bow_loss

    def get_loss_and_logits(
        self,
        input_ids: torch.LongTensor = None,
        role_ids: torch.LongTensor = None,
        turn_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.LongTensor] = None,
        labels: torch.LongTensor = None,
        t_weights=None,
        t=None,
        diffusion=None,
        gold_rc_loss_weight=0.0
    ):
        loss = defaultdict()
        bsz, _ = input_ids.shape
        
        if self.encoder_latent_size > 0:
            latent_ids = torch.arange(self.encoder_latent_size, dtype=torch.long, device=input_ids.device)
            latent_ids = latent_ids[None, :].expand(input_ids.shape[0], -1)
            encoder_latent_embed = self.embed_encoder_latent(latent_ids)
        else:
            encoder_latent_embed = None

        encoder_outputs = self.encoder(
                input_ids=input_ids,
                role_ids=role_ids,
                turn_ids=turn_ids,
                attention_mask=attention_mask,
                latent_embed=encoder_latent_embed
            )
        
        latent_ids = torch.arange(self.decoder_latent_size, dtype=torch.long, device=input_ids.device)
        latent_ids = latent_ids[None, :].expand(input_ids.shape[0], -1)
        decoder_latent_embed = self.embed_decoder_latent(latent_ids)
        decoder_turn_ids = torch.ones_like(decoder_input_ids, device=decoder_input_ids.device)
        decoder_role_ids = torch.ones_like(decoder_input_ids, device=decoder_input_ids.device)
        decoder_role_ids = decoder_role_ids * 2

        response_outputs = self.encoder(
            input_ids=decoder_input_ids,
            role_ids=decoder_role_ids,
            turn_ids=decoder_turn_ids,
            attention_mask=decoder_attention_mask,
            latent_embed=decoder_latent_embed
        )

        decoder_latent = response_outputs.latent_states
        decoder_latent = self.latent_proj(decoder_latent)
        
        model_kwargs={'encoder_hidden_state':encoder_outputs.last_hidden_state, 'encoder_attention_mask':attention_mask, 'encoder_latent': encoder_outputs.latent_states}
        diffusion_out, pred_x_start, x_t = diffusion.training_losses(self.LatentTransformer, decoder_latent, t, model_kwargs=model_kwargs)
        diffusion_loss = diffusion_out["mse"]
        # diffusion_loss = diffusion_loss.mean()

        if labels is not None:

            if self.with_diffusion_rc_loss:
                #进行梯度截断，decoder只负责重构，不给latent回传梯度
                pred_latent = pred_x_start.detach()
                decoder_outputs = self.decoder(
                    input_ids=decoder_input_ids,
                    encoder_hidden_states=encoder_outputs.last_hidden_state,
                    encoder_attention_mask=attention_mask,
                    decoder_latent=pred_latent,
                )
                diffusion_hidden_state = decoder_outputs.last_hidden_state
            else:
                diffusion_hidden_state = None
            
            if self.with_gold_rc_loss and gold_rc_loss_weight > 0.0:
                gold_latent = decoder_latent.detach()
                decoder_outputs = self.decoder(
                    input_ids=decoder_input_ids,
                    encoder_hidden_states=encoder_outputs.last_hidden_state,
                    encoder_attention_mask=attention_mask,
                    decoder_latent=gold_latent,
                )
                gold_hidden_state = decoder_outputs.last_hidden_state
            else:
                gold_hidden_state = None
            
            loss['diffusion_loss'] = diffusion_loss
            if self.with_tT_loss:
                loss['tT_loss'] = diffusion_out['tT_loss'].mean()
            if self.with_sim_loss:
                sim_loss1 = self.get_sim_loss(decoder_latent)
                sim_loss2 = self.get_sim_loss(pred_latent)
                loss['sim_loss'] = sim_loss1 + sim_loss2
            if self.with_clip_loss:
                encoder_latent = encoder_outputs.latent_states
                clip_loss = self.clip(encoder_latent.view(bsz, -1), decoder_latent.view(bsz, -1))
                loss['clip_loss'] = clip_loss
            if self.with_classifier_loss:
                classifier_loss = self.latent_classifier(gold_latent.detach(), pred_latent.detach())
                loss['classifier_loss'] = classifier_loss
            if self.with_noise_loss:
                mask_tmp = labels.masked_fill(~labels.eq(-100), 1.0)
                mask = mask_tmp.masked_fill(mask_tmp.eq(-100), 0.0)
                weight = mask.sum(-1).to(dtype=attention_mask.dtype)
                _shape = diffusion_hidden_state.shape[:2]
                time_step = diffusion.num_timesteps
                mask_position = t * weight / time_step
                mask_position = torch.ceil(mask_position).long()
                pad_id = 1
                mask_cond = torch.arange(_shape[1], device=mask.device)
                mask_position = (weight - mask_position).unsqueeze(1)
                noise_input_ids = decoder_input_ids.masked_fill(mask_cond > mask_position, pad_id)
                noise_attention_mask = decoder_attention_mask.masked_fill(mask_cond > mask_position, 0)
                noise_latent, _ = self.decoder(
                    input_ids=noise_input_ids,
                    attention_mask=noise_attention_mask,
                    encoder_hidden_states=encoder_outputs.last_hidden_state,
                    encoder_attention_mask=attention_mask,
                    )
                loss['noise_loss'] = ((x_t - noise_latent) ** 2).mean()
            if self.with_bow_loss:
                # loss['bow_loss'] = 0.5 * self.get_bow_loss(pred_latent, labels, decoder_attention_mask) + 0.5 * self.get_bow_loss(decoder_latent, labels, decoder_attention_mask)
                loss['bow_loss'] = self.get_bow_loss(decoder_latent, labels, decoder_attention_mask, t_weights)
        else:
            latent = pred_x_start.detach()
            decoder_outputs = self.decoder(
                input_ids=decoder_input_ids,
                encoder_hidden_states=encoder_outputs.last_hidden_state,
                encoder_attention_mask=attention_mask,
                decoder_latent=latent,
            )
            diffusion_hidden_state = decoder_outputs.last_hidden_state
            loss = None
            gold_hidden_state = None
        return loss, diffusion_hidden_state, gold_hidden_state
    
    def check_model(
        self,
        input_ids: torch.LongTensor = None,
        role_ids: torch.LongTensor = None,
        turn_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.LongTensor] = None,
        labels: torch.LongTensor = None,
        t=None,
        diffusion=None,
        rc_loss_weight=0.0
    ):
        loss = defaultdict()
        encoder_outputs = self.encoder(
                input_ids=input_ids,
                role_ids=role_ids,
                turn_ids=turn_ids,
                attention_mask=attention_mask,
            )
        decoder_latent, hidden_state_wo_latent = self.decoder(
                input_ids=decoder_input_ids,
                attention_mask=decoder_attention_mask,
                encoder_hidden_states=encoder_outputs.last_hidden_state,
                encoder_attention_mask=attention_mask,
            )
        
        decoder_latent = self.latent_proj(decoder_latent)

        model_kwargs={'encoder_hidden_state':encoder_outputs.last_hidden_state, 'encoder_attention_mask':attention_mask, 'encoder_latent': encoder_outputs.latent_states}
        diffusion_out, pred_x_start, x_t = diffusion.training_losses(self.LatentTransformer, decoder_latent, t, model_kwargs=model_kwargs)
        # diffusion_loss = (diffusion_out["mse"] * weights).mean()
        # noise = ((x_t - decoder_latent) ** 2).mean()
        # movement = mean_flat((x_t - decoder_latent) ** 2) - diffusion_out["mse"]
        movement = diffusion_out["mse"]
        loss["movement"] = movement


        # gold_latent = self.get_input_latent(decoder_latent)
        # decoder_outputs = self.decoder(
        #     input_ids=decoder_input_ids,
        #     encoder_hidden_states=encoder_outputs.last_hidden_state,
        #     encoder_attention_mask=attention_mask,
        #     decoder_latent=gold_latent,
        # )
        hidden_state = hidden_state_wo_latent

        pred_latent = pred_x_start
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            encoder_hidden_states=encoder_outputs.last_hidden_state,
            encoder_attention_mask=attention_mask,
            decoder_latent=pred_latent,
        )
        diffusion_hidden_state = decoder_outputs.last_hidden_state
        
        return loss, hidden_state, diffusion_hidden_state
    
    def get_generation_input(
        self,
        input_ids: torch.LongTensor = None,
        role_ids: torch.LongTensor = None,
        turn_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        decoder_input_ids: torch.LongTensor = None,
        decoder_attention_mask: torch.LongTensor = None,
        diffusion = None,
        guidance_scale=0.0
    ):
        encoder_outputs = self.encoder(
            input_ids=input_ids,
            role_ids=role_ids,
            turn_ids=turn_ids,
            attention_mask=attention_mask,
        )

        if decoder_input_ids is None:
            bsz = input_ids.shape[0]
            model_kwargs={'encoder_hidden_state':encoder_outputs.last_hidden_state, 'encoder_attention_mask':attention_mask, 'encoder_latent': encoder_outputs.latent_states}
            # cond_fn = self.clip.cond_fn(encoder_outputs.encoder_latent.view(bsz, -1), guidance_scale)
            cond_fn = None
            #此处不加torch.no_grad()会爆显存，加速用ddim_sample_loop,正常用p_sample_loop,带概率不加噪的用MLM_sample_loop
            with torch.no_grad():
                decoder_latent = diffusion.ddim_sample_loop(
                    self.LatentTransformer,
                    (input_ids.shape[0], self.decoder_latent_size, self.embed_dim),
                    device=input_ids.device,
                    progress=False,
                    model_kwargs=model_kwargs,
                    cond_fn=cond_fn
                )
        else:
            decoder_latent, hidden_state_wo_latent = self.decoder(
                input_ids=decoder_input_ids,
                attention_mask=decoder_attention_mask,
                encoder_hidden_states=encoder_outputs.last_hidden_state,
                encoder_attention_mask=attention_mask,
            )
            decoder_latent = self.latent_proj(decoder_latent)

            # model_kwargs={'encoder_hidden_state':encoder_outputs.last_hidden_state, 'encoder_attention_mask':attention_mask, 'encoder_latent': encoder_outputs.encoder_latent}
            # cond_fn = None
            # t = torch.tensor([1], device=input_ids.device)
            # t = t.long()
            # decoder_latent = diffusion.ddim_sample_check(
            #     self.LatentTransformer,
            #     (input_ids.shape[0], self.decoder.latent_size, self.embed_dim),
            #     device=input_ids.device,
            #     progress=False,
            #     model_kwargs=model_kwargs,
            #     cond_fn=cond_fn,
            #     t=t,
            #     x_start=decoder_latent,
            # )

        # print(cosin_sim(decoder_latent.view(decoder_latent.shape[0], -1), decoder_latent.view(decoder_latent.shape[0], -1)))
        # logger.info(cosin_sim(decoder_latent.view(decoder_latent.shape[0], -1), decoder_latent.view(decoder_latent.shape[0], -1)))
        # logger.info(cosin_sim(decoder_latent[0], decoder_latent[0]))
        # for i in range(decoder_latent.shape[0]):
            # print(cosin_sim(decoder_latent[i], decoder_latent[i]))
        return encoder_outputs, decoder_latent


    def forward(
        self,
        input_ids: torch.LongTensor = None,
        role_ids: torch.LongTensor = None,
        turn_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        decoder_head_mask: Optional[torch.Tensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[List[torch.FloatTensor]] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        decoder_latent:Optional[torch.Tensor] = None,
    ) -> Union[Tuple, Seq2SeqModelOutput]:

        # different to other models, Bart automatically creates decoder_input_ids from
        # input_ids if no decoder_input_ids are provided
        if decoder_input_ids is None and decoder_inputs_embeds is None:
            if input_ids is None:
                raise ValueError(
                    "If no `decoder_input_ids` or `decoder_inputs_embeds` are "
                    "passed, `input_ids` cannot be `None`. Please pass either "
                    "`input_ids` or `decoder_input_ids` or `decoder_inputs_embeds`."
                )

            decoder_input_ids = shift_tokens_right(
                input_ids, self.config.pad_token_id, self.config.decoder_start_token_id
            )

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if encoder_outputs is None:
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                role_ids=role_ids,
                turn_ids=turn_ids,
                attention_mask=attention_mask,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )

        # decoder outputs consists of (dec_features, past_key_value, dec_hidden, dec_attn)
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=encoder_outputs.last_hidden_state,
            encoder_attention_mask=attention_mask,
            head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            past_key_values=past_key_values,
            inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            decoder_latent=decoder_latent,
        )

        if not return_dict:
            return decoder_outputs + encoder_outputs

        return BartDiffusionModelOutput(
            last_hidden_state=decoder_outputs.last_hidden_state,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )

class BartDiffusionForConditionalGeneration(BartPretrainedModel):
    base_model_prefix = "model"
    _keys_to_ignore_on_load_missing = [r"final_logits_bias", r"lm_head.weight"]

    def __init__(self, config: BartDiffusionConfig):
        super().__init__(config)
        self.model = BartDiffusionModel(config)
        self.register_buffer("final_logits_bias", torch.zeros((1, self.model.shared.num_embeddings)))
        self.lm_head = nn.Linear(config.d_model, self.model.shared.num_embeddings, bias=False)

        # Initialize weights and apply final processing
        self.post_init()
    
    def init_from_bart(self, pretrained_model_path):
        bart_state_dict = torch.load(os.path.join(pretrained_model_path, "pytorch_model.bin"))
        missing_para = []
        for n, p in self.named_parameters():
            if "LatentTrans" in n:
                continue
            n = n[6:]
            if n in bart_state_dict:
                p.data.copy_(bart_state_dict[n].data)
            else:
                missing_para.append(n)
        # for n, p in self.model.LatentTransformer.named_parameters():
        #     # n = n[6:]
        #     bart_name = 'decoder.' + n
        #     if bart_name in bart_state_dict:
        #         p.data.copy_(bart_state_dict[bart_name].data)
        #     else:
        #         missing_para.append(n)

                # logger.info(n)
        logger.info('parameters not loading:')
        for para in missing_para:
            print(para)
        del bart_state_dict

    def get_encoder(self):
        return self.model.get_encoder()

    def get_decoder(self):
        return self.model.get_decoder()

    def resize_token_embeddings(self, new_num_tokens: int) -> nn.Embedding:
        new_embeddings = super().resize_token_embeddings(new_num_tokens)
        self._resize_final_logits_bias(new_num_tokens)
        return new_embeddings

    def _resize_final_logits_bias(self, new_num_tokens: int) -> None:
        old_num_tokens = self.final_logits_bias.shape[-1]
        if new_num_tokens <= old_num_tokens:
            new_bias = self.final_logits_bias[:, :new_num_tokens]
        else:
            extra_bias = torch.zeros((1, new_num_tokens - old_num_tokens), device=self.final_logits_bias.device)
            new_bias = torch.cat([self.final_logits_bias, extra_bias], dim=1)
        self.register_buffer("final_logits_bias", new_bias)

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings
    
    def get_loss(
        self,
        input_ids: torch.LongTensor = None,
        role_ids: torch.LongTensor = None,
        turn_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.LongTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        schedule_sampler=None,
        diffusion=None,
        gold_rc_loss_weight=0.0,
    ):
        t, t_weights = schedule_sampler.sample(input_ids.shape[0], input_ids.device)
        loss, diffusion_hidden_states, gold_hidden_state = self.model.get_loss_and_logits(
            input_ids=input_ids,
            role_ids=role_ids,
            turn_ids=turn_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            labels=labels,
            t_weights=t_weights,
            t=t,
            diffusion=diffusion,
            gold_rc_loss_weight=gold_rc_loss_weight,
        )
        
        if labels is not None:
            #方案1：按照token采取等待策略
            if isinstance(schedule_sampler, LossSecondMomentResampler):
                schedule_sampler.update_with_all_losses(t, loss['diffusion_loss'])
            loss['diffusion_loss'] = (loss['diffusion_loss'] * t_weights).mean()

            mask_tmp = labels.masked_fill(~labels.eq(-100), 1.0)
            mask = mask_tmp.masked_fill(mask_tmp.eq(-100), 0.0)
            weight = mask.sum(-1).to(dtype=input_ids.dtype)
            _shape = labels.shape[:2]

            loss_fct = CrossEntropyLoss(reduction='none')
            if diffusion_hidden_states is not None:
                diffusion_logits = self.lm_head(diffusion_hidden_states) + self.final_logits_bias
                diffusion_rc_loss = loss_fct(diffusion_logits.view(-1, self.config.vocab_size), labels.view(-1))
                diffusion_rc_loss = diffusion_rc_loss.view(_shape[0], _shape[1])
                diffusion_rc_loss = diffusion_rc_loss.sum(-1) / weight
                diffusion_rc_loss = diffusion_rc_loss * t_weights
                diffusion_rc_loss = diffusion_rc_loss.mean()
                loss['diffusion_rc_loss'] = diffusion_rc_loss

            if gold_hidden_state is not None:
                logits = self.lm_head(gold_hidden_state) + self.final_logits_bias
                gold_rc_loss = loss_fct(logits.view(-1, self.config.vocab_size), labels.view(-1))
                gold_rc_loss = gold_rc_loss.view(_shape[0], _shape[1])
                gold_rc_loss = gold_rc_loss.sum(-1) / weight * t_weights
                gold_rc_loss = gold_rc_loss.mean()
                loss['gold_rc_loss'] = gold_rc_loss

            #方案2：直接按照batch采取等待策略，一个batch内
            # loss_fct = CrossEntropyLoss()
            # if hidden_state is not None:
            #     logits = self.lm_head(hidden_state) + self.final_logits_bias
            #     rc_loss = loss_fct(logits.view(-1, self.config.vocab_size), labels.view(-1))
            #     loss['rc_loss'] = rc_loss
            # if gold_hidden_state is not None:
            #     gold_logits = self.lm_head(gold_hidden_state) + self.final_logits_bias
            #     gold_rc_loss = loss_fct(gold_logits.view(-1, self.config.vocab_size), labels.view(-1))
            #     loss['gold_rc_loss'] = gold_rc_loss
            # diffusion_rc_loss = loss_fct(diffusion_logits.view(-1, self.config.vocab_size), labels.view(-1))
            # loss['diffusion_rc_loss'] = diffusion_rc_loss
            return loss
        else:
            diffusion_logits = self.lm_head(diffusion_hidden_states) + self.final_logits_bias
            return diffusion_logits

    def check(
        self,
        input_ids: torch.LongTensor = None,
        role_ids: torch.LongTensor = None,
        turn_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.LongTensor] = None,
        labels: torch.LongTensor = None,
        schedule_sampler=None,
        diffusion=None,
        rc_loss_weight=0.0
    ):
        t, weights = schedule_sampler.sample(input_ids.shape[0], input_ids.device)
        loss, hidden_state, diffusion_hidden_states = self.model.check_model(
            input_ids=input_ids,
            role_ids=role_ids,
            turn_ids=turn_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            labels=labels,
            t=t,
            diffusion=diffusion,
            rc_loss_weight=rc_loss_weight,
        )

        mask_tmp = labels.masked_fill(~labels.eq(-100), 1.0)
        mask = mask_tmp.masked_fill(mask_tmp.eq(-100), 0.0)
        weight = mask.sum(-1).to(dtype=hidden_state.dtype)
        _shape = diffusion_hidden_states.shape[:2]

        loss_fct = CrossEntropyLoss(reduction='none')
        diffusion_logits = self.lm_head(diffusion_hidden_states) + self.final_logits_bias
        diffusion_rc_loss = loss_fct(diffusion_logits.view(-1, self.config.vocab_size), labels.view(-1))
        diffusion_rc_loss = diffusion_rc_loss.view(_shape[0], _shape[1])
        diffusion_rc_loss = diffusion_rc_loss.sum(-1) / weight

        logits = self.lm_head(hidden_state) + self.final_logits_bias
        rc_loss = loss_fct(logits.view(-1, self.config.vocab_size), labels.view(-1))
        rc_loss = rc_loss.view(_shape[0], _shape[1])
        rc_loss = rc_loss.sum(-1) / weight

        loss['rc_loss'] = rc_loss
        loss['diffusion_rc_loss'] = diffusion_rc_loss
        loss['t'] = t
        return loss


    def diffusion_generate(
        self,
        input_ids: torch.LongTensor = None,
        role_ids: torch.LongTensor = None,
        turn_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        max_length:int = None,
        min_length:int = None,
        num_beams:int = None,
        decoder_input_ids: torch.LongTensor = None,
        decoder_attention_mask: torch.LongTensor = None,
        diffusion = None,
    ):
        encoder_outputs, decoder_latent  = self.model.get_generation_input(
            input_ids=input_ids,
            role_ids=role_ids,
            turn_ids=turn_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask= decoder_attention_mask,
            diffusion=diffusion,
        )

        return self.generate(
            input_ids=input_ids,
            turn_ids=turn_ids,
            role_ids=role_ids,
            attention_mask=attention_mask,
            max_length=max_length,
            min_length=min_length,
            num_beams=num_beams,
            encoder_outputs=encoder_outputs,
            decoder_latent=decoder_latent,
        )

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        role_ids: torch.LongTensor = None,
        turn_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        decoder_head_mask: Optional[torch.Tensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[List[torch.FloatTensor]] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        decoder_latent:Optional[torch.Tensor] = None,
    ) -> Union[Tuple, Seq2SeqLMOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
            config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
            (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Returns:
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if labels is not None:
            if use_cache:
                logger.warning("The `use_cache` argument is changed to `False` since `labels` is provided.")
            use_cache = False
            if decoder_input_ids is None and decoder_inputs_embeds is None:
                decoder_input_ids = shift_tokens_right(
                    labels, self.config.pad_token_id, self.config.decoder_start_token_id
                )

        outputs = self.model(
            input_ids,
            role_ids,
            turn_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            encoder_outputs=encoder_outputs,
            decoder_attention_mask=decoder_attention_mask,
            head_mask=head_mask,
            decoder_head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            decoder_inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            decoder_latent=decoder_latent,
        )
        lm_logits = self.lm_head(outputs[0]) + self.final_logits_bias

        masked_lm_loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            masked_lm_loss = loss_fct(lm_logits.view(-1, self.config.vocab_size), labels.view(-1))

        if not return_dict:
            output = (lm_logits,) + outputs[1:]
            return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output

        return BartDiffusionOutput(
            loss=masked_lm_loss,
            logits=lm_logits,
            past_key_values=outputs.past_key_values,
            decoder_hidden_states=outputs.decoder_hidden_states,
            decoder_attentions=outputs.decoder_attentions,
            cross_attentions=outputs.cross_attentions,
            encoder_last_hidden_state=outputs.encoder_last_hidden_state,
            encoder_hidden_states=outputs.encoder_hidden_states,
            encoder_attentions=outputs.encoder_attentions,
        )

    def prepare_inputs_for_generation(
        self,
        decoder_input_ids,
        past=None,
        attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        cross_attn_head_mask=None,
        use_cache=None,
        encoder_outputs=None,
        decoder_latent=None,
        **kwargs
    ):
        # cut decoder_input_ids if past is used
        if past is not None:
            decoder_input_ids = decoder_input_ids[:, -1:]

        return {
            "input_ids": None,  # encoder_outputs is defined. input_ids not needed
            "encoder_outputs": encoder_outputs,
            "past_key_values": past,
            "decoder_input_ids": decoder_input_ids,
            "attention_mask": attention_mask,
            "head_mask": head_mask,
            "decoder_head_mask": decoder_head_mask,
            "cross_attn_head_mask": cross_attn_head_mask,
            "use_cache": use_cache,  # change this to avoid caching (presumably for debugging)
            "decoder_latent": decoder_latent
        }
    
    @staticmethod
    def _expand_inputs_for_generation(
        input_ids: torch.LongTensor,
        expand_size: int = 1,
        is_encoder_decoder: bool = False,
        attention_mask: Optional[torch.LongTensor] = None,
        encoder_outputs: Optional[ModelOutput] = None,
        **model_kwargs,
    ) -> Tuple[torch.LongTensor, Dict[str, Any]]:
        expanded_return_idx = (
            torch.arange(input_ids.shape[0]).view(-1, 1).repeat(1, expand_size).view(-1).to(input_ids.device)
        )
        input_ids = input_ids.index_select(0, expanded_return_idx)

        if "token_type_ids" in model_kwargs:
            token_type_ids = model_kwargs["token_type_ids"]
            model_kwargs["token_type_ids"] = token_type_ids.index_select(0, expanded_return_idx)

        if attention_mask is not None:
            model_kwargs["attention_mask"] = attention_mask.index_select(0, expanded_return_idx)

        if is_encoder_decoder:
            if encoder_outputs is None:
                raise ValueError("If `is_encoder_decoder` is True, make sure that `encoder_outputs` is defined.")
            encoder_outputs["last_hidden_state"] = encoder_outputs.last_hidden_state.index_select(
                0, expanded_return_idx.to(encoder_outputs.last_hidden_state.device)
            )
            model_kwargs["encoder_outputs"] = encoder_outputs
            decoder_latent = model_kwargs["decoder_latent"]
            model_kwargs["decoder_latent"] = decoder_latent.index_select(0, expanded_return_idx)
        return input_ids, model_kwargs
    
    def prepare_decoder_input_ids_from_labels(self, labels: torch.Tensor):
        return shift_tokens_right(labels, self.config.pad_token_id, self.config.decoder_start_token_id)

    @staticmethod
    def _reorder_cache(past, beam_idx):
        reordered_past = ()
        for layer_past in past:
            # cached cross_attention states don't have to be reordered -> they are always the same
            reordered_past += (
                tuple(past_state.index_select(0, beam_idx) for past_state in layer_past[:2]) + layer_past[2:],
            )
        return reordered_past