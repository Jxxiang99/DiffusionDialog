import copy
from dataclasses import dataclass
import imp
import math
import random
import warnings
from typing import Callable, List, Optional, Tuple, Union, Dict, Any
import pdb

import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
import os
import pdb
import numpy as np

from model.bart_diffusion.configuration_BartDiffusion import BartDiffusionConfig

class LatentCLIP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.encoder_latent_fc = nn.Linear(config.d_model * config.encoder_latent_size, config.d_model * config.decoder_latent_size)
        self.decoder_latent_fc = nn.Linear(config.d_model * config.decoder_latent_size, config.d_model * config.decoder_latent_size)
        self.decoder_latent_size = config.decoder_latent_size
        self.encoder_latent_size = config.encoder_latent_size
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
    
    def forward(self, encoder_latent, decoder_latent):
        bsz = encoder_latent.shape[0]
        encoder_latent = encoder_latent.view(bsz, -1)
        decoder_latent = decoder_latent.view(bsz, -1)
        encoder_latent_featrues = self.encoder_latent_fc(encoder_latent)
        decoder_latent_featrues = self.decoder_latent_fc(decoder_latent)
        # decoder_latent_featrues = decoder_latent

        encoder_latent_featrues = encoder_latent_featrues / (torch.linalg.norm(encoder_latent_featrues, dim=-1, keepdim=True) + 1e-12)
        decoder_latent_featrues = decoder_latent_featrues / (torch.linalg.norm(decoder_latent_featrues, dim=-1, keepdim=True) + 1e-12)

        logit_scale = self.logit_scale.exp()
        logits1 = logit_scale * encoder_latent_featrues @ decoder_latent_featrues.t()
        logits2 = logits1.t()

        bsz = encoder_latent.shape[0]
        label = torch.from_numpy(np.arange(bsz)).to(encoder_latent.device)
        loss1 = nn.CrossEntropyLoss()(logits1, label)
        loss2 = nn.CrossEntropyLoss()(logits2, label)
        loss = (loss1 + loss2) / 2
        return loss

    def cond_fn(self, encoder_latent, grad_scale:float) -> Callable[..., torch.Tensor]:
        bsz = encoder_latent.shape[0]
        with torch.no_grad():
            encoder_latent = encoder_latent.view(bsz, -1)
            encoder_latent_featrue = self.encoder_latent_fc(encoder_latent)

        def cond_fn(decoder_latent, t=None, grad_scale=grad_scale, **kwargs):
            #此处原文中还加了time step embedding,但是为了训练方便，这边没加
            with torch.enable_grad():
                x_var = decoder_latent.detach().requires_grad_(True)
                decoder_latent_featrue = self.decoder_latent_fc(x_var.view(bsz, -1))
                loss = torch.exp(self.logit_scale) * (encoder_latent_featrue * decoder_latent_featrue).sum()
                grad = torch.autograd.grad(loss, x_var)[0].detach()
            return grad * grad_scale

        return cond_fn