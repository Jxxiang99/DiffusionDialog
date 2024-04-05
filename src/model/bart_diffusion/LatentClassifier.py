from collections import defaultdict
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union, Dict, Any

import torch
import torch.utils.checkpoint
from torch import nn
import pdb
import os
import math
from model.bart_diffusion.configuration_BartDiffusion import BartDiffusionConfig
from transformers.activations import ACT2FN

class LatentClassifier(nn.Module):
    def __init__(self, config: BartDiffusionConfig) -> None:
        super().__init__()
        self.linear_size = config.d_model * config.decoder_latent_size
        self.latent_fn1 = nn.Linear(self.linear_size, 2 * self.linear_size)
        self.active_fc = ACT2FN[config.activation_function]
        self.latent_fn2 = nn.Linear(self.linear_size * 2, 2)
    
    def forward(self, gold_latent, pred_latent, weights=None):
        bsz = gold_latent.shape[0]
        gold_labels = torch.ones(bsz, device=gold_latent.device).long()
        pred_labels = torch.zeros(bsz, device=pred_latent.device).long()

        gold_latent_flat = gold_latent.view(bsz, -1)
        gold_latent_flat = self.latent_fn1(gold_latent_flat)
        gold_latent_flat = self.active_fc(gold_latent_flat)
        gold_latent_flat = self.latent_fn2(gold_latent_flat)

        pred_latent_flat = pred_latent.view(bsz, -1)
        pred_latent_flat = self.latent_fn1(pred_latent_flat)
        pred_latent_flat = self.active_fc(pred_latent_flat)
        pred_latent_flat = self.latent_fn2(pred_latent_flat)

        if weights is None:
            loss_fn = nn.CrossEntropyLoss()
            loss_gold = loss_fn(gold_latent_flat, gold_labels)
            loss_pred = loss_fn(pred_latent_flat, pred_labels)
        else:
            loss_fn = nn.CrossEntropyLoss(reduction='none')
            loss_gold = loss_fn(gold_latent_flat, gold_labels) * weights
            loss_pred = loss_fn(pred_latent_flat, pred_labels) * weights
            loss_gold = loss_gold.mean()
            loss_pred = loss_pred.mean()
        
        loss = loss_gold + loss_pred
        return loss