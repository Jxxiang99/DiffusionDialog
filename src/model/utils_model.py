import os
import math
import random
import warnings
import torch
import pdb
from typing import Dict, List, Optional, Tuple

import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn import CrossEntropyLoss
from accelerate.logging import get_logger
logger = get_logger(__name__)

def transfer_parameters(args, plm, model):
    plm_dict = plm.state_dict()
    model_dict = model.state_dict()
    plm_dict = {k: v for k , v in plm_dict.items() if k in model_dict}
    model_dict.update(plm_dict)
    model.load_state_dict(model_dict)
    logger.info('saving init model parameters')
    torch.save(model.state_dict(), args.checkpoint)
    del plm
    del plm_dict
    return model

class BaseModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.loss_weight = {'rc_loss': 1.0}
    
    def init_loss_weight(self, args):
        for key in self.loss_weight.keys():
            if self.loss_weight[key] > 0.0:
                continue
            loss_name = key + '_weight'
            loss_weight = getattr(args, loss_name, 0.0)
            self.loss_weight[key] = loss_weight
    
    def change_loss_dict(self, config):
        if getattr(config, 'loss_dict', None) is not None:
            for key in self.loss_weight.keys():
                name = 'with_' + key
                if self.loss_weight[key] != 0.0:
                    config.loss_dict[name] = True
                else:
                    config.loss_dict[name] = False
        return config

    def loss_weight_multi(self, loss):
        for key in loss.keys():
            loss[key] = loss[key] * self.loss_weight[key]
        return loss
    
    def show_loss_weight(self):
        logger.info(f"loss weight:{self.loss_weight}")