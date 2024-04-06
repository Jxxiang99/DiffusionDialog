from collections import defaultdict
import imp
import os
import torch
import accelerate.logging as logging
import torch.nn as nn
import torch.nn.functional as F
from transformers import BartForConditionalGeneration, BartTokenizerFast, BartConfig
from accelerate import Accelerator
from transformers.modeling_outputs import BaseModelOutput
from model.bart_base.configuration_bart import BartBaseConfig
from model.bart_base.modeling_BartBase import BartBaseForConditionalGeneration
from model.bart_diffusion.configuration_BartDiffusion import BartDiffusionConfig
from model.bart_diffusion.modeling_BartDiffusion import BartDiffusionForConditionalGeneration
from model.bart_diffusion.modeling_BartDiffusion_top import BartDiffusionForConditionalGenerationTOP
from model.bart_diffusion.LatentTransformer import LatentTransformer
from model.utils_model import BaseModel
from model.bart_diffusion.diffusion_utils import get_gaussian_diffusion_and_sampler
import pdb

accelerator = Accelerator()

logger = logging.get_logger(__name__)

class BartBase(BaseModel):
    def __init__(self, args, checkpoint=None) -> None:
        super(BartBase, self).__init__()
        self.args = args
        self.loss_weight = {'rc_loss' : 1.0, 'masked_lm_loss' : 0.0}
        if args.with_masked_lm_loss == True:
            if args.masked_lm_loss_weight != 0:
                self.loss_weight['masked_lm_loss'] = args.masked_lm_loss_weight
            else:
                raise ValueError("with masked_lm_loss, you should specify you masked_lm_loss_weight")
        if checkpoint:
            plm_path = checkpoint
            self.tokenizer = BartTokenizerFast.from_pretrained(plm_path)
            self.model = BartBaseForConditionalGeneration.from_pretrained(plm_path)
        else:
            plm_path = args.plm_init_path
            config = BartBaseConfig.from_pretrained(plm_path)
            if args.with_masked_lm_loss:
                config.with_masked_lm_loss = True
            self.tokenizer = BartTokenizerFast.from_pretrained(plm_path)
            self.model = BartBaseForConditionalGeneration(config)
            # self.model.init_weights()
            self.model.init_from_bart(plm_path)
        self.vocab_size = len(self.tokenizer)

    def forward(self, input_ids, attention_mask, labels, decoder_input_ids, turn_ids, role_ids, masked_labels):
        with accelerator.autocast():
            outputs = self.model(
                input_ids=input_ids, 
                decoder_input_ids=decoder_input_ids,
                turn_ids=turn_ids,
                role_ids=role_ids,
                attention_mask=attention_mask, 
                masked_labels=masked_labels,
                labels=labels, 
            )
        mle_loss = outputs.loss
        loss = defaultdict()
        loss['rc_loss'] = mle_loss
        if outputs.masked_loss is not None:
            loss['masked_lm_loss'] = outputs.masked_loss * self.loss_weight['masked_lm_loss']
        return loss

    def eval_loss(self, input_ids, attention_mask, labels, decoder_input_ids, turn_ids, role_ids, masked_labels):
        with accelerator.autocast():
            outputs = self.model(
                input_ids=input_ids, 
                turn_ids=turn_ids,
                role_ids=role_ids,
                attention_mask=attention_mask, 
                decoder_input_ids=decoder_input_ids
            )

        logits = outputs.logits
        mle_loss = F.cross_entropy(logits.view(-1, self.vocab_size), labels.view(-1), reduction="none")
        mask_tmp = labels.masked_fill(~labels.eq(-100), 1.0)
        mask = mask_tmp.masked_fill(mask_tmp.eq(-100), 0.0)

        # sum
        mle_loss_sum = torch.sum(mle_loss)
        token_num_sum = torch.sum(mask)
        return mle_loss_sum, token_num_sum

    def generate(self, input_ids, attention_mask, labels, decoder_input_ids, turn_ids, role_ids, masked_labels):
        outputs = self.model.generate(
            input_ids=input_ids, 
            turn_ids=turn_ids,
            role_ids=role_ids,
            attention_mask=attention_mask, 
            max_length=128, 
            min_length=1, 
            num_beams=self.args.beam_size
        )
        return outputs

class BartDiffusion(BaseModel):
    def __init__(self, args, checkpoint=None) -> None:
        super(BartDiffusion, self).__init__()
        self.args = args
        self.loss_weight = {'rc_loss': 0.0,
                            'gold_rc_loss': 0.0,
                            'diffusion_loss': 1.0,
                            'diffusion_rc_loss': 1.0,
                            'sim_loss': 0.0,
                            'classifier_loss': 0.0,
                            'noise_loss':0.0,
                            'bow_loss':1.0,
                            'tT_loss':0.0}
        self.init_loss_weight(args)
        if checkpoint:
            logger.info(f"loading the checkpoint {checkpoint}")
            plm_path = checkpoint
            self.tokenizer = BartTokenizerFast.from_pretrained(plm_path)
            self.model = BartDiffusionForConditionalGeneration.from_pretrained(plm_path)
        else:
            plm_path = args.plm_init_path
            config = BartDiffusionConfig.from_pretrained(plm_path)
            config = self.change_loss_dict(config)
            self.tokenizer = BartTokenizerFast.from_pretrained(plm_path)
            self.model = BartDiffusionForConditionalGeneration(config)
            self.model.init_from_bart(plm_path)
        
        self.diffuison_para = ['LatentTransformer', 'latent_fn', 'model.final_layer_norm', 'embed_latents']
        self.diffusion, self.schedule_sampler = get_gaussian_diffusion_and_sampler()
        self.ddim_diffusion, self.ddim_schedule_sampler = get_gaussian_diffusion_and_sampler(timestep_respacing="ddim50")
        # self.ddim_diffusion, self.ddim_schedule_sampler = get_gaussian_diffusion_and_sampler()
        self.vocab_size = len(self.tokenizer)

    def forward(self, input_ids, attention_mask, labels, decoder_input_ids, turn_ids, role_ids, masked_labels, decoder_attention_mask):
        with accelerator.autocast():
            loss = self.model.get_loss(
                input_ids=input_ids, 
                turn_ids=turn_ids,
                role_ids=role_ids,
                attention_mask=attention_mask,
                decoder_input_ids=decoder_input_ids,
                labels=labels,
                decoder_attention_mask=decoder_attention_mask,
                schedule_sampler=self.schedule_sampler,
                diffusion=self.diffusion,
                gold_rc_loss_weight=self.loss_weight['gold_rc_loss']
            )

        # if loss['diffusion_loss'] < 0.1 and self.loss_weight['diffusion_loss'] > 0.002:
        #     self.loss_weight['diffusion_loss'] -= 0.001
        # elif self.loss_weight['diffusion_loss'] < 1.0:
        #     self.loss_weight['diffusion_loss'] += 0.001
        # if ((1.2 * loss['gold_rc_loss']) > loss['diffusion_rc_loss']):
        #     if self.loss_weight['gold_rc_loss'] < 1.0:
        #         self.loss_weight['gold_rc_loss'] += 0.001
        #     if self.loss_weight['diffusion_loss'] > 0.002:
        #         self.loss_weight['diffusion_loss'] -= 0.001
        #     if self.loss_weight['diffusion_rc_loss'] > 0.002:
        #         self.loss_weight['diffusion_rc_loss'] -= 0.001
        #         # self.loss_weight['noise_loss'] -=0.001
        # elif ((1.2 * loss['gold_rc_loss']) < loss['diffusion_rc_loss']):
        #     if self.loss_weight['gold_rc_loss'] > 0.002:
        #         self.loss_weight['gold_rc_loss'] -= 0.001
        #     if self.loss_weight['diffusion_loss'] < 1.0:
        #         self.loss_weight['diffusion_loss'] += 0.001
        #     if self.loss_weight['diffusion_rc_loss'] < 1.0:
        #         self.loss_weight['diffusion_rc_loss'] += 0.001

        loss = self.loss_weight_multi(loss)
        return loss

    def eval_loss(self, input_ids, attention_mask, labels, decoder_input_ids, turn_ids, role_ids, masked_labels, decoder_attention_mask):
        with accelerator.autocast():
            logits = self.model.get_loss(
                input_ids=input_ids, 
                turn_ids=turn_ids,
                role_ids=role_ids,
                attention_mask=attention_mask, 
                decoder_input_ids=decoder_input_ids,
                decoder_attention_mask=decoder_attention_mask,
                schedule_sampler=self.schedule_sampler,
                diffusion=self.diffusion,
            )

        mle_loss = F.cross_entropy(logits.view(-1, self.vocab_size), labels.view(-1), reduction="none")
        mask_tmp = labels.masked_fill(~labels.eq(-100), 1.0)
        mask = mask_tmp.masked_fill(mask_tmp.eq(-100), 0.0)

        # sum
        mle_loss_sum = torch.sum(mle_loss)
        token_num_sum = torch.sum(mask)
        return mle_loss_sum, token_num_sum

    def generate(self, input_ids, attention_mask, labels, decoder_input_ids, turn_ids, role_ids, masked_labels, decoder_attention_mask, teacher_forcing = False):
        #更改diffusion，配合模型文件的采样函数，更改采样方式和步长
        if teacher_forcing :
            outputs = self.model.diffusion_generate(
            input_ids=input_ids, 
            turn_ids=turn_ids,
            role_ids=role_ids,
            attention_mask=attention_mask,
            max_length=128,
            min_length=1,
            num_beams=self.args.beam_size,
            diffusion=self.ddim_diffusion,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask
        )
        else:
            outputs = self.model.diffusion_generate(
                input_ids=input_ids, 
                turn_ids=turn_ids,
                role_ids=role_ids,
                attention_mask=attention_mask,
                max_length=128,
                min_length=1,
                num_beams=self.args.beam_size,
                diffusion=self.ddim_diffusion
            )
        return outputs

    def check(self, input_ids, attention_mask, labels, decoder_input_ids, turn_ids, role_ids, masked_labels, decoder_attention_mask):
        loss = self.model.check(
                input_ids=input_ids, 
                turn_ids=turn_ids,
                role_ids=role_ids,
                attention_mask=attention_mask,
                decoder_input_ids=decoder_input_ids,
                labels=labels,
                decoder_attention_mask=decoder_attention_mask,
                schedule_sampler=self.schedule_sampler,
                diffusion=self.diffusion,
                rc_loss_weight=self.loss_weight['rc_loss']
            )
        
        return loss

class BartDiffusionTOP(BaseModel):
    def __init__(self, args, checkpoint=None) -> None:
        super(BartDiffusionTOP, self).__init__()
        self.args = args
        self.loss_weight = {'rc_loss': 0.001,
                            'diffusion_loss': 1.0,
                            'diffusion_rc_loss': 1.0,
                            'bow_loss': 0.0,
                            'sim_loss': 0.0,
                            'tT_loss': 0.0}
        self.init_loss_weight(args)
        if checkpoint:
            logger.info(f"loading the checkpoint {checkpoint}")
            plm_path = checkpoint
            self.tokenizer = BartTokenizerFast.from_pretrained(plm_path)
            self.model = BartDiffusionForConditionalGenerationTOP.from_pretrained(plm_path)
        else:
            plm_path = args.plm_init_path
            config = BartDiffusionConfig.from_pretrained(plm_path)
            config = self.change_loss_dict(config)
            self.tokenizer = BartTokenizerFast.from_pretrained(plm_path)
            self.model = BartDiffusionForConditionalGenerationTOP(config)
            self.model.init_from_bart(plm_path)
        
        self.diffusion, self.schedule_sampler = get_gaussian_diffusion_and_sampler()
        self.ddim_diffusion, self.ddim_schedule_sampler = get_gaussian_diffusion_and_sampler(timestep_respacing="ddim1000")
        # self.ddim_diffusion, self.ddim_schedule_sampler = get_gaussian_diffusion_and_sampler()

        self.vocab_size = len(self.tokenizer)

    def forward(self, input_ids, attention_mask, labels, decoder_input_ids, turn_ids, role_ids, masked_labels, decoder_attention_mask):
        with accelerator.autocast():
            loss = self.model.get_loss(
                input_ids=input_ids, 
                turn_ids=turn_ids,
                role_ids=role_ids,
                attention_mask=attention_mask,
                decoder_input_ids=decoder_input_ids,
                labels=labels,
                decoder_attention_mask=decoder_attention_mask,
                schedule_sampler=self.schedule_sampler,
                diffusion=self.diffusion,
                rc_loss_weight=1.0
            )

        if (1.2 * loss['rc_loss']) < loss['diffusion_rc_loss']:
            if self.loss_weight['rc_loss'] > 0.002:
                self.loss_weight['rc_loss'] -= 0.001
            if self.loss_weight['diffusion_loss'] < 1.0:
                self.loss_weight['diffusion_loss'] += 0.001
            if self.loss_weight['diffusion_rc_loss'] < 1.0:
                self.loss_weight['diffusion_rc_loss'] += 0.001
            # self.loss_weight['diffusion_loss'] += 0.001
        elif (1.2 * loss['rc_loss']) >= loss['diffusion_rc_loss']:
            if self.loss_weight['rc_loss'] < 1.0:
                self.loss_weight['rc_loss'] += 0.001
            if self.loss_weight['diffusion_loss'] > 0.002:
                self.loss_weight['diffusion_loss'] -= 0.001
            if self.loss_weight['diffusion_rc_loss'] > 0.002:
                self.loss_weight['diffusion_rc_loss'] -= 0.001
            # self.loss_weight['diffusion_loss'] -= 0.001

        loss = self.loss_weight_multi(loss)
        return loss

    def eval_loss(self, input_ids, attention_mask, labels, decoder_input_ids, turn_ids, role_ids, masked_labels, decoder_attention_mask):
        with accelerator.autocast():
            logits = self.model.get_loss(
                input_ids=input_ids, 
                turn_ids=turn_ids,
                role_ids=role_ids,
                attention_mask=attention_mask, 
                decoder_input_ids=decoder_input_ids,
                decoder_attention_mask=decoder_attention_mask,
                schedule_sampler=self.schedule_sampler,
                diffusion=self.diffusion,
            )

        mle_loss = F.cross_entropy(logits.view(-1, self.vocab_size), labels.view(-1), reduction="none")
        mask_tmp = labels.masked_fill(~labels.eq(-100), 1.0)
        mask = mask_tmp.masked_fill(mask_tmp.eq(-100), 0.0)

        # sum
        mle_loss_sum = torch.sum(mle_loss)
        token_num_sum = torch.sum(mask)
        return mle_loss_sum, token_num_sum

    def generate(self, input_ids, attention_mask, labels, decoder_input_ids, turn_ids, role_ids, masked_labels, decoder_attention_mask, teacher_forcing = False):
        if teacher_forcing :
            outputs = self.model.diffusion_generate(
            input_ids=input_ids, 
            turn_ids=turn_ids,
            role_ids=role_ids,
            attention_mask=attention_mask,
            max_length=128,
            min_length=1,
            num_beams=self.args.beam_size,
            diffusion=self.ddim_diffusion,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask
        )
        else:
            outputs = self.model.diffusion_generate(
                input_ids=input_ids, 
                turn_ids=turn_ids,
                role_ids=role_ids,
                attention_mask=attention_mask,
                max_length=128,
                min_length=1,
                num_beams=self.args.beam_size,
                diffusion=self.ddim_diffusion
            )
        return outputs