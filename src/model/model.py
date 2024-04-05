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
from model.bart_vae.modeling_BartVAE import BartVAEForConditionalGeneration
from model.bart_vae.configuration_BartVAE import BartVAEConfig
from model.modeling_doha import DoHAForConditionalGeneration
from model.bart_diffusion.configuration_BartDiffusion import BartDiffusionConfig
from model.bart_diffusion.modeling_BartDiffusion import BartDiffusionForConditionalGeneration
from model.bart_diffusion.modeling_BartDiffusion_top import BartDiffusionForConditionalGenerationTOP
from model.bart_diffusion.LatentTransformer import LatentTransformer
from model.modeling_doha import DoHAForConditionalGeneration
from model.utils_model import BaseModel
from model.bart_diffusion.diffusion_utils import get_gaussian_diffusion_and_sampler
import pdb

accelerator = Accelerator()

logger = logging.get_logger(__name__)


class BartDialogModel(nn.Module):
    def __init__(self, args, checkpoint=None):
        super(BartDialogModel, self).__init__()
        self.args = args

        if checkpoint:
            plm_path = checkpoint
        else:
            plm_path = args.plm_init_path

        self.tokenizer = BartTokenizerFast.from_pretrained(plm_path)
        self.model = BartForConditionalGeneration.from_pretrained(plm_path)

        for token in ["[CONTEXT]", "[DOCUMENT]", "[RESPONSE]"]:

            if token in self.tokenizer.vocab:
                logger.info(f"{token} token exists.")
            else:
                self.tokenizer.add_tokens([token])
                self.model.resize_token_embeddings(len(self.tokenizer))
                logger.info(f"Add {token} token to tokenizer.")

        self.embed_dim = self.model.config.hidden_size
        self.pad_token = self.tokenizer.pad_token
        self.pad_token_id = self.tokenizer.pad_token_id
        self.eos_token = self.tokenizer.eos_token
        self.eos_token_id = self.tokenizer.eos_token_id

        self.vocab_size = len(self.tokenizer)

    def forward(self, input_ids, attention_mask, labels, *args, **kwargs):
        with accelerator.autocast():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        mle_loss = outputs.loss

        return mle_loss

    def eval_loss(self, input_ids, attention_mask, labels, *args, **kwargs):
        with accelerator.autocast():
            outputs = self.model(
                input_ids=input_ids, attention_mask=attention_mask, decoder_input_ids=kwargs["decoder_input_ids"]
            )

        logits = outputs.logits
        mle_loss = F.cross_entropy(logits.view(-1, self.vocab_size), labels.view(-1), reduction="none")
        mask_tmp = labels.masked_fill(~labels.eq(-100), 1.0)
        mask = mask_tmp.masked_fill(mask_tmp.eq(-100), 0.0)

        # sum
        mle_loss_sum = torch.sum(mle_loss)
        token_num_sum = torch.sum(mask)
        return mle_loss_sum, token_num_sum

    def generate(self, input_ids, attention_mask, *args, **kwargs):
        outputs = self.model.generate(
            input_ids=input_ids, attention_mask=attention_mask, max_length=128, min_length=1, num_beams=self.args.beam_size,
        )
        return outputs


class PlanBartDialogModel(nn.Module):
    def __init__(self, args, checkpoint=None):
        super(PlanBartDialogModel, self).__init__()
        self.args = args

        if checkpoint:
            plm_path = checkpoint
        else:
            plm_path = args.plm_init_path

        self.tokenizer = BartTokenizerFast.from_pretrained(plm_path)
        self.model = BartForConditionalGeneration.from_pretrained(plm_path)

        for token in ["[CONTEXT]", "[DOCUMENT]", "[PLAN]", "[RESPONSE]"]:

            if token in self.tokenizer.vocab:
                logger.info(f"{token} token exists.")
            else:
                self.tokenizer.add_tokens([token])
                self.model.resize_token_embeddings(len(self.tokenizer))
                logger.info(f"Add {token} token to tokenizer.")

        self.embed_dim = self.model.config.hidden_size
        self.pad_token = self.tokenizer.pad_token
        self.pad_token_id = self.tokenizer.pad_token_id
        self.eos_token = self.tokenizer.eos_token
        self.eos_token_id = self.tokenizer.eos_token_id

        self.vocab_size = len(self.tokenizer)

    def forward(self, input_ids, attention_mask, labels, *args, **kwargs):
        with accelerator.autocast():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        mle_loss = outputs.loss

        return mle_loss

    def eval_loss(self, input_ids, attention_mask, labels, *args, **kwargs):
        with accelerator.autocast():
            outputs = self.model(
                input_ids=input_ids, attention_mask=attention_mask, decoder_input_ids=kwargs["decoder_input_ids"]
            )

        logits = outputs.logits
        mle_loss = F.cross_entropy(logits.view(-1, self.vocab_size), labels.view(-1), reduction="none")
        mask_tmp = labels.masked_fill(~labels.eq(-100), 1.0)
        mask = mask_tmp.masked_fill(mask_tmp.eq(-100), 0.0)

        # sum
        mle_loss_sum = torch.sum(mle_loss)
        token_num_sum = torch.sum(mask)
        return mle_loss_sum, token_num_sum

    def generate(self, input_ids, attention_mask, *args, **kwargs):
        outputs = self.model.generate(
            input_ids=input_ids, attention_mask=attention_mask, max_length=128, min_length=1, num_beams=self.args.beam_size,
        )
        return outputs

    @staticmethod
    def post_process(generated_sents):
        res = []
        for sent in generated_sents:
            res.append(sent.split("[RESPONSE]")[-1].strip())
        return res


class CoDR(nn.Module):
    def __init__(self, args, checkpoint=None):
        super(CoDR, self).__init__()
        self.args = args

        if checkpoint:
            plm_path = checkpoint
        else:
            plm_path = args.plm_init_path

        self.tokenizer = BartTokenizerFast.from_pretrained(plm_path)
        self.model = BartForConditionalGeneration.from_pretrained(plm_path)

        for token in ["[CONTEXT]", "[DOCUMENT]"]:

            if token in self.tokenizer.vocab:
                logger.info(f"{token} token exists.")
            else:
                self.tokenizer.add_tokens([token])
                self.model.resize_token_embeddings(len(self.tokenizer))
                logger.info(f"Add {token} token to tokenizer.")

        self.embed_dim = self.model.config.hidden_size
        self.pad_token = self.tokenizer.pad_token
        self.pad_token_id = self.tokenizer.pad_token_id
        self.eos_token = self.tokenizer.eos_token
        self.eos_token_id = self.tokenizer.eos_token_id

        self.vocab_size = len(self.tokenizer)

    def encode(self, input_ids, attention_mask, doc_input_ids, doc_attention_mask):
        source_reps = self.model.get_encoder()(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        source_reps = source_reps.last_hidden_state

        doc_reps = self.model.get_encoder()(input_ids=doc_input_ids, attention_mask=doc_attention_mask, return_dict=True)
        doc_reps = doc_reps.last_hidden_state

        reps = torch.cat([source_reps, doc_reps], dim=1)
        attention_mask = torch.cat([attention_mask, doc_attention_mask], dim=1)

        return BaseModelOutput(last_hidden_state=reps), attention_mask

    def forward(self, input_ids, attention_mask, doc_input_ids, doc_attention_mask, labels, *args, **kwargs):
        with accelerator.autocast():
            source_reps, source_mask = self.encode(input_ids, attention_mask, doc_input_ids, doc_attention_mask)
            outputs = self.model(input_ids=None, attention_mask=source_mask, encoder_outputs=(source_reps,), labels=labels)
        mle_loss = outputs.loss

        return mle_loss

    def eval_loss(self, input_ids, attention_mask, doc_input_ids, doc_attention_mask, labels, *args, **kwargs):
        with accelerator.autocast():
            source_reps, source_mask = self.encode(input_ids, attention_mask, doc_input_ids, doc_attention_mask)
            outputs = self.model(input_ids=None, attention_mask=source_mask, encoder_outputs=(source_reps,), labels=labels)

        logits = outputs.logits
        mle_loss = F.cross_entropy(logits.view(-1, self.vocab_size), labels.view(-1), reduction="none")
        mask_tmp = labels.masked_fill(~labels.eq(-100), 1.0)
        mask = mask_tmp.masked_fill(mask_tmp.eq(-100), 0.0)

        # sum
        mle_loss_sum = torch.sum(mle_loss)
        token_num_sum = torch.sum(mask)
        return mle_loss_sum, token_num_sum

    def generate(self, input_ids, attention_mask, doc_input_ids, doc_attention_mask, *args, **kwargs):
        source_reps, source_mask = self.encode(input_ids, attention_mask, doc_input_ids, doc_attention_mask)
        outputs = self.model.generate(
            inputs=None,
            attention_mask=source_mask,
            encoder_outputs=source_reps,
            max_length=128,
            min_length=1,
            num_beams=self.args.beam_size,
        )
        return outputs

class DoHA(nn.Module):
    def __init__(self, args, checkpoint=None):
        super(DoHA, self).__init__()
        self.args = args

        if checkpoint:
            plm_path = checkpoint
            self.tokenizer = BartTokenizerFast.from_pretrained(plm_path)
            self.model = DoHAForConditionalGeneration.from_pretrained(plm_path)
        else:
            plm_path = args.plm_init_path
            config = BartConfig.from_pretrained(plm_path)
            self.tokenizer = BartTokenizerFast.from_pretrained(plm_path)
            self.model = DoHAForConditionalGeneration(config)
            self.model.init_from_bart(plm_path)

        for token in ["[CONTEXT]", "[DOCUMENT]"]:

            if token in self.tokenizer.vocab:
                logger.info(f"{token} token exists.")
            else:
                self.tokenizer.add_tokens([token])
                self.model.resize_token_embeddings(len(self.tokenizer))
                logger.info(f"Add {token} token to tokenizer.")

        self.embed_dim = self.model.config.hidden_size
        self.pad_token = self.tokenizer.pad_token
        self.pad_token_id = self.tokenizer.pad_token_id
        self.eos_token = self.tokenizer.eos_token
        self.eos_token_id = self.tokenizer.eos_token_id
        self.vocab_size = len(self.tokenizer)

    def encode(self, input_ids, attention_mask, doc_input_ids, doc_attention_mask, return_dict=False):
        source_reps = self.model.get_encoder()(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        source_reps = source_reps.last_hidden_state

        doc_reps = self.model.get_encoder()(input_ids=doc_input_ids, attention_mask=doc_attention_mask, return_dict=True)
        doc_reps = doc_reps.last_hidden_state

        if return_dict:
            return BaseModelOutput(last_hidden_state=source_reps), BaseModelOutput(last_hidden_state=doc_reps)
        else:
            return source_reps, doc_reps

    def forward(self, input_ids, attention_mask, doc_input_ids, doc_attention_mask, labels, *args, **kwargs):
        with accelerator.autocast():
            source_reps, doc_reps = self.encode(input_ids, attention_mask, doc_input_ids, doc_attention_mask)
            outputs = self.model(attention_mask=(attention_mask, doc_attention_mask), encoder_outputs=(source_reps, doc_reps), labels=labels, return_dict=False)
        mle_loss = outputs[0]

        return mle_loss

    def eval_loss(self, input_ids, attention_mask, doc_input_ids, doc_attention_mask, labels, *args, **kwargs):
        with accelerator.autocast():
            source_reps, doc_reps = self.encode(input_ids, attention_mask, doc_input_ids, doc_attention_mask)
            outputs = self.model(attention_mask=(attention_mask, doc_attention_mask), encoder_outputs=(source_reps, doc_reps), labels=labels, return_dict=False)

        logits = outputs[1]
        mle_loss = F.cross_entropy(logits.view(-1, self.vocab_size), labels.view(-1), reduction="none")
        mask_tmp = labels.masked_fill(~labels.eq(-100), 1.0)
        mask = mask_tmp.masked_fill(mask_tmp.eq(-100), 0.0)

        # sum
        mle_loss_sum = torch.sum(mle_loss)
        token_num_sum = torch.sum(mask)
        return mle_loss_sum, token_num_sum

    def generate(self, input_ids, attention_mask, doc_input_ids, doc_attention_mask, *args, **kwargs):
        source_reps, doc_reps = self.encode(input_ids, attention_mask, doc_input_ids, doc_attention_mask)
        outputs = self.model.generate(
            inputs=None,
            attention_mask=(attention_mask, doc_attention_mask),
            encoder_outputs=(source_reps, doc_reps),
            max_length=128,
            min_length=1,
            num_beams=self.args.beam_size,
            return_dict_in_generate=False,
        )
        return outputs

class BartVAE(nn.Module):
    def __init__(self, args, checkpoint=None) -> None:
        super(BartVAE, self).__init__()
        self.args = args

        self.loss_weight = {'rc_loss': 1.0, 'masked_lm_loss': 0.0, 'kl_loss': 0.0, 'bow_loss': 0.0}
        if args.with_masked_lm_loss:
            self.loss_weight['masked_lm_loss'] = args.masked_lm_loss_weight
        if args.with_kl_loss:
            self.loss_weight['kl_loss'] = args.kl_loss_weight
        if args.with_bow_loss:
            self.loss_weight['bow_loss'] = args.bow_loss_weight
            self.kl_target = args.kl_target

        if checkpoint:
            plm_path = checkpoint
            self.tokenizer = BartTokenizerFast.from_pretrained(plm_path)
            self.model = BartVAEForConditionalGeneration.from_pretrained(plm_path)
        else:
            plm_path = args.plm_init_path
            config = BartVAEConfig.from_pretrained(plm_path)

            if args.with_masked_lm_loss:
                config.with_masked_lm_loss = True
            if args.with_kl_loss:
                config.with_kl_loss = True
                config.kl_target = self.kl_target
            if args.with_bow_loss:
                config.with_bow_loss = True

            self.tokenizer = BartTokenizerFast.from_pretrained(plm_path)
            self.model = BartVAEForConditionalGeneration(config)
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
                determin=False,
            )
        mle_loss = outputs.loss
        loss = defaultdict()
        loss['rc_loss'] = mle_loss

        if outputs.kl_loss is not None:
            loss['kl_loss'] = outputs.kl_loss * self.loss_weight['kl_loss']
        if outputs.masked_loss is not None:
            loss['masked_lm_loss'] = outputs.masked_loss * self.loss_weight['masked_lm_loss']
        if outputs.bow_loss is not None:
            loss['bow_loss'] = outputs.bow_loss * self.loss_weight['bow_loss']
            
        return loss

    def eval_loss(self, input_ids, attention_mask, labels, decoder_input_ids, turn_ids, role_ids, masked_labels):
        with accelerator.autocast():
            outputs = self.model(
                input_ids=input_ids, 
                turn_ids=turn_ids,
                role_ids=role_ids,
                attention_mask=attention_mask, 
                decoder_input_ids=decoder_input_ids,
                determin=True
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
            num_beams=self.args.beam_size,
            determin=True,
        )
        return outputs

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