from dataclasses import replace
from linecache import getline
from opcode import opname
import os
from urllib import response
import torch
import accelerate.logging as logging
from torch.utils.data import DataLoader, Dataset, IterableDataset
from accelerate import Accelerator
from typing import List
import zstandard as zstd
import msgspec
import io
import random
import numpy as np


accelerator = Accelerator()

logger = logging.get_logger(__name__)

PAD_INDEX = 0
MAX_SENTENCE = 15
def _infer_absolute_position_sentence_backward(input_ids, sep_id, pad_id, cls_id):

    # given context: <CLS> <turn1> [SEP] <turn2> <SEP> <turn3> <SEP> <turn4>
    # response: <response> </s>
    #            <CLS> <turn1> [SEP] <turn2> <SEP> <turn3>  <response>
    # index:       0      4      0      3      0      2         1
    # all token not belong to one specific turn can be seen as pad

    # input_ids = torch.tensor([
    #     [4, 101, 191, 218, 5, 224, 241, 260, 179, 9, 361, 730, 259, 5, 491, 429, 407, 395, 1],
    #     [4, 101, 191, 229, 228, 9, 261, 270, 819, 5, 631, 830, 929, 5, 415, 402, 490, 1, 1],
    # ])

    positions = torch.cumsum(input_ids.flip(1).eq(sep_id).int(), dim=1).flip(1) + 1
    positions[input_ids == cls_id] = PAD_INDEX
    positions[input_ids == pad_id] = PAD_INDEX
    positions[input_ids == sep_id] = PAD_INDEX
    # prevent array subscript out of index
    positions[positions > MAX_SENTENCE] = MAX_SENTENCE

    return positions

def _infer_absolute_position_role_backward(input_ids, sep_id, pad_id, cls_id):

    # given context: <CLS> <turn1> [SEP] <turn2> <SEP> <turn3> <SEP> <turn4>
    # response: <response> </s>
    #
    #            <CLS> <turn1> [SEP] <turn2> <SEP> <turn3> <SEP> <turn4> <response>
    # index:       0      1      0       2     0      1      0      2        1

    # input_ids = torch.tensor([
    #     [4, 101, 191, 218, 5, 224, 241, 260, 179, 9, 361, 730, 259, 5, 491, 429, 407, 395, 1],
    #     [4, 101, 191, 229, 228, 9, 261, 270, 819, 5, 631, 830, 929, 5, 415, 402, 490, 1, 1],
    # ])

    positions = (torch.cumsum(input_ids.flip(1).eq(sep_id).int(), dim=1).flip(1) + 1) % 2 + 1

    positions[input_ids == cls_id] = PAD_INDEX
    positions[input_ids == pad_id] = PAD_INDEX
    positions[input_ids == sep_id] = PAD_INDEX

    return positions

class RedditDataset(Dataset):
    def __init__(self, args, path, tokenizer):

        self.tokenizer = tokenizer
        self.context_list = []
        self.response_list = []
        data_dir = args.dataset_path
        max_context_turns = args.max_context_turn

        # load data
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if line:
                    line = line.strip()
                    # print(line)
                    if '\t' not in line:
                        print(path)
                        print(line)
                    context, response = line.split("\t")
                    context = context.split("__eou__")
                    context = str(tokenizer.sep_token).join(context[-max_context_turns:])
                    self.context_list.append(context)
                    self.response_list.append(response)

    def __len__(self):
        return len(self.context_list)

    def __getitem__(self, index: int):
        return {"context": self.context_list[index], "response": self.response_list[index]}

class BartBaseDialogCollator:
    def __init__(self, args, tokenizer, model):
        self.tokenizer = tokenizer
        self.model = model
        self.max_source_length = args.max_source_length
        self.max_target_length = args.max_target_length
        self.sep_id = tokenizer.sep_token_id
        self.pad_id = tokenizer.pad_token_id
        self.cls_id = tokenizer.cls_token_id
        self.mask_id = tokenizer.mask_token_id
        self.replace_probs = [0.85, 0.15]
        self.change_probs = [0.8, 0.1, 0.1]
        self.nspecials = 4
        self.with_masked_lm_loss = args.with_masked_lm_loss
    
    def get_mask_input(self, input_ids):
        batch_size = input_ids.shape[0]
        seq_len = input_ids.shape[1]
        replace_probs = torch.tensor(self.replace_probs)
        replace_probs = replace_probs.repeat(batch_size, 1)
        change_probs = torch.tensor(self.change_probs)
        change_probs = change_probs.repeat(batch_size, 1)

        replace_masks = torch.multinomial(replace_probs, seq_len, replacement=True)
        replace_masks[input_ids == self.sep_id] = 0
        replace_masks[input_ids == self.cls_id] = 0
        replace_masks[input_ids == self.pad_id] = 0

        change_masks = torch.multinomial(change_probs, seq_len, replacement=True)

        rand = input_ids.clone().random_(self.nspecials, self.tokenizer.vocab_size)
        masked_input_ids = input_ids.clone()
        masked_input_ids[(replace_masks == 1) & (change_masks == 0)] = self.mask_id
        masked_input_ids[(replace_masks == 1) & (change_masks == 1)] = rand[(replace_masks == 1) & (change_masks == 1)]

        masked_labels = input_ids.clone()
        masked_labels[masked_labels == self.pad_id] = -100
        masked_labels[replace_masks == 0] = -100
        # masked_labels[replace_masks == 2] = -100
        return masked_input_ids, masked_labels

    def __call__(self, features):
        context = []
        response = []
        for d in features:
            context.append(d["context"])
            response.append(d["response"])

        model_inputs = self.tokenizer(
            context, padding=True, max_length=self.max_source_length, truncation=True, return_tensors="pt"
        )
        with self.tokenizer.as_target_tokenizer():
            labels = self.tokenizer(
                response, padding=True, max_length=self.max_target_length, truncation=True, return_tensors="pt"
            )["input_ids"]

        labels[labels == self.pad_id] = -100
        model_inputs["labels"] = labels
        decoder_input_ids = self.model.prepare_decoder_input_ids_from_labels(labels=model_inputs["labels"])
        model_inputs["decoder_input_ids"] = decoder_input_ids
        model_inputs["turn_ids"] = _infer_absolute_position_sentence_backward(model_inputs["input_ids"], sep_id= self.sep_id, pad_id= self.pad_id, cls_id = self.cls_id)
        model_inputs["role_ids"] = _infer_absolute_position_role_backward(model_inputs["input_ids"], sep_id= self.sep_id, pad_id= self.pad_id, cls_id = self.cls_id)
        if self.with_masked_lm_loss:
            model_inputs["input_ids"], model_inputs["masked_labels"] = self.get_mask_input(model_inputs["input_ids"])
        else:
            model_inputs["masked_labels"] = None
        return model_inputs