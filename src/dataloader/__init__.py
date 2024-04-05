import os
import torch
import accelerate.logging as logging
from torch.utils.data import DataLoader, Dataset
from accelerate import Accelerator
from typing import List
from dataloader.dataset import *
from dataloader.collator import *
from tqdm.auto import tqdm

accelerator = Accelerator()

logger = logging.get_logger(__name__)


DATASET_MAP = {"daily": DailyDialogDataset, "wow": WoWDataset, "planwow": PlanWoWDataset, "reddit": RedditDataset, "persona": PersonaChatDataset}
COLLATOR_MAP = {"bart": DialogCollator, "docbart": DocDialogCollator , "planbart": PlanDocDialogCollator, "codr": ContextDrivenCollator, 'bartbase': BartBaseDialogCollator, 
                "bartvae": BartBaseDialogCollator, 'bartdiffusion': BartDiffusionDialogCollator, 'diffusion': BartDiffusionDialogCollator, 'bartdiffusion_top':BartDiffusionDialogCollator}


def get_dataloaders(args, tokenizer, model, split_list: List[str]):
    res_loaders = []

    collator = COLLATOR_MAP[args.model](args, tokenizer, model)

    for s in split_list:
        dataset = DATASET_MAP[args.data](args, s, tokenizer)

        if s == "train":
            if args.data == "reddit":
                dataloader = DataLoader(dataset, batch_size=args.train_batch_size, collate_fn=collator)
            else:
                dataloader = DataLoader(dataset, batch_size=args.train_batch_size, shuffle=True, collate_fn=collator)
        elif s in ["dev", "valid"]:
            dataloader = DataLoader(dataset, batch_size=args.dev_batch_size, collate_fn=collator)
        else:
            dataloader = DataLoader(dataset, batch_size=args.test_batch_size, collate_fn=collator)

        res_loaders.append(dataloader)

    return res_loaders
