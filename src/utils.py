import os
import sys
import time
import logging
import torch
import random
import numpy as np
from accelerate.logging import get_logger
from torch.utils.tensorboard import SummaryWriter

logger = get_logger(__name__)


def init_logger(path, name):
    log_path = os.path.join(path, f"{name}.log")
    file_handler = logging.FileHandler(log_path)
    stdout_handler = logging.StreamHandler(sys.stdout)
    handlers = [file_handler, stdout_handler]
    logging.basicConfig(
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO, handlers=handlers,
    )


def init_writer(args):
    current_time = time.strftime("%Y-%m-%d.%H-%M-%S")
    current_time += f".{args.model}"
    path = os.path.join(args.log_dir, f"{args.data}/{current_time}")
    os.makedirs(path)
    writer = SummaryWriter(log_dir=path)
    return writer, path


def print_config(args):
    logger.info("========Here is your configuration========")
    args = args.__dict__
    for key, value in args.items():
        logger.info(f"\t{key} = {value}")


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def print_model(model):
    # 打印模型信息
    # print("\nModel Structure")
    # print(model)
    logger.info(f"The model has {count_parameters(model)/1000**2:.1f}M trainable parameters")
    # print("\nModel Parameters")
    # for name, param in model.named_parameters():
    #     print("\t" + name + "\t", list(param.size()))

