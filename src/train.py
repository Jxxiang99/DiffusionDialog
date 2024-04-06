import os
import argparse
from utils import init_logger, init_writer, print_config, print_model
from accelerate import Accelerator
from accelerate.logging import get_logger
from model.model import *
from trainer import Trainer
from dataloader import get_dataloaders
from transformers import set_seed
import pdb

logger = get_logger(__name__)

MODEL_MAP = {"bartbase": BartBase, 'bartdiffusion': BartDiffusion}


def get_parser_config() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="The arguments of Conversation")

    # dataset
    parser.add_argument("--model", type=str, default="bart", choices=["bartbase", "bartdiffusion"])
    parser.add_argument("--data", type=str, default="daily", choices=["daily", "persona"])
    parser.add_argument("--train_set_split_name", type=str, default="train")
    parser.add_argument("--dev_set_split_name", type=str, default="dev")
    parser.add_argument("--test_set_split_name", type=str, default="test")

    # path
    parser.add_argument("--dataset_path", type=str, default="")
    parser.add_argument("--save_model_path", type=str, default="")
    parser.add_argument("--log_dir", type=str, default="")
    parser.add_argument("--bart_path", type=str, default="")

    # load checkpoint
    parser.add_argument("--load_checkpoint", default=False, action="store_true")
    parser.add_argument("--checkpoint", type=str, default="")

    # initialize pretrained model
    parser.add_argument("--plm_init_path", type=str, default="bart-large")

    # data process
    parser.add_argument("--max_source_length", type=int, default=512)
    parser.add_argument("--max_target_length", type=int, default=128)
    parser.add_argument("--max_context_turn", type=int, default=10, help="maximum number of conversation context turn")
    parser.add_argument("--mask_source", default=False, action="store_true", help='Whether to mask input')
    parser.add_argument("--mask_prob", type=float, default=0.1)

    #model
    parser.add_argument("--encoder_bow_loss_weight", type=float, default=0.0)
    parser.add_argument("--masked_lm_loss_weight", type=float, default=0.0)
    parser.add_argument("--with_masked_lm_loss", default=False, action="store_true")
    parser.add_argument("--with_kl_loss", default=False, action="store_true")
    parser.add_argument("--kl_target", type=float, default=5.0)
    parser.add_argument("--kl_loss_weight", type=float, default=0.0)
    parser.add_argument("--with_bow_loss", default=False, action="store_true")
    parser.add_argument("--bow_loss_weight", type=float, default=0.0)
    parser.add_argument("--clip_loss_weight", type=float, default=0.0)
    parser.add_argument("--diffusion_loss_weight", type=float, default=0.0)
    parser.add_argument("--lm_rc_loss_weight", type=float, default=0.0)
    
    
    # train
    parser.add_argument("--total_steps", type=int, default=30000)
    parser.add_argument("--logging_steps", type=int, default=20)
    parser.add_argument("--eval_steps", type=int, default=2000)
    parser.add_argument("--warmup_steps", type=int, default=2000)
    parser.add_argument("--grad_accum_steps", default=1, type=int)

    parser.add_argument("--train_batch_size", type=int, default=4)
    parser.add_argument("--dev_batch_size", type=int, default=8)
    parser.add_argument("--test_batch_size", type=int, default=8)

    parser.add_argument("--optimizer", type=str, default="adamw", choices=["adamw", "adafactor"])
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--clip_value", default=1, type=float)
    parser.add_argument("--train_state", type=str, default="bart", choices=["bart", "diffusion"])

    
    # generate
    parser.add_argument("--beam_size", type=int, default=3)

    # other
    parser.add_argument("--seed", type=int, default=2021)

    return parser.parse_args()


if __name__ == "__main__":

    args = get_parser_config()

    accelerator = Accelerator()

    if accelerator.is_main_process:
        writer, path = init_writer(args)
        init_logger(path, "train")
    else:
        path = None
        writer = None
    
    # writer = None
    set_seed(args.seed)
    print_config(args)

    # load model
    if args.load_checkpoint:
        logger.info("Load checkpoint for Model and Tokenizer...")
        model = MODEL_MAP[args.model](args, args.checkpoint)
        # path = os.path.basename(args.checkpoint)
        # resume_step = os.path.splitext(path)[0].replace(f"{args.model}_{args.data}_training_step_", "")
        # resume_step = int(resume_step)
        resume_step=None
    else:
        logger.info("Initiate Model and Tokenizer...")
        model = MODEL_MAP[args.model](args)
        resume_step = None
    print_model(model)

    if args.data != 'reddit':
        # load data
        (train_loader, dev_loader) = get_dataloaders(
            args, model.tokenizer, model.model, split_list=[args.train_set_split_name, args.dev_set_split_name]
        )
        trainer = Trainer(args, model, accelerator, writer, path)
        trainer.train(train_loader, dev_loader, resume_step)
    else:
        trainer = PreTrainer(args, model, accelerator, writer, path)
        trainer.train(resume_step)

    if accelerator.is_main_process:
        writer.close()
