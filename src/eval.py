import os
import torch
import argparse
from utils import init_logger, init_writer, print_config, print_model
from accelerate import Accelerator
from accelerate.logging import get_logger
from model.model import *
from evaluator import Evaluator
from dataloader import get_dataloaders
from transformers import set_seed
from evaluate_plato import NLTK_Evaluator
from eval_plato_MBR import MBR_Evaluator

logger = get_logger(__name__)

MODEL_MAP = {"bart": BartDialogModel, "docbart": BartDialogModel, "planbart": PlanBartDialogModel, "codr": CoDR, "bartbase": BartBase, 
             "bartvae": BartVAE, 'bartdiffusion': BartDiffusion, 'bartdiffusion_top': BartDiffusionTOP}
EVALUATOR_MAP = {"nlg-eval" : Evaluator, "nltk-eval" : NLTK_Evaluator, "mbr-eval" : MBR_Evaluator}


def get_parser_config() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="The arguments of Conversation")

    # dataset
    parser.add_argument("--model", type=str, default="bart", choices=["bart", "docbart", "planbart", "codr", "bartbase", "bartvae", "bartdiffusion", "diffusion", "bartdiffusion_top"])
    parser.add_argument("--data", type=str, default="daily", choices=["daily", "wow", "planwow", "persona"])
    parser.add_argument("--train_set_split_name", type=str, default="train")
    parser.add_argument("--dev_set_split_name", type=str, default="dev")
    parser.add_argument(
        "--test_set_split_name",
        type=str,
        default="test",
        help="if there are multiple test sets, please use commas to separate them. eg: test_seen,test_unseen",
    )
    parser.add_argument("--eval_way", type=str, default="nlg-eval", choices=["nlg-eval", "nltk-eval", "mbr-eval"])
    parser.add_argument("--tokenizer_path", type=str, default=None)

    # path
    parser.add_argument("--save_result_dir", type=str, default="")
    parser.add_argument("--dataset_path", type=str, default="")
    parser.add_argument("--log_dir", type=str, default="")
    parser.add_argument("--checkpoint_path", type=str, default="")
    parser.add_argument("--bart_path", type=str, default="")

    parser.add_argument("--start_step", type=int, default=8000)
    parser.add_argument("--end_step", type=int, default=20000)
    parser.add_argument("--interval_steps", type=int, default=2000)

    #model
    parser.add_argument("--encoder_bow_loss_weight", type=float, default=0.0)
    parser.add_argument("--masked_lm_loss_weight", type=float, default=0.0)
    parser.add_argument("--with_masked_lm_loss", default=False, action="store_true")
    parser.add_argument("--with_kl_loss", default=False, action="store_true")
    parser.add_argument("--kl_target", type=float, default=5.0)
    parser.add_argument("--kl_loss_weight", type=float, default=0.0)
    parser.add_argument("--with_bow_loss", default=False, action="store_true")
    parser.add_argument("--leak_generation", default=False, action="store_true")

    # data process
    parser.add_argument("--max_source_length", type=int, default=1024)
    parser.add_argument("--max_target_length", type=int, default=128)
    parser.add_argument("--max_context_turn", type=int, default=10, help="maximum number of conversation context turn")

    parser.add_argument("--dev_batch_size", type=int, default=8)
    parser.add_argument("--test_batch_size", type=int, default=8)

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
        init_logger(path, "eval")
    else:
        writer = None

    set_seed(args.seed)
    print_config(args)

    # eval dev to choose the best model
    # choose the lowest dev loss model as the bset model
    best_checkpoint = -1
    best_metrics = 10000
    for n in range(args.start_step, args.end_step + 1, args.interval_steps):
        logger.info(f"Evaluating dev set with step {n}")
        checkpoint = os.path.join(args.checkpoint_path, f"training_step_{n}")
        model = MODEL_MAP[args.model](args, checkpoint)

        # load data
        (dev_loader,) = get_dataloaders(args, model.tokenizer, model.model, split_list=[args.dev_set_split_name])

        evaluator = EVALUATOR_MAP[args.eval_way](args, model, accelerator, writer, save_result_path=path)
        res = evaluator.eval(dev_loader, n, split="dev", save_result=False)

        if res["loss"] < best_metrics:
            best_metrics = res["loss"]
            best_checkpoint = n

        accelerator.free_memory()
        torch.cuda.empty_cache()

    # eval test
    logger.info(f"Already chosen step {best_checkpoint} checkpoint as the best model.")
    checkpoint = os.path.join(args.checkpoint_path, f"training_step_{best_checkpoint}")
    model = MODEL_MAP[args.model](args, checkpoint)

    test_set_split_list = args.test_set_split_name.split(",")
    for test_set_split in test_set_split_list:
        logger.info(f"Evaluating {test_set_split} set:")
        (test_loader,) = get_dataloaders(args, model.tokenizer, model.model, split_list=[test_set_split])
        evaluator = EVALUATOR_MAP[args.eval_way](args, model, accelerator, writer, save_result_path=path)
        res = evaluator.eval(test_loader, best_checkpoint, split=test_set_split, save_result=True)

    if accelerator.is_main_process:
        writer.close()
