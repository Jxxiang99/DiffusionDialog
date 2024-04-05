import os
from statistics import mode
from turtle import forward
from numpy import outer
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
import matplotlib.pyplot as plt
import numpy as np
import pdb
import math
import torch as th
from tqdm import tqdm

logger = get_logger(__name__)
MODEL_MAP = {"bart": BartDialogModel, "docbart": BartDialogModel, "planbart": PlanBartDialogModel, "codr": CoDR, "bartbase": BartBase, 
             "bartvae": BartVAE, 'bartdiffusion': BartDiffusion, 'bartdiffusion_top': BartDiffusionTOP}
EVALUATOR_MAP = {"nlg-eval" : Evaluator, "nltk-eval" : NLTK_Evaluator}


def get_parser_config() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="The arguments of Conversation")

    # dataset
    parser.add_argument("--model", type=str, default="bart", choices=["bart", "docbart", "planbart", "codr", "bartbase", "bartvae", "bartdiffusion", "diffusion", "bartdiffusion_top"])
    parser.add_argument("--data", type=str, default="daily", choices=["daily", "wow", "planwow", "persona", "reddit"])
    parser.add_argument("--train_set_split_name", type=str, default="train")
    parser.add_argument("--dev_set_split_name", type=str, default="dev")
    parser.add_argument(
        "--test_set_split_name",
        type=str,
        default="test",
        help="if there are multiple test sets, please use commas to separate them. eg: test_seen,test_unseen",
    )
    parser.add_argument("--eval_way", type=str, default="nlg-eval", choices=["nlg-eval", "nltk-eval"])
    parser.add_argument("--tokenizer_path", type=str, default=None)

    # path
    parser.add_argument("--save_result_dir", type=str, default="")
    parser.add_argument("--dataset_path", type=str, default="")
    parser.add_argument("--log_dir", type=str, default="")
    parser.add_argument("--checkpoint", type=str, default="")
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
    parser.add_argument("--load_checkpoint", default=False, action="store_true")
    parser.add_argument("--change_diffusion", default=False, action="store_true")
    parser.add_argument("--train_state", type=str, default="bart", choices=["bart", "diffusion"])

    # data process
    parser.add_argument("--max_source_length", type=int, default=1024)
    parser.add_argument("--max_target_length", type=int, default=128)
    parser.add_argument("--max_context_turn", type=int, default=10, help="maximum number of conversation context turn")

    parser.add_argument("--dev_batch_size", type=int, default=8)
    parser.add_argument("--test_batch_size", type=int, default=8)
    parser.add_argument("--train_batch_size", type=int, default=2)
    # generate
    parser.add_argument("--beam_size", type=int, default=3)

    # other
    parser.add_argument("--seed", type=int, default=2021)

    return parser.parse_args()

def timestep_embedding(timesteps, dim, max_period=10000):
    """
    Create sinusoidal timestep embeddings.

    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                    These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """
    half = dim // 2
    freqs = th.exp(
        -math.log(max_period) * th.arange(start=0, end=half, dtype=th.float32) / half
    ).to(device=timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = th.cat([th.cos(args), th.sin(args)], dim=-1)
    if dim % 2:
        embedding = th.cat([embedding, th.zeros_like(embedding[:, :1])], dim=-1)
    return embedding
# print(timestep_embedding(torch.tensor([1, 110, 210, 310]), dim=128))

def batch_show(context, response, pred_response):
    for i in range(len(context)):
        print(f"context:{context[i].replace('<pad>', '')}")
        print(f"response:{response[i]}")
        print(f"pred_response:{pred_response[i]}")
        print()

def batch_show_pair(context, response, pred_response_1, pred_response_2):
    for i in range(len(context)):
        print(f"context:{context[i].replace('<pad>', '')}")
        print(f"response:{response[i]}")
        print(f"pred_response_1:{pred_response_1[i]}")
        print(f"pred_response_2:{pred_response_2[i]}")
        print()

def diversity_sents_show(samples = 3):
    args = get_parser_config()

    accelerator = Accelerator()

    # if accelerator.is_main_process:
    #     writer, path = init_writer(args)
    #     init_logger(path, "train")
    # else:
    #     writer = None

    set_seed(args.seed)
    print_config(args)

    # load model
    if args.load_checkpoint:
        logger.info("Load checkpoint for Model and Tokenizer...")
        model = MODEL_MAP[args.model](args, args.checkpoint)
        # path = os.path.basename(args.checkpoint)
        # resume_step = os.path.splitext(path)[0].replace(f"{args.model}_{args.data}_training_step_", "")
        # resume_step = int(resume_step)
    else:
        logger.info("Initiate Model and Tokenizer...")
        model = MODEL_MAP[args.model](args)
        # resume_step = None
    print_model(model)
    # load data
    (train_loader, dev_loader) = get_dataloaders(
        args, model.tokenizer, model.model, split_list=[args.train_set_split_name, args.dev_set_split_name]
    )
    model, train_loader, dev_loader = accelerator.prepare(model, train_loader, dev_loader)

    # trainer = Trainer(args, model, accelerator, writer)
    # trainer.train(train_loader, dev_loader, resume_step)
    model.eval()
    pred_response_list = []
    for batch_data in tqdm(dev_loader):
        context_ids = batch_data['input_ids']
        response_ids = batch_data['decoder_input_ids']
        context = model.tokenizer.batch_decode(context_ids, skip_special_tokens=False)
        response = model.tokenizer.batch_decode(response_ids, skip_special_tokens=True)
        # pred_response = model.generate(**batch_data, teacher_forcing=True)
        for i in range(samples):
            pred_response = model.generate(**batch_data)
            pred_response = model.tokenizer.batch_decode(pred_response, skip_special_tokens=True)
            pred_response_list = pred_response_list + [pred_response]
        for bsz in range(len(context)):
            print(f"context:{context[bsz].replace('<pad>', '')}")
            print(f"response:{response[bsz]}")
            for sample in range(samples):
                print(f"pred_response_{sample}:{pred_response_list[sample][bsz]}")
            print()
        break
    return

def check_generate_sents():
    args = get_parser_config()

    accelerator = Accelerator()

    # if accelerator.is_main_process:
    #     writer, path = init_writer(args)
    #     init_logger(path, "train")
    # else:
    #     writer = None

    set_seed(args.seed)
    print_config(args)

    # load model
    if args.load_checkpoint:
        logger.info("Load checkpoint for Model and Tokenizer...")
        model = MODEL_MAP[args.model](args, args.checkpoint)
        # path = os.path.basename(args.checkpoint)
        # resume_step = os.path.splitext(path)[0].replace(f"{args.model}_{args.data}_training_step_", "")
        # resume_step = int(resume_step)
    else:
        logger.info("Initiate Model and Tokenizer...")
        model = MODEL_MAP[args.model](args)
        # resume_step = None
    print_model(model)
    # load data
    (train_loader, dev_loader) = get_dataloaders(
        args, model.tokenizer, model.model, split_list=[args.train_set_split_name, args.dev_set_split_name]
    )
    model, train_loader, dev_loader = accelerator.prepare(model, train_loader, dev_loader)

    # trainer = Trainer(args, model, accelerator, writer)
    # trainer.train(train_loader, dev_loader, resume_step)
    count = 0
    model.eval()
    writer = ([],[],[])
    for batch_data in tqdm(train_loader):
        context_ids = batch_data['input_ids']
        response_ids = batch_data['decoder_input_ids']
        context = model.tokenizer.batch_decode(context_ids, skip_special_tokens=False)
        response = model.tokenizer.batch_decode(response_ids, skip_special_tokens=True)
        # pred_response = model.generate(**batch_data, teacher_forcing=True)
        pred_response = model.generate(**batch_data)
        pred_response_1 = model.tokenizer.batch_decode(pred_response, skip_special_tokens=True)
        print("1.")
        # batch_show(context, response, pred_response)
        pred_response = model.generate(**batch_data)
        pred_response_2 = model.tokenizer.batch_decode(pred_response, skip_special_tokens=True)
        print("2.")
        batch_show_pair(context, response, pred_response_1, pred_response_2)
        break
    return

def plot_check():
    # n = 10
    # x = np.random.rand(n) * 2# 随机产生10个0~2之间的x坐标
    # y = np.random.rand(n) * 2# 随机产生10个0~2之间的y坐标
    # # 2.创建一张figure
    # fig = plt.figure(1)
    # # 6. 正式绘制散点图：scatter
    # plt.scatter(x, y, alpha=0.5, marker='o')
    # plt.xlabel('t')
    # #设置Y轴标签
    # plt.ylabel('diffusion_loss')
    # # 8. 设置图标题：title
    # plt.title('test')
    # plt.savefig('test.png', dpi=200, bbox_inches='tight', transparent=False)

    args = get_parser_config()
    accelerator = Accelerator()
    set_seed(args.seed)
    print_config(args)

    # load model
    if args.load_checkpoint:
        logger.info("Load checkpoint for Model and Tokenizer...")
        model = MODEL_MAP[args.model](args, args.checkpoint)
        # path = os.path.basename(args.checkpoint)
        # resume_step = os.path.splitext(path)[0].replace(f"{args.model}_{args.data}_training_step_", "")
        # resume_step = int(resume_step)
    else:
        logger.info("Initiate Model and Tokenizer...")
        model = MODEL_MAP[args.model](args)
        # resume_step = None
    print_model(model)
    # load data
    (train_loader, dev_loader) = get_dataloaders(
        args, model.tokenizer, model.model, split_list=[args.train_set_split_name, args.dev_set_split_name]
    )
    model, train_loader, dev_loader = accelerator.prepare(model, train_loader, dev_loader)

    # trainer = Trainer(args, model, accelerator, writer)
    # trainer.train(train_loader, dev_loader, resume_step)
    count = 0
    model.eval()
    t = [0 for _ in range(1000)]
    diffusion_rc_loss = [0 for _ in range(1000)]
    rc_loss = [0 for _ in range(1000)]
    movement = [0 for _ in range(1000)]
    for batch_data in tqdm(train_loader):
        with torch.no_grad():
            loss = model.check(**batch_data)
            for t_s, diffusion_rc_loss_s, rc_loss_s, movement_s in zip(loss['t'], loss['diffusion_rc_loss'], loss['rc_loss'], loss['movement']):
                t[t_s] += 1
                diffusion_rc_loss[t_s] += diffusion_rc_loss_s.item()
                rc_loss[t_s] += rc_loss_s.item()
                movement[t_s] += movement_s.item()
        
    
    time_step = [i for i in range(1000)]
    for i in range(1000):
        if t[i] > 0:
            diffusion_rc_loss[i] = diffusion_rc_loss[i] / t[i]
            rc_loss[i] = rc_loss[i] / t[i]
            movement[i] = movement[i] / t[i]

    fig1 = plt.figure(1)
    plt.scatter(time_step, diffusion_rc_loss, alpha=0.5, c="r", label='diffusion_rc_loss')
    plt.scatter(time_step, rc_loss, alpha=0.5, c="b", label='rc_loss')
    plt.xlabel('t')
    plt.title('average rc_loss at different time step')
    plt.savefig('dev_1.png', dpi=200, bbox_inches='tight', transparent=False)

    fig2 = plt.figure(2)
    plt.scatter(time_step, movement, alpha=0.5, c="r")
    plt.xlabel('t')
    plt.ylabel('movement')
    plt.title('movement at different time step')
    plt.savefig('dev_2.png', dpi=200, bbox_inches='tight', transparent=False)
    return

def check_reddit(path):
    lines = []
    with open(path, 'r') as f:
        while True:
            line = f.readline()
            lines.append(line)
            if line == '':
                break
    pdb.set_trace()

class Block1(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(3, 3)
    
    def forward(self, x):
        return self.linear(x)
    
class Block2(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(3, 3)
    
    def forward(self, x):
        return self.linear(x)

class Block3(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(3, 3)
    
    def forward(self, x):
        return self.linear(x)

class Net(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.block1 = Block1()
        self.block2 = Block2()
        self.block3 = Block3()
    
    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = F.linear(x, self.block3.linear.weight)
        return x

if __name__ == "__main__":
    
    # a = torch.randn([3, 1, 3, 3])
    # b = torch.tensor([[0, 1], [1, 0], [1, 1]])
    # b = b[:,None,None,:].repeat(1, 1, 2, 1)
    # a[:,:,1:,1:][b == 1] = 0
    # print(a)
    # pdb.set_trace()
    # plot_check()
    diversity_sents_show(10)
    # for name, para in net.named_parameters():
    #     if 'block3' in name:
    #         para.requires_grad = False
    # z = net(x)
    # z = z.sum()
    # z.backward()
    # print(net.block1.linear.weight.grad)
    # print(net.block2.linear.weight.grad)
    # print(net.block3.linear.weight.grad)
    # a = {}
    # a['yes'] += 2
    # print(a)
    # a['yes'] += 2
    # print(a)
