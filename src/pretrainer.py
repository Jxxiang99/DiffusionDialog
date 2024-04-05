import imp
import os
import time
import torch
import accelerate.logging as logging
import numpy as np
import pdb

from torch.optim import AdamW
from transformers import get_scheduler, Adafactor
# from pretrain.dataloader import *
from dataloader import *
from torch.utils.data import DataLoader, Dataset

COLLATOR_MAP = {'bartbase': BartBaseDialogCollator, 'bartdiffusion': BartDiffusionDialogCollator}


logger = logging.get_logger(__name__)


class PreTrainer:
    def __init__(self, args, model, accelerator, writer, save_model_path):
        self.args = args
        self.model = model
        self.accelerator = accelerator
        self.writer = writer
        self.set_optimizer()
        self.set_scheduler()
        self.save_model_path = save_model_path

    def set_optimizer(self):
        optimizer_name = self.args.optimizer.lower()
        if optimizer_name == "adamw":
            no_decay = ["bias", "LayerNorm.weight"]
            # it's always good practice to set no decay to biase and LayerNorm parameters
            optimizer_grouped_parameters = [
                {
                    "params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
                    "weight_decay": 1e-8,
                },
                {
                    "params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
                    "weight_decay": 0.0,
                },
            ]
            self.optimizer = AdamW(optimizer_grouped_parameters, lr=self.args.learning_rate)
        elif optimizer_name == "adafactor":
            self.optimizer = Adafactor(
                self.model.parameters(), scale_parameter=False, relative_step=False, warmup_init=False, lr=1e-3
            )
        else:
            raise NameError("Unknown optimizer")

    def set_scheduler(self):
        self.scheduler = get_scheduler(
            "constant",
            self.optimizer,
            self.args.warmup_steps * self.accelerator.num_processes,
            self.args.total_steps * self.accelerator.num_processes,
        )

    def print_basic_info(self, train_loader, dev_loader):
        logger.info("Start Training...")
        total_batch_size = self.args.train_batch_size * self.accelerator.num_processes * self.args.grad_accum_steps
        logger.info("Train Instances Size:%d" % len(train_loader.dataset))
        logger.info("  Dev Instances Size:%d" % len(dev_loader.dataset))
        logger.info(f"  1 epoch has {len(train_loader.dataset)/total_batch_size} steps")
        logger.info(f"  Instantaneous batch size per device = {self.args.train_batch_size}")
        logger.info(f"  Gradient Accumulation steps = {self.args.grad_accum_steps}")
        logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
        logger.info(f"  Total optimization steps = {self.args.total_steps}")

    def print_remaining_time(self, start_time, end_time, step):
        avg_time_per_step = (end_time - start_time) / (self.args.logging_steps)
        remaining_step = self.args.total_steps - step
        remaining_hour = avg_time_per_step * remaining_step / 3600
        logger.info(f"Estimated remaining time: {remaining_hour:.3f} hours")

    def model_update(self):
        self.accelerator.clip_grad_value_(self.model.parameters(), self.args.clip_value)
        self.optimizer.step()
        self.scheduler.step()
        self.optimizer.zero_grad()
        self.model.zero_grad()

    def save_model(self, effective_step):
        # save_name = f"{self.args.data}/{self.args.model}_{self.args.data}_training_step_{effective_step}"
        # save_name = os.path.join(self.args.save_model_path, save_name)
        # if not os.path.exists(save_name):
        #     os.makedirs(save_name)
        save_name = f"training_step_{effective_step}"
        save_name = os.path.join(self.save_model_path, save_name)
        if not os.path.exists(save_name):
            os.makedirs(save_name)
        
        logger.info(f"Saving model to {save_name}...")
        unwrapped_model = self.accelerator.unwrap_model(self.model)
        unwrapped_model.model.save_pretrained(save_name, save_function=self.accelerator.save)
        unwrapped_model.tokenizer.save_pretrained(save_name)
        logger.info("Model Saved!")
    


    def sum_up_ListDictLoss(self, loss):
        loss_dict = {name:0.0 for name in loss[0].keys()}
        for slice in loss:
            for key, value in slice.items():
                loss_dict[key] += value.item()
        for key, value in loss_dict.items():
            loss_dict[key] = value / (self.args.logging_steps * self.args.grad_accum_steps)
            loss_dict[key] = round(loss_dict[key], 3)
        return loss_dict

    def train(self, resume_step=None):
        collator = COLLATOR_MAP[self.args.model](self.args, self.model.tokenizer, self.model.model)
        files = os.listdir(self.args.dataset_path)
        train_files = []
        for f in files:
            if 'train' in f:
                train_files.append(os.path.join(self.args.dataset_path, f))
            elif 'valid' in f:
                valid_path = os.path.join(self.args.dataset_path, f)
        dev_dataset = RedditDataset(self.args, valid_path, self.model.tokenizer)
        dev_loader = DataLoader(dev_dataset, batch_size=self.args.dev_batch_size, collate_fn=collator)
        self.model, self.optimizer, self.scheduler, dev_loader = self.accelerator.prepare(
            getattr(self.model, "module", self.model), self.optimizer, self.scheduler, dev_loader
        )

        step = 0
        effective_step = 0
        epoch = 0
        batch_loss_history = []

        while True:
            epoch += 1
            self.model.train()
            self.model.zero_grad()

            # if self.args.load_checkpoint and resume_step is not None:
            #     effective_step = resume_step
            start_time = time.time()

            logger.info(f"=========Epoch: {epoch}=========")
            for train_path in train_files:
                logger.info(f"loading the dataset {train_path}")
                train_dataset = RedditDataset(self.args, train_path, getattr(self.model, "module", self.model).tokenizer)
                train_loader = DataLoader(train_dataset, batch_size=self.args.train_batch_size, collate_fn=collator, shuffle=True)
                train_loader= self.accelerator.prepare(train_loader)

                self.model.train()
                self.model.zero_grad()
                self.print_basic_info(train_loader, dev_loader)
                for batch in train_loader:
                    step += 1

                    # if self.args.load_checkpoint:
                    #     if resume_step is not None and effective_step < resume_step:
                    #         if step % self.args.grad_accum_steps == 0:
                    #             effective_step += 1
                    #             if effective_step % 1000 == 0:
                    #                 logger.info(f"skip steps {effective_step}/{resume_step}")
                    #                 start_time = time.time()
                    #         continue

                    loss_dict = self.model(**batch)
                    batch_loss_history.append(loss_dict)
                    loss = 0.0
                    for name, value in loss_dict.items():
                        loss += value
                    loss = loss / self.args.grad_accum_steps
                    self.accelerator.backward(loss)

                    # batch_loss = self.model(**batch)
                    # batch_loss_history.append(batch_loss)
                    # batch_loss = batch_loss / self.args.grad_accum_steps
                    # self.accelerator.backward(batch_loss)

                    if step % self.args.grad_accum_steps == 0:
                        self.model_update()
                        effective_step += 1

                        # 间隔打印信息 print intermediate result
                        if effective_step % self.args.logging_steps == 0:
                            end_time = time.time()
                            # batch_loss_history = torch.stack(self.accelerator.gather(batch_loss_history))

                            one_train_loss = self.sum_up_ListDictLoss(batch_loss_history)

                            if self.accelerator.is_main_process:
                                for key, value in one_train_loss.items():
                                    self.writer.add_scalar(key, value, effective_step)
                                learning_rate = self.optimizer.state_dict()["param_groups"][0]["lr"]
                                getattr(self.model, "module", self.model).show_loss_weight()
                                self.writer.add_scalar(
                                    "learning_rate", learning_rate, effective_step
                                )
                                logger.info(
                                    f"At train step {effective_step} Train Loss: {one_train_loss}, consumes time: {end_time-start_time:.3f}, learning rate: {learning_rate}"
                                )
                                self.print_remaining_time(start_time, end_time, effective_step)

                            batch_loss_history = []

                            start_time = time.time()

                        # Eval
                        if effective_step % self.args.eval_steps == 0:

                            one_dev_loss = self.eval(dev_loader)

                            if self.accelerator.is_main_process:
                                self.writer.add_scalar("dev/loss", one_dev_loss, effective_step)
                                one_dev_ppl = round(float(np.exp(one_dev_loss)), 3)
                                logger.info(
                                    f"At training steps {effective_step}, dev loss={one_dev_loss:.3f}, ppl={one_dev_ppl:.3f}"
                                )

                                self.save_model(effective_step)

                            start_time = time.time()

                    # 检查是否到达训练step上限
                    if effective_step >= self.args.total_steps:
                        break

                del train_dataset
                del train_loader
                  
                if effective_step >= self.args.total_steps:
                    break

            if effective_step >= self.args.total_steps:
                    break

        if self.accelerator.is_main_process:
            self.writer.close()

    def eval(self, dev_loader):
        self.model.eval()
        # 评估开发集
        logger.info("Eval dev dataset perplexity ...")
        batch_loss_history = []
        n_total_words = []

        for batch_i, dev_batch in enumerate(dev_loader):

            if batch_i == 0:
                self.generate_sentence(**dev_batch)
            
            if batch_i % self.args.logging_steps == 0:
                logger.info(f"have evaled {batch_i} steps")

            with torch.no_grad():
                batch_loss, n_words = getattr(self.model, "module", self.model).eval_loss(**dev_batch)

            batch_loss_history.append(batch_loss)
            n_total_words.append(n_words)

        batch_loss_history = torch.stack(self.accelerator.gather(batch_loss_history))
        n_total_words = torch.stack(self.accelerator.gather(n_total_words))

        eval_loss = batch_loss_history.sum() / n_total_words.sum()
        self.model.train()
        return eval_loss.item()

    def generate_sentence(self, **kwargs):
        self.model.eval()

        with torch.no_grad():
            generated = getattr(self.model, "module", self.model).generate(**kwargs)

        labels = kwargs["labels"]
        input_ids = kwargs["input_ids"]
        labels[labels == -100] = 0
        input_ids = self.accelerator.gather(self.accelerator.pad_across_processes(input_ids, getattr(self.model, "module", self.model).tokenizer.pad_token_id))
        labels = self.accelerator.gather(self.accelerator.pad_across_processes(labels, getattr(self.model, "module", self.model).tokenizer.pad_token_id))
        generated = self.accelerator.gather(self.accelerator.pad_across_processes(generated, getattr(self.model, "module", self.model).tokenizer.pad_token_id))

        input_sents = getattr(self.model, "module", self.model).tokenizer.batch_decode(input_ids, skip_special_tokens=True)
        target_sents = getattr(self.model, "module", self.model).tokenizer.batch_decode(labels, skip_special_tokens=True)
        output_sents = getattr(self.model, "module", self.model).tokenizer.batch_decode(generated, skip_special_tokens=True)

        for input_sent, target_sent, output_sent in zip(input_sents, target_sents, output_sents):
            s = "\n".join(
                ["Input sentence: " + input_sent, "Ground truth: " + target_sent, "Generated response: " + output_sent + "\n"]
            )
            logger.info(s)
