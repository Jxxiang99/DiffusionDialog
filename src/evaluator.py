import os
import torch
import accelerate.logging as logging
import numpy as np
from tqdm import tqdm
from nlgeval import NLGEval
from typing import List
from metrics import F1Metric
from datetime import datetime


logger = logging.get_logger(__name__)


class Evaluator:
    def __init__(self, args, model, accelerator, writer, save_result_path):
        self.args = args
        self.model = model
        self.accelerator = accelerator
        self.writer = writer
        logger.info("Evaluater initialize ready")
        self.save_result_path = save_result_path

    @staticmethod
    def eval_distinct(corpus):
        unigrams = []
        bigrams = []
        for n, rep in enumerate(corpus):
            rep = rep.strip()
            temp = rep.split(" ")
            unigrams += temp
            for i in range(len(temp) - 1):
                bigrams.append(temp[i] + " " + temp[i + 1])
        distink_1 = len(set(unigrams)) * 1.0 / len(unigrams)
        distink_2 = len(set(bigrams)) * 1.0 / len(bigrams)
        return distink_1, distink_2

    @staticmethod
    def eval_f1(pred: List[str], gold: List[str]):
        return F1Metric.compute_all_pairs(pred, gold)

    def save_result(self, gold, pred):
        current_time = datetime.now().strftime("%b%d_%H-%M-%S")
        save_result_dir = self.save_result_path
        if not os.path.exists(save_result_dir):
            os.makedirs(save_result_dir)
        gold_path = f"{current_time}_gold.txt"
        gold_path = os.path.join(save_result_dir, gold_path)
        with open(gold_path, "w", encoding="utf-8") as outf:
            for g in gold:
                outf.write(g + "\n")
        pred_path = f"{current_time}_pred.txt"
        pred_path = os.path.join(save_result_dir, pred_path)
        with open(pred_path, "w", encoding="utf-8") as outf:
            for p in pred:
                outf.write(p + "\n")

        logger.info(f"Save gold result to {gold_path}")
        logger.info(f"Save pred result to {pred_path}")

    def _compute_metrics(self, pred_sents_all, gold_sents_all):
        p, r, f = self.eval_f1(pred_sents_all, gold_sents_all)

        gold_dist1, gold_dist2 = self.eval_distinct(gold_sents_all)
        pred_dist1, pred_dist2 = self.eval_distinct(pred_sents_all)

        nlgeval = NLGEval(no_skipthoughts=True, no_glove=True)

        gold_sents_all = [gold_sents_all]

        res = nlgeval.compute_metrics(ref_list=gold_sents_all, hyp_list=pred_sents_all)
        res["precision"] = p
        res["recall"] = r
        res["f1"] = f
        res["gold_dist1"] = gold_dist1
        res["gold_dist2"] = gold_dist2
        res["pred_dist1"] = pred_dist1
        res["pred_dist2"] = pred_dist2

        return res

    def eval(self, dataloader, step, split=None, save_result=True):
        self.model, dataloader = self.accelerator.prepare(self.model, dataloader)
        self.model.eval()
        logger.info("Evaluate perplexity...")
        batch_loss_history = []
        n_total_words = []
        for batch in dataloader:

            with torch.no_grad():
                batch_loss, n_words = getattr(self.model, "module", self.model).eval_loss(**batch)

            batch_loss_history.append(batch_loss)
            n_total_words.append(n_words)

        batch_loss_history = torch.stack(self.accelerator.gather(batch_loss_history))
        n_total_words = torch.stack(self.accelerator.gather(n_total_words))

        eval_loss = batch_loss_history.sum() / n_total_words.sum()
        logger.info(f"Bits per word: {eval_loss.item():.3f}")

        word_perplexity = np.exp(eval_loss.item())
        logger.info(f"Perplexity is {word_perplexity:.3f}.")

        pred_sents_all, gold_sents_all = self.predict(dataloader)

        # post_process
        if hasattr(getattr(self.model, "module", self.model), "post_process"):
            pred_sents_all = getattr(self.model, "module", self.model).post_process(pred_sents_all)
            gold_sents_all = getattr(self.model, "module", self.model).post_process(gold_sents_all)

        if self.accelerator.is_main_process and save_result:
            self.save_result(gold_sents_all, pred_sents_all)

        res = self._compute_metrics(pred_sents_all, gold_sents_all)
        res["loss"] = eval_loss.item()
        res["ppl"] = word_perplexity

        logger.info(f"On step {step}, res of {split}:")
        for k, v in res.items():
            logger.info(f"{k}:\t{v:.3f}")
            if self.accelerator.is_main_process:
                self.writer.add_scalar(f"{split}/{k}", v, step)

        return res

    def predict(self, dataloader):

        self.model.eval()

        gold_sents_all = []
        pred_sents_all = []

        progress_bar = tqdm(range(len(dataloader)), disable=not self.accelerator.is_local_main_process)

        for i, batch in enumerate(dataloader):
            progress_bar.update(1)
            labels = batch["labels"]

            with torch.no_grad():
                generated = getattr(self.model, "module", self.model).generate(**batch)

            # TODO: 有未知bug，第一个batch会导致encoder中第一个linear层输出[non],导致后续解码错误
            if i == 0:
                with torch.no_grad():
                    generated = getattr(self.model, "module", self.model).generate(**batch)

            labels[labels == -100] = 0
            pred_temp = getattr(self.model, "module", self.model).tokenizer.batch_decode(
                self.accelerator.gather(self.accelerator.pad_across_processes(generated, 1)), skip_special_tokens=True,
            )
            gold_temp = getattr(self.model, "module", self.model).tokenizer.batch_decode(
                self.accelerator.gather(self.accelerator.pad_across_processes(labels, 1)), skip_special_tokens=True,
            )

            pred_sents_all += pred_temp
            gold_sents_all += gold_temp

        if len(pred_sents_all) > len(dataloader.dataset):
            print('length not balanced')
            pred_sents_all = pred_sents_all[: len(dataloader.dataset)]
            gold_sents_all = gold_sents_all[: len(dataloader.dataset)]

        return (pred_sents_all, gold_sents_all)
