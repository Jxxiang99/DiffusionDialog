from collections import Counter

from nltk.translate import bleu_score
from nltk.translate.bleu_score import SmoothingFunction
import numpy as np
import argparse
import torch
import os
from datetime import datetime
import accelerate.logging as logging
from tqdm import tqdm
from typing import List
from transformers import AutoTokenizer
from metrics import F1Metric
import pdb
from collections import defaultdict
logger = logging.get_logger(__name__)


# references:
# https://github.com/lemuria-wchen/Research/blob/master/NLP/Dialogue-PLATO/plato/metrics/metrics.py
# https://github.com/lemuria-wchen/Research/blob/master/NLP/Dialogue-PLATO/tools/dstc7_avsd_eval.py
# https://github.com/lemuria-wchen/Research/blob/master/NLP/Dialogue-PLATO/tools/knowledge_f1.py

# This script integrates all evaluation methods proposed in the Plato article.
# ACL 2020: https://www.aclweb.org/anthology/2020.acl-main.9.pdf
# Repository: https://github.com/PaddlePaddle/Research/tree/master/NLP/Dialogue-PLATO
def tokenize_sents(sents, tokenizer):
    transfered_sents = []
    for sent in sents:
        tokens = tokenizer.tokenize(sent.strip())
        sent = ' '.join(tokens)
        sent = sent.replace(' ##', '')
        sent = sent.split()
        transfered_sents.append(sent)
    return transfered_sents

class NLTK_Evaluator:
    def __init__(self, args, model, accelerator, writer, save_result_path):
        self.args = args
        self.model = model
        self.accelerator = accelerator
        self.writer = writer
        self.tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
        logger.info("Evaluater initialize ready")
        self.save_result_path = save_result_path

    @staticmethod
    def eval_distinct(corpus):
        intra_dist1, intra_dist2 = [], []
        unigrams_all, bigrams_all = Counter(), Counter()
        for hyp in corpus:
            unigrams = Counter(hyp)
            bigrams = Counter(zip(hyp, hyp[1:]))
            intra_dist1.append((len(unigrams)+1e-12) / (len(hyp)+1e-5))
            intra_dist2.append((len(bigrams)+1e-12) / (max(0, len(hyp)-1)+1e-5))
            unigrams_all.update(unigrams)
            bigrams_all.update(bigrams)
        inter_dist1 = (len(unigrams_all)+1e-12) / (sum(unigrams_all.values())+1e-5)
        inter_dist2 = (len(bigrams_all)+1e-12) / (sum(bigrams_all.values())+1e-5)
        intra_dist1 = np.average(intra_dist1)
        intra_dist2 = np.average(intra_dist2)
        return intra_dist1, intra_dist2, inter_dist1, inter_dist2

    @staticmethod
    def eval_f1(pred: List[str], gold: List[str]):
        return F1Metric.compute_all_pairs(pred, gold)

    @staticmethod
    def eval_bleu(hyps, refs) -> tuple:
        bleu_1, bleu_2 = [], []
        for hyp, ref in zip(hyps, refs):
            try:
                score = bleu_score.sentence_bleu(
                    [ref], hyp, smoothing_function=SmoothingFunction().method7, weights=[1, 0, 0, 0])
            except Exception as e:
                print(e)
                score = 0
            bleu_1.append(score)
            try:
                score = bleu_score.sentence_bleu(
                    [ref], hyp, smoothing_function=SmoothingFunction().method7, weights=[0.5, 0.5, 0, 0])
            except Exception as e:
                print(e)
                score = 0
            bleu_2.append(score)
        bleu_1, bleu_2 = np.average(bleu_1), np.average(bleu_2)
        return bleu_1, bleu_2

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

        gold_tokenized = tokenize_sents(gold_sents_all, self.tokenizer)
        pred_tokenized = tokenize_sents(pred_sents_all, self.tokenizer)

        pred_intra_dist1, pred_intra_dist2, pred_inter_dist1, pred_inter_dist2 = self.eval_distinct(pred_tokenized)


        gold_sents_all = [gold_sents_all]

        blue1, blue2 = self.eval_bleu(pred_tokenized, gold_tokenized)

        res = defaultdict()
        res["Bleu_1"] = blue1
        res["Bleu_2"] = blue2
        res["dist1"] = pred_inter_dist1
        res["dist2"] = pred_inter_dist2

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

        if save_result is True:
            pred_sents_all, gold_sents_all = self.predict(dataloader)

            # post_process
            if hasattr(getattr(self.model, "module", self.model), "post_process"):
                pred_sents_all = getattr(self.model, "module", self.model).post_process(pred_sents_all)
                gold_sents_all = getattr(self.model, "module", self.model).post_process(gold_sents_all)

            if self.accelerator.is_main_process and save_result:
                self.save_result(gold_sents_all, pred_sents_all)

            res = self._compute_metrics(pred_sents_all, gold_sents_all)
        else:
            res = defaultdict()
            
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
