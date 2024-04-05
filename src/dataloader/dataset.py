import os
import torch
import accelerate.logging as logging
from torch.utils.data import DataLoader, Dataset, IterableDataset
# import zstandard as zstd
import io
# import msgspec
from accelerate import Accelerator
from typing import List
import pdb


accelerator = Accelerator()

logger = logging.get_logger(__name__)


class DailyDialogDataset(Dataset):
    def __init__(self, args, split, tokenizer):

        self.tokenizer = tokenizer
        self.context_list = []
        self.response_list = []
        data_dir = args.dataset_path
        max_context_turns = args.max_context_turn

        # load data
        with open(os.path.join(data_dir, f"dial.{split}"), "r", encoding="utf-8") as f:
            for line in f:
                if line:
                    line = line.strip()
                    context, response = line.split("\t")
                    context = context.split("__eou__")
                    context = str(tokenizer.sep_token).join(context[-max_context_turns:])
                    self.context_list.append(context)
                    self.response_list.append(response)

    def __len__(self):
        return len(self.context_list)

    def __getitem__(self, index: int):
        return {"context": self.context_list[index], "response": self.response_list[index]}

class PersonaChatDataset(Dataset):
    def __init__(self, args, split, tokenizer):

        self.tokenizer = tokenizer
        self.context_list = []
        self.response_list = []
        data_dir = args.dataset_path
        max_context_turns = args.max_context_turn

        # load data
        with open(os.path.join(data_dir, f"dial.{split}"), "r", encoding="utf-8") as f:
            for line in f:
                if line:
                    line = line.strip()
                    persona, context, response = line.split("\t")
                    context = context.split("__eou__")
                    context = str(tokenizer.sep_token).join(context[-max_context_turns:])
                    self.context_list.append(context)
                    self.response_list.append(response)

    def __len__(self):
        return len(self.context_list)

    def __getitem__(self, index: int):
        return {"context": self.context_list[index], "response": self.response_list[index]}


class WoWDataset(Dataset):
    def __init__(self, data_dir, split, tokenizer, max_context_turns):

        self.tokenizer = tokenizer
        self.context_list = []
        self.document_list = []
        self.response_list = []

        # load data
        with open(os.path.join(data_dir, f"{split}.txt"), "r", encoding="utf-8") as f:
            for line in f:
                if line:
                    line = line.strip().split("\t")
                    context = line[0]
                    document = line[1]
                    response = line[2]

                    context_list = context.split(" [SEP] ")
                    context = str(tokenizer.eos_token).join(context_list[-max_context_turns:])

                    document_list = document.split(" [SEP] ")
                    document = str(tokenizer.eos_token).join(document_list)

                    self.context_list.append(context)
                    self.document_list.append(document)
                    self.response_list.append(response)

    def __len__(self):
        return len(self.context_list)

    def __getitem__(self, index: int):
        return {
            "context": self.context_list[index],
            "document": self.document_list[index],
            "response": self.response_list[index],
        }


class PlanWoWDataset(Dataset):
    def __init__(self, data_dir, split, tokenizer, max_context_turns):

        self.tokenizer = tokenizer
        self.context_list = []
        self.document_list = []
        self.plan_list = []
        self.response_list = []

        # load data
        with open(os.path.join(data_dir, f"{split}.txt.plan"), "r", encoding="utf-8") as f:
            for line in f:
                if line:
                    line = line.strip().split("\t")
                    context = line[0]
                    document = line[1]
                    response = line[2]
                    plan = line[3]

                    context_list = context.split(" [SEP] ")
                    context = str(tokenizer.eos_token).join(context_list[-max_context_turns:])

                    document_list = document.split(" [SEP] ")
                    document = str(tokenizer.eos_token).join(document_list)

                    self.context_list.append(context)
                    self.document_list.append(document)
                    self.plan_list.append(plan)
                    self.response_list.append(response)

    def __len__(self):
        return len(self.context_list)

    def __getitem__(self, index: int):
        return {
            "context": self.context_list[index],
            "document": self.document_list[index],
            "plan": self.plan_list[index],
            "response": self.response_list[index],
        }

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

# class RedditDataset(IterableDataset):
#     def __init__(self, args, split, tokenizer) -> None:
#         super().__init__()
#         self.files = []

#         data_dir = args.dataset_path
#         if data_dir is not None:
#             files = os.listdir(data_dir)
#             for f in files:
#                 if f.startswith("DLGS") and f.endswith(f"{split}.zst"):
#                     self.files.append(os.path.join(data_dir, f))
#         else:
#             raise NotImplementedError

#         self.files = sorted(self.files)
#         self.tokenizer = tokenizer
#         self.max_context_turns = args.max_context_turn
    
#     def __iter__(self):
#         for filepath in self.files:
#             print("generating examples from = %s", filepath)
#             fh = open(filepath, "rb")
#             dctx = zstd.ZstdDecompressor(max_window_size=2147483648)
#             stream_reader = dctx.stream_reader(fh)
#             f = io.TextIOWrapper(stream_reader, encoding="utf-8")
#             decoder = msgspec.json.Decoder()

#             for line in f:
#                 if line:
#                     sample = decoder.decode(line.encode("utf-8"))
#                     context = sample['context']
#                     context = str(self.tokenizer.sep_token).join(context[-self.max_context_turns:])
#                     response = sample['response']
#                     yield {"context" : context, "response" : response}
#             fh.close()