import os
from dataclasses import dataclass
from functools import partial
from typing import Literal

import tiktoken
import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

from datasets import Dataset as HF_Dataset
from datasets import load_dataset
from shared.utils import setup_logger

logger = setup_logger(__name__)


def pad_or_truncate(sequence, max_length, pad_value):
    """
    Pads the sequence to the max_length with pad_value or truncates it if necessary.
    """
    seq_len = len(sequence)
    mask = [1] * seq_len
    if seq_len < max_length:
        return sequence + [pad_value] * (max_length - seq_len), mask + [0] * (
            max_length - seq_len
        )
    else:
        return sequence[:max_length], mask[:max_length]


def collate_fn(batch, max_sequence_length: int = 1024, eos_token_id: int = 50256):
    x, y, mask = [], [], []
    max_batch_sequence = 0
    for sample in batch:
        max_batch_sequence = max(max_batch_sequence, len(sample["input_ids"]))
    max_sequence_length = min(max_sequence_length, max_batch_sequence)

    for sample in batch:
        input_ids, mask_input_ids = pad_or_truncate(
            sample["input_ids"], max_sequence_length, eos_token_id
        )
        label_ids, _ = pad_or_truncate(sample["label_ids"], max_sequence_length, -100)

        x.extend([input_ids])
        mask.extend([mask_input_ids])
        y.extend([label_ids])

    return (
        torch.tensor(x, dtype=torch.long),
        torch.tensor(mask, dtype=torch.long),
        torch.tensor(y, dtype=torch.long),
    )


@dataclass
class SFTColumnsMapping(object):
    prompt: str
    answer: str


class HuggingFaceDataset(Dataset):
    def __init__(
        self,
        dataset: HF_Dataset,
        tokenizer: tiktoken.Encoding,
        column_mapping: SFTColumnsMapping,
    ) -> None:
        self.dataset: HF_Dataset = dataset
        self.tokenizer: tiktoken.Encoding = tokenizer
        self.prompt: str = column_mapping.prompt
        self.answer: str = column_mapping.answer
        self.prompt_key: str = "Question: "
        self.answer_key: str = "\n\nAnswer: "
        self.instruction_ids: list[int] = self.tokenizer.encode_ordinary(
            self.prompt_key
        )
        self.answer_ids: list[int] = self.tokenizer.encode_ordinary(self.answer_key)
        self.eos_token_id = tokenizer._special_tokens["<|endoftext|>"]

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> dict[str, int]:
        prompt_ids = self.tokenizer.encode_ordinary(self.dataset[idx][self.prompt])
        label_ids = self.tokenizer.encode_ordinary(self.dataset[idx][self.answer])
        labels = (
            len(self.instruction_ids + prompt_ids + self.answer_ids) * [-100]
            + label_ids
            + [self.eos_token_id]
        )
        input_ids = (
            self.instruction_ids
            + prompt_ids
            + self.answer_ids
            + label_ids
            + [self.eos_token_id]
        )
        return {
            "prompt": self.dataset[idx][self.prompt],
            "answer": self.dataset[idx][self.answer],
            "input_ids": input_ids,
            "label_ids": labels,
        }


def create_dataloader(
    dataset_name: str,
    tokenizer: tiktoken.Encoding,
    batch_size: int = 32,
    max_sequence_length: int = 1024,
    split: Literal["train", "test", "validation"] = "train",
    shuffle: bool = False,
    columns_mapping: SFTColumnsMapping = SFTColumnsMapping("instruction", "response"),
):
    """
    Create a dataloader for the DPO task.
    dataset_name: str = "Dahoas/full-hh-rlhf"
    """
    ds = load_dataset(dataset_name, split=split)
    dataset = HuggingFaceDataset(ds, tokenizer, columns_mapping)
    sampler = DistributedSampler(
        dataset=dataset,
        num_replicas=int(os.getenv("WORLD_SIZE", 1)),
        rank=int(os.getenv("LOCAL_RANK", 0)),
        shuffle=shuffle,
        drop_last=True,
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=partial(collate_fn, max_sequence_length=max_sequence_length),
        sampler=sampler,
    )
