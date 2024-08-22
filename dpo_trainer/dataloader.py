import os
from dataclasses import dataclass
from functools import partial
from typing import Literal

import tiktoken
import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from shared.utils import setup_logger

from datasets import Dataset as HF_Dataset
from datasets import load_dataset

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
    for sample in batch:
        max_sequence_length = min(
            max_sequence_length,
            max(len(sample["chosen_ids"]), len(sample["rejected_ids"])),
        )

    for sample in batch:
        chosen_ids, mask_chosen = pad_or_truncate(
            sample["chosen_ids"], max_sequence_length, eos_token_id
        )
        labels_chosen, _ = pad_or_truncate(
            sample["labels_chosen"], max_sequence_length, -100
        )

        rejected_ids, mask_rejected = pad_or_truncate(
            sample["rejected_ids"], max_sequence_length, eos_token_id
        )
        labels_rejected, _ = pad_or_truncate(
            sample["labels_rejected"], max_sequence_length, -100
        )

        x.extend([chosen_ids, rejected_ids])
        mask.extend([mask_chosen, mask_rejected])
        y.extend([labels_chosen, labels_rejected])

    return (
        torch.tensor(x, dtype=torch.long),
        torch.tensor(mask, dtype=torch.long),
        torch.tensor(y, dtype=torch.long),
    )


@dataclass
class DPOColumnsMapping(object):
    prompt: str
    chosen: str
    rejected: str


class HuggingFaceDataset(Dataset):
    def __init__(
        self,
        dataset: HF_Dataset,
        tokenizer: tiktoken.Encoding,
        column_mapping: DPOColumnsMapping,
    ) -> None:
        self.dataset: HF_Dataset = dataset
        self.tokenizer: tiktoken.Encoding = tokenizer
        self.prompt: str = column_mapping.prompt
        self.chosen: str = column_mapping.chosen
        self.rejected: str = column_mapping.rejected
        self.instruction_key: str = "Question: "
        self.answer_key: str = "\n\nAnswer: "
        self.instruction_ids: list[int] = self.tokenizer.encode_ordinary(
            self.instruction_key
        )
        self.answer_ids: list[int] = self.tokenizer.encode_ordinary(self.answer_key)

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> dict[str, int]:
        prompt_ids = self.tokenizer.encode_ordinary(self.dataset[idx][self.prompt])
        chosen_ids = self.tokenizer.encode_ordinary(self.dataset[idx][self.chosen])
        rejected_ids = self.tokenizer.encode_ordinary(self.dataset[idx][self.rejected])
        labels_chosen = (
            len(self.instruction_ids + prompt_ids + self.answer_ids) * [-100]
            + chosen_ids
        )
        labels_rejected = (
            len(self.instruction_ids + prompt_ids + self.answer_ids) * [-100]
            + rejected_ids
        )

        rejected_ids = (
            self.instruction_ids + prompt_ids + self.answer_ids + rejected_ids
        )
        chosen_ids = self.instruction_ids + prompt_ids + self.answer_ids + chosen_ids
        return {
            "chosen_ids": chosen_ids,
            "labels_chosen": labels_chosen,
            "rejected_ids": rejected_ids,
            "labels_rejected": labels_rejected,
        }


def create_dataloader(
    dataset_name: str,
    tokenizer: tiktoken.Encoding,
    batch_size: int = 32,
    max_sequence_length: int = 1024,
    split: Literal["train", "test", "validation"] = "train",
    shuffle: bool = False,
    columns_mapping: DPOColumnsMapping = DPOColumnsMapping(
        "prompt", "chosen", "rejected"
    ),
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
