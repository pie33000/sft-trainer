from dataclasses import dataclass
from functools import partial
from typing import Literal

import tiktoken
import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

from datasets import Dataset as HF_Dataset
from datasets import load_dataset
from utils import setup_logger

logger = setup_logger(__name__)


# bug to fix with label and tokens ids size, how is it possible to have different sizes?
def collate_fn(batch, max_sequence_length: int = 1024):
    x, y = [], []
    for i in range(len(batch)):
        if len(batch[i]["chosen_ids"]) <= max_sequence_length:
            batch[i]["chosen_ids"] += [50256] * (
                max_sequence_length - len(batch[i]["chosen_ids"])
            )
            batch[i]["labels_chosen"] += [-100] * (
                max_sequence_length - len(batch[i]["labels_chosen"])
            )
        else:
            batch[i]["chosen_ids"] = batch[i]["chosen_ids"][:max_sequence_length]
            batch[i]["labels_chosen"] = batch[i]["labels_chosen"][:max_sequence_length]
        if len(batch[i]["rejected_ids"]) <= max_sequence_length:
            batch[i]["rejected_ids"] += [50256] * (
                max_sequence_length - len(batch[i]["rejected_ids"])
            )
            batch[i]["labels_rejected"] += [-100] * (
                max_sequence_length - len(batch[i]["labels_rejected"])
            )
        else:
            batch[i]["rejected_ids"] = batch[i]["rejected_ids"][:max_sequence_length]
            batch[i]["labels_rejected"] = batch[i]["labels_rejected"][
                :max_sequence_length
            ]
        x.append(batch[i]["chosen_ids"])
        y.append(batch[i]["labels_chosen"])
        x.append(batch[i]["rejected_ids"])
        y.append(batch[i]["labels_rejected"])
    return torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)


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
    rank: int = 0,
    num_replicas: int = 1,
    shuffle: bool = True,
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
        num_replicas=num_replicas,
        rank=rank,
        shuffle=shuffle,
        drop_last=True,
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=partial(collate_fn, max_sequence_length=max_sequence_length),
        sampler=sampler,
    )
