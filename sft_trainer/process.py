import multiprocessing as mp
import os
from math import log2
from typing import Literal

import numpy as np
import tiktoken
from tqdm import tqdm

from datasets import Dataset, IterableDataset, load_dataset
from dpo_trainer.utils import get_tokenizer, setup_logger

# Create or get a logger
logger = setup_logger(__name__)


class DataProcessing:
    def __init__(
        self,
        dataset: Dataset | IterableDataset,
        tokenizer: tiktoken.core.Encoding,
        split: Literal["train", "validation"] = "train",
        shard_size: int | None = None,
        max_sequence_length: int = 256,
        instruction_key="query",
        answer_key="response",
        user_format: str = "### User:",
        assistant_format: str = "### Assistant:",
        output_name_dir: str = "gsm8k",
        nprocs: int = max(1, os.cpu_count() // 2),
        mode: Literal["instruction", "json"] = "instruction",
    ):
        assert log2(
            max_sequence_length
        ).is_integer(), "max_sequence_length must be a power of 2"

        assert isinstance(dataset, Dataset) or isinstance(
            dataset, IterableDataset
        ), "dataset must be a HF Dataset or IterableDataset object"

        assert isinstance(
            tokenizer, tiktoken.core.Encoding
        ), "tokenizer must be a tiktoken Encoding object"

        logger.info(f"User format: {user_format}, Assistant format: {assistant_format}")
        logger.info("Max sequence length: %s", max_sequence_length)
        logger.info("Number of processes: %s", nprocs)
        logger.info("Output directory: %s", output_name_dir)

        self.dataset = dataset
        self.tokenizer = tokenizer
        self.max_sequence_length = max_sequence_length

        self.split = split

        self.output_name_dir = output_name_dir
        self.output_dir_path = os.path.join("datasets", self.output_name_dir)
        os.makedirs(self.output_dir_path, exist_ok=True)
        self.root_filename = os.path.join(
            self.output_dir_path, f"{self.output_name_dir}_{split}_"
        )
        
        self.shard_size = self._calculate_shard_size(shard_size)
        logger.info("Shard size: %s", self.shard_size)

        self.instruction_key = instruction_key
        self.answer_key = answer_key
        self.user_format = user_format
        self.assistant_format = assistant_format

        self.nprocs = nprocs

    def _calculate_shard_size(self, shard_size: int | None) -> int:
        if shard_size is None:
            if isinstance(self.dataset, Dataset):
                shard_size = min(int(1e6), len(self.dataset))
            else:
                shard_size = int(1e6)
        return shard_size

    def tokenize(self, row) -> np.ndarray:
        tokens = self.tokenizer.encode_ordinary(self.user_format)
        tokens.extend(
            self.tokenizer.encode_ordinary(" " + row[self.instruction_key] + " ")
        )
        tokens.extend(self.tokenizer.encode_ordinary(self.assistant_format))
        tokens.extend(self.tokenizer.encode_ordinary(" " + row[self.answer_key]))
        tokens_np = np.array(tokens)
        assert (0 <= tokens_np).all() and (
            tokens_np < 2**16
        ).all(), "token dictionary too large for uint16"

        tokens_np_uint16 = tokens_np.astype(np.uint16)
        return tokens_np_uint16

    def _validate_max_sequence_length(self, tokens, eot):
        if len(tokens) > self.max_sequence_length:
            tokens = tokens[: self.max_sequence_length - 1]
            tokens = np.append(tokens, np.array([eot]), axis=0)
        elif len(tokens) < self.max_sequence_length:
            padding_array = np.array([eot] * (self.max_sequence_length - len(tokens)))
            tokens = np.append(tokens, padding_array, axis=0)
        assert (
            len(tokens) == self.max_sequence_length
        ), f"Token length mismatch the max sequence length {self.max_sequence_length}"
        return tokens

    def _create_np_array(self):
        return np.empty((self.shard_size, self.max_sequence_length), dtype=np.uint16)

    def process(self):
        dataset_length = len(self.dataset) if hasattr(self.dataset, "__len__") else None
        with tqdm(total=dataset_length, desc="Processing", unit="chunks") as pbar:
            with mp.Pool(self.nprocs) as pool:
                eot = self.tokenizer._special_tokens["<|endofprompt|>"]

                shard_index, row_count = 0, 0
                arr_tokens_np = self._create_np_array()

                for tokens in pool.imap(self.tokenize, self.dataset, chunksize=16):
                    tokens = self._validate_max_sequence_length(tokens, eot)
                    if row_count == self.shard_size:
                        self._write_datafile(
                            f"{self.root_filename}{shard_index:06d}",
                            arr_tokens_np[:row_count, :],
                        )
                        arr_tokens_np = self._create_np_array()
                        row_count = 0
                        shard_index += 1
                    arr_tokens_np[row_count, :] = tokens
                    row_count += 1
                    pbar.update()

                if row_count != 0:
                    self._write_datafile(
                        f"{self.root_filename}{shard_index:06d}",
                        arr_tokens_np[:row_count, :],
                    )

    @staticmethod
    def _write_datafile(filename: str, tokens_np: np.ndarray) -> None:
        np.save(filename, tokens_np)


if __name__ == "__main__":
    split = "train"
    # dataset = load_dataset("openai/gsm8k", "main", split="train", streaming=True)
    dataset = load_dataset("databricks/databricks-dolly-15k", split="train", num_proc=8)
    enc = get_tokenizer()

    dp = DataProcessing(
        dataset=dataset,
        max_sequence_length=1024,
        split=split,
        output_name_dir="databricks-dolly-15k",
        tokenizer=enc,
        instruction_key="instruction",
        answer_key="response",
        nprocs=14,
    )
    dp.process()
    print("Done!")
