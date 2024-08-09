import numpy as np
import torch
from datasets import load_dataset

from utils import get_tokenizer


class DataLoaderLite:
    def __init__(
        self,
        dataset,
        tokenizer,
        max_length: int,
        batch_size: int = 32,
        instruction_key: str = "question",
        answer_key: str = "answer",
        process_rank: int = 0,
        num_processes: int = 8,
    ):
        self.dataset = dataset
        self.data = []
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.batch_size = batch_size
        self.instruction_key = instruction_key
        self.answer_key = answer_key
        self.max_shard_size = int(1e6)

        self._eot = self.tokenizer._special_tokens["<|endoftext|>"]
        # self._eop = self.tokenizer._special_tokens["<|endofprompt|>"]
        self._eop = self.tokenizer._special_tokens["<|endoftext|>"]

        self.process_rank = process_rank
        self.num_processes = num_processes

        self.reset()

        # Shard the dataset
        self.get_shard()
        self.data = torch.tensor(self.data, dtype=torch.uint32)

    def reset(self):
        self.current_shard_start_index = 0
        self.current_shard_token_index = 0
        self.current_position = self.batch_size * self.max_length * self.process_rank

    def tokenize(self, row):
        # tokenizes a single document and returns a numpy array of uint16 tokens
        tokens = [self._eot]  # the special <|endoftext|> token delimits all documents
        tokens.extend(self.tokenizer.encode_ordinary(row[self.instruction_key]))
        tokens.extend([self._eop])
        tokens.extend(self.tokenizer.encode_ordinary(row[self.answer_key]))
        tokens_np = np.array(tokens)
        assert (0 <= tokens_np).all() and (
            tokens_np < 2**16
        ).all(), "token dictionary too large for uint16"
        tokens_np_uint16 = tokens_np.astype(np.uint16)
        return tokens_np_uint16

    def get_shard(self):
        self.data = []
        idx = 0
        while True:
            tokens = self.tokenize(self.dataset[idx])
            if self.current_shard_token_index > 0:
                tokens = tokens[self.current_shard_token_index :]
                self.current_shard_token_index = 0
            if len(self.data) + len(tokens) < self.max_shard_size:
                self.data.extend(tokens)
                idx += 1
            else:
                break
        nb_tokens = self.max_shard_size - len(self.data)
        self.data.extend(tokens[:nb_tokens])
        self.current_shard_start_index = idx
        self.current_shard_token_index = nb_tokens

    def next_batch(self):
        B, T = self.batch_size, self.max_length
        buf = self.data[self.current_position : self.current_position + B * T + 1]
        x = (buf[:-1]).view(B, T)  # inputs
        y = (buf[1:]).view(B, T)  # targets
        # advance the position in the tensor
        self.current_position += B * T * self.num_processes
        # if loading the next batch would be out of bounds, advance to next shard
        if self.current_position + (B * T * self.num_processes + 1) > len(self.data):
            return None, None
        return x, y
