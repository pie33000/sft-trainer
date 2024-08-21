import os

import numpy as np
import tiktoken
import torch

from dpo_trainer.utils import setup_logger

logger = setup_logger(__name__)


class DataLoaderLite:
    def __init__(
        self,
        tokenizer: tiktoken.Encoding,
        batch_size: int,
        data_folder: str,
        process_rank: int,
        num_processes: int,
        split: str,
        master_process: bool,
        instruction_template: str = "### User:",
        response_template: str = "### Assistant:",
        verbose: int = 0,
    ):
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.process_rank = process_rank
        self.num_processes = num_processes
        assert split in {"train", "val"}

        # get the shard filenames
        shards = os.listdir(data_folder)
        shards = [s for s in shards if split in s]
        shards = sorted(shards)
        shards = [os.path.join(data_folder, s) for s in shards]
        self.shards = shards
        assert len(shards) > 0, f"no shards found for split {split}"

        if master_process:
            logger.info(f"found {len(shards)} shards for split {split}")

        self.instruction_template = instruction_template
        self.response_template = response_template

        logger.info(f"Insruction template: {self.instruction_template}")
        logger.info(f"Response template: {self.response_template}")

        self.instruction_template_tokens = self.tokenizer.encode_ordinary(
            self.instruction_template
        )
        self.response_template_tokens = self.tokenizer.encode_ordinary(
            self.response_template
        )

        self.ignore_index = -100
        logger.info(f"Ignored indexes will be replaced by {self.ignore_index}")

        self.verbose = verbose

        self.reset()

    @staticmethod
    def load_tokens(filename: str) -> tuple[torch.tensor, torch.tensor]:
        npt = np.load(filename)
        npt = npt.astype(np.int32)
        ptt = torch.tensor(npt, dtype=torch.long)
        return ptt

    def ignore_indexes(self, y: torch.tensor) -> torch.tensor:
        attention_mask = torch.ones_like(y)
        for row in range(y.size(0)):
            attention_mask[row] = attention_mask[row].masked_fill(
                y[row] == self.tokenizer._special_tokens["<|endofprompt|>"], 0
            )
            response_token_ids_idxs = []
            human_token_ids_idxs = []
            for assistant_idx in torch.where(
                y[row] == self.response_template_tokens[0]
            )[0]:
                if (
                    self.response_template_tokens
                    == y[row][
                        assistant_idx : assistant_idx
                        + len(self.response_template_tokens)
                    ].tolist()
                ):
                    response_token_ids_idxs.append(
                        assistant_idx + len(self.response_template_tokens)
                    )
            if len(response_token_ids_idxs) == 0 and self.verbose == 1:
                logger.warning(
                    f"Could not find response key `{self.response_template}` in the "
                    f"following instance: {self.tokenizer.decode(y[row].detach().cpu().numpy())} "
                    f"This instance will be ignored in loss calculation. "
                    f"Note, if this happens often, consider increasing the `max_seq_length`."
                )
                y[row, :] = self.ignore_index
            human_token_ids = self.instruction_template_tokens
            for human_idx in np.where(y[row] == human_token_ids[0])[0]:
                if (
                    human_token_ids
                    == y[row][human_idx : human_idx + len(human_token_ids)].tolist()
                ):
                    human_token_ids_idxs.append(human_idx)
            if (
                len(human_token_ids_idxs) > 0
                and len(response_token_ids_idxs) > 0
                and human_token_ids_idxs[0] > response_token_ids_idxs[0]
            ):
                human_token_ids_idxs = [0] + human_token_ids_idxs
            for idx, (start, end) in enumerate(
                zip(human_token_ids_idxs, response_token_ids_idxs)
            ):
                if idx != 0:
                    y[row, start:end] = self.ignore_index
                else:
                    y[row, :end] = self.ignore_index

                if len(response_token_ids_idxs) < len(human_token_ids_idxs):
                    y[row, human_token_ids_idxs[-1] :] = self.ignore_index
            # Set padding token to ignore index, except the first one
            y[row, y[row] == self.tokenizer._special_tokens["<|endofprompt|>"]] = (
                self.ignore_index
            )
        return y, attention_mask

    def reset(self):
        # state, init at shard zero
        self.current_shard = 0
        self.tokens = self.load_tokens(self.shards[self.current_shard])
        self.current_position = self.batch_size * self.process_rank

    def next_batch(self):
        B = self.batch_size
        tokens_tensor = self.tokens[
            self.current_position : self.current_position + B, :
        ]
        x = tokens_tensor.view(B, -1).clone()[:, :-1]
        y = tokens_tensor.view(B, -1).clone()[:, 1:]
        y, mask = self.ignore_indexes(y)
        self.current_position += B * self.num_processes
        if self.current_position + (B * self.num_processes + 1) > len(self.tokens):
            self.current_shard = (self.current_shard + 1) % len(self.shards)
            self.tokens = self.load_tokens(self.shards[self.current_shard])
            self.current_position = B * self.process_rank
        return x, y, mask
