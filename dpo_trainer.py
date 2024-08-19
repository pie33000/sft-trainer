from functools import partial
from typing import Optional

import tiktoken
import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from torchtune.modules.loss import DPOLoss
from transformers import GPT2LMHeadModel

from datasets import Dataset as HF_Dataset
from datasets import load_dataset
from model import WrappedModel, dynamic_model
from utils import setup_logger

logger = setup_logger(__name__)


# Dahoas/full-hh-rlhf
ds = load_dataset("Dahoas/full-hh-rlhf", split="train")
enc = tiktoken.encoding_for_model("gpt2")


def collate_fn(batch, left_pad=False):
    max_length = len(max(batch, key=len))
    for i in range(len(batch)):
        if left_pad:
            batch[i] = [50256] * (max_length - len(batch[i])) + batch[i]
        else:
            batch[i] += [50256] * (max_length - len(batch[i]))
    return torch.tensor(batch, dtype=torch.long)


class HuggingFaceDataset(Dataset):
    def __init__(
        self, dataset: HF_Dataset, tokenizer: tiktoken.Encoding, column_name: str
    ) -> None:
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.column_name = column_name

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        prompt = self.tokenizer.encode_ordinary(self.dataset[idx][self.column_name])
        return prompt


dataset = HuggingFaceDataset(ds, enc, "prompt")
sampler = DistributedSampler(
    dataset=dataset, num_replicas=1, rank=0, shuffle=True, drop_last=True
)
chose_dataset = HuggingFaceDataset(ds, enc, "chosen")
rejected_dataset = HuggingFaceDataset(ds, enc, "rejected")
dataloader = DataLoader(
    dataset,
    batch_size=4,
    collate_fn=partial(collate_fn, left_pad=True),
    sampler=sampler,
)
chosen_dataloader = DataLoader(
    chose_dataset, batch_size=4, collate_fn=partial(collate_fn, left_pad=False)
)
rejected_dataloader = DataLoader(
    rejected_dataset, batch_size=4, collate_fn=partial(collate_fn, left_pad=False)
)


class DPOTrainer:
    def __init__(
        self,
        model,
        ref_model,
        tokenizer: tiktoken.Encoding,
        dataloader,
        chosen_dataloader,
        rejected_dataloader,
    ) -> None:
        self.device = "mps"

        self.model = dynamic_model(model).to(self.device)
        if ref_model is None:
            logger.warning(
                "Reference model is not provided. Using the model as reference."
            )
            ref_model = model
        self.ref_model = dynamic_model(ref_model).to(self.device)
        self.tokenizer = tokenizer
        self.dataloader = dataloader
        self.chosen_dataloader = chosen_dataloader
        self.rejected_dataloader = rejected_dataloader

        self.loss = DPOLoss()
        self.optimizer = self.model.configure_optimizers(
            weight_decay=0.01,
            learning_rate=5e-5,
            device_type="mps",
            master_process=True,
        )

    def train(
        self, max_length: int = 1024, do_sampling: bool = False, top_k: int = 50
    ) -> None:
        for x, x_chosen, x_rejected in zip(
            self.dataloader, self.chosen_dataloader, self.rejected_dataloader
        ):
            self.optimizer.zero_grad()
            x, x_chosen, x_rejected = (
                x.to(self.device),
                x_chosen.to(self.device),
                x_rejected.to(self.device),
            )
            y_chosen, y_rejected = (
                torch.cat([x, x_chosen], dim=-1),
                torch.cat([x, x_rejected], dim=-1),
            )

            _, logits, mask = self.generate(
                x,
                self.model,
                max_length=max_length,
                do_sampling=do_sampling,
                top_k=top_k,
            )
            _, logits_ref, mask_ref = self.generate(
                x,
                self.ref_model,
                max_length=max_length,
                do_sampling=do_sampling,
                top_k=top_k,
            )
            log_probs_choosen = self.compute_log_probs(logits, y_chosen, mask=mask)
            log_probs_rejected = self.compute_log_probs(logits, y_rejected, mask=mask)

            log_probs_choosen_ref = self.compute_log_probs(
                logits_ref, y_chosen, mask=mask_ref
            )
            log_probs_rejected_ref = self.compute_log_probs(
                logits_ref, y_rejected, mask=mask_ref
            )
            loss, chosen_reward, rejected_reward = self.compute_loss(
                policy_chosen_logps=log_probs_choosen,
                policy_rejected_logps=log_probs_rejected,
                reference_chosen_logps=log_probs_choosen_ref,
                reference_rejected_logps=log_probs_rejected_ref,
            )
            reward_accuracy = (chosen_reward > rejected_reward).float()
            logger.info(f"Reward accuracy: {reward_accuracy.item():05f}")

            loss.backward()
            self.optimizer.step()

    def generate(
        self,
        x: list[list[int]],
        model: WrappedModel,
        max_length: int,
        do_sampling: bool = True,
        top_k: int = 50,
    ) -> tuple[torch.LongTensor, torch.FloatTensor, torch.LongTensor]:
        x, logits, mask = model.generate(
            x,
            max_length,
            do_sampling,
            top_k,
            self.tokenizer._special_tokens["<|endoftext|>"],
        )
        return x, logits, mask

    @staticmethod
    def compute_log_probs(
        logits: torch.FloatTensor,
        labels: torch.LongTensor,
        label_pad_token_id: int = -100,
        mask: Optional[torch.LongTensor] = None,
    ) -> tuple[torch.FloatTensor, torch.LongTensor]:
        B, T, V = logits.size()
        B_L, T_L = labels.size()

        if B != B_L:
            raise ValueError("Batch size mismatch")
        if T < T_L:
            raise ValueError(
                "Logits sequence length is shorter than labels sequence length"
            )
        if T > T_L:
            logits = logits[:, :T_L, :]
            mask = mask[:, : T_L - 1]

        labels = labels[:, 1:].clone()
        logits = logits[:, :-1, :]

        per_token_logps = torch.gather(
            logits.log_softmax(-1), dim=2, index=labels.unsqueeze(2)
        ).squeeze(2)
        # The mask remove the log probabilities of the padding tokens and the instruction tokens
        return (per_token_logps * mask).sum(-1), mask.sum(-1)

    def compute_loss(
        self,
        policy_chosen_logps: torch.FloatTensor,
        policy_rejected_logps: torch.FloatTensor,
        reference_chosen_logps: torch.FloatTensor,
        reference_rejected_logps: torch.FloatTensor,
    ) -> tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        loss, chosen_reward, rejected_reward = self.loss(
            policy_chosen_logps=policy_chosen_logps,
            policy_rejected_logps=policy_rejected_logps,
            reference_chosen_logps=reference_chosen_logps,
            reference_rejected_logps=reference_rejected_logps,
        )

        return loss, chosen_reward, rejected_reward

    def compute_metrics(self) -> None:
        raise NotImplementedError()


enc = tiktoken.encoding_for_model("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")
dpo_trainer = DPOTrainer(
    model, model, enc, dataloader, chosen_dataloader, rejected_dataloader
)
dpo_trainer.train(max_length=1024, do_sampling=True, top_k=50)
