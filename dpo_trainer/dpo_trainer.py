import math
from copy import deepcopy

import tiktoken
import torch
from dataloader import create_dataloader
from model import dynamic_model
from torch.nn import CrossEntropyLoss
from torch.optim.lr_scheduler import LinearLR
from torchtune.modules.loss import DPOLoss
from transformers import GPT2LMHeadModel
from utils import setup_logger

logger = setup_logger(__name__)


class DPOTrainer:
    def __init__(
        self, model, ref_model, tokenizer: tiktoken.Encoding, dataloader
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

        self.loss = DPOLoss()
        self.optimizer = self.model.configure_optimizers(
            weight_decay=0.01,
            learning_rate=5e-4,
            device_type="mps",
            master_process=True,
        )
        self.scheduler = LinearLR(
            self.optimizer, start_factor=0.1, total_iters=len(dataloader)
        )

    def train(self) -> None:
        for step, batch in enumerate(self.dataloader):
            x, mask, y = batch
            x, mask, y = x.to(self.device), mask.to(self.device), y.to(self.device)
            self.optimizer.zero_grad()
            # keep only odd index in x and y to get chosen values
            logits = self.model(
                x,
                attention_mask=mask,
                use_cache=False,
            )[0]
            log_probs, _ = self.compute_log_probs(logits, y)

            with torch.no_grad():
                logits_ref = self.ref_model(
                    x,
                    attention_mask=mask,
                    use_cache=False,
                )[0]
                log_probs_ref, _ = self.compute_log_probs(logits_ref, y)

            nll_loss = self.compute_chosen_cross_entropy(logits[0::2], y[0::2])

            loss, chosen_reward, rejected_reward = self.compute_loss(
                policy_chosen_logps=log_probs[0::2].view(-1),
                policy_rejected_logps=log_probs[1::2].view(-1),
                reference_chosen_logps=log_probs_ref[0::2].view(-1),
                reference_rejected_logps=log_probs_ref[1::2].view(-1),
            )
            loss = loss.mean()
            reward_accuracy = (chosen_reward > rejected_reward).float().mean()
            lr = self.optimizer.param_groups[0]["lr"]
            if step % 10 == 0:
                print(
                    f"Step: {step} | Loss: {loss.item():05f} | Reward accuracy: {reward_accuracy.item():05f} | "
                    f"chosen_reward/ratio {chosen_reward.mean().item():05f} | rejected_reward/ratio {rejected_reward.mean().item():05f} | "
                    f"Logp_chosen: {log_probs[0::2].mean().item():05f} | Logp_rejected: {log_probs[1::2].mean().item():05f} | "
                    f"NLL loss: {nll_loss.item():05f} | LR: {lr}"
                )

            loss.backward()
            self.optimizer.step()
            self.scheduler.step()

    def compute_chosen_cross_entropy(
        self, logits: torch.FloatTensor, labels: torch.LongTensor
    ) -> torch.FloatTensor:
        logits = logits[..., :-1, :].contiguous()
        labels = labels[..., 1:].contiguous()
        # Flatten the tokens
        loss_fct = CrossEntropyLoss()
        logits = logits.view(-1, logits.shape[-1])
        labels = labels.view(-1)
        # Enable model parallelism
        labels = labels.to(logits.device)
        loss = loss_fct(logits, labels)
        return loss

    @staticmethod
    def compute_log_probs(
        logits: torch.FloatTensor,
        labels: torch.LongTensor,
        label_pad_token_id: int = -100,
    ) -> tuple[torch.FloatTensor, torch.LongTensor]:
        if logits.shape[:-1] != labels.shape:
            raise ValueError(
                "Logits (batch and sequence length dim) and labels must have the same shape."
            )

        labels = labels[:, 1:].clone()
        logits = logits[:, :-1, :]

        loss_mask = labels != label_pad_token_id

        # dummy token; we'll ignore the losses on these tokens later
        labels[labels == label_pad_token_id] = 0

        per_token_logps = torch.gather(
            logits.log_softmax(-1), dim=2, index=labels.unsqueeze(2)
        ).squeeze(2)

        return (per_token_logps * loss_mask).sum(-1), loss_mask.sum(-1)

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
dataloader = create_dataloader("Dahoas/full-hh-rlhf", enc, batch_size=16)
dpo_trainer = DPOTrainer(model, deepcopy(model), enc, dataloader)
dpo_trainer.train()
