import tiktoken
import torch
from torchtune.modules.loss import DPOLoss

from model import WrappedModel
from utils import setup_logger

logger = setup_logger(__name__)


class DPOTrainer:
    def __init__(self, model, ref_model, tokenizer: tiktoken.Encoding, dataset) -> None:
        self.device = "mps"

        self.model = model.to(self.device)
        self.ref_model = ref_model.to(self.device)
        self.tokenizer = tokenizer
        self.dataset = dataset

        self.loss = DPOLoss()

    def train(self) -> None:
        max_length = 256
        do_sampling = True
        top_k = 50
        dataloader = ...
        for x_choosen, x_rejected, y_choosen, y_rejected in dataloader.next_batch():
            self.optimizer.zero_grad()
            x_choosen, x_rejected = (
                x_choosen.to(self.device),
                x_rejected.to(self.device),
            )
            y_choosen, y_rejected = (
                y_choosen.to(self.device),
                y_rejected.to(self.device),
            )

            y_pred_choosen, logits_choosen = self.generate(
                x_choosen,
                self.model,
                max_length=max_length,
                do_sampling=do_sampling,
                top_k=top_k,
            )
            y_pred_choosen_ref, logits_choosen_ref = self.generate(
                x_choosen,
                self.ref_model,
                max_length=max_length,
                do_sampling=do_sampling,
                top_k=top_k,
            )
            y_pred_rejected, logits_rejected = self.generate(
                x_rejected,
                self.model,
                max_length=max_length,
                do_sampling=do_sampling,
                top_k=top_k,
            )
            y_pred_rejected_ref, logits_rejected_ref = self.generate(
                x_rejected,
                self.ref_model,
                max_length=max_length,
                do_sampling=do_sampling,
                top_k=top_k,
            )
            log_probs_choosen = self.compute_log_probs(logits_choosen, y_choosen)
            log_probs_rejected = self.compute_log_probs(logits_rejected, y_rejected)

            log_probs_choosen_ref = self.compute_log_probs(
                logits_choosen_ref, y_choosen
            )
            log_probs_rejected_ref = self.compute_log_probs(
                logits_rejected_ref, y_rejected
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
    ) -> tuple[torch.LongTensor, torch.FloatTensor]:
        x, logits = model.generate(
            x,
            max_length,
            do_sampling,
            top_k,
            self.tokenizer._special_tokens["<|endoftext|>"],
        )
        return x, logits

    def compute_log_probs(
        self,
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

    def compute_metrics(self) -> None: ...


dop_trainer = DPOTrainer(None, None, None, None)
enc = tiktoken.encoding_for_model("gpt2")
text = "Hello, my name is"
tokens = enc.encode_ordinary(text)
logits = torch.randn(1, 4, 50257)
labels = torch.tensor(tokens, dtype=torch.long)[1:].view(1, -1)

log_probs = dop_trainer.compute_log_probs(logits, labels)
print(log_probs)
