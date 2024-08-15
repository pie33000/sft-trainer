import tiktoken
import torch
from torchtune.modules.loss import DPOLoss

from model import WrappedModel


class DPOTrainer:
    def __init__(self, model, ref_model, tokenizer, dataset) -> None:
        self.model = model
        self.ref_model = ref_model
        self.tokenizer = tokenizer
        self.dataset = dataset

        self.loss = DPOLoss()

    def train(self) -> None: ...

    @staticmethod
    def generate(
        x: torch.LongTensor,
        model: WrappedModel,
        max_length: int,
        do_sampling: bool = True,
        top_k: int = 50,
    ) -> torch.LongTensor:
        # Create bacthing support and add padding token to be sure the generation have all the same size
        x_generation = torch.zeros((x.shape[0], max_length), dtype=torch.long)
        for idx, batch in enumerate(x):
            # Generate the next token
            tokens = model.generate(
                x=batch,
                max_length=max_length,
                do_sampling=do_sampling,
                top_k=top_k,
            )
            if len(tokens) < max_length:
                tokens = torch.cat(
                    [tokens, torch.tensor([model.tokenizer.eot_token_id])],
                    dim=-1,
                )
            x_generation[idx, :] = tokens
        return x_generation

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
