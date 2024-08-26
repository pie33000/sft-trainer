import inspect
from typing import Optional, Type

import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import setup_logger

logger = setup_logger(__name__)


class WrappedModel(nn.Module):
    """
    Wraps an existing PyTorch model.
    """

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model
        self.print_model_parameters()

    def print_model_parameters(self):
        total_params = int(sum(p.numel() for p in self.model.parameters()) // 1e6)
        print(f"Total number of parameters: {total_params}M")

    def forward(self, *args, **kwargs):
        targets = kwargs.pop("targets", None)
        shift_labels = kwargs.pop("shift_labels", False)
        logits = self.model(*args, **kwargs)
        if hasattr(logits, "logits"):
            logits = logits.logits
        loss = None
        if targets is not None:
            if shift_labels:
                logits = logits[:, :-1, :].contiguous()
                targets = targets[:, 1:].contiguous()
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
            )
        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        x: list | list[list[int]],
        max_length: int,
        do_sampling: bool = True,
        top_k: int = 50,
        temperature: float = 1.0,
        eos_token_id: int = 50256,
        seed: Optional[int] = None,
    ) -> tuple[torch.LongTensor, torch.FloatTensor]:
        x, mask = self.prepare_inputs(x, max_length)
        B, T = x.size()
        generator = None
        if seed is not None:
            generator = torch.Generator()
            generator.manual_seed(seed)
        for _ in range(T, max_length):
            logits = self.model(x)  # (B, seq_len, vocab_size)
            if hasattr(logits, "logits"):
                logits = logits.logits
            if do_sampling:
                next_token_id = self.top_k_logits(
                    logits[:, -1, :] / temperature, k=top_k, generator=generator
                )  # (B, 1)
            else:
                next_token_id = torch.argmax(logits, dim=-1)
            x = torch.cat([x, next_token_id], dim=-1)
        x, mask = self.post_process_inputs(x, mask, eos_token_id)
        return x, logits, mask

    def prepare_inputs(
        self, x: torch.LongTensor | list[list[int]], max_length: int
    ) -> tuple[torch.LongTensor, torch.LongTensor]:
        max_instruction_length = x.shape[-1]
        if max_instruction_length > max_length:
            max_instruction_length = max_length
            logger.warning(
                f"Instruction length {max_instruction_length} is longer than max sequence length {max_length}. ",
                "Consider to increase the max sequence length.",
            )
        if x.dim() == 1:
            # create a batch of size 1
            x = torch.tensor(x, dtype=torch.long)
            x = x.unsqueeze(0)
        mask = torch.ones(size=(x.shape[0], max_length), dtype=torch.long)
        mask[:, :max_instruction_length] = 0
        return x, mask.to(x.device)

    def post_process_inputs(
        self, x: torch.LongTensor, mask: torch.LongTensor, eos_token_id: int
    ) -> torch.LongTensor:
        B, T = x.size()
        mask_copy = mask.clone()
        mask = x * mask
        mask = torch.eq(mask, eos_token_id)
        eos_token_positions = torch.argmax(mask.int(), dim=-1)
        eos_token_positions[~mask.any(dim=1)] = (
            T + 1
        )  # set it to position out of the matrix shape
        indices = torch.arange(T).expand(B, T).to(x.device)
        mask_to_fill = indices >= eos_token_positions.unsqueeze(1)
        x[mask_to_fill] = eos_token_id
        return x, mask_copy | mask_to_fill

    @staticmethod
    def left_pad_batch_sequence(
        x: list[int], max_length: int, eos_token_id: int
    ) -> list[int]:
        if len(x) < max_length:
            x = [eos_token_id] * (max_length - len(x)) + x
        else:
            x = x[:max_length]
        return x

    def top_k_logits(
        self, logits: torch.FloatTensor, k: int = 50, generator=None
    ) -> torch.FloatTensor:
        # logits (d_model, vocab_size)
        if k == 0:
            return logits
        top_k_logits, top_k_indices = torch.topk(logits, k, dim=-1)
        probs = F.softmax(top_k_logits, dim=-1)

        next_token = torch.multinomial(probs, num_samples=1, generator=generator)

        # Get the corresponding token id
        next_token_id = top_k_indices.gather(-1, next_token)

        return next_token_id

    def configure_optimizers(
        self,
        weight_decay: float,
        learning_rate: float,
        device_type: str,
        master_process: bool,
    ) -> torch.optim.Optimizer:
        # start with all of the candidate parameters (that require grad)
        param_dict = {pn: p for pn, p in self.named_parameters()}
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": nodecay_params, "weight_decay": 0.0},
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        if master_process:
            print(
                f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters"
            )
            print(
                f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters"
            )
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = "fused" in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == "cuda"
        if master_process:
            print(f"using fused AdamW: {use_fused}")
        optimizer = torch.optim.AdamW(
            optim_groups, lr=learning_rate, betas=(0.9, 0.95), eps=1e-8, fused=use_fused
        )
        return optimizer


def dynamic_model(model_class: Type[nn.Module], **kwargs) -> nn.Module:
    """
    Function to create a dynamic model instance.
    Args:
        model_class (Type[nn.Module]): The model class to instantiate.
        **kwargs: Additional arguments to pass to the model class.
    Returns:
        nn.Module: The instantiated model.
    """
    return WrappedModel(model_class, **kwargs)
