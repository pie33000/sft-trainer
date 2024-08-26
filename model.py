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
