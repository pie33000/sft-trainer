import os
import time
from copy import deepcopy
from typing import TypeVar

import tiktoken
import torch
import torch.distributed as dist
from config import DPOConfig
from model import dynamic_model
from torch.distributed import destroy_process_group, init_process_group
from torch.nn import CrossEntropyLoss, Module
from torch.optim.lr_scheduler import LinearLR
from torch.utils.data import DataLoader
from torchtune.modules.loss import DPOLoss
from utils import setup_logger

logger = setup_logger(__name__)

ModelType = TypeVar("ModelType", bound=Module)


class DPOTrainer:
    def __init__(
        self,
        model,
        tokenizer: tiktoken.Encoding,
        dataloader: DataLoader,
        config: DPOConfig,
        ref_model: ModelType | None = None,
        val_dataloader: DataLoader | None = None,
    ) -> None:
        self.optimizer_cfg = config.optimizer_config
        self.dpo_loss_config = config.dpo_loss_config
        self.training_config = config.training_config
        self.ddp_config = config.ddp_config
        self.ddp_config.num_processes = int(os.getenv("WORLD_SIZE", 1))
        self.ddp_config.process_rank = int(os.getenv("LOCAL_RANK", 0))

        self.device = (
            "cpu"
            if config.device == "cpu"
            else f"{config.device}:{self.ddp_config.process_rank}"
        )
        self.device_type = "cuda" if self.device.startswith("cuda") else "cpu"

        if self.ddp_config.num_processes > 1 and self.device_type == "cuda":
            self.ddp_config.master_process = self.ddp_config.process_rank == 0
            init_process_group(backend="nccl")
            torch.cuda.set_device(self.device)

        os.makedirs(self.training_config.checkpoint_path, exist_ok=True)
        os.makedirs(self.training_config.log_path, exist_ok=True)

        self.model = dynamic_model(model).to(self.device)
        if ref_model is None:
            logger.warning(
                "Reference model is not provided. Using the model as reference."
            )
            ref_model = deepcopy(model)
        self.ref_model = dynamic_model(ref_model).to(self.device)
        self.tokenizer = tokenizer
        self.dataloader = dataloader
        self.val_dataloader = val_dataloader

        self.optimizer = self.model.configure_optimizers(
            weight_decay=self.optimizer_cfg.weight_decay,
            learning_rate=self.optimizer_cfg.learning_rate,
            device_type=self.device,
            master_process=self.ddp_config.master_process,
        )
        self.scheduler = LinearLR(
            self.optimizer, start_factor=0.1, total_iters=len(dataloader)
        )
        self.loss = DPOLoss(
            beta=self.dpo_loss_config.beta,
            label_smoothing=self.dpo_loss_config.label_smoothing,
            loss_type=self.dpo_loss_config.loss_type,
        )

    def train(self) -> None:
        loss_accum = 0
        for step, batch in enumerate(self.dataloader):
            t0 = time.time()
            x, mask, y = batch
            x, mask, y = x.to(self.device), mask.to(self.device), y.to(self.device)
            self.optimizer.zero_grad()
            # keep only odd index in x and y to get chosen values
            self.model.train()
            if self.ddp_config.num_processes > 1:
                self.model.require_backward_grad_sync = (
                    step % self.optimizer_cfg.accumulation_steps == 0 and step != 0
                )
                with torch.autocast(device_type=self.device_type, dtype=torch.bfloat16):
                    logits = self.model(
                        x,
                        attention_mask=mask,
                        use_cache=False,
                    )[0]
                    with torch.no_grad():
                        logits_ref = self.ref_model(
                            x,
                            attention_mask=mask,
                            use_cache=False,
                        )[0]
            else:
                logits = logits = self.model(
                    x,
                    attention_mask=mask,
                    use_cache=False,
                )[0]
                with torch.no_grad():
                    logits_ref = self.ref_model(
                        x,
                        attention_mask=mask,
                        use_cache=False,
                    )[0]
            log_probs, _ = self.compute_log_probs(logits, y)
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
            if (
                step % self.training_config.step_log_training_loss == 0
                and step != 0
                and self.ddp_config.master_process
            ):
                process_time = time.time() - t0
                raw_processed_per_s = (
                    self.dataloader.batch_size
                    * self.optimizer_cfg.accumulation_steps
                    / process_time
                )
                margin = abs(log_probs[0::2].mean().item()) - abs(
                    log_probs[1::2].mean().item()
                )
                margin_ref = abs(log_probs_ref[0::2].mean().item()) - abs(
                    log_probs_ref[1::2].mean().item()
                )
                print(
                    f"Step: {step} | Loss: {loss.item():05f} | Reward accuracy: {reward_accuracy.item():05f} | "
                    f"chosen_reward/ratio {chosen_reward.mean().item():05f} | rejected_reward/ratio {rejected_reward.mean().item():05f} | "
                    f"Logp_chosen: {log_probs[0::2].mean().item():05f} | Logp_rejected: {log_probs[1::2].mean().item():05f} | "
                    f"Logp_chosen/ref: {log_probs_ref[0::2].mean().item():05f} | Logp_rejected/ref: {log_probs_ref[1::2].mean().item():05f} | "
                    f"Margin: {margin:05f} | Margin/ref: {margin_ref:05f} | "
                    f"NLL loss: {nll_loss.item():05f} | LR: {lr} | raw/s : {raw_processed_per_s:.2f}"
                )
                self.save_logs(
                    step=step,
                    loss=loss.item(),
                    reward_accuracy=reward_accuracy.item(),
                    logp_chosen=log_probs[0::2].mean().item(),
                    logp_rejected=log_probs[1::2].mean().item(),
                    logp_chosen_ref=log_probs_ref[0::2].mean().item(),
                    logp_rejected_ref=log_probs_ref[1::2].mean().item(),
                    margin=margin,
                    margin_ref=margin_ref,
                    nll_loss=nll_loss.item(),
                    lr=lr,
                    row_per_s=raw_processed_per_s,
                )
            if step % self.training_config.step_save_model == 0 and step != 0:
                self.save_checkpoint(step, loss.item())

            loss = loss
            loss_accum += loss.detach()
            loss.backward()
            self.scheduler.step()

            if self.ddp_config.num_processes > 1:
                dist.all_reduce(
                    loss_accum / self.optimizer_cfg.accumulation_steps,
                    op=dist.ReduceOp.AVG,
                )
            if self.device_type == "cuda":
                torch.cuda.synchronize()

        self.save_checkpoint(step, loss.item())
        destroy_process_group()

    def compute_chosen_cross_entropy(
        self, logits: torch.FloatTensor, labels: torch.LongTensor
    ) -> torch.FloatTensor:
        logits = logits[..., :-1, :].contiguous()
        labels = labels[..., 1:].contiguous()
        loss_fct = CrossEntropyLoss()
        logits = logits.view(-1, logits.shape[-1])
        labels = labels.view(-1)
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

    def compute_validation_loss(self):
        raise NotImplementedError()

    def save_checkpoint(self, step: int, loss: float) -> None:
        torch.save(
            {
                "step": step,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "loss": loss,
            },
            os.path.join(self.training_config.checkpoint_path, f"checkpoint_{step}.pt"),
        )

    def save_logs(
        self,
        step,
        loss: float,
        reward_accuracy: float,
        logp_chosen: float,
        logp_rejected: float,
        logp_chosen_ref: float,
        logp_rejected_ref: float,
        margin: float,
        margin_ref: float,
        nll_loss: float,
        lr: float,
        row_per_s: float,
    ) -> None:
        with open(os.path.join(self.training_config.log_path, "logs.txt"), "a") as f:
            if step - self.training_config.step_log_training_loss == 0:
                f.write(
                    "step; loss; reward_accuracy; logp_chosen; "
                    "logp_rejected; logp_chosen/ref; logp_rejected/ref; "
                    "margin; margin/ref; nll_loss; lr; row/s\n"
                )
            f.write(
                f"{step}; {loss}; {reward_accuracy}; {logp_chosen}; {logp_rejected}; "
                f"{logp_chosen_ref}; {logp_rejected_ref}; {margin}; {margin_ref}; "
                f"{nll_loss}; {lr}; {row_per_s}\n"
            )


# implement the evaluation
