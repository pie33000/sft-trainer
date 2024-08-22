import math
import os
import time
from dataclasses import dataclass

import tiktoken
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import OmegaConf
from tiktoken.core import Encoding
from torch.distributed import destroy_process_group, init_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM

from sft_trainer.config import SFTConfig
from sft_trainer.dataloader import SFTColumnsMapping, create_dataloader
from shared.model import dynamic_model
from shared.utils import get_tokenizer

# dataset to use
# openai/gsm8k
# meta-math/MetaMathQA
# databricks-dolly-15k
# https://huggingface.co/datasets/teknium/openhermes?row=25
# https://huggingface.co/datasets/HuggingFaceH4/ultrachat_200k?row=9


@dataclass
class TrainingConfig:
    batch_size: int
    sequence_length: int
    min_lr: float = 6e-4 * 0.1
    max_lr: float = 6e-4
    warmer_steps: int = 50
    max_steps: int = 500
    validation_steps: int = 10
    checkpoint_steps: int = 20


class SFTTrainer(nn.Module):
    def __init__(
        self,
        model,
        encoder: Encoding,
        dataloader: DataLoader,
        config: SFTConfig,
        val_dataloader: DataLoader | None = None,
    ):
        super(SFTTrainer, self).__init__()
        self.config = config
        self.create_log_dir()
        self.encoder: tiktoken.Encoding = encoder
        self.dataloader = dataloader
        self.val_dataloader = val_dataloader
        self.config.ddp_config.num_processes = int(os.getenv("WORLD_SIZE", 1))
        self.config.ddp_config.process_rank = int(os.getenv("LOCAL_RANK", 0))
        self.config.ddp_config.master_process = self.config.ddp_config.process_rank == 0

        self.device = (
            "cpu"
            if config.device == "cpu"
            else f"{config.device}:{self.config.ddp_config.process_rank}"
        )
        self.device_type = "cuda" if self.device.startswith("cuda") else "cpu"
        self.model = dynamic_model(model).to(self.device)
        self.raw_model = self.model

        if self.config.ddp_config.num_processes > 1:
            init_process_group(backend="nccl")
            torch.cuda.set_device(self.device)
            self.model = DDP(
                self.model, device_ids=[self.config.ddp_config.process_rank]
            )
            self.raw_model = self.model.module

        self.optimizer = self.raw_model.configure_optimizers(
            weight_decay=self.config.optimizer_config.weight_decay,
            learning_rate=self.config.optimizer_config.learning_rate,
            device_type=self.device,
            master_process=self.config.ddp_config.master_process,
        )

        self.grad_accum_steps = max(
            10,
            10
            // (
                self.config.training_config.batch_size
                * self.config.training_config.sequence_length
                * self.config.ddp_config.num_processes
            ),
        )

    def create_log_dir(self):
        os.makedirs(self.config.training_config.log_path, exist_ok=True)

    def get_lr(self, iter):
        # 1) linear warmup for warmup_iters steps
        if iter < self.config.optimizer_config.warmup_steps:
            return (
                self.config.optimizer_config.max_lr
                * (iter + 1)
                / self.config.optimizer_config.warmup_steps
            )
        # 3) in between, use cosine decay down to min learning rate
        decay_ratio = (iter - self.config.optimizer_config.warmup_steps) / (
            self.config.training_config.max_steps
            - self.config.optimizer_config.warmup_steps
        )
        assert 0 <= decay_ratio <= 1
        coeff = 0.5 * (
            1.0 + math.cos(math.pi * decay_ratio)
        )  # coeff starts at 1 and goes to 0
        return self.config.optimizer_config.min_lr + coeff * (
            self.config.optimizer_config.max_lr - self.config.optimizer_config.min_lr
        )

    def calculate_validation_loss(self, step):
        self.model.eval()
        with torch.no_grad():
            val_loss_accum = 0.0
            val_loss_steps = 5
            for _ in range(val_loss_steps):
                for batch in self.val_dataloader:
                    x, mask, y = batch
                    x, mask, y = (
                        x.to(self.device),
                        mask.to(self.device),
                        y.to(self.device),
                    )
                    if self.config.ddp_config.num_processes > 1:
                        with torch.autocast(
                            device_type=self.device_type, dtype=torch.bfloat16
                        ):
                            logits, loss = self.model(
                                x, targets=y, attention_mask=mask, shift_labels=True
                            )
                    else:
                        logits, loss = self.model(
                            x, targets=y, attention_mask=mask, shift_labels=True
                        )
                    loss = loss / val_loss_steps
                    val_loss_accum += loss.detach()
        if self.config.ddp_config.num_processes > 1:
            dist.all_reduce(val_loss_accum, op=dist.ReduceOp.AVG)
        if self.config.ddp_config.master_process:
            print(f"validation loss: {val_loss_accum.item():.4f}")
            with open(
                os.path.join(self.config.training_config.log_path, "sft.txt"), "a"
            ) as f:
                f.write(f"{step} val {val_loss_accum.item():.4f}\n")
            if step > 0 and (step % self.config.training_config.step_save_model == 0):
                # optionally write model checkpoints
                checkpoint_path = os.path.join("log", f"model_{step:05d}.pt")
                checkpoint = {
                    "model": self.raw_model.state_dict(),
                    "step": step,
                    "val_loss": val_loss_accum.item(),
                }
                torch.save(checkpoint, checkpoint_path)

    def generate(self):
        self.model.eval()
        num_return_sequences = 4
        max_length = 32
        tokens = self.val_dataloader.dataset.instruction_ids
        tokens.extend(self.encoder.encode_ordinary(" Who is Emmanuel Macron?\n "))
        tokens.extend(self.val_dataloader.dataset.answer_ids)
        tokens = torch.tensor(tokens, dtype=torch.long)
        tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1)
        xgen = tokens.to(self.device, dtype=torch.long)
        # sample_rng = torch.Generator(device=self.device)
        # sample_rng.manual_seed(42 + self.config.ddp_config.process_rank)
        while xgen.size(1) < max_length:
            # forward the model to get the logits
            with torch.no_grad():
                if self.config.ddp_config.num_processes > 1:
                    with torch.autocast(
                        device_type=self.device_type, dtype=torch.bfloat16
                    ):
                        logits, loss = self.model(xgen)  # (B, T, vocab_size)
                else:
                    logits, loss = self.model(xgen)  # (B, T, vocab_size)
                # take the logits at the last position
                logits = logits[:, -1, :]  # (B, vocab_size)
                # get the probabilities
                probs = F.softmax(logits, dim=-1)
                # do top-k sampling of 50 (huggingface pipeline default)
                # topk_probs here becomes (5, 50), topk_indices is (5, 50)
                topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
                # select a token from the top-k probabilities
                # note: multinomial does not demand the input to sum to 1
                ix = torch.multinomial(topk_probs, 1)  # (B, 1)
                # gather the corresponding indices
                xcol = torch.gather(topk_indices, -1, ix)  # (B, 1)
                # append to the sequence
                xgen = torch.cat((xgen, xcol), dim=1)
        # print the generated text
        for i in range(num_return_sequences):
            tokens = xgen[i, :max_length].tolist()
            decoded = self.encoder.decode(tokens)
            print(f"rank {self.config.ddp_config.process_rank} sample {i}: {decoded}")

    def train(self):
        loss_accum = 0
        for step, batch in enumerate(self.dataloader):
            t0 = time.time()
            if step > 0 and step % self.config.training_config.step_log_eval_loss == 0:
                self.calculate_validation_loss(step)
                self.generate()
            self.model.train()
            if step == 0:
                self.optimizer.zero_grad()
            x, mask, y = batch
            x, y, mask = (
                x.to(self.device),
                y.to(self.device),
                mask.to(self.device),
            )
            if self.config.ddp_config.num_processes > 1:
                self.model.require_backward_grad_sync = (
                    step % self.config.optimizer_config.accumulation_steps == 0
                    and step > 0
                )
            if self.config.ddp_config.num_processes > 1:
                with torch.autocast(device_type=self.device_type, dtype=torch.bfloat16):
                    logits, loss = self.model(
                        x, targets=y, attention_mask=mask, shift_labels=True
                    )
            else:
                logits, loss = self.model(
                    x, targets=y, attention_mask=mask, shift_labels=True
                )
            loss = loss / self.config.optimizer_config.accumulation_steps
            loss_accum += loss.detach()
            loss.backward()

            if self.config.ddp_config.num_processes > 1:
                dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG)
            norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            # determine and set the learning rate for this iteration

            if step % self.config.optimizer_config.accumulation_steps == 0 and step > 0:
                self.optimizer.step()
                self.optimizer.zero_grad()
                lr = self.get_lr(step)
                for param_group in self.optimizer.param_groups:
                    param_group["lr"] = lr
                if self.device_type == "cuda":
                    torch.cuda.synchronize()
                t1 = time.time()
                dt = t1 - t0  # time difference in seconds
                tokens_processed = (
                    self.config.training_config.batch_size
                    * self.config.training_config.sequence_length
                    * self.config.optimizer_config.accumulation_steps
                    * self.config.ddp_config.num_processes
                )
                tokens_per_sec = tokens_processed / dt
                if (
                    self.config.ddp_config.master_process
                    and step % self.config.training_config.step_log_training_loss == 0
                    and step > 0
                ):
                    print(
                        f"step {step:5d} | loss: {loss_accum.item():.6f} | lr {lr:.4e} | norm: {norm:.4f} | dt: {dt*1000:.2f}ms | tok/sec: {tokens_per_sec:.2f}"
                    )
                    with open(
                        os.path.join(self.config.training_config.log_path, "sft.txt"),
                        "a",
                    ) as f:
                        f.write(f"{step} train {loss_accum.item():.6f}\n")

                loss_accum = 0
        if self.config.ddp_config.num_processes > 1:
            destroy_process_group()


model = AutoModelForCausalLM.from_pretrained("openai-community/gpt2")
enc = get_tokenizer("gpt2")
dataloader = create_dataloader(
    "teknium/openhermes",
    enc,
    batch_size=4,
    columns_mapping=SFTColumnsMapping(prompt="instruction", answer="output"),
)
val_dataloader = create_dataloader(
    "teknium/openhermes",
    enc,
    batch_size=4,
    columns_mapping=SFTColumnsMapping(prompt="instruction", answer="output"),
    split="train[:10]",
)

conf = SFTConfig(**OmegaConf.load("sft_trainer/config.yml"))
trainer = SFTTrainer(model, enc, dataloader, conf, val_dataloader=val_dataloader)
trainer.train()
