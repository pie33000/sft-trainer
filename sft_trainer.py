import math
import os
import time
from dataclasses import dataclass

import torch
import torch.distributed as dist
import torch.nn as nn
from datasets import (
    Dataset,
    DatasetDict,
    IterableDataset,
    IterableDatasetDict,
    load_dataset,
)
from tiktoken.core import Encoding
from torch.distributed import destroy_process_group, init_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import AutoModelForCausalLM

from dataloader import DataLoaderLite
from model import dynamic_model
from utils import get_tokenizer

# dataset to use
# openai/gsm8k
# meta-math/MetaMathQA


@dataclass
class SFTDataset:
    dataset: DatasetDict | Dataset | IterableDatasetDict | IterableDataset
    instruction_key: str
    answer_key: str


@dataclass
class TrainingConfig:
    batch_size: int
    sequence_length: int
    min_lr: float = 6e-4 * 0.1
    max_lr: float = 6e-4
    warmer_steps: int = 10
    max_steps: int = 500


class SFTTrainer(nn.Module):
    def __init__(
        self,
        model,
        encoder: Encoding,
        sft_dataset: SFTDataset,
        training_config: TrainingConfig,
        device,
        process_rank=0,
        num_processes=8,
        ddp_local_rank=0,
    ):
        super(SFTTrainer, self).__init__()
        self.create_log_dir()
        self.model_wrapped = dynamic_model(model)
        print(dir(self.model_wrapped))
        print(type(self.model_wrapped))
        self.model = DDP(self.model_wrapped, device_ids=[ddp_local_rank])
        self.raw_model = self.model.module

        self.optimizer = self.raw_model.configure_optimizers(
            weight_decay=0.1,
            learning_rate=0.0001,
            device_type=device,
            master_process=True,
        )

        self.encoder = encoder
        self.device = device
        self.sft_dataset = sft_dataset
        self.training_config = training_config

        self.current_dataset_idx = 0

        self.process_rank = process_rank
        self.master_process = self.process_rank == 0
        self.num_processes = num_processes
        self.ddp_local_rank = ddp_local_rank
        self.grad_accum_steps = 20000 // (
            self.training_config.batch_size
            * self.training_config.sequence_length
            * self.num_processes
        )

    def create_log_dir(self):
        log_dir = "log"
        os.makedirs(log_dir, exist_ok=True)

    def get_lr(self, iter):
        # 1) linear warmup for warmup_iters steps
        if iter < self.training_config.warmer_steps:
            return (
                self.training_config.max_lr
                * (iter + 1)
                / self.training_config.warmer_steps
            )
        # 2) if it > lr_decay_iters, return min learning rate
        if iter > self.training_config.max_steps:
            return self.training_config.min_lr
        # 3) in between, use cosine decay down to min learning rate
        decay_ratio = (iter - self.training_config.warmer_steps) / (
            self.training_config.max_steps - self.training_config.warmer_steps
        )
        assert 0 <= decay_ratio <= 1
        coeff = 0.5 * (
            1.0 + math.cos(math.pi * decay_ratio)
        )  # coeff starts at 1 and goes to 0
        return self.training_config.min_lr + coeff * (
            self.training_config.max_lr - self.training_config.min_lr
        )

    def load_data(self) -> DataLoaderLite:
        dataloader = DataLoaderLite(
            self.sft_dataset.dataset,
            self.encoder,
            max_length=self.training_config.sequence_length,
            batch_size=self.training_config.batch_size,
            instruction_key=self.sft_dataset.instruction_key,
            answer_key=self.sft_dataset.answer_key,
            process_rank=self.process_rank,
            num_processes=self.num_processes,
        )
        return dataloader

    def train_v1(self):
        dataloader = self.load_data()
        while True:
            x, y = dataloader.next_batch()
            if x is None or y is None:
                break
            x = x.to(torch.int64)
            y = y.to(torch.int64)
            self.optimizer.zero_grad()
            y_pred, loss = self.model(x, targets=y)
            print(loss)
            loss.backward()
            self.optimizer.step()
            # self.scheduler.step()

    def train(self):
        dataloader = self.load_data()
        for step in range(self.training_config.max_steps):
            t0 = time.time()
            last_step = step == self.training_config.max_steps - 1

            # do one step of the optimization
            self.model.train()
            self.optimizer.zero_grad()
            loss_accum = 0.0
            for micro_step in range(self.grad_accum_steps):
                x, y = dataloader.next_batch()
                if x is None or y is None:
                    dataloader.reset()
                    x, y = dataloader.next_batch()
                x, y = x.to(device), y.to(device)
                # added after video, this field is also used by the forward pass.

                self.model.require_backward_grad_sync = (
                    micro_step == self.grad_accum_steps - 1
                )
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    logits, loss = self.model(x, targets=y)
                # we have to scale the loss to account for gradient accumulation,
                # because the gradients just add on each successive backward().
                # addition of gradients corresponds to a SUM in the objective, but
                # instead of a SUM we want MEAN. Scale the loss here so it comes out right
                loss = loss / self.grad_accum_steps
                loss_accum += loss.detach()
                loss.backward()

            dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG)
            norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            # determine and set the learning rate for this iteration
            lr = self.get_lr(step)
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = lr
            self.optimizer.step()
            torch.cuda.synchronize()  # wait for the GPU to finish work
            t1 = time.time()
            dt = t1 - t0  # time difference in seconds
            tokens_processed = (
                self.training_config.batch_size
                * self.training_config.sequence_length
                * self.grad_accum_steps
                * ddp_world_size
            )
            tokens_per_sec = tokens_processed / dt
            if self.master_process:
                print(
                    f"step {step:5d} | loss: {loss_accum.item():.6f} | lr {lr:.4e} | norm: {norm:.4f} | dt: {dt*1000:.2f}ms | tok/sec: {tokens_per_sec:.2f}"
                )
                with open("log/log.txt", "a") as f:
                    f.write(f"{step} train {loss_accum.item():.6f}\n")


ddp_rank = int(os.environ["RANK"])
ddp_local_rank = int(os.environ["LOCAL_RANK"])
ddp_world_size = int(os.environ["WORLD_SIZE"])
device = f"cuda:{ddp_local_rank}"

print("-----------------------------------------------------------------------------\n")
print(
    f"Init device {device}, rank {ddp_rank}, local rank {ddp_local_rank}, world size {ddp_world_size}"
)
print("-----------------------------------------------------------------------------\n")

torch.cuda.set_device(device)
init_process_group(backend="nccl")

enc = get_tokenizer()
ds = load_dataset("meta-math/MetaMathQA")
sft_dataset = SFTDataset(dataset=ds, instruction_key="query", answer_key="response")
training_config = TrainingConfig(batch_size=4, sequence_length=512)
model = AutoModelForCausalLM.from_pretrained("openai-community/gpt2").to(device)
trainer = SFTTrainer(
    model=model,
    encoder=enc,
    sft_dataset=sft_dataset,
    training_config=training_config,
    device=device,
    num_processes=ddp_world_size,
    process_rank=ddp_rank,
    ddp_local_rank=ddp_local_rank,
)
print("-----------------------------------------------------------------------------\n")
print("Training ....")
trainer.train()
print("-----------------------------------------------------------------------------\n")

print("-----------------------------------------------------------------------------\n")
print("Destroying process group ....")
destroy_process_group()
print("-----------------------------------------------------------------------------\n")
