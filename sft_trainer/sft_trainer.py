import math
import os
import time
from dataclasses import dataclass

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from tiktoken.core import Encoding
from torch.distributed import destroy_process_group, init_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import AutoModelForCausalLM

from sft_trainer.dataloader import DataLoaderLite
from model import dynamic_model
from utils import get_tokenizer

# dataset to use
# openai/gsm8k
# meta-math/MetaMathQA
# databricks-dolly-15k
# https://huggingface.co/datasets/teknium/openhermes?row=25
# https://huggingface.co/datasets/HuggingFaceH4/ultrachat_200k?row=9


@dataclass
class SFTDataset:
    dataset_folder: str
    instruction_template: str
    response_template: str


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
        sft_dataset: SFTDataset,
        training_config: TrainingConfig,
        device,
        device_type,
        process_rank=0,
        num_processes=8,
        ddp_local_rank=0,
        is_ddp_run=False,
    ):
        super(SFTTrainer, self).__init__()
        self.create_log_dir()
        self.is_ddp_run = is_ddp_run
        self.model = dynamic_model(model)
        if self.is_ddp_run:
            self.model = DDP(self.model_wrapped, device_ids=[ddp_local_rank])
        self.raw_model = self.model.module if self.is_ddp_run else self.model

        self.optimizer = self.raw_model.configure_optimizers(
            weight_decay=0.1,
            learning_rate=0.0001,
            device_type=device,
            master_process=True,
        )

        self.encoder = encoder
        self.device = device
        self.device_type = device_type
        self.sft_dataset = sft_dataset
        self.training_config = training_config

        self.current_dataset_idx = 0

        self.process_rank = process_rank
        self.master_process = self.process_rank == 0
        self.num_processes = num_processes
        self.ddp_local_rank = ddp_local_rank
        self.grad_accum_steps = max(
            10,
            10
            // (
                self.training_config.batch_size
                * self.training_config.sequence_length
                * self.num_processes
            ),
        )
        print(self.grad_accum_steps)

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

    def load_data(self) -> tuple[DataLoaderLite, DataLoaderLite]:
        train_dataloader = DataLoaderLite(
            self.encoder,
            batch_size=self.training_config.batch_size,
            data_folder=self.sft_dataset.dataset_folder,
            process_rank=self.process_rank,
            num_processes=self.num_processes,
            split="train",
            master_process=self.master_process,
            instruction_template=self.sft_dataset.instruction_template,
            response_template=self.sft_dataset.response_template,
        )
        # improve the validation process (leakage)
        val_dataloader = DataLoaderLite(
            self.encoder,
            batch_size=1,
            data_folder=self.sft_dataset.dataset_folder,
            process_rank=self.process_rank,
            num_processes=self.num_processes,
            split="train",
            master_process=self.master_process,
            instruction_template=self.sft_dataset.instruction_template,
            response_template=self.sft_dataset.response_template,
        )
        return train_dataloader, val_dataloader

    def calculate_validation_loss(self, dataloader, step, last_step):
        self.model.eval()
        dataloader.reset()
        with torch.no_grad():
            val_loss_accum = 0.0
            val_loss_steps = 20
            for _ in range(val_loss_steps):
                x, y, mask = dataloader.next_batch()
                x, y = (
                    x.to(device, dtype=torch.long),
                    y.to(device, dtype=torch.long),
                )
                if self.is_ddp_run:
                    with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                        logits, loss = self.model(x, targets=y, attention_mask=mask)
                else:
                    logits, loss = self.model(x, targets=y)
                loss = loss / val_loss_steps
                val_loss_accum += loss.detach()
        if self.is_ddp_run:
            dist.all_reduce(val_loss_accum, op=dist.ReduceOp.AVG)
        if self.master_process:
            print(f"validation loss: {val_loss_accum.item():.4f}")
            with open("log/log.txt", "a") as f:
                f.write(f"{step} val {val_loss_accum.item():.4f}\n")
            if step > 0 and (
                step % self.training_config.checkpoint_steps == 0 or last_step
            ):
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
        tokens = enc.encode_ordinary(" " + self.sft_dataset.instruction_template + " ")
        tokens.extend(enc.encode_ordinary("Who is Emmanuel Macron?"))
        tokens.extend(enc.encode_ordinary(" " + self.sft_dataset.response_template))
        tokens = torch.tensor(tokens, dtype=torch.long)
        tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1)
        xgen = tokens.to(self.device, dtype=torch.long)
        sample_rng = torch.Generator(device=self.device)
        sample_rng.manual_seed(42 + self.process_rank)
        while xgen.size(1) < max_length:
            # forward the model to get the logits
            with torch.no_grad():
                if self.is_ddp_run:
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
                ix = torch.multinomial(topk_probs, 1, generator=sample_rng)  # (B, 1)
                # gather the corresponding indices
                xcol = torch.gather(topk_indices, -1, ix)  # (B, 1)
                # append to the sequence
                xgen = torch.cat((xgen, xcol), dim=1)
        # print the generated text
        for i in range(num_return_sequences):
            tokens = xgen[i, :max_length].tolist()
            decoded = enc.decode(tokens)
            print(f"rank {self.process_rank} sample {i}: {decoded}")

    def train(self):
        dataloader, val_dataloader = self.load_data()
        for step in range(self.training_config.max_steps):
            t0 = time.time()
            last_step = step == self.training_config.max_steps - 1
            if step % self.training_config.validation_steps == 0 or last_step:
                self.calculate_validation_loss(val_dataloader, step, last_step)
            if (step > 0 and step % 10 == 0) or last_step:
                self.generate()
            # do one step of the optimization
            self.model.train()
            self.optimizer.zero_grad()
            loss_accum = 0.0
            for micro_step in range(self.grad_accum_steps):
                x, y, mask = dataloader.next_batch()
                if x is None or y is None:
                    dataloader.reset()
                    x, y, mask = dataloader.next_batch()
                x, y, mask = (
                    x.to(device, dtype=torch.long),
                    y.to(device, dtype=torch.long),
                    mask.to(device, dtype=torch.long),
                )
                # added after video, this field is also used by the forward pass.
                if self.is_ddp_run:
                    self.model.require_backward_grad_sync = (
                        micro_step == self.grad_accum_steps - 1
                    )
                if self.is_ddp_run:
                    with torch.autocast(
                        device_type=self.device_type, dtype=torch.bfloat16
                    ):
                        logits, loss = self.model(x, targets=y, attention_mask=mask)
                else:
                    logits, loss = self.model(x, targets=y, attention_mask=mask)
                loss = loss / self.grad_accum_steps
                loss_accum += loss.detach()
                loss.backward()

            if self.is_ddp_run:
                dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG)
            norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            # determine and set the learning rate for this iteration
            lr = self.get_lr(step)
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = lr
            self.optimizer.step()
            if self.device_type == "cuda":
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


ddp_rank = int(os.getenv("RANK", -1))
is_ddp_run = ddp_rank != -1
if is_ddp_run:
    init_process_group(backend="nccl")
    print("Starting CUDA GPU run ...")
    ddp_local_rank = int(os.environ["LOCAL_RANK"])
    ddp_world_size = int(os.environ["WORLD_SIZE"])
    device = f"cuda:{ddp_local_rank}"
    device_type = "cuda" if device.startswith("cuda") else "cpu"
    torch.cuda.set_device(device)
else:
    print("Starting MPS GPU run ...")
    ddp_rank = 0
    ddp_local_rank = 0
    ddp_world_size = 1
    master_process = True
    device = "cpu"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    device_type = "cuda" if device.startswith("cuda") else "mps"
    print(f"using device: {device}")


print("-----------------------------------------------------------------------------\n")
print(
    f"Init device {device}, rank {ddp_rank}, local rank {ddp_local_rank}, world size {ddp_world_size}"
)
print("-----------------------------------------------------------------------------\n")

enc = get_tokenizer()
sft_dataset = SFTDataset(
    dataset_folder="datasets/databricks-dolly-15k",
    instruction_template="### User:",
    response_template="### Assistant:",
)
training_config = TrainingConfig(batch_size=4, sequence_length=1024)
model = AutoModelForCausalLM.from_pretrained("openai-community/gpt2").to(device)
# TO DO: set issue with padding put label to -100 and create a mask at 0
trainer = SFTTrainer(
    model=model,
    encoder=enc,
    sft_dataset=sft_dataset,
    training_config=training_config,
    device=device,
    device_type=device_type,
    num_processes=ddp_world_size,
    process_rank=ddp_rank,
    ddp_local_rank=ddp_local_rank,
    is_ddp_run=is_ddp_run,
)
print("-----------------------------------------------------------------------------\n")
print("Training ....")
trainer.train()
print("-----------------------------------------------------------------------------\n")

print("-----------------------------------------------------------------------------\n")
print("Destroying process group ....")
if is_ddp_run:
    destroy_process_group()
print("-----------------------------------------------------------------------------\n")
