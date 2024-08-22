from omegaconf import OmegaConf
from transformers import GPT2LMHeadModel

from dpo_trainer.config import DPOConfig
from dpo_trainer.dataloader import create_dataloader
from dpo_trainer.trainer import DPOTrainer
from dpo_trainer.utils import get_tokenizer

if __name__ == "__main__":
    model_name = "gpt2"
    enc = get_tokenizer(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name)

    config = DPOConfig(**OmegaConf.load("dpo_trainer/config.yml"))

    dataloader = create_dataloader(
        "Dahoas/full-hh-rlhf", enc, batch_size=config.training_config.batch_size
    )

    dpo_trainer = DPOTrainer(model, enc, dataloader, config)
    dpo_trainer.train()


# torchrun --standalone --nproc_per_node=2 dpo_trainer/__main__.py
