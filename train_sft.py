from omegaconf import OmegaConf
from transformers import AutoModelForCausalLM

from sft_trainer.config import SFTConfig
from sft_trainer.dataloader import SFTColumnsMapping, create_dataloader
from sft_trainer.trainer import SFTTrainer
from utils import get_tokenizer

model = AutoModelForCausalLM.from_pretrained("openai-community/gpt2")
enc = get_tokenizer("gpt2")

conf = SFTConfig(**OmegaConf.load("sft_trainer/config.yml"))
dataloader = create_dataloader(
    "Open-Orca/OpenOrca",
    enc,
    batch_size=conf.training_config.batch_size,
    split="train[:50%]",
    columns_mapping=SFTColumnsMapping(prompt="question", answer="response"),
)
trainer = SFTTrainer(model, enc, dataloader, conf)
trainer.train()
