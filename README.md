# Fine-Tuning LLMs - Distributed Training

This guide walks you through the setup and usage of the DPO (Direct Preference Optimization) Trainer to align your Large Language Model (LLM) with user preferences on multiple GPUs.

## Installation

First, create a new Conda environment and install the necessary dependencies:

    conda create -n llm-ft python=3.10
    conda activate llm-ft
    pip install .


## Usage
### DPO Trainer

The DPO Trainer allows you to fine-tune your LLM to better reflect user preferences at scale. The following steps demonstrate how to set up and run the training process.

**Imports**

Ensure you import the necessary modules:

    from omegaconf import OmegaConf
    from transformers import GPT2LMHeadModel

    from dpo_trainer.config import DPOConfig
    from dpo_trainer.dataloader import create_dataloader
    from dpo_trainer.trainer import DPOTrainer
    from dpo_trainer.utils import get_tokenizer

**Set Up Training**

Begin by specifying the model name, initializing the tokenizer, and loading the model. Then, load the training configuration and create a data loader.
    
    model_name = "gpt2"
    enc = get_tokenizer(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name)

    config = DPOConfig(**OmegaConf.load("dpo_trainer/config.yml"))

    dataloader = create_dataloader(
        "Dahoas/full-hh-rlhf", enc, batch_size=config.training_config.batch_size
    )

    dpo_trainer = DPOTrainer(model, enc, dataloader, config)

**Train a model**

To begin training your model on a single GPU (cuda, mps) with the DPO Trainer, simply call the train method:

    dpo_trainer.train()

To train a model on multiple GPUs (cuda), you can use torchrun and specify the number of gpus to use, an example with 8 GPUs.

    torchrun --standalone --nproc_per_node=8 dpo_trainer/__main__.py




     
       