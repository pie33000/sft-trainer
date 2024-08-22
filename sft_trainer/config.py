from dataclasses import dataclass


@dataclass
class OptimizerConfig:
    weight_decay: float
    learning_rate: float
    min_lr: float
    max_lr: float
    warmup_steps: int
    accumulation_steps: int


@dataclass
class TrainingConfig:
    max_steps: int
    batch_size: int
    epochs: int
    sequence_length: int
    log_path: str
    step_log_training_loss: int
    step_log_eval_loss: int
    step_save_model: int
    checkpoint_path: str


@dataclass
class DDPConfig:
    master_process: bool
    num_processes: int
    process_rank: int


@dataclass
class SFTConfig:
    device: str
    optimizer_config: OptimizerConfig
    training_config: TrainingConfig
    ddp_config: DDPConfig
