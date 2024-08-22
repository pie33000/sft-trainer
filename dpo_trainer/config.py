from dataclasses import dataclass


@dataclass
class OptimizerConfig:
    weight_decay: float
    learning_rate: float
    accumulation_steps: int


@dataclass
class DPOLossConfig:
    beta: float
    label_smoothing: float
    loss_type: str


@dataclass
class TrainingConfig:
    batch_size: int
    epochs: int
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
class DPOConfig:
    device: str
    optimizer_config: OptimizerConfig
    dpo_loss_config: DPOLossConfig
    training_config: TrainingConfig
    ddp_config: DDPConfig
