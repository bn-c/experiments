"""Training and model configuration."""

from __future__ import annotations

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class ModelConfig(BaseSettings):
    """TRM-MLP hyperparameters (paper defaults for Sudoku-Extreme)."""

    d_model: int = 512
    n_layers: int = 2
    mlp_expansion: int = 4
    n_latent: int = Field(default=6, description='Inner z updates per latent_recursion (paper n=6)')
    T: int = Field(default=3, description='Outer deep-recursion blocks (T-1 no-grad, 1 grad)')
    max_supervision_steps: int = Field(default=16, description='Deep supervision steps N_sup')
    n_digits: int = 10
    n_classes: int = 9

    model_config = SettingsConfigDict(env_prefix='TRM_')


class TrainConfig(BaseSettings):
    """Optimizer and data settings."""

    batch_size: int = 32
    epochs: int = 1
    lr: float = 1e-4
    weight_decay: float = 1.0
    ema_decay: float = 0.999
    train_size: int = 1000
    num_workers: int = 0
    pin_memory: bool = False
    checkpoint_dir: Path = Path('checkpoints')
    seed: int = 42

    model_config = SettingsConfigDict(env_prefix='TRM_TRAIN_')


def tiny_model_config() -> ModelConfig:
    """Small model for CPU smoke runs."""
    return ModelConfig(d_model=64, n_latent=2, T=2, max_supervision_steps=2, mlp_expansion=2)


def tiny_train_config() -> TrainConfig:
    """Small batch / few samples for smoke training."""
    return TrainConfig(batch_size=4, epochs=1, train_size=32)
