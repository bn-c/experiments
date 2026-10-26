"""Configuration for the application."""

from pathlib import Path

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Config(BaseSettings):
    """Configuration for the application."""

    model_config = SettingsConfigDict(env_file='.env')

    huggingface_api_token: str | None = Field(
        description='Optional HF token (public sudoku-extreme works without it).',
        default=None,
    )
    wandb_api_key: str | None = Field(description='The API key for the Weights and Biases API.', default=None)
    wandb_project: str | None = Field(description='The project name for the Weights and Biases API.', default=None)
    wandb_entity: str | None = Field(description='The entity name for the Weights and Biases API.', default=None)
    data_dir: Path = Field(default=Path('data'), description='Directory where datasets are stored.')


class ArchitectureConfig(BaseModel):
    """Architecture hyperparameters for the Sudoku JEPA model."""

    d_model: int = Field(default=512, description='Transformer hidden dimension.')
    n_layers: int = Field(default=8, description='Number of encoder Transformer layers.')
    n_heads: int = Field(default=8, description='Number of attention heads.')
    d_ffn: int = Field(default=2048, description='Feed-forward network inner dimension.')
    dropout: float = Field(default=0.1, description='Dropout rate.')
    d_latent: int = Field(default=256, description='Latent variable z dimension.')
    predictor_hidden: int = Field(default=1024, description='Predictor MLP hidden dimension.')
    decoder_layers: int = Field(default=4, description='Number of decoder Transformer layers.')
    decoder_heads: int = Field(default=8, description='Number of decoder attention heads.')
    decoder_d_cell: int = Field(default=128, description='Per-cell dimension in decoder.')


class TrainingConfig(BaseModel):
    """Training hyperparameters."""

    batch_size: int = Field(default=512, description='Training batch size.')
    lr: float = Field(default=3e-4, description='Peak learning rate.')
    weight_decay: float = Field(default=0.01, description='AdamW weight decay.')
    warmup_steps: int = Field(default=2000, description='Linear warmup steps.')
    epochs: int = Field(default=50, description='Total training epochs.')
    grad_clip_norm: float = Field(default=1.0, description='Max gradient norm for clipping.')
    ema_momentum_start: float = Field(default=0.996, description='Initial EMA momentum for target encoder.')
    ema_momentum_end: float = Field(default=1.0, description='Final EMA momentum for target encoder.')
    decode_loss_weight: float = Field(default=1.0, description='Weight for auxiliary decoder cross-entropy loss.')
    vicreg_sim_weight: float = Field(default=25.0, description='VICReg similarity (invariance) loss weight.')
    vicreg_var_weight: float = Field(default=1.0, description='VICReg variance loss weight.')
    vicreg_cov_weight: float = Field(default=0.01, description='VICReg covariance loss weight.')
    num_workers: int = Field(default=4, description='DataLoader worker count.')
    pin_memory: bool = Field(default=True, description='Pin memory in DataLoader.')
    checkpoint_dir: Path = Field(default=Path('checkpoints'), description='Directory for saving checkpoints.')
    keep_top_k: int = Field(default=3, description='Number of best checkpoints to keep.')
    checkpoint_interval_sec: int = Field(
        default=900,
        description='Minimum seconds between validation checkpoints (default 15 min; use ~1200 for 20 min).',
    )
    val_size: int = Field(default=500_000, description='Validation set size.')
    test_size: int = Field(default=500_000, description='Test set size.')
    split_seed: int = Field(default=42, description='Random seed for train/val/test split.')
    langevin_steps: int = Field(default=50, description='Langevin dynamics steps at inference.')
    langevin_lr: float = Field(default=0.01, description='Langevin dynamics step size.')
    langevin_noise_scale: float = Field(default=0.005, description='Langevin dynamics noise scale.')
    n_chains: int = Field(default=8, description='Number of parallel Langevin chains.')
    z_noise_scale: float = Field(
        default=0.1,
        description='Noise scale for z during training (z = z_encoder(z_target) + noise * scale). '
        'With L2-normalized z_encoder output (unit norm) and d_latent=256, '
        'scale=0.1 gives noise L2 ≈ 1.6 for ~0.6:1 SNR.',
    )
    constraint_loss_weight: float = Field(
        default=0.1, description='Weight for Sudoku constraint penalty during training.'
    )


def tiny_architecture() -> ArchitectureConfig:
    """Return a small architecture for CPU / smoke training (~1M params)."""
    return ArchitectureConfig(
        d_model=128,
        n_layers=2,
        n_heads=4,
        d_ffn=256,
        dropout=0.1,
        d_latent=64,
        predictor_hidden=256,
        decoder_layers=1,
        decoder_heads=4,
        decoder_d_cell=64,
    )


config = Config()
