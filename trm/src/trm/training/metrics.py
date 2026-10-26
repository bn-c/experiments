"""Experiment tracking integration with Weights & Biases."""

import logging
from datetime import UTC, datetime
from pathlib import Path

from trm.config import AppConfig

logger = logging.getLogger(__name__)

_wandb_available = False
try:
    import wandb

    _wandb_available = True
except ImportError:
    pass


def init_wandb(app_cfg: AppConfig, run_name: str | None = None, extra_config: dict | None = None) -> bool:
    """
    Initialize a W&B run if credentials are configured.

    Args:
        app_cfg: Application config with wandb_api_key, wandb_project, wandb_entity.
        run_name: Optional name for the run.
        extra_config: Optional dict of hyperparameters to log.

    Returns:
        True if wandb was initialized, False otherwise.

    """
    if not _wandb_available or not app_cfg.wandb_api_key:
        logger.info('W&B not configured, skipping initialization')
        return False

    timestamp = datetime.now(tz=UTC).strftime('%Y%m%d-%H%M%S')
    name = f'{run_name}-{timestamp}' if run_name else timestamp

    wandb.login(key=app_cfg.wandb_api_key)
    wandb.init(
        project=app_cfg.wandb_project,
        entity=app_cfg.wandb_entity,
        name=name,
        config=extra_config or {},
    )
    return True


def log_train_step(loss: float, step: int) -> None:
    """Log per-batch training loss to W&B."""
    if not _wandb_available or wandb.run is None:
        return

    wandb.log({'train/loss': loss}, step=step)


def log_epoch(
    epoch: int,
    epoch_loss: float,
    step: int,
    *,
    cell_accuracy: float,
    puzzle_accuracy: float,
) -> None:
    """Log epoch-level training loss and EMA train accuracies to W&B."""
    if not _wandb_available or wandb.run is None:
        return

    wandb.log(
        {
            'train/epoch_loss': epoch_loss,
            'train/epoch': epoch,
            'train/epoch_cell_accuracy': cell_accuracy,
            'train/epoch_puzzle_accuracy': puzzle_accuracy,
        },
        step=step,
    )


def log_validation(cell_accuracy: float, puzzle_accuracy: float, step: int) -> None:
    """Log validation accuracy metrics to W&B."""
    if not _wandb_available or wandb.run is None:
        return

    wandb.log(
        {
            'val/cell_accuracy': cell_accuracy,
            'val/puzzle_accuracy': puzzle_accuracy,
        },
        step=step,
    )


def upload_checkpoint_to_wandb(checkpoint_path: Path) -> None:
    """
    Upload a checkpoint file as a W&B model artifact.

    Args:
        checkpoint_path: Path to the checkpoint .pt file.

    """
    if not _wandb_available or wandb.run is None:
        return

    artifact = wandb.Artifact(
        name=f'model-{wandb.run.id}',
        type='model',
    )
    artifact.add_file(str(checkpoint_path))
    wandb.log_artifact(artifact)
    logger.info('Uploaded checkpoint to W&B: %s', checkpoint_path.name)


def finish_wandb() -> None:
    """Finalize the W&B run, ensuring all artifacts are uploaded."""
    if not _wandb_available or wandb.run is None:
        return
    wandb.finish()
