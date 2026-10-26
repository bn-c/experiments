import torch
from torch.optim.lr_scheduler import LinearLR

from ebm.training.checkpoint import (
    LAST_CHECKPOINT_NAME,
    CheckpointManager,
    _CheckpointData,
    resolve_resume_position,
)


def _make_data(epoch: int = 0, val_energy: float = 1.0, cell_accuracy: float = 0.1) -> _CheckpointData:
    model = torch.nn.Linear(4, 4)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    return _CheckpointData(
        model=model,
        optimizer=optimizer,
        epoch=epoch,
        step=0,
        val_energy=val_energy,
        cell_accuracy=cell_accuracy,
    )


def test_save_creates_file(tmp_path):
    mgr = CheckpointManager(tmp_path, keep_top_k=3)
    path = mgr.save(_make_data(epoch=0, cell_accuracy=0.5))
    assert path is not None
    assert path.exists()


def test_keeps_top_k(tmp_path):
    mgr = CheckpointManager(tmp_path, keep_top_k=2)
    mgr.save(_make_data(epoch=0, cell_accuracy=0.3))
    mgr.save(_make_data(epoch=1, cell_accuracy=0.5))
    mgr.save(_make_data(epoch=2, cell_accuracy=0.8))

    checkpoint_files = list(tmp_path.glob('*.pt'))
    assert len(checkpoint_files) == 2


def test_rejects_worse_than_top_k(tmp_path):
    mgr = CheckpointManager(tmp_path, keep_top_k=2)
    mgr.save(_make_data(epoch=0, cell_accuracy=0.5))
    mgr.save(_make_data(epoch=1, cell_accuracy=0.8))
    path = mgr.save(_make_data(epoch=2, cell_accuracy=0.1))
    assert path is None


def test_best_path_empty(tmp_path):
    mgr = CheckpointManager(tmp_path, keep_top_k=3)
    assert mgr.best_path is None


def test_best_path_returns_highest_accuracy(tmp_path):
    mgr = CheckpointManager(tmp_path, keep_top_k=3)
    mgr.save(_make_data(epoch=0, cell_accuracy=0.3))
    mgr.save(_make_data(epoch=1, cell_accuracy=0.8))
    mgr.save(_make_data(epoch=2, cell_accuracy=0.5))
    best = mgr.best_path
    assert best is not None
    assert 'acc0.8000' in best.name


def test_save_last_and_load_scheduler(tmp_path):
    mgr = CheckpointManager(tmp_path, keep_top_k=3)
    data = _make_data(epoch=2, cell_accuracy=0.42)
    scheduler = LinearLR(data.optimizer, start_factor=1.0, total_iters=10)
    for _ in range(3):
        scheduler.step()

    last_path = mgr.save_last(data, scheduler)
    assert last_path.name == LAST_CHECKPOINT_NAME

    new_model = torch.nn.Linear(4, 4)
    new_optimizer = torch.optim.SGD(new_model.parameters(), lr=0.01)
    new_scheduler = LinearLR(new_optimizer, start_factor=1.0, total_iters=10)
    meta = CheckpointManager.load(last_path, new_model, new_optimizer, new_scheduler)
    assert meta['epoch'] == 2
    assert new_scheduler.state_dict() == scheduler.state_dict()


def test_resolve_resume_position():
    assert resolve_resume_position(1, None, 100) == (2, 0)
    assert resolve_resume_position(1, 100, 100) == (2, 0)
    assert resolve_resume_position(2, 50, 100) == (2, 50)


def test_save_last_persists_batch_in_epoch(tmp_path):
    mgr = CheckpointManager(tmp_path, keep_top_k=1)
    data = _make_data(epoch=3, cell_accuracy=0.5)
    data.batch_in_epoch = 42
    scheduler = LinearLR(data.optimizer, start_factor=1.0, total_iters=10)
    path = mgr.save_last(data, scheduler)
    meta = torch.load(path, weights_only=True)
    assert meta['batch_in_epoch'] == 42


def test_load_restores_model(tmp_path):
    mgr = CheckpointManager(tmp_path, keep_top_k=3)
    data = _make_data(epoch=5, cell_accuracy=0.5)
    original_state = {k: v.clone() for k, v in data.model.state_dict().items()}
    path = mgr.save(data)
    assert path is not None

    new_model = torch.nn.Linear(4, 4)
    checkpoint = CheckpointManager.load(path, new_model)
    assert checkpoint['epoch'] == 5
    for k, v in new_model.state_dict().items():
        assert torch.allclose(v, original_state[k])
