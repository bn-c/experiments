"""TRM-MLP for 9x9 Sudoku (shared net, latent recursion, deep supervision)."""

from __future__ import annotations

import torch
from torch import nn

from trm.config import ModelConfig
from trm.model.mlp_mixer import MLPMixerBackbone

SEQ_LEN = 81


class TRMMLP(nn.Module):
    """
    Tiny Recursive Model with an MLP-Mixer backbone.

    Recursion (Algorithm 3, Jolicoeur-Martineau et al. 2025):
        for i in range(n): z = net(x, y, z)
        y = net(y, z)

    ``deep_recursion`` runs the block ``T-1`` times under ``no_grad``, then once with grad.
    """

    def __init__(self, cfg: ModelConfig) -> None:
        """Build embeddings, MLP-Mixer backbone, LM head, and halt Q-head."""
        super().__init__()
        self.cfg = cfg
        self.puzzle_embed = nn.Embedding(cfg.n_digits, cfg.d_model)
        self.answer_embed = nn.Embedding(cfg.n_digits, cfg.d_model)
        self.pos_embed = nn.Embedding(SEQ_LEN, cfg.d_model)
        self.puzzle_bias = nn.Parameter(torch.zeros(1, 1, cfg.d_model))
        self.backbone = MLPMixerBackbone(SEQ_LEN, cfg.d_model, cfg.n_layers, cfg.mlp_expansion)
        self.out_norm = nn.LayerNorm(cfg.d_model)
        self.y_norm = nn.LayerNorm(cfg.d_model)
        self.z_norm = nn.LayerNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.n_classes)
        # Halt Q-head (paper ACT, no continue): BCE target = sequence fully correct.
        self.q_head = nn.Linear(cfg.d_model, 1)
        with torch.no_grad():
            self.q_head.weight.zero_()
            self.q_head.bias.fill_(-5.0)
        self.y_init = nn.Parameter(torch.zeros(1, SEQ_LEN, cfg.d_model))
        self.z_init = nn.Parameter(torch.zeros(1, SEQ_LEN, cfg.d_model))

    def embed_puzzle(self, puzzle: torch.Tensor) -> torch.Tensor:
        """Embed puzzle digits ``(B, 81)`` → ``(B, 81, D)``."""
        pos = torch.arange(SEQ_LEN, device=puzzle.device)
        return self.puzzle_embed(puzzle) + self.pos_embed(pos) + self.puzzle_bias

    def _net_latent(self, x: torch.Tensor, y: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        return self.backbone(x + y + z)

    def _net_refine(self, y: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        return self.backbone(y + z)

    def latent_recursion(self, x: torch.Tensor, y: torch.Tensor, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply ``n`` latent ``z`` updates then one ``y`` refine."""
        for _ in range(self.cfg.n_latent):
            z = self._net_latent(x, y, z)
        y = self._net_refine(y, z)
        return y, z

    def deep_recursion(self, x: torch.Tensor, y: torch.Tensor, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Run ``T-1`` no-grad recursions then one with grad (live tensors for heads)."""
        cfg = self.cfg
        if cfg.T > 1:
            with torch.no_grad():
                for _ in range(cfg.T - 1):
                    y, z = self.latent_recursion(x, y, z)
        y, z = self.latent_recursion(x, y, z)
        return y, z

    def logits_from_y(self, y: torch.Tensor) -> torch.Tensor:
        """Map answer embeddings to per-cell class logits ``(B, 81, 9)``."""
        return self.head(self.out_norm(y))

    def initial_state(self, batch_size: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        """Expand learned ``y`` / ``z`` initial states for a batch."""
        y = self.y_init.expand(batch_size, -1, -1).to(device)
        z = self.z_init.expand(batch_size, -1, -1).to(device)
        return y, z

    def supervision_step(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        z: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        One deep-supervision / ACT micro-step.

        Matches paper Algorithm 3 / upstream: heads see live ``y`` (grads flow through
        the last ``latent_recursion``); carry returned for the next step is detached.

        Returns ``(y, z, logits, q_halt_logits)`` where ``q_halt_logits`` is ``(B,)``.
        """
        y, z = self.deep_recursion(x, y, z)
        y = self.y_norm(y)
        z = self.z_norm(z)
        logits = self.logits_from_y(y)
        q_halt = self.q_head(y.mean(dim=1)).squeeze(-1)
        # Detach carry only after heads — ``y.detach()`` must not feed the heads.
        return y.detach(), z.detach(), logits, q_halt

    def forward_supervised(
        self,
        puzzle: torch.Tensor,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """
        Deep supervision forward for training.

        Returns final ``y`` and a list of logits after each supervision step.
        """
        x = self.embed_puzzle(puzzle)
        batch_size = puzzle.shape[0]
        y, z = self.initial_state(batch_size, puzzle.device)
        step_logits: list[torch.Tensor] = []
        for _ in range(self.cfg.max_supervision_steps):
            y, z, logits, _q_halt = self.supervision_step(x, y, z)
            step_logits.append(logits)
        return y, step_logits

    def forward_inference(self, puzzle: torch.Tensor) -> torch.Tensor:
        """Run ``max_supervision_steps`` improvements and return final logits."""
        x = self.embed_puzzle(puzzle)
        y, z = self.initial_state(puzzle.shape[0], puzzle.device)
        with torch.no_grad():
            for _ in range(self.cfg.max_supervision_steps):
                y, z, logits, _q_halt = self.supervision_step(x, y, z)
        return logits
