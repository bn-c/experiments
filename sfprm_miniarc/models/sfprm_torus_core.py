"""Weight-tied toroidal fixed-point reasoner (SFPRM torus A/B fork).

Parallel copy of sfprm_core with product-of-circles geometry from toroidal_ops.
Sphere baseline (sfprm_core / spherical_ops) is untouched. Not paper-faithful.

``project_to_torus`` bakes unit ambient (pair radius 1/√n_pairs) for latents,
weights, and Q/K. Attention: project → RoPE → re-project over full ambient
(interleaved pairs; RoPE half-split leaves the torus) — no extra Q/K scale.

Desk default dims match sphere: d_model=384, n_heads=8, d_ff=1536, n_layers=2.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.toroidal_ops import chordal_residual, guarded_slerp, project_to_torus

# Desk 16GB default (paper ratios: heads=8, MLP×4, L_layers=2).
DEFAULT_D_MODEL = 384
DEFAULT_N_HEADS = 8
DEFAULT_D_FF = 1536  # 4 × d_model
DEFAULT_N_LAYERS = 2
DEFAULT_VOCAB = 12  # colors 0–9, PAD=10, SEP=11 (paper Mini-ARC vocab 12)
PAD_ID = 10
SEP_ID = 11


def logit(p: float) -> float:
    p = min(max(p, 1e-6), 1.0 - 1e-6)
    return math.log(p / (1.0 - p))


class RotaryEmbedding(nn.Module):
    """Standard RoPE on head_dim (even)."""

    def __init__(self, head_dim: int, max_seq_len: int = 256, base: float = 10000.0):
        super().__init__()
        if head_dim % 2 != 0:
            raise ValueError(f"RoPE head_dim must be even, got {head_dim}")
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.max_seq_len = max_seq_len
        self._build_cache(max_seq_len)

    def _build_cache(self, seq_len: int) -> None:
        t = torch.arange(seq_len, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)  # [S, head_dim/2]
        emb = torch.cat([freqs, freqs], dim=-1)
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :], persistent=False)
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :], persistent=False)
        self.max_seq_len = seq_len

    def forward(self, seq_len: int) -> tuple[torch.Tensor, torch.Tensor]:
        if seq_len > self.max_seq_len:
            self._build_cache(seq_len)
        return self.cos_cached[:, :, :seq_len], self.sin_cached[:, :, :seq_len]


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rope(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    # q,k: [B, H, S, Dh]
    q = q * cos + _rotate_half(q) * sin
    k = k * cos + _rotate_half(k) * sin
    return q, k


class CausalDepthwiseConv1d(nn.Module):
    """Bias-free depthwise 1-D causal conv (paper: k=4)."""

    def __init__(self, d_model: int, kernel_size: int = 4):
        super().__init__()
        self.kernel_size = kernel_size
        self.conv = nn.Conv1d(
            d_model,
            d_model,
            kernel_size=kernel_size,
            groups=d_model,
            bias=False,
            padding=0,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, S, D] → conv over sequence
        x_t = x.transpose(1, 2)  # [B, D, S]
        x_t = F.pad(x_t, (self.kernel_size - 1, 0))
        return self.conv(x_t).transpose(1, 2)


class SwiGLU(nn.Module):
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.w_gate = nn.Linear(d_model, d_ff, bias=False)
        self.w_up = nn.Linear(d_model, d_ff, bias=False)
        self.w_down = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))


class NGPTLayer(nn.Module):
    """One nGPT layer: QK-norm attention + SwiGLU MLP, each with eigen-LR retraction."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        rope: RotaryEmbedding,
        conv_kernel: int = 4,
        weight_dropout: float = 0.0,
    ):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f"d_model={d_model} must be divisible by n_heads={n_heads}")
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.rope = rope
        self.weight_dropout = weight_dropout

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

        self.conv = CausalDepthwiseConv1d(d_model, kernel_size=conv_kernel)
        self.mlp = SwiGLU(d_model, d_ff)

        # Eigen learning rates (per hidden dim), init 0.5 as paper
        self.alpha_A = nn.Parameter(torch.full((d_model,), 0.5))
        self.alpha_M = nn.Parameter(torch.full((d_model,), 0.5))
        # Per-dimension QK scale, init 1
        self.s_qk = nn.Parameter(torch.ones(self.head_dim))

    def _drop_weight(self, w: torch.Tensor) -> torch.Tensor:
        if not self.training or self.weight_dropout <= 0:
            return w
        return F.dropout(w, p=self.weight_dropout, training=True)

    def _linear(self, linear: nn.Linear, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self._drop_weight(linear.weight), linear.bias)

    def forward(
        self,
        z: torch.Tensor,
        *,
        variational_mask: torch.Tensor | None = None,
        pad_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        B, S, D = z.shape
        if variational_mask is not None:
            z = z * variational_mask

        q = project_to_torus(self._linear(self.q_proj, z))
        k = project_to_torus(self._linear(self.k_proj, z))
        v = self._linear(self.v_proj, z)

        q = q.view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, S, self.n_heads, self.head_dim).transpose(1, 2)

        cos, sin = self.rope(S)
        cos = cos.to(dtype=q.dtype, device=q.device)
        sin = sin.to(dtype=q.dtype, device=q.device)
        q, k = apply_rope(q, k, cos, sin)
        # RoPE half-split leaves interleaved pairs; retract over full ambient so
        # project_to_torus alone yields ‖q‖₂≈1 (no duplicate 1/√(d/2) scale).
        q = project_to_torus(q.transpose(1, 2).contiguous().view(B, S, D))
        k = project_to_torus(k.transpose(1, 2).contiguous().view(B, S, D))
        q = q.view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, S, self.n_heads, self.head_dim).transpose(1, 2)

        # nGPT-style: unit-ambient Q/K → logits are s_qk-scaled cosines (no /√d_h).
        scale = self.s_qk.to(dtype=q.dtype)
        logits = (q * scale) @ k.transpose(-2, -1)
        if pad_mask is not None:
            # pad_mask [B, S] True=valid; block attention to PAD keys
            key_ok = pad_mask[:, None, None, :]  # [B, 1, 1, S]
            logits = logits.masked_fill(~key_ok, torch.finfo(logits.dtype).min)
        attn = torch.softmax(logits, dim=-1)
        if pad_mask is not None:
            # Zero weights on all-masked rows (numerical safety) and pad queries
            attn = torch.nan_to_num(attn, nan=0.0)
            attn = attn * pad_mask[:, None, :, None].to(dtype=attn.dtype)
        attn_out = (attn @ v).transpose(1, 2).contiguous().view(B, S, D)
        attn_out = project_to_torus(self._linear(self.out_proj, attn_out))

        # Eigen-LR attention retraction
        z = project_to_torus(z + self.alpha_A * (attn_out - z))

        # Causal conv + SwiGLU MLP branch (weight dropout on Linear mats)
        h = project_to_torus(z + self.conv(z))
        mlp_out = project_to_torus(
            F.linear(
                F.silu(F.linear(h, self._drop_weight(self.mlp.w_gate.weight)))
                * F.linear(h, self._drop_weight(self.mlp.w_up.weight)),
                self._drop_weight(self.mlp.w_down.weight),
            )
        )
        z = project_to_torus(z + self.alpha_M * (mlp_out - z))
        return z


class NormalizedTransformerBlock(nn.Module):
    """Stack of ``n_layers`` nGPT layers (weight-tied across solver iterations)."""

    def __init__(
        self,
        d_model: int = DEFAULT_D_MODEL,
        n_heads: int = DEFAULT_N_HEADS,
        d_ff: int = DEFAULT_D_FF,
        n_layers: int = DEFAULT_N_LAYERS,
        max_seq_len: int = 128,
        conv_kernel: int = 4,
        weight_dropout: float = 0.05,
        variational_dropout: float = 0.2,
    ):
        super().__init__()
        self.d_model = d_model
        self.variational_dropout = variational_dropout
        self.rope = RotaryEmbedding(d_model // n_heads, max_seq_len=max_seq_len)
        self.layers = nn.ModuleList(
            [
                NGPTLayer(
                    d_model,
                    n_heads,
                    d_ff,
                    self.rope,
                    conv_kernel=conv_kernel,
                    weight_dropout=weight_dropout,
                )
                for _ in range(n_layers)
            ]
        )

    def sample_variational_mask(
        self,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor | None:
        """Bernoulli mask shared across sequence *and* solver iters (locked)."""
        if not self.training or self.variational_dropout <= 0:
            return None
        keep = 1.0 - self.variational_dropout
        return (
            torch.bernoulli(torch.full((batch_size, 1, self.d_model), keep, device=device, dtype=dtype))
            / keep
        )

    def forward(
        self,
        z: torch.Tensor,
        *,
        variational_mask: torch.Tensor | None = None,
        pad_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # Caller should pass a locked mask across solver iters. Fallback samples once
        # per forward only when none is provided (single-shot calls).
        if variational_mask is None:
            variational_mask = self.sample_variational_mask(z.shape[0], z.device, z.dtype)
        if variational_mask is not None:
            z = z * variational_mask
        for layer in self.layers:
            z = layer(z, variational_mask=None, pad_mask=pad_mask)
        return z


class SFPRMTorus(nn.Module):
    """Toroidal Fixed-Point Reasoner (product-of-circles; parallel to sphere SFPRM)."""

    def __init__(
        self,
        vocab_size: int = DEFAULT_VOCAB,
        d_model: int = DEFAULT_D_MODEL,
        n_heads: int = DEFAULT_N_HEADS,
        d_ff: int = DEFAULT_D_FF,
        n_layers: int = DEFAULT_N_LAYERS,
        max_seq_len: int = 128,
        n_puzzles: int = 149,
        puzzle_emb_len: int = 1,
        num_classes: int = 10,
        weight_dropout: float = 0.05,
        variational_dropout: float = 0.2,
        lambda_init: float = 0.1,
        s_z_init: float = 16.0,
    ):
        super().__init__()
        if d_model % 2 != 0:
            raise ValueError(f"torus d_model must be even, got {d_model}")
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.puzzle_emb_len = puzzle_emb_len
        self.n_puzzles = n_puzzles
        self.variational_dropout = variational_dropout

        self.token_emb = nn.Embedding(vocab_size, d_model)
        # Per-puzzle embedding (length 1 for Mini-ARC); prepended to the sequence.
        self.puzzle_emb = nn.Embedding(max(n_puzzles, 1), d_model * puzzle_emb_len)

        self.core_block = NormalizedTransformerBlock(
            d_model=d_model,
            n_heads=n_heads,
            d_ff=d_ff,
            n_layers=n_layers,
            max_seq_len=max_seq_len + puzzle_emb_len,
            weight_dropout=weight_dropout,
            variational_dropout=variational_dropout,
        )

        # λ = sigmoid(raw); init so sigmoid(raw) = lambda_init (paper: 0.1, not σ(0)=0.5)
        self.raw_lambda = nn.Parameter(torch.tensor(logit(lambda_init)))
        self.s_z = nn.Parameter(torch.tensor(s_z_init))
        # Unnormalized readout head (paper B.2)
        self.head = nn.Linear(d_model, num_classes, bias=False)
        # Q-head on detached state (ARC / Mini-ARC; loss weight 0.5 in train)
        self.q_head = nn.Linear(d_model, 1)
        nn.init.zeros_(self.q_head.weight)
        nn.init.constant_(self.q_head.bias, -5.0)

        self.reset_nGPT_weights()

    def _is_unnormalized_head(self, module: nn.Module) -> bool:
        # Paper B.2: readout is unnormalized; Q-head is a scalar probe — leave both alone.
        return module is self.head or module is self.q_head

    def reset_nGPT_weights(self) -> None:
        """Initialize Linear / Embedding rows with per-pair torus renorm."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=1.0 / math.sqrt(m.weight.shape[1]))
                if not self._is_unnormalized_head(m):
                    with torch.no_grad():
                        m.weight.copy_(project_to_torus(m.weight))
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, mean=0.0, std=1.0 / math.sqrt(self.d_model))
                with torch.no_grad():
                    m.weight.copy_(project_to_torus(m.weight))
            elif isinstance(m, nn.Conv1d):
                nn.init.normal_(m.weight, mean=0.0, std=1.0 / math.sqrt(m.kernel_size[0]))
                with torch.no_grad():
                    w = m.weight.view(m.weight.shape[0], -1)
                    m.weight.copy_(project_to_torus(w).view_as(m.weight))
        # Re-apply Q-head prior after generic Linear init
        nn.init.zeros_(self.q_head.weight)
        nn.init.constant_(self.q_head.bias, -5.0)

    @torch.no_grad()
    def renorm_weights(self) -> None:
        """Post-optimizer-step per-pair row renorm (torus only; puzzle emb excluded)."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                if self._is_unnormalized_head(m):
                    continue
                m.weight.copy_(project_to_torus(m.weight))
            elif isinstance(m, nn.Embedding) and m is not self.puzzle_emb:
                m.weight.copy_(project_to_torus(m.weight))
            elif isinstance(m, nn.Conv1d):
                w = m.weight.view(m.weight.shape[0], -1)
                m.weight.copy_(project_to_torus(w).view_as(m.weight))

    @property
    def lambda_param(self) -> torch.Tensor:
        return torch.sigmoid(self.raw_lambda)

    def get_input_embedding(self, x: torch.Tensor, puzzle_ids: torch.Tensor | None = None) -> torch.Tensor:
        """
        Token embeddings + optional puzzle embedding prepended.

        Returns z_input of shape [B, puzzle_emb_len + S, D] on the product torus.
        """
        seq_len = x.shape[1]
        if seq_len + self.puzzle_emb_len > self.max_seq_len + self.puzzle_emb_len:
            raise ValueError(f"sequence length {seq_len} exceeds max_seq_len={self.max_seq_len}")
        tok = self.token_emb(x)  # [B, S, D]
        if self.puzzle_emb_len > 0:
            if puzzle_ids is None:
                puzzle_ids = torch.zeros(x.shape[0], dtype=torch.long, device=x.device)
            pe = self.puzzle_emb(puzzle_ids.clamp(0, self.n_puzzles - 1))
            pe = pe.view(x.shape[0], self.puzzle_emb_len, self.d_model)
            tok = torch.cat([pe, tok], dim=1)
        return project_to_torus(tok)

    def expand_query_mask(self, query_mask: torch.Tensor) -> torch.Tensor:
        """Shift query mask right by puzzle_emb_len (False on puzzle tokens)."""
        if self.puzzle_emb_len <= 0:
            return query_mask
        B = query_mask.shape[0]
        pad = torch.zeros(B, self.puzzle_emb_len, dtype=torch.bool, device=query_mask.device)
        return torch.cat([pad, query_mask], dim=1)

    def token_pad_mask(self, x: torch.Tensor, pad_id: int = PAD_ID) -> torch.Tensor:
        """
        Valid-token mask for sequences (+ puzzle emb), shape [B, puzzle+S].

        True = real content (puzzle emb + non-PAD tokens). PAD slots are False
        so chordal halt / attention ignore them.
        """
        valid = x != pad_id  # [B, S]
        if self.puzzle_emb_len <= 0:
            return valid
        pe = torch.ones(x.shape[0], self.puzzle_emb_len, dtype=torch.bool, device=x.device)
        return torch.cat([pe, valid], dim=1)

    def sample_variational_mask(self, batch_size: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor | None:
        """Sample a locked variational mask once per segment (reuse across solver iters)."""
        return self.core_block.sample_variational_mask(batch_size, device, dtype)

    def propose(
        self,
        z: torch.Tensor,
        z_input: torch.Tensor,
        *,
        variational_mask: torch.Tensor | None = None,
        pad_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """F_θ(z; x) = f_θ(Slerp_λ(z, x)) — undamped proposal (for residual / halt)."""
        z_inj = guarded_slerp(z, z_input, self.lambda_param)
        return self.core_block(z_inj, variational_mask=variational_mask, pad_mask=pad_mask)

    def step(
        self,
        z: torch.Tensor,
        z_input: torch.Tensor,
        eta: torch.Tensor,
        *,
        variational_mask: torch.Tensor | None = None,
        pad_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        One damped solver iteration.

        Returns (z_next, z_tilde proposal, per-sample mean-pair chordal residual).
        ``eta`` is per-sample in (0, 1], shape [B] or broadcastable.
        Pass a locked ``variational_mask`` across iters within a BPTT segment.
        """
        z_tilde = self.propose(z, z_input, variational_mask=variational_mask, pad_mask=pad_mask)
        residual = chordal_residual(z, z_tilde, token_mask=pad_mask)
        z_next = guarded_slerp(z, z_tilde, eta)
        return z_next, z_tilde, residual

    def decode_logits(self, z: torch.Tensor) -> torch.Tensor:
        return self.s_z * self.head(z)

    def q_logits(self, z: torch.Tensor) -> torch.Tensor:
        """Q-head on mean-pooled state (call with detached z)."""
        return self.q_head(z.mean(dim=1)).squeeze(-1)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
