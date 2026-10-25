"""
Small Vision Transformer that consumes the 12-channel DRF biomimetic front-end
output directly (no RGB patch embedding) instead of raw pixels.

Each of the 12 retinal pathway channels is patchified independently (shared
conv weights, as if it were its own single-channel image) rather than fused
together by a single in_chans=12 conv — the pathways are qualitatively
different signals (luminance, chroma, motion, ...), not RGB-like co-located
color planes, so mixing them in one convolution's receptive field would blend
unrelated signals together at the very first layer. A learned per-channel
embedding (added to each pathway's tokens) lets the encoder tell the 12
pathways apart; a shared spatial position embedding is reused across channels.

DRFViT = DRFBiomimeticFrontEnd(target_resolution=(64, 64))  ->  (B, 12, 64, 64)
         -> per-channel patchify -> ViT encoder -> [CLS] -> linear head
"""

from typing import Tuple
import warnings

import torch
import torch.nn as nn

from .frontend import DRFBiomimeticFrontEnd


class PatchEmbed(nn.Module):
    """
    Per-channel conv patchify with independent weights: (B, C, H, W) -> (B, C, N, embed_dim).

    A grouped conv (groups=in_chans) gives each of the C channels its own
    learned patchify kernel, rather than sharing one kernel across channels
    (shared weights) or mixing channels together in one kernel's receptive
    field (a plain in_chans=C conv).
    """

    def __init__(self, in_chans: int, img_size: int, patch_size: int, embed_dim: int):
        super().__init__()
        assert img_size % patch_size == 0, "img_size must be divisible by patch_size"
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.grid_size = img_size // patch_size
        self.num_patches = self.grid_size ** 2
        self.proj = nn.Conv2d(
            in_chans, in_chans * embed_dim, kernel_size=patch_size, stride=patch_size, groups=in_chans
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        x = self.proj(x)                                    # (B, C*embed_dim, grid, grid)
        x = x.view(b, c, self.embed_dim, self.num_patches)   # (B, C, embed_dim, N)
        return x.transpose(2, 3)                             # (B, C, N, embed_dim)


class TransformerBlock(nn.Module):
    """Pre-LN transformer block: MHSA + MLP, both with residual connections."""

    def __init__(self, embed_dim: int, num_heads: int, mlp_ratio: float, dropout: float):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(embed_dim)
        hidden = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        attn_out, _ = self.attn(h, h, h, need_weights=False)
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x


class VisionTransformer(nn.Module):
    """Minimal ViT encoder with a learned [CLS] token and classification head."""

    def __init__(
        self,
        in_chans: int = 12,
        img_size: int = 64,
        patch_size: int = 16,
        num_classes: int = 100,
        embed_dim: int = 192,
        depth: int = 6,
        num_heads: int = 3,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.patch_embed = PatchEmbed(in_chans, img_size, patch_size, embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        # Shared spatial position embedding (same grid position means the same
        # thing in every pathway) + a per-channel embedding distinguishing the
        # 12 pathways, analogous to ChannelViT-style channel tokens.
        self.pos_embed = nn.Parameter(torch.zeros(1, 1, num_patches, embed_dim))
        self.channel_embed = nn.Parameter(torch.zeros(1, in_chans, 1, embed_dim))
        self.cls_pos_embed = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_drop = nn.Dropout(dropout)

        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout) for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.channel_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.zeros_(m.bias)
            nn.init.ones_(m.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b = x.shape[0]
        x = self.patch_embed(x)                      # (B, C, N, embed_dim)
        x = x + self.pos_embed + self.channel_embed   # broadcast over (C) and (N) respectively
        x = x.flatten(1, 2)                           # (B, C*N, embed_dim)

        cls_tokens = self.cls_token.expand(b, -1, -1) + self.cls_pos_embed
        x = torch.cat([cls_tokens, x], dim=1)
        x = self.pos_drop(x)

        for block in self.blocks:
            x = block(x)

        x = self.norm(x)
        return self.head(x[:, 0])  # classify from [CLS]


def _rescale_per_channel_01(x: torch.Tensor, channel_min: torch.Tensor, channel_max: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    Fixed, global per-channel min-max rescale to [0, 1] (clamped), using
    dataset-wide statistics rather than each sample's own min/max.

    The 12 DRF pathways have wildly different native ranges (e.g. sustained
    luminance ~0.8, signed chroma ~+-0.8, all-zero motion channels on static
    images), so without this each per-pathway patch kernel would start from a
    different activation scale. Unlike per-sample min-max, this does NOT force
    every image to touch exactly 0 and 1 — relative contrast and absolute
    brightness/motion magnitude across samples are preserved; only
    out-of-calibration-range outliers get clamped.

    Note: `PatchEmbed` uses a *grouped* conv (independent weights per pathway),
    so this is a conditioning aid for optimisation, not a prerequisite for
    keeping channels distinguishable — the per-channel embedding below is what
    tells the 12 pathways apart.
    """
    scaled = (x - channel_min) / (channel_max - channel_min).clamp_min(eps)
    return scaled.clamp(0.0, 1.0)


class DRFViT(nn.Module):
    """
    DRF biomimetic front-end (frozen, zero-parameter) feeding a small ViT.

    The front-end is treated as a fixed, non-trainable preprocessing stage:
    every image is pushed through it as an independent "first frame" (motion
    channels 5-11 are consequently all-zero, which is correct for static
    i.i.d. classification images rather than video).
    """

    def __init__(
        self,
        num_classes: int = 100,
        drf_resolution: Tuple[int, int] = (64, 64),
        patch_size: int = 16,
        embed_dim: int = 192,
        depth: int = 6,
        num_heads: int = 3,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
    ):
        super().__init__()
        # The front-end is parameter-free (it is an analytic filter bank), so there is
        # nothing to freeze here: `frontend.parameters()` is empty by construction.
        self.frontend = DRFBiomimeticFrontEnd(target_resolution=drf_resolution).eval()

        # Global per-channel calibration stats for _rescale_per_channel_01, set
        # by calibrate(). Buffers so they're saved/restored with state_dict()
        # (i.e. checkpoints and --continue carry calibration forward for free).
        self.register_buffer("channel_min", torch.zeros(1, 12, 1, 1))
        self.register_buffer("channel_max", torch.ones(1, 12, 1, 1))
        self.calibrated = False

        self.vit = VisionTransformer(
            in_chans=12,
            img_size=drf_resolution[0],
            patch_size=patch_size,
            num_classes=num_classes,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
        )

    @torch.no_grad()
    def calibrate(self, loader, device, max_batches: int = 20) -> None:
        """
        Runs `max_batches` batches of raw RGB images through the frozen DRF
        front-end to compute a fixed, dataset-wide per-channel min/max, used
        thereafter by forward() instead of per-sample min-max rescaling.

        Call this once before training. `scripts/train_vit_imagenet100.py` does it
        automatically before the first epoch (and skips it when resuming, because the
        buffers are restored from the checkpoint).

        Raises:
            ValueError: if no batch could be read (e.g. an empty loader), which would
                otherwise leave `channel_min` at +inf and produce NaNs.
        """
        running_min = torch.full((1, 12, 1, 1), float("inf"), device=device)
        running_max = torch.full((1, 12, 1, 1), float("-inf"), device=device)
        n_batches = 0
        for i, (images, _labels) in enumerate(loader):
            if i >= max_batches:
                break
            self.frontend.reset()
            features = self.frontend(images.to(device))  # (B, 12, H, W)
            batch_min = features.amin(dim=(0, 2, 3), keepdim=True)
            batch_max = features.amax(dim=(0, 2, 3), keepdim=True)
            running_min = torch.minimum(running_min, batch_min)
            running_max = torch.maximum(running_max, batch_max)
            n_batches += 1

        if n_batches == 0:
            raise ValueError("calibrate() received no batches; refusing to leave min/max unset.")

        self.channel_min.copy_(running_min)
        self.channel_max.copy_(running_max)
        self.calibrated = True

    def forward(self, x_rgb: torch.Tensor) -> torch.Tensor:
        if not self.calibrated:
            warnings.warn(
                "DRFViT.forward() called before calibrate(); per-channel rescaling falls "
                "back to the (0, 1) identity range, so pathway scales are not normalised. "
                "Load a calibrated checkpoint or call calibrate(train_loader, device) first.",
                RuntimeWarning,
                stacklevel=2,
            )
        self.frontend.reset()  # each batch call is an independent set of static images
        with torch.no_grad():
            features = self.frontend(x_rgb)  # (B, 12, H, W)
            features = _rescale_per_channel_01(features, self.channel_min, self.channel_max)
        return self.vit(features)
