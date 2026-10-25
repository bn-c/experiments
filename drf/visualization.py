"""Rendering helpers that colorize the 12 DRF channels into a dashboard image.

Requires the optional `numpy` and `Pillow` dependencies.
"""

import numpy as np
import torch
from PIL import Image, ImageDraw

from .frontend import DRFBiomimeticFrontEnd

CHANNEL_SUBTITLES = (
    "Parvo Tonic (L+M)",
    "Midget Opponent (L-M)",
    "Bistratified (S-LM)",
    "High-Freq Bright Edges",
    "High-Freq Dark Edges",
    "Phasic Flicker (dY/dt)",
    "DSGC (0 deg, Left->Right)",
    "DSGC (180 deg, Right->Left)",
    "DSGC (90 deg, Down->Up)",
    "DSGC (270 deg, Up->Down)",
    "Object Motion Sensitive",
    "Collision Approach (PV5)",
)

BIPOLAR_CHANNELS = ("CHROMA_RG", "CHROMA_BY", "TRANSIENT")

PANEL_SIZE = 180
PANEL_HEIGHT = 220
GUTTER = 10
HEADER_HEIGHT = 55


def colorize_unipolar(tensor_2d: np.ndarray) -> np.ndarray:
    """Colorizes non-negative activations with a black -> red -> yellow hot palette."""
    val = np.maximum(0.0, tensor_2d)
    vmax = np.percentile(val, 99.5) if np.max(val) > 1e-4 else 1.0
    vmax = max(vmax, 1e-4)
    norm = np.clip(val / vmax, 0.0, 1.0)

    r = np.clip(1.5 * norm, 0.0, 1.0)
    g = np.clip(2.0 * norm - 0.5, 0.0, 1.0)
    b = np.clip(3.0 * norm - 1.5, 0.0, 1.0)

    rgb = np.stack([r, g, b], axis=-1)
    return (rgb * 255.0).astype(np.uint8)


def colorize_bipolar(tensor_2d: np.ndarray, channel_name: str) -> np.ndarray:
    """
    Colorizes signed activations around zero.
    - CHROMA_RG: green (negative) -> black -> red (positive)
    - CHROMA_BY: yellow (negative) -> black -> blue (positive)
    - TRANSIENT: cyan (darkening) -> black -> orange (brightening)
    """
    vmax = np.percentile(np.abs(tensor_2d), 99.0) if np.max(np.abs(tensor_2d)) > 1e-4 else 1.0
    vmax = max(vmax, 1e-4)
    norm = np.clip(tensor_2d / vmax, -1.0, 1.0)

    h, w = norm.shape
    rgb = np.zeros((h, w, 3), dtype=np.float32)

    pos = np.maximum(0.0, norm)
    neg = np.maximum(0.0, -norm)

    if channel_name == "CHROMA_RG":
        rgb[..., 0] = pos
        rgb[..., 1] = neg
    elif channel_name == "CHROMA_BY":
        rgb[..., 2] = pos
        rgb[..., 0] = neg * 0.9
        rgb[..., 1] = neg * 0.9
    else:
        rgb[..., 0] = pos
        rgb[..., 1] = pos * 0.6 + neg * 0.5
        rgb[..., 2] = neg

    return (np.clip(rgb, 0.0, 1.0) * 255.0).astype(np.uint8)


def colorize_channel(channel_data: np.ndarray, channel_idx: int) -> np.ndarray:
    """Colorizes one channel according to its polarity."""
    name = DRFBiomimeticFrontEnd.CHANNEL_NAMES[channel_idx]
    if name in BIPOLAR_CHANNELS:
        return colorize_bipolar(channel_data, name)
    if name == "SUSTAINED_LUM":
        vmax = np.percentile(channel_data, 99.0) if np.max(channel_data) > 1e-4 else 1.0
        vmax = max(vmax, 1e-4)
        gray = (np.clip(channel_data / vmax, 0.0, 1.0) * 255.0).astype(np.uint8)
        return np.stack([gray, gray, gray], axis=-1)
    return colorize_unipolar(channel_data)


def channel_frame_to_image(channel_data: np.ndarray, channel_idx: int) -> Image.Image:
    """Colorizes a single (H, W) channel frame into a standalone RGB image."""
    return Image.fromarray(colorize_channel(channel_data, channel_idx))


def render_channel_panel(rgb_data: np.ndarray, title: str, subtitle: str = "") -> Image.Image:
    """Renders a single labeled channel square."""
    img = Image.fromarray(rgb_data).resize((PANEL_SIZE, PANEL_SIZE), Image.Resampling.NEAREST)
    panel = Image.new("RGB", (PANEL_SIZE, PANEL_HEIGHT), (20, 24, 33))
    panel.paste(img, (0, 0))

    draw = ImageDraw.Draw(panel)
    draw.text((6, 184), title, fill=(240, 240, 245))
    if subtitle:
        draw.text((6, 202), subtitle, fill=(160, 165, 180))
    draw.rectangle([0, 0, PANEL_SIZE - 1, PANEL_HEIGHT - 1], outline=(60, 65, 80))
    return panel


def create_12_channel_grid(
    original_rgb: np.ndarray,
    channels_tensor: torch.Tensor,
    header_text: str = ""
) -> Image.Image:
    """
    Composes a 4x4 dashboard:
      Row 0: raw input, SUSTAINED_LUM, CHROMA_RG, CHROMA_BY
      Row 1: SUBUNIT_ON, SUBUNIT_OFF, TRANSIENT, DIFF_MOTION
      Row 2: DS_RIGHT, DS_LEFT, DS_UP, DS_DOWN
      Row 3: LOOMING + per-channel statistics card
    """
    ch_data = channels_tensor.squeeze(0).cpu().numpy()
    channel_names = DRFBiomimeticFrontEnd.CHANNEL_NAMES

    panels = [
        render_channel_panel(
            colorize_channel(ch_data[i], i),
            f"{i}: {channel_names[i]}",
            CHANNEL_SUBTITLES[i],
        )
        for i in range(12)
    ]

    orig_pil = Image.fromarray(original_rgb).resize((PANEL_SIZE, PANEL_SIZE), Image.Resampling.BILINEAR)
    orig_panel = Image.new("RGB", (PANEL_SIZE, PANEL_HEIGHT), (20, 24, 33))
    orig_panel.paste(orig_pil, (0, 0))
    draw = ImageDraw.Draw(orig_panel)
    draw.text((6, 184), "RAW INPUT", fill=(100, 220, 255))
    draw.text((6, 202), f"{original_rgb.shape[1]}x{original_rgb.shape[0]} RGB", fill=(160, 165, 180))
    draw.rectangle([0, 0, PANEL_SIZE - 1, PANEL_HEIGHT - 1], outline=(80, 120, 180))

    canvas_w = 4 * PANEL_SIZE + 5 * GUTTER
    canvas_h = HEADER_HEIGHT + 4 * PANEL_HEIGHT + 5 * GUTTER
    canvas = Image.new("RGB", (canvas_w, canvas_h), (12, 14, 20))
    draw_canvas = ImageDraw.Draw(canvas)

    if header_text:
        grid_h, grid_w = channels_tensor.shape[-2:]
        draw_canvas.text((15, 15), header_text, fill=(255, 255, 255))
        draw_canvas.text(
            (15, 33),
            f"Dynamic Receptive Field Biomimetic Front-End "
            f"(12 Retinal Ganglion Channels @ {grid_w}x{grid_h})",
            fill=(160, 170, 190),
        )

    rows = [
        [orig_panel, panels[0], panels[1], panels[2]],
        [panels[3], panels[4], panels[5], panels[10]],
        [panels[6], panels[7], panels[8], panels[9]],
    ]
    for r_idx, row in enumerate(rows):
        y = HEADER_HEIGHT + r_idx * (PANEL_HEIGHT + GUTTER)
        for c_idx, item in enumerate(row):
            canvas.paste(item, (GUTTER + c_idx * (PANEL_SIZE + GUTTER), y))

    y3 = HEADER_HEIGHT + 3 * (PANEL_HEIGHT + GUTTER)
    canvas.paste(panels[11], (GUTTER, y3))

    info_w = PANEL_SIZE * 3 + 2 * GUTTER
    info_panel = Image.new("RGB", (info_w, PANEL_HEIGHT), (20, 24, 33))
    draw_info = ImageDraw.Draw(info_panel)
    draw_info.rectangle([0, 0, info_w - 1, PANEL_HEIGHT - 1], outline=(60, 65, 80))
    draw_info.text((15, 15), "BIOMIMETIC RETINAL STATE ANALYSIS", fill=(100, 200, 255))

    stats = [
        f"{channel_names[i]}: {float(ch_data[i].mean()):.3f} (max: {float(ch_data[i].max()):.3f})"
        for i in range(12)
    ]
    draw_info.text((15, 45), "\n".join(stats[:4]), fill=(210, 215, 230))
    draw_info.text((190, 45), "\n".join(stats[4:8]), fill=(210, 215, 230))
    draw_info.text((370, 45), "\n".join(stats[8:]), fill=(210, 215, 230))

    canvas.paste(info_panel, (GUTTER + PANEL_SIZE + GUTTER, y3))
    return canvas


def save_channel_images(channels_tensor: torch.Tensor, output_dir, prefix: str) -> list:
    """Saves each of the 12 channels as its own colorized PNG. Returns the written paths."""
    ch_data = channels_tensor.squeeze(0).cpu().numpy()
    names = DRFBiomimeticFrontEnd.CHANNEL_NAMES
    paths = []
    for i in range(12):
        path = output_dir / f"{prefix}_{i:02d}_{names[i].lower()}.png"
        channel_frame_to_image(ch_data[i], i).save(path)
        paths.append(path)
    return paths


def save_channel_videos(channel_frames: list, output_dir, prefix: str, duration_ms: int = 100) -> list:
    """
    Saves each of the 12 channels as its own animated GIF.

    Args:
        channel_frames: list of (12, H, W) numpy arrays, one per video frame.
        output_dir, prefix: destination naming.
        duration_ms: per-frame GIF duration.
    """
    names = DRFBiomimeticFrontEnd.CHANNEL_NAMES
    paths = []
    for i in range(12):
        frames = [channel_frame_to_image(frame[i], i) for frame in channel_frames]
        path = output_dir / f"{prefix}_{i:02d}_{names[i].lower()}.gif"
        frames[0].save(path, save_all=True, append_images=frames[1:], duration=duration_ms, loop=0)
        paths.append(path)
    return paths

