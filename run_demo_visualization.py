"""
Demo script to test DRFBiomimeticFrontEnd on real-world sample image and video.

Generates:
1. Static picture 12-channel decomposition (t=0)
2. Saccadic eye-movement 12-channel decomposition (t=1)
3. Dynamic video 12-channel animated stream (GIF)
4. Traffic camera video 12-channel animated stream (GIF)

All outputs are saved to the workspace and copied to the conversation artifact directory.
"""

import os
import shutil
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch
import imageio.v3 as iio

from src.models.drf_frontend import DRFBiomimeticFrontEnd

# Destination artifact directory
ARTIFACT_DIR = Path(r"C:\Users\bachnc1\.gemini\antigravity\brain\d02de693-9770-4d88-a61a-bf2e4b9a741d")
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)


def colorize_unipolar(tensor_2d: np.ndarray) -> np.ndarray:
    """
    Colorizes non-negative feature activations [0, max] using an 'inferno'-like colormap:
    Black -> Dark Purple -> Hot Red -> Orange -> Bright Yellow.
    Input: (H, W) float
    Output: (H, W, 3) uint8
    """
    val = np.maximum(0.0, tensor_2d)
    vmax = np.percentile(val, 99.5) if np.max(val) > 1e-4 else 1.0
    vmax = max(vmax, 1e-4)
    norm = np.clip(val / vmax, 0.0, 1.0)

    # 5-stage hot palette
    r = np.clip(1.5 * norm, 0.0, 1.0)
    g = np.clip(2.0 * norm - 0.5, 0.0, 1.0)
    b = np.clip(3.0 * norm - 1.5, 0.0, 1.0)

    rgb = np.stack([r, g, b], axis=-1)
    return (rgb * 255.0).astype(np.uint8)


def colorize_bipolar(tensor_2d: np.ndarray, channel_name: str) -> np.ndarray:
    """
    Colorizes signed bipolar activations [-max, +max].
    - For CHROMA_RG: Green (negative) -> Black (zero) -> Red (positive)
    - For CHROMA_BY: Yellow (negative) -> Black (zero) -> Blue (positive)
    - For TRANSIENT: Blue/Cyan (darkening -) -> Black (0) -> Orange/Red (brightening +)
    """
    vmax = np.percentile(np.abs(tensor_2d), 99.0) if np.max(np.abs(tensor_2d)) > 1e-4 else 1.0
    vmax = max(vmax, 1e-4)
    norm = np.clip(tensor_2d / vmax, -1.0, 1.0)

    H, W = norm.shape
    rgb = np.zeros((H, W, 3), dtype=np.float32)

    pos = np.maximum(0.0, norm)
    neg = np.maximum(0.0, -norm)

    if channel_name == "CHROMA_RG":
        # Positive = Red, Negative = Green
        rgb[..., 0] = pos
        rgb[..., 1] = neg
    elif channel_name == "CHROMA_BY":
        # Positive = Blue, Negative = Yellow (R+G)
        rgb[..., 2] = pos
        rgb[..., 0] = neg * 0.9
        rgb[..., 1] = neg * 0.9
    else:  # TRANSIENT
        # Brightening = Warm Yellow/Red, Darkening = Cold Cyan/Blue
        rgb[..., 0] = pos
        rgb[..., 1] = pos * 0.6 + neg * 0.5
        rgb[..., 2] = neg

    return (np.clip(rgb, 0.0, 1.0) * 255.0).astype(np.uint8)


def colorize_channel(channel_data: np.ndarray, channel_idx: int) -> np.ndarray:
    """Colorizes channel depending on whether it is unipolar or bipolar."""
    names = DRFBiomimeticFrontEnd.CHANNEL_NAMES
    name = names[channel_idx]
    if name in ["CHROMA_RG", "CHROMA_BY", "TRANSIENT"]:
        return colorize_bipolar(channel_data, name)
    elif name == "SUSTAINED_LUM":
        # Grayscale representation for luminance
        vmax = np.percentile(channel_data, 99.0) if np.max(channel_data) > 1e-4 else 1.0
        vmax = max(vmax, 1e-4)
        norm = np.clip(channel_data / vmax, 0.0, 1.0)
        gray = (norm * 255.0).astype(np.uint8)
        return np.stack([gray, gray, gray], axis=-1)
    else:
        return colorize_unipolar(channel_data)


def render_channel_panel(rgb_data: np.ndarray, title: str, subtitle: str = "") -> Image.Image:
    """Renders a single labeled channel square."""
    img = Image.fromarray(rgb_data).resize((180, 180), Image.Resampling.NEAREST)
    # Add title header
    panel = Image.new("RGB", (180, 220), (20, 24, 33))
    panel.paste(img, (0, 0))

    draw = ImageDraw.Draw(panel)
    draw.text((6, 184), title, fill=(240, 240, 245))
    if subtitle:
        draw.text((6, 202), subtitle, fill=(160, 165, 180))

    # Border
    draw.rectangle([0, 0, 179, 219], outline=(60, 65, 80))
    return panel


def create_12_channel_grid(
    original_rgb: np.ndarray,
    channels_tensor: torch.Tensor,
    header_text: str = ""
) -> Image.Image:
    """
    Composes a 4x4 clean dashboard:
    Row 0: Original RGB (expanded), Sustained Lum, Chroma RG, Chroma BY
    Row 1: Subunit ON, Subunit OFF, Transient, OMS
    Row 2: DS Right, DS Left, DS Up, DS Down
    Row 3: Looming, + summary panels
    """
    ch_data = channels_tensor.squeeze(0).cpu().numpy()  # (12, 128, 128)
    channel_names = DRFBiomimeticFrontEnd.CHANNEL_NAMES
    subtitles = [
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
    ]

    panels = []
    for i in range(12):
        c_rgb = colorize_channel(ch_data[i], i)
        panel = render_channel_panel(c_rgb, f"{i}: {channel_names[i]}", subtitles[i])
        panels.append(panel)

    # Original RGB panel (resized to 180x180 with label)
    orig_pil = Image.fromarray(original_rgb).resize((180, 180), Image.Resampling.BILINEAR)
    orig_panel = Image.new("RGB", (180, 220), (20, 24, 33))
    orig_panel.paste(orig_pil, (0, 0))
    draw = ImageDraw.Draw(orig_panel)
    draw.text((6, 184), "RAW INPUT", fill=(100, 220, 255))
    draw.text((6, 202), f"{original_rgb.shape[1]}x{original_rgb.shape[0]} RGB", fill=(160, 165, 180))
    draw.rectangle([0, 0, 179, 219], outline=(80, 120, 180))

    # Grid layout: 4 columns x 4 rows
    # Total width = 4 * 180 + 5 * 10 = 720 + 50 = 770
    # Total height = header(50) + 4 * 220 + 5 * 10 = 50 + 880 + 50 = 980
    W_canvas = 4 * 180 + 5 * 10
    H_canvas = 55 + 4 * 220 + 5 * 10
    canvas = Image.new("RGB", (W_canvas, H_canvas), (12, 14, 20))
    draw_canvas = ImageDraw.Draw(canvas)

    if header_text:
        draw_canvas.text((15, 15), header_text, fill=(255, 255, 255))
        draw_canvas.text((15, 33), "Dynamic Receptive Field Biomimetic Front-End (12 Retinal Ganglion Channels @ 128x128)", fill=(160, 170, 190))

    # Layout:
    # Row 0: [ORIGINAL RGB], [0: SUSTAINED_LUM], [1: CHROMA_RG], [2: CHROMA_BY]
    # Row 1: [3: SUBUNIT_ON], [4: SUBUNIT_OFF], [5: TRANSIENT], [10: DIFF_MOTION]
    # Row 2: [6: DS_RIGHT], [7: DS_LEFT], [8: DS_UP], [9: DS_DOWN]
    # Row 3: [11: LOOMING], [INFO STATS 1], [INFO STATS 2], [INFO STATS 3]

    row0 = [orig_panel, panels[0], panels[1], panels[2]]
    row1 = [panels[3], panels[4], panels[5], panels[10]]
    row2 = [panels[6], panels[7], panels[8], panels[9]]
    row3_items = [panels[11]]

    rows = [row0, row1, row2]
    y_start = 55

    for r_idx, row in enumerate(rows):
        y = y_start + r_idx * (220 + 10)
        for c_idx, item in enumerate(row):
            x = 10 + c_idx * (180 + 10)
            canvas.paste(item, (x, y))

    # Row 3
    y3 = y_start + 3 * (220 + 10)
    canvas.paste(panels[11], (10, y3))

    # Create summary info cards in remaining 3 slots
    info_panel = Image.new("RGB", (180 * 3 + 20, 220), (20, 24, 33))
    draw_info = ImageDraw.Draw(info_panel)
    draw_info.rectangle([0, 0, 180 * 3 + 19, 219], outline=(60, 65, 80))
    draw_info.text((15, 15), "BIOMIMETIC RETINAL STATE ANALYSIS", fill=(100, 200, 255))
    
    # Compute active channel metrics
    active_m = [f"{channel_names[i]}: {float(ch_data[i].mean()):.3f} (max: {float(ch_data[i].max()):.3f})" for i in range(12)]
    col1 = "\n".join(active_m[:4])
    col2 = "\n".join(active_m[4:8])
    col3 = "\n".join(active_m[8:])

    draw_info.text((15, 45), col1, fill=(210, 215, 230))
    draw_info.text((190, 45), col2, fill=(210, 215, 230))
    draw_info.text((370, 45), col3, fill=(210, 215, 230))

    canvas.paste(info_panel, (10 + 180 + 10, y3))
    return canvas


def process_image_demo():
    """Processes static picture and micro-saccadic shift test."""
    print("[1/4] Processing sample picture from the internet...")
    img_path = Path("sample_online.jpg")
    img_pil = Image.open(img_path).convert("RGB")
    rgb_np = np.array(img_pil)

    # Tensor in [0, 1]
    x_tensor = torch.from_numpy(rgb_np).permute(2, 0, 1).unsqueeze(0).float() / 255.0

    frontend = DRFBiomimeticFrontEnd()
    frontend.eval()

    # Pass 1: Static initial frame (t=0)
    frontend.reset()
    with torch.no_grad():
        out_static = frontend(x_tensor)

    grid_static = create_12_channel_grid(
        rgb_np, out_static, "DRF-FE Static Regime (t=0) - Form & Color Streams Active"
    )
    static_save_path = OUTPUT_DIR / "picture_channels_static.png"
    grid_static.save(static_save_path)
    shutil.copy(static_save_path, ARTIFACT_DIR / "picture_channels_static.png")
    print(f"  -> Saved {static_save_path}")

    # Pass 2: Simulated Saccade (Frame shifted by 6 pixels to the right)
    rgb_shifted = np.roll(rgb_np, shift=6, axis=1)
    x_shifted = torch.from_numpy(rgb_shifted).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    with torch.no_grad():
        out_saccade = frontend(x_shifted)

    grid_saccade = create_12_channel_grid(
        rgb_shifted, out_saccade, "DRF-FE Saccadic Shift (t=1) - Motion & Transient Streams Firing"
    )
    saccade_save_path = OUTPUT_DIR / "picture_channels_saccade.png"
    grid_saccade.save(saccade_save_path)
    shutil.copy(saccade_save_path, ARTIFACT_DIR / "picture_channels_saccade.png")
    print(f"  -> Saved {saccade_save_path}")


def process_video_demo(video_path: str, output_name: str, max_frames: int = 40):
    """Processes a video file frame-by-frame and renders an animated GIF grid."""
    print(f"Processing video: {video_path} (up to {max_frames} frames)...")
    if not os.path.exists(video_path):
        print(f"  Video file {video_path} not found, skipping.")
        return

    frontend = DRFBiomimeticFrontEnd()
    frontend.eval()
    frontend.reset()

    frames_grid = []
    frame_idx = 0

    try:
        reader = iio.imiter(video_path)
        for frame in reader:
            if frame_idx >= max_frames:
                break

            # Convert frame to (1, 3, H, W) in [0, 1]
            rgb_np = frame[..., :3]  # Ensure 3 channels
            x_tensor = torch.from_numpy(rgb_np).permute(2, 0, 1).unsqueeze(0).float() / 255.0

            with torch.no_grad():
                out_t = frontend(x_tensor)

            grid_img = create_12_channel_grid(
                rgb_np,
                out_t,
                f"DRF-FE Temporal Video Stream | Frame #{frame_idx:03d}"
            )
            # Resize canvas slightly for smaller GIF size
            small_grid = grid_img.resize((grid_img.width // 2, grid_img.height // 2), Image.Resampling.BILINEAR)
            frames_grid.append(small_grid)

            frame_idx += 1
            if frame_idx % 10 == 0:
                print(f"  Processed {frame_idx} frames...")

    except Exception as e:
        print(f"  Error reading video {video_path}: {e}")

    if frames_grid:
        gif_path = OUTPUT_DIR / f"{output_name}.gif"
        frames_grid[0].save(
            gif_path,
            save_all=True,
            append_images=frames_grid[1:],
            duration=100,  # 10 fps
            loop=0
        )
        shutil.copy(gif_path, ARTIFACT_DIR / f"{output_name}.gif")
        print(f"  -> Saved animated GIF to {gif_path} ({len(frames_grid)} frames, {os.path.getsize(gif_path):,} bytes)")

        # Also save the peak motion frame as a static PNG
        peak_idx = min(15, len(frames_grid) - 1)
        png_path = OUTPUT_DIR / f"{output_name}_snapshot.png"
        frames_grid[peak_idx].save(png_path)
        shutil.copy(png_path, ARTIFACT_DIR / f"{output_name}_snapshot.png")
        print(f"  -> Saved snapshot PNG to {png_path}")


def main():
    print("=== Running DRF-FE Real-World Image & Video Diagnostic Suite ===")
    process_image_demo()
    process_video_demo("sample_online_video.mp4", "sample_online_video_channels", max_frames=30)
    process_video_demo("car-detection.mp4", "traffic_car_channels", max_frames=35)
    print("=== Diagnostic Suite Completed Successfully! ===")


if __name__ == "__main__":
    main()
