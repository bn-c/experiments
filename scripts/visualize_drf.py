"""
Renders the 12 DRF channels for an image and/or a video.

Examples:
    python scripts/visualize_drf.py --image sample.jpg
    python scripts/visualize_drf.py --video clip.mp4 --max-frames 30
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from drf import DRFBiomimeticFrontEnd
from drf.visualization import create_12_channel_grid, save_channel_images, save_channel_videos


def to_tensor(rgb_np: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(rgb_np).permute(2, 0, 1).unsqueeze(0).float() / 255.0


def process_image(image_path: Path, output_dir: Path) -> None:
    """Renders the static frame (t=0) and a simulated 6 px saccade (t=1), plus 12 per-channel PNGs."""
    rgb_np = np.array(Image.open(image_path).convert("RGB"))

    frontend = DRFBiomimeticFrontEnd().eval()
    frontend.reset()

    out_static = frontend(to_tensor(rgb_np))
    static_path = output_dir / f"{image_path.stem}_static.png"
    create_12_channel_grid(
        rgb_np, out_static, "DRF-FE Static Regime (t=0) - Form & Color Streams Active"
    ).save(static_path)
    print(f"  -> {static_path}")
    for path in save_channel_images(out_static, output_dir, f"{image_path.stem}_static"):
        print(f"  -> {path}")

    rgb_shifted = np.roll(rgb_np, shift=6, axis=1)
    out_saccade = frontend(to_tensor(rgb_shifted))
    saccade_path = output_dir / f"{image_path.stem}_saccade.png"
    create_12_channel_grid(
        rgb_shifted, out_saccade, "DRF-FE Saccadic Shift (t=1) - Motion & Transient Streams Firing"
    ).save(saccade_path)
    print(f"  -> {saccade_path}")
    for path in save_channel_images(out_saccade, output_dir, f"{image_path.stem}_saccade"):
        print(f"  -> {path}")


def process_video(video_path: Path, output_dir: Path, max_frames: int) -> None:
    """Streams a video frame-by-frame and renders a combined GIF plus 12 per-channel GIFs."""
    import imageio.v3 as iio

    frontend = DRFBiomimeticFrontEnd().eval()
    frontend.reset()

    frames = []
    channel_frames = []
    for frame_idx, frame in enumerate(iio.imiter(video_path)):
        if frame_idx >= max_frames:
            break
        rgb_np = frame[..., :3]
        out_t = frontend(to_tensor(rgb_np))
        grid = create_12_channel_grid(
            rgb_np, out_t, f"DRF-FE Temporal Video Stream | Frame #{frame_idx:03d}"
        )
        frames.append(grid.resize((grid.width // 2, grid.height // 2), Image.Resampling.BILINEAR))
        channel_frames.append(out_t.squeeze(0).cpu().numpy())

    if not frames:
        print(f"  no frames decoded from {video_path}")
        return

    gif_path = output_dir / f"{video_path.stem}_channels.gif"
    frames[0].save(gif_path, save_all=True, append_images=frames[1:], duration=100, loop=0)
    print(f"  -> {gif_path} ({len(frames)} frames)")

    snapshot_path = output_dir / f"{video_path.stem}_channels_snapshot.png"
    frames[min(15, len(frames) - 1)].save(snapshot_path)
    print(f"  -> {snapshot_path}")

    for path in save_channel_videos(channel_frames, output_dir, video_path.stem):
        print(f"  -> {path}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Visualize DRF-FE channel decompositions.")
    parser.add_argument("--image", type=Path, help="Image file to decompose.")
    parser.add_argument("--video", type=Path, help="Video file to decompose frame-by-frame.")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--max-frames", type=int, default=30)
    args = parser.parse_args()

    if args.image is None and args.video is None:
        parser.error("provide at least one of --image / --video")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.image is not None:
        print(f"Processing image {args.image}...")
        process_image(args.image, args.output_dir)
    if args.video is not None:
        print(f"Processing video {args.video} (up to {args.max_frames} frames)...")
        process_video(args.video, args.output_dir, args.max_frames)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
