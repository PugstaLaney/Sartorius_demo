"""
Generate a synthetic time-lapse from one source image.

Why synthetic first: real Cell Tracking Challenge datasets require license
clicks. We can iterate on the tracker logic faster with synthesized motion
where we control the ground truth. Real data goes in as a second step.

How it works:
    1. Run Cellpose on the source image to find every cell
    2. For each cell, cut out its bounding box (the actual pixels) + mask
    3. Build a clean background canvas (median color of the original)
    4. For each frame f in 0..N-1:
        - For each cell, compute a new (x, y) position:
            new_x = original_x + per_cell_velocity_x * f + small_jitter
            new_y = original_y + per_cell_velocity_y * f + small_jitter
        - Paste the cell's pixels at the new position, respecting its mask
    5. Save each frame with an Incucyte-style filename so the existing
       filename parser will exercise correctly

The output frames sit in data/timelapse/<run_name>/ and are timestamped
on 5-minute intervals (00d00h00m, 00d00h05m, ...) to mimic Incucyte cadence.

Usage:
    py synthetic_timelapse.py
    py synthetic_timelapse.py --frames 20 --drift 12 --seed 7
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

# This script lives in dev_scripts/ but imports from backend/. Make backend
# importable by adding it to Python's module search path at runtime.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))

import numpy as np
from PIL import Image

from inference import CellSegmenter


def _format_timestamp(frame_index: int, minutes_per_frame: int = 5) -> str:
    """Render a frame index as a `<NN>d<HH>h<MM>m` Incucyte-style timestamp."""
    total_minutes = frame_index * minutes_per_frame
    days = total_minutes // (60 * 24)
    hours = (total_minutes // 60) % 24
    minutes = total_minutes % 60
    return f"{days:02d}d{hours:02d}h{minutes:02d}m"


def _extract_cell_patches(image: np.ndarray, mask: np.ndarray) -> list[dict]:
    """
    For each cell ID in the mask, return its pixel patch, its mask patch,
    and its original centroid. We need all three to paste it back at new
    positions on a synthesized frame.
    """
    patches = []
    n_cells = int(mask.max())
    for cell_id in range(1, n_cells + 1):
        ys, xs = np.where(mask == cell_id)
        if ys.size == 0:
            continue
        y0, y1 = ys.min(), ys.max() + 1
        x0, x1 = xs.min(), xs.max() + 1
        cell_mask = (mask[y0:y1, x0:x1] == cell_id)
        cell_pixels = image[y0:y1, x0:x1].copy()
        patches.append({
            "id": cell_id,
            "pixels": cell_pixels,        # H x W x 3 RGB
            "mask": cell_mask,            # H x W bool
            "y0": int(y0),                # original top-left in source
            "x0": int(x0),
            "centroid_y": float((ys.min() + ys.max()) / 2.0),
            "centroid_x": float((xs.min() + xs.max()) / 2.0),
        })
    return patches


def _render_frame(
    background: np.ndarray,
    patches: list[dict],
    offsets: list[tuple[int, int]],
) -> np.ndarray:
    """
    Build one frame by pasting each cell patch onto the background at its
    new (offset) position. Offsets are integer pixel shifts in (dy, dx).
    """
    out = background.copy()
    H, W = background.shape[:2]

    for patch, (dy, dx) in zip(patches, offsets):
        ph, pw = patch["mask"].shape
        new_y = patch["y0"] + dy
        new_x = patch["x0"] + dx

        # Clip to image bounds — cells that drift off the edge get cropped.
        src_y0 = max(0, -new_y)
        src_x0 = max(0, -new_x)
        dst_y0 = max(0, new_y)
        dst_x0 = max(0, new_x)
        dst_y1 = min(H, new_y + ph)
        dst_x1 = min(W, new_x + pw)
        src_y1 = src_y0 + (dst_y1 - dst_y0)
        src_x1 = src_x0 + (dst_x1 - dst_x0)

        if dst_y1 <= dst_y0 or dst_x1 <= dst_x0:
            continue  # entirely off-canvas

        m = patch["mask"][src_y0:src_y1, src_x0:src_x1]
        p = patch["pixels"][src_y0:src_y1, src_x0:src_x1]
        out[dst_y0:dst_y1, dst_x0:dst_x1][m] = p[m]

    return out


def generate(
    source_path: Path,
    out_dir: Path,
    n_frames: int,
    drift_px: float,
    jitter_px: float,
    cell_type: str,
    well: str,
    location: str,
    crop: str,
    seed: int,
) -> None:
    rng = np.random.default_rng(seed)

    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load + segment the source image.
    print(f"Loading source image: {source_path}", flush=True)
    img = Image.open(source_path).convert("RGB")
    arr = np.array(img)

    print("Running segmentation to find cells...", flush=True)
    segmenter = CellSegmenter()
    result = segmenter.segment(source_path.read_bytes())
    print(f"  found {result.cell_count} cells", flush=True)

    patches = _extract_cell_patches(arr, result.mask)
    if not patches:
        print("No cells extracted; can't synthesize. Pick a different source.", file=sys.stderr)
        return

    # Background = median color of pixels that are NOT inside any cell.
    bg_mask = (result.mask == 0)
    bg_color = np.median(arr[bg_mask], axis=0).astype(np.uint8)
    background = np.broadcast_to(bg_color, arr.shape).copy()

    # Per-cell velocity: each cell drifts in its own random direction.
    # |v| ~ Uniform(0, drift_px) pixels per frame, direction Uniform(0, 2*pi).
    n = len(patches)
    speeds = rng.uniform(0, drift_px, size=n)
    angles = rng.uniform(0, 2 * np.pi, size=n)
    vx = speeds * np.cos(angles)
    vy = speeds * np.sin(angles)

    print(f"Generating {n_frames} frames into {out_dir}", flush=True)
    for f in range(n_frames):
        jitter_y = rng.normal(0, jitter_px, size=n)
        jitter_x = rng.normal(0, jitter_px, size=n)
        offsets = [
            (int(round(vy[i] * f + jitter_y[i])), int(round(vx[i] * f + jitter_x[i])))
            for i in range(n)
        ]
        frame = _render_frame(background, patches, offsets)

        timestamp = _format_timestamp(f)
        filename = f"{cell_type}_Phase_{well}_{location}_{timestamp}_{crop}.png"
        out_path = out_dir / filename
        Image.fromarray(frame).save(out_path)
        print(f"  [{f+1:02d}/{n_frames:02d}] {filename}", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    project_root = Path(__file__).resolve().parent.parent
    parser.add_argument(
        "--source",
        type=Path,
        default=project_root / "data" / "sample_images" / "cells_demo.png",
        help="Source image to seed the synthesis from.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=project_root / "data" / "timelapse" / "synthetic_run",
        help="Where to write the generated frames.",
    )
    parser.add_argument("--frames", type=int, default=12)
    parser.add_argument("--drift", type=float, default=6.0,
                        help="Maximum per-cell drift speed in pixels per frame.")
    parser.add_argument("--jitter", type=float, default=1.5,
                        help="Per-frame random jitter (stddev in pixels).")
    parser.add_argument("--cell-type", default="Synth")
    parser.add_argument("--well", default="A1")
    parser.add_argument("--location", default="1")
    parser.add_argument("--crop", default="1")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    generate(
        source_path=args.source,
        out_dir=args.out,
        n_frames=args.frames,
        drift_px=args.drift,
        jitter_px=args.jitter,
        cell_type=args.cell_type,
        well=args.well,
        location=args.location,
        crop=args.crop,
        seed=args.seed,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
