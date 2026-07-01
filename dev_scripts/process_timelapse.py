"""
End-to-end time-lapse processor.

Pipeline:
    1. Take a folder of time-lapse image frames (e.g. data/timelapse/synthetic_run/)
    2. Sort them by filename (Incucyte's timestamp encoding sorts naturally)
    3. Segment each frame with the existing CellSegmenter
    4. Feed the per-cell observations to the Tracker frame-by-frame
    5. Emit tracks.json + an animated GIF visualization

The output sits alongside the input frames in a `_results/` subfolder, so
running this script multiple times against the same input is idempotent.

Usage:
    py process_timelapse.py
    py process_timelapse.py --input data/timelapse/synthetic_run --max-distance 40
"""

from __future__ import annotations

import argparse
import io
import json
import random
import sys
from pathlib import Path

# This script lives in dev_scripts/ but imports from backend/. Make backend
# importable by adding it to Python's module search path at runtime.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from inference import CellSegmenter
from morphology import per_cell_stats
from tracker import Tracker


IMAGE_EXTENSIONS = {".tif", ".tiff", ".png", ".jpg", ".jpeg"}


def _color_for_track(track_id: int) -> tuple[int, int, int]:
    """Deterministic color per track ID. Same ID → same color across frames."""
    rng = random.Random(track_id)
    return (rng.randint(60, 255), rng.randint(60, 255), rng.randint(60, 255))


def _draw_overlay(
    base_image: Image.Image,
    mask: np.ndarray,
    track_ids: list[int],
    track_history: dict[int, list[tuple[float, float]]],
) -> Image.Image:
    """
    Paint each cell on top of the original image using its TRACK color (not
    its per-frame cell ID), and draw a trailing line for each track showing
    where it has been in recent frames.
    """
    rgba = base_image.convert("RGBA")
    overlay = Image.new("RGBA", rgba.size, (0, 0, 0, 0))

    # Fill each cell with its track color.
    arr = np.array(overlay)
    n_cells = int(mask.max())
    for cell_idx in range(1, n_cells + 1):
        if cell_idx - 1 >= len(track_ids):
            break
        tid = track_ids[cell_idx - 1]
        color = _color_for_track(tid)
        arr[mask == cell_idx, 0:3] = color
        arr[mask == cell_idx, 3] = 130  # semi-transparent
    overlay = Image.fromarray(arr, mode="RGBA")

    composed = Image.alpha_composite(rgba, overlay)

    # Draw trajectory lines + track ID labels on top.
    draw = ImageDraw.Draw(composed)
    try:
        font = ImageFont.truetype("arial.ttf", 11)
    except Exception:
        font = ImageFont.load_default()

    for tid, history in track_history.items():
        if len(history) < 2:
            continue
        color = _color_for_track(tid)
        # Trail: last ~6 frames of this track.
        recent = history[-6:]
        for (x0, y0), (x1, y1) in zip(recent[:-1], recent[1:]):
            draw.line([(x0, y0), (x1, y1)], fill=color + (220,), width=2)

    # Track ID labels at the most recent position.
    for tid, history in track_history.items():
        if not history:
            continue
        x, y = history[-1]
        draw.text((x + 3, y + 3), str(tid), fill=(255, 255, 255, 255), font=font)

    return composed.convert("RGB")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    project_root = Path(__file__).resolve().parent.parent
    parser.add_argument(
        "--input",
        type=Path,
        default=project_root / "data" / "timelapse" / "synthetic_run",
        help="Folder of time-lapse frames, sorted by filename.",
    )
    parser.add_argument("--max-distance", type=float, default=30.0,
                        help="Max pixels a cell can move between frames and still be linked.")
    parser.add_argument("--area-weight", type=float, default=0.01,
                        help="How heavily to penalize area mismatches in the cost matrix.")
    parser.add_argument("--gif-duration-ms", type=int, default=400,
                        help="Per-frame duration in the output GIF.")
    args = parser.parse_args()

    if not args.input.exists():
        print(f"Input folder not found: {args.input}", file=sys.stderr)
        print("Generate one with: py synthetic_timelapse.py", file=sys.stderr)
        return 1

    frame_paths = sorted(
        p for p in args.input.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not frame_paths:
        print(f"No image frames in {args.input}", file=sys.stderr)
        return 1

    results_dir = args.input / "_results"
    results_dir.mkdir(exist_ok=True)
    overlay_dir = results_dir / "overlays"
    overlay_dir.mkdir(exist_ok=True)

    print(f"Loading segmentation model...", flush=True)
    segmenter = CellSegmenter()
    print(f"  device: {segmenter.device}", flush=True)

    tracker = Tracker(max_distance=args.max_distance, area_weight=args.area_weight)
    track_history: dict[int, list[tuple[float, float]]] = {}
    overlay_frames: list[Image.Image] = []

    print(f"Processing {len(frame_paths)} frames from {args.input}", flush=True)
    for f, frame_path in enumerate(frame_paths):
        image_bytes = frame_path.read_bytes()
        result = segmenter.segment(image_bytes)
        cells = per_cell_stats(result.mask)

        track_ids = tracker.update(f, cells)

        # Update the history dict for the visualizer.
        for cell, tid in zip(cells, track_ids):
            track_history.setdefault(tid, []).append(
                (float(cell["centroid_x"]), float(cell["centroid_y"]))
            )

        # Render overlay frame.
        base = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        overlay = _draw_overlay(base, result.mask, track_ids, track_history)
        overlay_path = overlay_dir / f"frame_{f:03d}.png"
        overlay.save(overlay_path)
        overlay_frames.append(overlay)

        print(
            f"  [{f+1:02d}/{len(frame_paths):02d}] {frame_path.name}: "
            f"{result.cell_count} cells, {len(tracker.tracks)} tracks so far",
            flush=True,
        )

    # Dump the tracks JSON.
    tracks_path = results_dir / "tracks.json"
    tracks_path.write_text(json.dumps(tracker.dump(), indent=2))
    print(f"\nWrote {tracks_path}", flush=True)

    # Render the animated GIF.
    gif_path = results_dir / "timelapse.gif"
    if overlay_frames:
        overlay_frames[0].save(
            gif_path,
            save_all=True,
            append_images=overlay_frames[1:],
            duration=args.gif_duration_ms,
            loop=0,
            optimize=False,
        )
        print(f"Wrote {gif_path}", flush=True)

    # Summary.
    n_full_length = sum(1 for t in tracker.tracks.values() if len(t.points) == len(frame_paths))
    n_partial = len(tracker.tracks) - n_full_length
    print(f"\nSummary:")
    print(f"  Frames processed:        {len(frame_paths)}")
    print(f"  Total tracks created:    {len(tracker.tracks)}")
    print(f"  Full-length tracks:      {n_full_length} (present in every frame)")
    print(f"  Partial tracks:          {n_partial} (appeared/disappeared partway through)")
    print(f"\nOpen {gif_path} to see the result.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
