"""
Shared overlay rendering for time-lapse pipelines.

Both `process_timelapse.py` (CLI, file output) and `main.py` (HTTP, JSON
output) need to draw the same overlay images — cells colored by track ID
plus trailing motion lines per track. Putting that here keeps them in sync.
"""

from __future__ import annotations

import random

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def color_for_track(track_id: int) -> tuple[int, int, int]:
    """
    Deterministic RGB color for a given track ID. Same ID -> same color
    every frame, so a cell visually keeps its identity as the user scrubs.
    """
    rng = random.Random(track_id)
    return (rng.randint(60, 255), rng.randint(60, 255), rng.randint(60, 255))


def render_overlay(
    base_image: Image.Image,
    mask: np.ndarray,
    track_ids: list[int],
    track_history: dict[int, list[tuple[float, float]]],
    trail_length: int = 6,
) -> Image.Image:
    """
    Paint each cell using its TRACK color (not its per-frame cell ID), draw
    a trailing motion line for each track using the last `trail_length`
    positions, and label each visible cell with its track ID.

    base_image       Original frame to draw on
    mask             Instance mask (0=bg, 1..N=cell IDs as Cellpose returns them)
    track_ids        track_ids[i] is the track ID assigned to cell with mask ID i+1
    track_history    track_id -> list of (x, y) centroids over time, ordered by frame.
                     Pass only the history UP TO AND INCLUDING the current frame
                     if you want the trail to show "where it has been so far."
    """
    rgba = base_image.convert("RGBA")
    overlay = Image.new("RGBA", rgba.size, (0, 0, 0, 0))

    # Tint each cell area with the track color, semi-transparent.
    arr = np.array(overlay)
    n_cells = int(mask.max())
    for cell_idx in range(1, n_cells + 1):
        if cell_idx - 1 >= len(track_ids):
            break
        tid = track_ids[cell_idx - 1]
        r, g, b = color_for_track(tid)
        arr[mask == cell_idx, 0:3] = (r, g, b)
        arr[mask == cell_idx, 3] = 130
    overlay = Image.fromarray(arr, mode="RGBA")

    composed = Image.alpha_composite(rgba, overlay)

    # Trail lines + ID labels.
    draw = ImageDraw.Draw(composed)
    try:
        font = ImageFont.truetype("arial.ttf", 11)
    except Exception:
        font = ImageFont.load_default()

    for tid, history in track_history.items():
        if len(history) < 2:
            continue
        color = color_for_track(tid)
        recent = history[-trail_length:]
        for (x0, y0), (x1, y1) in zip(recent[:-1], recent[1:]):
            draw.line([(x0, y0), (x1, y1)], fill=color + (220,), width=2)

    for tid, history in track_history.items():
        if not history:
            continue
        x, y = history[-1]
        draw.text((x + 3, y + 3), str(tid), fill=(255, 255, 255, 255), font=font)

    return composed.convert("RGB")
