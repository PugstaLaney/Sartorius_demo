"""
Render segmented cells over the original frame using track-derived colors
and motion trails. Produces the annotated overlays shown in the WPF panel.

Role in the architecture
------------------------
Layer:       Leaf rendering module (no other backend module imports from this one)
Called by:   main.py (inside /track_timelapse), dev_scripts/process_timelapse.py
Depends on:  numpy, PIL (external only)
Runs when:   Once per frame during time-lapse processing, after segmentation
             and tracking have completed for that frame

Shared code so main.py's HTTP endpoint and dev_scripts/process_timelapse.py's
CLI orchestrator draw identical overlays. This module knows nothing about
segmentation, tracking algorithms, or JSON. It takes a base image, a mask,
track IDs, and history, and returns an annotated PIL image.

Both `dev_scripts/process_timelapse.py` (CLI, file output) and `main.py`
(HTTP, JSON output) need to draw the same overlay images. Cells are colored
by track ID plus trailing motion lines per track. Putting the shared code
here keeps them consistent.

See learning_materials/05_layered_architecture.md. This module is a good
example of "extract shared code into a leaf module when two orchestrators
would otherwise duplicate it."
"""

from __future__ import annotations


# =============================================================================
# IMPORTS
# =============================================================================
# This is a leaf module. No imports from other backend files. Both callers
# reuse it via `from track_visualization import ...`.

import random

import numpy as np
from PIL import Image, ImageDraw, ImageFont


# =============================================================================
# COLOR PICKING
# =============================================================================
# Every track needs a distinct visual color, and — critically — the SAME
# track needs to keep the SAME color across frames so a viewer can track it
# visually. We achieve this by seeding a random generator with the track ID:
# same input, same output, every time.

def color_for_track(track_id: int) -> tuple[int, int, int]:
    """
    Deterministic RGB color for a given track ID. Same ID -> same color
    every frame, so a cell visually keeps its identity as the user scrubs.
    """
    # Seed with track_id so the sequence is deterministic. Range 60-255
    # keeps colors visible against a dark background (never near-black).
    rng = random.Random(track_id)
    return (rng.randint(60, 255), rng.randint(60, 255), rng.randint(60, 255))


# =============================================================================
# THE OVERLAY RENDERER
# =============================================================================
# Given a base image, a mask, per-cell track IDs, and history, produce the
# annotated frame the client displays. Three visual elements:
#   1. Cell fills — each cell tinted with its track's color, semi-transparent
#   2. Motion trails — a short line following each track's recent centroids
#   3. Track ID labels — small number next to each visible cell

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
    # -------------------------------------------------------------------------
    # SUBSECTION: build a transparent overlay to tint each cell
    # -------------------------------------------------------------------------
    # We work with RGBA (Red/Green/Blue/Alpha) — alpha is the transparency
    # channel. Base image gets converted to RGBA, then we build a transparent
    # canvas the same size and fill in each cell's pixels with its color.
    rgba = base_image.convert("RGBA")
    overlay = Image.new("RGBA", rgba.size, (0, 0, 0, 0))     # fully transparent

    # Convert the overlay to a numpy array so we can use fast boolean-mask
    # assignment: `arr[mask == cell_idx, ...] = color`.
    arr = np.array(overlay)
    n_cells = int(mask.max())
    for cell_idx in range(1, n_cells + 1):
        # Guard against out-of-range indices (defensive; should not happen).
        if cell_idx - 1 >= len(track_ids):
            break

        # Look up this cell's track color.
        tid = track_ids[cell_idx - 1]
        r, g, b = color_for_track(tid)

        # Paint every pixel where mask == cell_idx. Alpha=130 means ~50% visible.
        arr[mask == cell_idx, 0:3] = (r, g, b)
        arr[mask == cell_idx, 3] = 130

    # Convert back to a PIL Image and composite over the base.
    overlay = Image.fromarray(arr, mode="RGBA")
    composed = Image.alpha_composite(rgba, overlay)

    # -------------------------------------------------------------------------
    # SUBSECTION: draw motion trails on top of the composed image
    # -------------------------------------------------------------------------
    # ImageDraw lets us draw shapes and text over an image. We try to load
    # a real font; if arial isn't available (some Linux systems), fall back
    # to PIL's built-in bitmap font.
    draw = ImageDraw.Draw(composed)
    try:
        font = ImageFont.truetype("arial.ttf", 11)
    except Exception:
        font = ImageFont.load_default()

    # For each track with at least 2 points, draw a polyline connecting the
    # last `trail_length` positions. The trail visually shows where the cell
    # has been drifting.
    for tid, history in track_history.items():
        if len(history) < 2:
            continue
        color = color_for_track(tid)
        recent = history[-trail_length:]
        # zip(recent[:-1], recent[1:]) yields consecutive pairs of points.
        for (x0, y0), (x1, y1) in zip(recent[:-1], recent[1:]):
            draw.line([(x0, y0), (x1, y1)], fill=color + (220,), width=2)

    # -------------------------------------------------------------------------
    # SUBSECTION: draw track ID labels at each cell's current position
    # -------------------------------------------------------------------------
    for tid, history in track_history.items():
        if not history:
            continue
        x, y = history[-1]                        # most recent centroid
        # White text with a small offset from the centroid so it doesn't
        # sit exactly on top of the cell.
        draw.text((x + 3, y + 3), str(tid), fill=(255, 255, 255, 255), font=font)

    # Convert back to RGB for consistent output (no need for alpha in the
    # final image; the overlay has already been composited in).
    return composed.convert("RGB")
