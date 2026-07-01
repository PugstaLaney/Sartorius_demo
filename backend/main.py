"""
FastAPI inference service.

Responsibilities:
  - Accept image uploads
  - Hand them to the model wrapper
  - Return a JSON response with the mask (PNG-encoded) and metrics
  - Expose health + metrics endpoints for observability

This file deliberately knows nothing about PyTorch or Cellpose internals.
That separation is what makes the model swappable.
"""

from __future__ import annotations

import base64
import io
from collections import deque
from contextlib import asynccontextmanager
from statistics import mean

import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image

from inference import CellSegmenter
from morphology import per_cell_stats, summary_stats
from gif_io import extract_frames
from tracker import Tracker
from track_visualization import render_overlay


# Module-level state that lives for the lifetime of the process.
# Keeping the model here (vs reloading per request) is THE deployment win.
SEGMENTER: CellSegmenter | None = None
LATENCY_WINDOW: deque[float] = deque(maxlen=100)  # Rolling window for /metrics


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI lifespan hook. Runs once at startup and once at shutdown.
    Loading the model here means the FIRST request is fast — the model
    is already warm in GPU memory before any user hits the API.
    """
    global SEGMENTER
    print("Loading Cellpose model...")
    SEGMENTER = CellSegmenter()
    print(f"Model ready on device: {SEGMENTER.device}")
    yield
    print("Shutting down.")


app = FastAPI(
    title="Sartorius Cell Segmentation Service",
    description="PyTorch + Cellpose deployed as a containerizable inference service.",
    version="0.1.0",
    lifespan=lifespan,
)

# CORS so the frontend (served from a different port) can call this.
# In production this would be locked down to the known frontend origin.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def _mask_to_png_base64(mask: np.ndarray) -> str:
    """
    Convert an integer instance mask into a colored PNG and return as base64.
    We map each cell ID to a distinct color so the frontend can overlay it
    on the original image without re-running any computation.
    """
    H, W = mask.shape
    rgba = np.zeros((H, W, 4), dtype=np.uint8)

    # Deterministic colors per cell ID. Hash-based so the same cell ID
    # always gets the same color (useful when comparing across frames).
    for cell_id in range(1, int(mask.max()) + 1):
        rng = np.random.default_rng(seed=cell_id)
        color = rng.integers(60, 255, size=3, dtype=np.uint8)
        rgba[mask == cell_id, 0:3] = color
        rgba[mask == cell_id, 3] = 140  # Semi-transparent overlay

    buf = io.BytesIO()
    Image.fromarray(rgba, mode="RGBA").save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


@app.get("/health")
async def health():
    """Liveness probe. Returns 200 once the model is loaded."""
    if SEGMENTER is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")
    return {"status": "ok", "device": SEGMENTER.device}


@app.get("/metrics")
async def metrics():
    """
    Minimal observability surface. In production this would be Prometheus
    format; for the demo, JSON is fine and demonstrates the concept.
    """
    if not LATENCY_WINDOW:
        return {"requests": 0, "avg_latency_ms": None}
    return {
        "requests": len(LATENCY_WINDOW),
        "avg_latency_ms": round(mean(LATENCY_WINDOW), 2),
        "min_latency_ms": round(min(LATENCY_WINDOW), 2),
        "max_latency_ms": round(max(LATENCY_WINDOW), 2),
        "device": SEGMENTER.device if SEGMENTER else None,
    }


@app.post("/segment")
async def segment(file: UploadFile = File(...)):
    """
    Main inference endpoint. Accepts an image upload, returns JSON containing:
      - cell_count: integer
      - inference_ms: float (the model itself, not network overhead)
      - mask_png_base64: colored mask overlay, ready to render
      - device: cuda or cpu
    """
    if SEGMENTER is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    image_bytes = await file.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Empty upload")

    try:
        result = SEGMENTER.segment(image_bytes)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Inference failed: {exc}")

    LATENCY_WINDOW.append(result.inference_ms)

    # Compute per-cell morphology so both ingest paths (HTTP + folder-watcher)
    # surface the same data shape. Downstream clients shouldn't have to care
    # which path produced the result.
    cells = per_cell_stats(result.mask)

    return JSONResponse({
        "cell_count": result.cell_count,
        "inference_ms": round(result.inference_ms, 2),
        "device": result.device,
        "mask_png_base64": _mask_to_png_base64(result.mask),
        "summary": summary_stats(cells),
        "per_cell": cells,
    })


def _image_to_png_base64(img: Image.Image) -> str:
    """Encode a PIL image as base64 PNG (used for time-lapse frame transport)."""
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


@app.post("/track_timelapse")
async def track_timelapse(file: UploadFile = File(...)):
    """
    Process an animated GIF as a time-lapse: segment each frame, link cells
    across frames with the Hungarian-assignment tracker, and return every
    frame's data in one response so the client can scrub a slider without
    making more requests.

    Response shape:
        {
          "n_frames": int,
          "frames": [
            {
              "frame_index": int,
              "inference_ms": float,
              "cell_count": int,
              "original_png_base64": str,   # what the client uploaded, per frame
              "overlay_png_base64": str,    # cells colored by track ID + trails
              "per_cell": [
                  {"cell_id": int, "track_id": int, "area_px": int, ...}, ...
              ],
              "summary": {...},
            }, ...
          ],
          "tracks_summary": {
            "n_tracks": int,
            "full_length": int,
            "partial": int,
          },
          "device": "cuda" | "cpu",
        }
    """
    if SEGMENTER is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    blob = await file.read()
    if not blob:
        raise HTTPException(status_code=400, detail="Empty upload")

    try:
        frame_bytes_list = extract_frames(blob)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Could not parse GIF: {exc}")

    if not frame_bytes_list:
        raise HTTPException(status_code=400, detail="GIF contained no frames")

    tracker = Tracker(max_distance=30.0, area_weight=0.01)
    track_history: dict[int, list[tuple[float, float]]] = {}
    out_frames: list[dict] = []

    for f_idx, png_bytes in enumerate(frame_bytes_list):
        try:
            seg = SEGMENTER.segment(png_bytes)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Frame {f_idx} failed: {exc}")

        cells = per_cell_stats(seg.mask)
        track_ids = tracker.update(f_idx, cells)

        # Attach track_id to each cell row so the client can show it directly.
        for cell, tid in zip(cells, track_ids):
            cell["track_id"] = int(tid)
            track_history.setdefault(tid, []).append(
                (float(cell["centroid_x"]), float(cell["centroid_y"]))
            )

        base = Image.open(io.BytesIO(png_bytes)).convert("RGB")
        overlay = render_overlay(base, seg.mask, track_ids, track_history)

        out_frames.append({
            "frame_index": f_idx,
            "inference_ms": round(seg.inference_ms, 2),
            "cell_count": seg.cell_count,
            "original_png_base64": _image_to_png_base64(base),
            "overlay_png_base64": _image_to_png_base64(overlay),
            "per_cell": cells,
            "summary": summary_stats(cells),
        })

    n_full = sum(1 for t in tracker.tracks.values() if len(t.points) == len(frame_bytes_list))
    n_partial = len(tracker.tracks) - n_full

    return JSONResponse({
        "n_frames": len(out_frames),
        "frames": out_frames,
        "tracks_summary": {
            "n_tracks": len(tracker.tracks),
            "full_length": n_full,
            "partial": n_partial,
        },
        "device": SEGMENTER.device,
    })
