"""
FastAPI application: the HTTP entry point for the segmentation service.

Role in the architecture
------------------------
Layer:       Orchestrator + HTTP entry point (top of the local import graph)
Imported by: uvicorn (the ASGI server that runs the service)
Depends on:  inference, morphology, tracker, gif_io, track_visualization
Runs when:   Once at startup (creates the SEGMENTER singleton via the
             lifespan hook), then per-request thereafter for each endpoint

This file composes the leaf modules into HTTP endpoints. It does NOT contain
any inference math, tracking algorithm, or image processing of its own. It
marshals bytes in from HTTP, calls the right leaf modules in the right order,
and marshals JSON back out.

Endpoints:
  GET  /health           liveness probe
  GET  /metrics          rolling-window latency stats
  POST /segment          single-image inference (used by the WPF single-image path)
  POST /track_timelapse  multi-frame inference + Hungarian tracking (GIF input)

This file deliberately knows nothing about PyTorch or Cellpose internals.
That separation is what makes the model swappable.

See learning_materials/03_http_and_json_boundaries.md for how requests
travel from the WPF client into this file, and 07_async_and_concurrency.md
for what `async def` is actually giving us.
"""

from __future__ import annotations


# =============================================================================
# IMPORTS
# =============================================================================
# Layout: stdlib, then third-party (FastAPI/PIL/numpy), then our own modules.
# The fact that our own imports (`from inference import ...`) are at the
# bottom is a small visual reminder that they sit at the top of the local
# import graph — main.py depends on them, they don't depend on us.

import base64                                # encode binary as ASCII for JSON transport
import io                                    # BytesIO buffers for in-memory image encoding
from collections import deque                # bounded-size list for the LATENCY_WINDOW
from contextlib import asynccontextmanager   # decorator for FastAPI's lifespan hook
from statistics import mean                  # arithmetic mean for /metrics

import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image

from inference import CellSegmenter                  # the model wrapper
from morphology import per_cell_stats, summary_stats # per-cell measurements
from gif_io import extract_frames                    # GIF -> list of frame PNGs
from tracker import Tracker                          # Hungarian cell tracker
from track_visualization import render_overlay       # cells colored by track ID


# =============================================================================
# MODULE-LEVEL STATE
# =============================================================================
# Variables declared at module scope live for the entire lifetime of the
# Python process. When uvicorn starts, this file gets imported once and these
# names get created. They stay in memory until the process shuts down.
#
# SEGMENTER is the star of the show. It's a single CellSegmenter instance
# created during startup (see the lifespan hook below). Every request handler
# reads from this same instance — the model is loaded ONCE, reused MANY times.

SEGMENTER: CellSegmenter | None = None

# LATENCY_WINDOW is a rolling window of the last N request latencies. `deque`
# with maxlen automatically drops oldest entries when we append past the cap.
# This gives us cheap observability for the /metrics endpoint without needing
# a real metrics database.
LATENCY_WINDOW: deque[float] = deque(maxlen=100)


# =============================================================================
# LIFESPAN HOOK (STARTUP + SHUTDOWN)
# =============================================================================
# `@asynccontextmanager` turns this generator into a context manager FastAPI
# knows how to use. The code BEFORE `yield` runs at startup; the code AFTER
# runs at shutdown.
#
# This is the single most important pattern in the file. Loading the model
# here (instead of per-request) is the difference between:
#   - Every request pays ~10s of model loading, OR
#   - Model loads once, every request pays ~650ms of inference.
#
# See learning_materials/02_classes_and_lifetime.md.

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI lifespan hook. Runs once at startup and once at shutdown.
    Loading the model here means the FIRST request is fast — the model
    is already warm in GPU memory before any user hits the API.
    """
    global SEGMENTER
    print("Loading Cellpose model...")
    SEGMENTER = CellSegmenter()             # <-- this line is the ~10-second startup cost
    print(f"Model ready on device: {SEGMENTER.device}")
    yield                                    # <-- FastAPI serves traffic here until shutdown
    print("Shutting down.")


# =============================================================================
# APP CONSTRUCTION AND MIDDLEWARE
# =============================================================================
# Create the FastAPI app instance and attach middleware that runs on every
# request/response.

app = FastAPI(
    title="Sartorius Cell Segmentation Service",
    description="PyTorch + Cellpose deployed as a containerizable inference service.",
    version="0.1.0",
    lifespan=lifespan,                       # register the startup/shutdown hook above
)

# CORS ("Cross-Origin Resource Sharing") tells the browser whether it's
# allowed to call this API from a different origin (protocol + host + port).
# The WPF client doesn't need CORS (it isn't a browser), but the HTML/JS
# frontend running on :8080 does — it lives at a different port from :8000
# where this API listens. Setting allow_origins=["*"] means "any origin can
# call us." In production this would be locked down to specific hosts.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# =============================================================================
# PRIVATE HELPERS: IMAGE ENCODING
# =============================================================================
# These are small utilities used by the endpoints below. Naming them with a
# leading underscore is a Python convention meaning "for internal use in this
# module, not intended to be imported by other modules."

def _mask_to_png_base64(mask: np.ndarray) -> str:
    """
    Convert an integer instance mask into a colored PNG and return as base64.
    We map each cell ID to a distinct color so the frontend can overlay it
    on the original image without re-running any computation.
    """
    H, W = mask.shape
    rgba = np.zeros((H, W, 4), dtype=np.uint8)   # RGBA canvas, starts all transparent

    # Deterministic colors per cell ID. Seeding the RNG with the cell ID
    # means cell 5 always gets the same color, in any image, forever. That's
    # useful when comparing outputs across frames or across runs.
    for cell_id in range(1, int(mask.max()) + 1):
        rng = np.random.default_rng(seed=cell_id)
        color = rng.integers(60, 255, size=3, dtype=np.uint8)
        rgba[mask == cell_id, 0:3] = color        # paint RGB channels
        rgba[mask == cell_id, 3] = 140            # paint alpha (0=fully transparent, 255=opaque)

    # Encode as PNG in memory, then base64-encode so the bytes can travel
    # inside a JSON string. Base64 blows up size by ~33%, but keeps everything
    # inside one JSON blob — the frontend gets image + metadata atomically.
    buf = io.BytesIO()
    Image.fromarray(rgba, mode="RGBA").save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _image_to_png_base64(img: Image.Image) -> str:
    """Encode a PIL image as base64 PNG (used for time-lapse frame transport)."""
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


# =============================================================================
# ENDPOINT: GET /health
# =============================================================================
# Liveness probe. Any monitoring tool (or Kubernetes readiness check) hits
# this to ask "are you alive and ready to serve?" We return 200 iff the
# model has finished loading.

@app.get("/health")
async def health():
    """Liveness probe. Returns 200 once the model is loaded."""
    if SEGMENTER is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")
    return {"status": "ok", "device": SEGMENTER.device}


# =============================================================================
# ENDPOINT: GET /metrics
# =============================================================================
# Cheap observability. Returns aggregate stats over the recent request
# window. In a production service this would be Prometheus-formatted and
# scraped by a monitoring system; for a demo, JSON on demand is fine.

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


# =============================================================================
# ENDPOINT: POST /segment
# =============================================================================
# Single-image inference. This is what the WPF window hits when you drag a
# TIFF/PNG onto the drop panel and click Run segmentation.

@app.post("/segment")
async def segment(file: UploadFile = File(...)):
    """
    Main inference endpoint. Accepts an image upload, returns JSON containing:
      - cell_count: integer
      - inference_ms: float (the model itself, not network overhead)
      - mask_png_base64: colored mask overlay, ready to render
      - device: cuda or cpu
    """
    # Guard: model not loaded yet (startup race, or something went wrong).
    if SEGMENTER is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    # STEP 1: read the upload into memory.
    # `await` yields control back to the event loop while the bytes are
    # streaming off the socket. That's the async win — other requests can
    # be handled during this I/O wait.
    image_bytes = await file.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Empty upload")

    # STEP 2: run inference.
    # This calls into inference.py, which uses the already-loaded model.
    # If anything blows up (bad image format, CUDA out of memory, etc.),
    # we translate it into a clean 500 response instead of a stack trace.
    try:
        result = SEGMENTER.segment(image_bytes)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Inference failed: {exc}")

    # STEP 3: record the latency for the /metrics endpoint.
    LATENCY_WINDOW.append(result.inference_ms)

    # STEP 4: compute per-cell morphology.
    # We do this here (rather than inside inference.py) because it's a
    # response-shaping concern, not a model concern. Both HTTP and folder-
    # watcher paths surface the same shape, so downstream clients don't
    # need to know which path produced the result.
    cells = per_cell_stats(result.mask)

    # STEP 5: build and return the JSON response.
    # Every field here maps to a property in the WPF's SegmentResponse
    # C# class. Adding a field here without updating that class means the
    # client will silently ignore it. Removing one means the client fails.
    return JSONResponse({
        "cell_count": result.cell_count,
        "inference_ms": round(result.inference_ms, 2),
        "device": result.device,
        "mask_png_base64": _mask_to_png_base64(result.mask),
        "summary": summary_stats(cells),
        "per_cell": cells,
    })


# =============================================================================
# ENDPOINT: POST /track_timelapse
# =============================================================================
# Multi-frame inference. Client uploads a GIF; we extract each frame, segment
# it, feed the segmentation to the Hungarian tracker, render an overlay for
# each frame, and return every frame's data in one response.
#
# Why one big response instead of streaming? The WPF client caches the whole
# response and lets the user scrub a slider through the frames — with the
# data pre-loaded, scrubbing feels instant. See MainWindow.xaml.cs for how
# the client handles this.

@app.post("/track_timelapse")
async def track_timelapse(file: UploadFile = File(...)):
    """
    Process an animated GIF as a time-lapse: segment each frame, link cells
    across frames with the Hungarian-assignment tracker, and return every
    frame's data in one response so the client can scrub a slider without
    making more requests.

    Response shape (see WPF's TimelapseResponse class for the C# mirror):
        {
          "n_frames": int,
          "frames": [
            {
              "frame_index": int,
              "inference_ms": float,
              "cell_count": int,
              "original_png_base64": str,
              "overlay_png_base64": str,
              "per_cell": [...],
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
    # -------------------------------------------------------------------------
    # SUBSECTION: input handling
    # -------------------------------------------------------------------------
    if SEGMENTER is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    blob = await file.read()
    if not blob:
        raise HTTPException(status_code=400, detail="Empty upload")

    # Extract the GIF's frames as a list of PNG-encoded bytes. If the upload
    # isn't a valid GIF (or is malformed), we return 400.
    try:
        frame_bytes_list = extract_frames(blob)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Could not parse GIF: {exc}")

    if not frame_bytes_list:
        raise HTTPException(status_code=400, detail="GIF contained no frames")

    # -------------------------------------------------------------------------
    # SUBSECTION: per-frame processing loop
    # -------------------------------------------------------------------------
    # Build a fresh Tracker for THIS request (not shared across requests —
    # each time-lapse has its own set of tracks). Also a fresh history dict
    # to feed the overlay renderer.
    tracker = Tracker(max_distance=30.0, area_weight=0.01)
    track_history: dict[int, list[tuple[float, float]]] = {}
    out_frames: list[dict] = []

    for f_idx, png_bytes in enumerate(frame_bytes_list):
        # Segment this frame with the shared model.
        try:
            seg = SEGMENTER.segment(png_bytes)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Frame {f_idx} failed: {exc}")

        # Compute per-cell morphology and hand it to the tracker. The tracker
        # returns track IDs in the same order as `cells`, so we can zip them
        # to attach track_id to each cell for the JSON response.
        cells = per_cell_stats(seg.mask)
        track_ids = tracker.update(f_idx, cells)

        # Enrich each cell dict with its track_id and update the history dict
        # so `render_overlay` can draw motion trails for each track.
        for cell, tid in zip(cells, track_ids):
            cell["track_id"] = int(tid)
            track_history.setdefault(tid, []).append(
                (float(cell["centroid_x"]), float(cell["centroid_y"]))
            )

        # Render the two images the client will show for this frame:
        # the original (so the user sees the source) and the overlay (with
        # cells colored by track ID + motion trails drawn).
        base = Image.open(io.BytesIO(png_bytes)).convert("RGB")
        overlay = render_overlay(base, seg.mask, track_ids, track_history)

        # Package this frame's slice of the response.
        out_frames.append({
            "frame_index": f_idx,
            "inference_ms": round(seg.inference_ms, 2),
            "cell_count": seg.cell_count,
            "original_png_base64": _image_to_png_base64(base),
            "overlay_png_base64": _image_to_png_base64(overlay),
            "per_cell": cells,
            "summary": summary_stats(cells),
        })

    # -------------------------------------------------------------------------
    # SUBSECTION: tracks summary + final response
    # -------------------------------------------------------------------------
    # "Full-length" = track present in every frame. That's the honest metric
    # for "how many cells did we successfully follow start to end." The rest
    # (partial) are tracks that appeared partway or dropped out early.
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
