"""
Cellpose model wrapper.

This file exists to isolate the model lifecycle from the web framework.
main.py knows nothing about PyTorch; it just calls CellSegmenter.segment().
That separation is the whole point of a deployment-engineering codebase:
the model can be swapped (different architecture, different weights, even a
different framework) without touching the API surface.
"""

from __future__ import annotations

import io
import time
from dataclasses import dataclass

import numpy as np
import torch
from cellpose import models
from PIL import Image


@dataclass
class SegmentationResult:
    """What we return to the API layer for every inference call."""
    mask: np.ndarray           # H x W array, 0 = background, 1..N = cell IDs
    cell_count: int            # Number of distinct cells found
    inference_ms: float        # Wall-clock time the model itself took
    device: str                # "cuda" or "cpu" — useful for the /metrics view


class CellSegmenter:
    """
    Loads the Cellpose model ONCE at startup. Reusing the loaded model
    across requests is the single biggest latency win versus reloading
    per request (which would cost ~5-10s every call).
    """

    def __init__(self, model_type: str = "cyto3", use_gpu: bool | None = None):
        # Resolve device. None = auto-detect (use GPU if CUDA is available).
        if use_gpu is None:
            use_gpu = torch.cuda.is_available()
        self.device = "cuda" if use_gpu else "cpu"

        # Cellpose wraps a U-Net. "cyto3" is the latest general cytoplasm model.
        # Other built-in options: "cyto2", "nuclei". For Sartorius-style
        # phase-contrast images, "cyto3" is the right default.
        self.model = models.Cellpose(gpu=use_gpu, model_type=model_type)
        self.model_type = model_type

    def segment(self, image_bytes: bytes) -> SegmentationResult:
        """
        Run segmentation on a single image. Accepts raw bytes (whatever the
        web framework hands us) and returns a structured result.
        """
        # Decode bytes into a numpy array. PIL handles PNG, JPEG, TIFF.
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        arr = np.array(img)

        # Cellpose expects a list of images and returns lists in return.
        # channels=[0, 0] = grayscale segmentation (no nuclei channel).
        # diameter=None = let Cellpose estimate cell size automatically.
        start = time.perf_counter()
        masks, _flows, _styles, _diams = self.model.eval(
            [arr],
            channels=[0, 0],
            diameter=None,
        )
        elapsed_ms = (time.perf_counter() - start) * 1000.0

        mask = masks[0]                              # Unwrap the single-image list
        cell_count = int(mask.max())                 # Cell IDs are 1..N, 0 = bg

        return SegmentationResult(
            mask=mask,
            cell_count=cell_count,
            inference_ms=elapsed_ms,
            device=self.device,
        )
