"""
Cellpose model wrapper.

This file exists to isolate the model lifecycle from the web framework.
main.py knows nothing about PyTorch; it just calls CellSegmenter.segment().
That separation is the whole point of a deployment-engineering codebase:
the model can be swapped (different architecture, different weights, even a
different framework) without touching the API surface.

See learning_materials/02_classes_and_lifetime.md for what the class + init
pattern is doing under the hood, and 04_arrays_tensors_and_gpus.md for how
the numpy/tensor conversions inside `segment()` work.
"""

from __future__ import annotations


# =============================================================================
# IMPORTS
# =============================================================================
# stdlib first, then third-party. We keep this file "leaf-level": nothing here
# imports from another module in `backend/`. That's why main.py and watcher.py
# can both use CellSegmenter without dragging FastAPI or watchdog behavior
# into each other.

import io                                   # BytesIO wraps raw bytes so PIL can read them like a file
import time                                 # perf_counter() gives us high-resolution wall-clock timing
from dataclasses import dataclass           # decorator that auto-generates __init__/__repr__/__eq__

import numpy as np                          # numerical arrays; images become numpy arrays here
import torch                                # only used to check whether a CUDA GPU is available
from cellpose import models                 # the actual segmentation model library
from PIL import Image                       # image loading (PNG/JPEG/TIFF decoding)


# =============================================================================
# RETURN-VALUE CONTAINER
# =============================================================================
# A `@dataclass` is a shortcut for "a class that just holds a bundle of named
# values, no behavior." Python auto-writes the __init__, so we don't have to
# type out `self.mask = mask; self.cell_count = cell_count; ...` manually.
#
# The value of using this instead of a plain dict:
#   - IDE autocomplete on .mask, .cell_count, etc.
#   - Type checkers can validate that main.py accesses only real fields
#   - A typo like `result.cel_count` fails loudly at read time, not later

@dataclass
class SegmentationResult:
    """What we return to the API layer for every inference call."""
    mask: np.ndarray           # H x W array. 0 = background, 1..N = cell IDs.
    cell_count: int            # Number of distinct cells found in this image.
    inference_ms: float        # Wall-clock time the model itself took (excludes I/O).
    device: str                # "cuda" or "cpu" — useful for the /metrics view.


# =============================================================================
# THE WORKER CLASS
# =============================================================================
# `CellSegmenter` is a "stateful worker" — an object that carries expensive-to-
# build state (the loaded model) and exposes a cheap method (segment()) that
# reuses that state across many calls.
#
# Why a class rather than two top-level functions:
#   - The loaded model is ~500 MB in GPU VRAM. We load it ONCE at startup.
#   - Every subsequent segment() call reuses the loaded model.
#   - Bundling the model + methods together makes that reuse the default
#     rather than something the caller has to remember.

class CellSegmenter:
    """
    Loads the Cellpose model ONCE at startup. Reusing the loaded model
    across requests is the single biggest latency win versus reloading
    per request (which would cost ~5-10s every call).
    """

    # -------------------------------------------------------------------------
    # SUBSECTION: __init__ — runs once when someone writes CellSegmenter(...)
    # -------------------------------------------------------------------------
    # `__init__` is a reserved method name Python looks for at instance
    # creation time. When main.py runs `SEGMENTER = CellSegmenter()`, Python:
    #   1. Creates a new empty CellSegmenter object
    #   2. Automatically calls __init__(new_object, ...) — `self` IS that
    #      new object
    #   3. Whatever we set on `self` inside __init__ gets stored on the
    #      instance so it's available later
    #
    # After __init__ finishes, the returned instance carries:
    #   - self.device      -> "cuda" or "cpu"
    #   - self.model       -> the loaded Cellpose model (references VRAM weights)
    #   - self.model_type  -> which pretrained model we loaded (e.g. "cyto3")
    def __init__(self, model_type: str = "cyto3", use_gpu: bool | None = None):
        # Resolve device.
        # `use_gpu=None` is the "figure it out for me" signal. If the caller
        # doesn't specify, we auto-detect based on whether CUDA is available.
        # This lets the same code run on your RTX 3060 (GPU) and on a CPU-only
        # laptop without changes.
        if use_gpu is None:
            use_gpu = torch.cuda.is_available()
        self.device = "cuda" if use_gpu else "cpu"

        # Load the model.
        # This is the SLOW line (~10 seconds). It pulls weights from disk
        # (or downloads them the first time), builds the PyTorch network in
        # memory, and — if use_gpu=True — pins the weights to GPU VRAM.
        #
        # "cyto3" = Cellpose's latest general-purpose cytoplasm model,
        # released by the Cellpose team at Janelia in 2024. Alternative
        # built-in options include "cyto2" and "nuclei".
        self.model = models.Cellpose(gpu=use_gpu, model_type=model_type)
        self.model_type = model_type

    # -------------------------------------------------------------------------
    # SUBSECTION: segment() — runs once per HTTP request
    # -------------------------------------------------------------------------
    # This method is what actually does inference. It DOES NOT reload the
    # model — it uses `self.model`, which was loaded during __init__.
    #
    # See learning_materials/01_image_journey.md for the full trace of what
    # happens to the image bytes on their way through this method.
    def segment(self, image_bytes: bytes) -> SegmentationResult:
        """
        Run segmentation on a single image. Accepts raw bytes (whatever the
        web framework hands us) and returns a structured result.
        """
        # Step 1: decode the raw file bytes into a PIL Image, then into a
        # numpy array. `.convert("RGB")` guarantees 3 channels regardless of
        # whether the source was grayscale, RGBA, or palette-indexed.
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        arr = np.array(img)   # shape: (H, W, 3), dtype: uint8

        # Step 2: run the model. Cellpose's API is batch-oriented, so we
        # wrap our single image in a list and unpack the first result.
        #   channels=[0, 0] = "use one grayscale channel, no nuclei stain."
        #   diameter=None   = "estimate cell diameter automatically."
        #
        # Under the hood, this line:
        #   - Copies `arr` from RAM to GPU VRAM as a PyTorch tensor
        #   - Runs the U-Net forward pass (~600ms on RTX 3060)
        #   - Computes the flow field, follows flows to attractor points
        #   - Returns the labeled mask back on CPU as a numpy array
        start = time.perf_counter()
        masks, _flows, _styles, _diams = self.model.eval(
            [arr],
            channels=[0, 0],
            diameter=None,
        )
        elapsed_ms = (time.perf_counter() - start) * 1000.0

        # Step 3: unpack the results. `masks` is a list because Cellpose
        # accepted a list; we only sent one image so we take [0].
        mask = masks[0]                          # shape: (H, W), dtype: int32
        cell_count = int(mask.max())             # cells are labeled 1..N, so max() == N

        # Step 4: return a SegmentationResult. This is a lightweight wrapper
        # holding REFERENCES to `mask` and the scalar metadata. It does not
        # copy the mask — main.py reads `result.mask` and the same numpy
        # array is right there.
        return SegmentationResult(
            mask=mask,
            cell_count=cell_count,
            inference_ms=elapsed_ms,
            device=self.device,
        )
