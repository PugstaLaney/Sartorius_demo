"""
Quick sanity check that the inference pipeline works end-to-end without
needing the FastAPI server running.

Run from the project root:
    py dev_scripts/smoke_test.py

Expected output: CUDA detected, model loads, segmentation runs, cell count > 0.
If this passes, the FastAPI service will also work.
"""

from __future__ import annotations

import sys
from pathlib import Path

# This script lives in dev_scripts/ but imports from backend/. Make backend
# importable by adding it to Python's module search path at runtime.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))

import numpy as np
import torch
from PIL import Image

from inference import CellSegmenter


def main() -> int:
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  device: {torch.cuda.get_device_name(0)}")

    print("\nLoading Cellpose model...")
    segmenter = CellSegmenter()
    print(f"  device: {segmenter.device}")

    # Find a sample image. Prefer something you dropped into data/sample_images.
    sample_dir = Path(__file__).parent.parent / "data" / "sample_images"
    candidates = [
        p for p in sample_dir.glob("*")
        if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".tif", ".tiff"}
    ]

    if not candidates:
        # Fall back to a Cellpose built-in example.
        import cellpose, os
        builtin = Path(os.path.dirname(cellpose.__file__)) / "data" / "rgb_2D_tif.tif"
        if builtin.exists():
            candidates = [builtin]
        else:
            # As a last resort, synthesize a fake image with circles.
            print("\nNo sample image found, synthesizing one.")
            arr = np.zeros((512, 512, 3), dtype=np.uint8)
            for cx, cy in [(120, 120), (250, 200), (400, 300), (180, 400)]:
                y, x = np.ogrid[:512, :512]
                arr[(x - cx) ** 2 + (y - cy) ** 2 <= 35 ** 2] = (200, 200, 200)
            tmp = sample_dir / "_synthetic.png"
            sample_dir.mkdir(parents=True, exist_ok=True)
            Image.fromarray(arr).save(tmp)
            candidates = [tmp]

    sample = candidates[0]
    print(f"\nRunning segmentation on: {sample.name}")

    with open(sample, "rb") as f:
        result = segmenter.segment(f.read())

    print(f"  cells detected: {result.cell_count}")
    print(f"  inference: {result.inference_ms:.1f} ms")
    print(f"  device: {result.device}")
    print(f"  mask shape: {result.mask.shape}")

    if result.cell_count == 0:
        print("\nWARNING: zero cells detected. Either the sample image is unusual")
        print("or something is wrong. Drop a real microscopy image into")
        print("data/sample_images/ and re-run.")
        return 1

    print("\nSmoke test PASSED.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
