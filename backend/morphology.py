"""
Per-cell morphology extraction from an instance segmentation mask.

This is the "real-time insights" piece — once we have a mask with each cell
labeled by a unique integer ID, downstream measurements are just bookkeeping.
We delegate the geometric heavy lifting to scikit-image's regionprops_table,
which is the standard tool for this in biological image analysis.

See learning_materials/04_arrays_tensors_and_gpus.md for how the input
`mask` array is structured (shape/dtype), and 01_image_journey.md for where
this fits into the request flow.
"""

from __future__ import annotations


# =============================================================================
# IMPORTS
# =============================================================================
# This is a leaf module — nothing here imports from another `backend/` file.
# That makes it trivially testable: you can pass in a fabricated numpy mask
# in a REPL and get out a list of dicts. No model, no HTTP, no framework.

from typing import Any                              # Any = "value of any type" for type hints

import numpy as np
from skimage.measure import regionprops_table       # THE standard morphology extractor


# =============================================================================
# CONFIG: WHICH MEASUREMENTS TO SURFACE
# =============================================================================
# We deliberately keep this list short. Every property here is something a
# biologist would actually look at. scikit-image supports dozens more
# (moments, feret diameter, orientation, etc.), but a big list just gives
# you noise in the JSON response.
#
#   area         total pixel count of the cell — proxy for cell size
#   perimeter    boundary length in pixels — proxy for membrane extent
#   eccentricity 0 = perfect circle, 1 = highly elongated
#   centroid     (y, x) center of mass — useful for downstream tracking
#   solidity     area / convex_hull_area — 1.0 means convex, lower = ragged

PROPERTIES = ("label", "area", "perimeter", "eccentricity", "centroid", "solidity")


# =============================================================================
# PER-CELL STATS
# =============================================================================
# Input: an instance mask (2D numpy array where each pixel is 0 = background
# or an integer cell ID 1..N).
# Output: a list of dicts, one per cell, with the measurements above.

def per_cell_stats(mask: np.ndarray) -> list[dict[str, Any]]:
    """
    Return one dict per cell with morphology measurements. Cells are
    identified by the integer IDs in the mask (1..N, 0 = background).
    Empty mask returns an empty list.
    """
    # If the mask has no cells (all zeros), skip the work and return early.
    # This also avoids a scikit-image edge case with empty masks.
    if mask.max() == 0:
        return []

    # regionprops_table returns a dict of column-oriented arrays:
    #   {"label": [1, 2, 3, ...], "area": [440, 297, ...], ...}
    # It's fast because the geometry is computed in vectorized C code.
    props = regionprops_table(mask, properties=PROPERTIES)

    # Transpose column-oriented -> row-oriented so each dict describes one cell.
    # Note the trailing '-0' and '-1' on centroid: scikit-image returns
    # centroid as two columns ("centroid-0" is y, "centroid-1" is x).
    n_cells = len(props["label"])
    return [
        {
            "cell_id": int(props["label"][i]),
            "area_px": int(props["area"][i]),
            "perimeter_px": float(round(props["perimeter"][i], 2)),
            "eccentricity": float(round(props["eccentricity"][i], 3)),
            "solidity": float(round(props["solidity"][i], 3)),
            "centroid_y": float(round(props["centroid-0"][i], 1)),
            "centroid_x": float(round(props["centroid-1"][i], 1)),
        }
        for i in range(n_cells)
    ]


# =============================================================================
# AGGREGATE SUMMARY
# =============================================================================
# The per-cell list can be long (100+ entries). This function reduces it to a
# small summary block — count, area distribution, mean eccentricity —
# so callers can show a one-line "there are 101 cells averaging 709 px" without
# scanning the array.

def summary_stats(per_cell: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Aggregate the per-cell list into a single summary block. Useful for the
    JSON sidecar so a downstream consumer doesn't need to walk the array
    just to get "how many cells, average size, etc."
    """
    # Empty input -> minimal output. Downstream code checks .count to decide
    # whether to display anything.
    if not per_cell:
        return {"count": 0}

    # Convert the two columns we aggregate into numpy arrays for vectorized
    # mean/median/min/max. Faster than Python-level loops for large N.
    areas = np.array([c["area_px"] for c in per_cell])
    eccentricities = np.array([c["eccentricity"] for c in per_cell])

    return {
        "count": len(per_cell),
        "area_px": {
            "mean": float(round(areas.mean(), 1)),
            "median": float(round(np.median(areas), 1)),
            "min": int(areas.min()),
            "max": int(areas.max()),
        },
        "eccentricity": {
            "mean": float(round(eccentricities.mean(), 3)),
        },
    }
