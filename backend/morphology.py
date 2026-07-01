"""
Per-cell morphology extraction from an instance segmentation mask.

This is the "real-time insights" piece — once we have a mask with each cell
labeled by a unique integer ID, downstream measurements are just bookkeeping.
We delegate the geometric heavy lifting to scikit-image's regionprops_table,
which is the standard tool for this in biological image analysis.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from skimage.measure import regionprops_table


# The set of per-cell measurements we surface. Keep this list short and
# meaningful — each one is something a biologist would actually look at.
#
#   area         total pixel count of the cell — proxy for cell size
#   perimeter    boundary length in pixels — proxy for membrane extent
#   eccentricity 0 = perfect circle, 1 = highly elongated
#   centroid     (y, x) center of mass — useful for downstream tracking
#   solidity     area / convex_hull_area — 1.0 means convex, lower = ragged
PROPERTIES = ("label", "area", "perimeter", "eccentricity", "centroid", "solidity")


def per_cell_stats(mask: np.ndarray) -> list[dict[str, Any]]:
    """
    Return one dict per cell with morphology measurements. Cells are
    identified by the integer IDs in the mask (1..N, 0 = background).
    Empty mask returns an empty list.
    """
    if mask.max() == 0:
        return []

    props = regionprops_table(mask, properties=PROPERTIES)

    # regionprops_table returns column-oriented arrays. Transpose into rows.
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


def summary_stats(per_cell: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Aggregate the per-cell list into a single summary block. Useful for the
    JSON sidecar so a downstream consumer doesn't need to walk the array
    just to get "how many cells, average size, etc."
    """
    if not per_cell:
        return {"count": 0}

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
