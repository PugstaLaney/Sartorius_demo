"""
Cell tracker — link instance-segmentation results across time.

Problem: We segment each frame independently. The segmenter has no memory,
so it assigns cell ID 1..N fresh in each frame. ID 5 in frame 0 has no
relationship to ID 5 in frame 1. We need to figure out which cell IDs
across frames correspond to the SAME physical cell so we can build
trajectories.

Algorithm: frame-to-frame Hungarian (Kuhn-Munkres) assignment on a cost
matrix. Each candidate match between a cell in frame N and a cell in frame
N+1 has a cost based on how far the centroids moved plus how much the area
changed. The Hungarian solver finds the globally optimal assignment.

Why Hungarian and not greedy nearest-neighbor:
    - Greedy can produce conflicts (cell A and cell B both pick cell X)
    - Hungarian guarantees each cell is matched at most once globally
    - It's the same algorithm TrackMate uses for its simple tracker

Why not learned features (re-id embeddings):
    - Honest scope: a deep tracker is a separate research project
    - For slow-moving cells under microscopy, centroid + area is plenty
    - If we want better later, swap link_frames() internals without
      changing the Tracker class interface

Reference: linear_sum_assignment from scipy.optimize implements Jonker-Volgenant
(an O(n^3) variant of Hungarian).

See learning_materials/06_hungarian_tracking.md for a full walkthrough.
"""

from __future__ import annotations


# =============================================================================
# IMPORTS
# =============================================================================

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy.optimize import linear_sum_assignment    # THE Hungarian solver

# A "cell observation" here is the per-cell dict produced by morphology.py.
# We only require centroid_x, centroid_y, and area_px. Anything else passes
# through untouched. Using a type alias makes the function signatures cleaner.
Cell = dict[str, Any]


# =============================================================================
# COST MATRIX CONSTRUCTION
# =============================================================================
# The Hungarian algorithm needs a matrix where cost[i, j] tells it "how bad
# would it be to match prev-cell i with current-cell j?" This function
# builds that matrix.

def _cost_matrix(
    prev_cells: list[Cell],
    curr_cells: list[Cell],
    max_distance: float,
    area_weight: float,
) -> np.ndarray:
    """
    Build the cost matrix for matching prev frame against curr frame.

    cost[i, j] = euclidean(centroid_i, centroid_j) + area_weight * |area_i - area_j|

    Pairs with centroid distance > max_distance get cost = +inf so the
    solver won't pick them. After solving, we filter out any matches that
    still have +inf cost — those are "no link" pairs.
    """
    n_prev, n_curr = len(prev_cells), len(curr_cells)

    # Start with +inf everywhere. Any pair we can't (or won't) match stays
    # at +inf and gets rejected later.
    cost = np.full((n_prev, n_curr), np.inf, dtype=np.float64)

    # Pull out the relevant columns as numpy arrays. Doing the math in numpy
    # (vectorized) is drastically faster than a Python loop over cell pairs.
    prev_xy = np.array([(c["centroid_x"], c["centroid_y"]) for c in prev_cells])
    curr_xy = np.array([(c["centroid_x"], c["centroid_y"]) for c in curr_cells])
    prev_area = np.array([c["area_px"] for c in prev_cells], dtype=np.float64)
    curr_area = np.array([c["area_px"] for c in curr_cells], dtype=np.float64)

    # Compute the pairwise distance matrix all at once using broadcasting.
    # prev_xy[:, 0, None] has shape (n_prev, 1). curr_xy[None, :, 0] has
    # shape (1, n_curr). Subtracting them broadcasts to (n_prev, n_curr).
    dx = prev_xy[:, 0, None] - curr_xy[None, :, 0]
    dy = prev_xy[:, 1, None] - curr_xy[None, :, 1]
    dist = np.sqrt(dx * dx + dy * dy)                     # euclidean distance
    area_diff = np.abs(prev_area[:, None] - curr_area[None, :])

    # Only fill in cost for pairs within reach. Everyone else stays at +inf.
    within_range = dist <= max_distance
    cost[within_range] = dist[within_range] + area_weight * area_diff[within_range]
    return cost


# =============================================================================
# FRAME-TO-FRAME LINKING
# =============================================================================
# Takes two lists of cells (prev frame, current frame) and figures out which
# pairs correspond to the same physical cell.

def link_frames(
    prev_cells: list[Cell],
    curr_cells: list[Cell],
    max_distance: float = 30.0,
    area_weight: float = 0.01,
) -> tuple[list[tuple[int, int]], list[int], list[int]]:
    """
    Return:
        matched   - list of (prev_idx, curr_idx) pairs that are linked
        new_curr  - indices into curr_cells that did not match anything
                    (these become new track IDs)
        lost_prev - indices into prev_cells that did not match anything
                    (their tracks are lost / cell left the frame)
    """
    # Edge case: one side has no cells. Nothing to match.
    if not prev_cells or not curr_cells:
        return [], list(range(len(curr_cells))), list(range(len(prev_cells)))

    # Build the cost matrix using the helper above.
    cost = _cost_matrix(prev_cells, curr_cells, max_distance, area_weight)

    # linear_sum_assignment cannot handle +inf, so replace with a "large but
    # finite" sentinel that's still way bigger than any legitimate cost.
    # This is a common pattern with the Hungarian algorithm.
    BIG = 1e9
    safe_cost = np.where(np.isinf(cost), BIG, cost)
    row_ind, col_ind = linear_sum_assignment(safe_cost)

    # After the solver returns, filter out any "matches" that were actually
    # forced to pick a +inf pair (i.e., no valid match existed for that row
    # or column). These count as "unmatched," not as real links.
    matched: list[tuple[int, int]] = []
    matched_prev: set[int] = set()
    matched_curr: set[int] = set()
    for r, c in zip(row_ind, col_ind):
        if cost[r, c] < np.inf:                    # was actually within range
            matched.append((int(r), int(c)))
            matched_prev.add(int(r))
            matched_curr.add(int(c))

    # Anything in the current frame that didn't get matched is a "new track."
    # Anything in the previous frame that didn't get matched is a "lost track."
    new_curr = [j for j in range(len(curr_cells)) if j not in matched_curr]
    lost_prev = [i for i in range(len(prev_cells)) if i not in matched_prev]
    return matched, new_curr, lost_prev


# =============================================================================
# DATA CONTAINERS FOR TRACK STATE
# =============================================================================
# TrackPoint = one observation of a cell in one frame.
# Track     = the ordered list of observations for a single cell over time.
# Both use @dataclass to avoid boilerplate.

@dataclass
class TrackPoint:
    """One observation of a cell in one frame, with its centroid + area."""
    frame: int
    centroid_x: float
    centroid_y: float
    area_px: int


@dataclass
class Track:
    """A single cell observed across multiple frames."""
    track_id: int
    # `field(default_factory=list)` is required for mutable defaults on
    # dataclasses. Without it, all instances would share the same list
    # (a classic Python footgun).
    points: list[TrackPoint] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Serialize this track for the JSON response."""
        return {
            "track_id": self.track_id,
            "first_frame": self.points[0].frame if self.points else None,
            "last_frame": self.points[-1].frame if self.points else None,
            "length": len(self.points),
            "points": [
                {"frame": p.frame, "x": p.centroid_x, "y": p.centroid_y, "area_px": p.area_px}
                for p in self.points
            ],
        }


# =============================================================================
# THE STATEFUL TRACKER
# =============================================================================
# `Tracker` is a stateful worker that carries "state from the previous frame"
# between calls. Same pattern as CellSegmenter: expensive-to-construct state
# lives on the instance; cheap methods reuse it.

class Tracker:
    """
    Stateful tracker: feed it segmentation results frame by frame, it returns
    the track IDs assigned to each cell in that frame and maintains a growing
    `tracks` dict you can dump at the end.

    Typical use:
        tr = Tracker(max_distance=30)
        for f, cells in enumerate(per_frame_cells):
            ids = tr.update(f, cells)
            # ids[k] is the track ID assigned to cells[k]
        tracks_json = tr.dump()
    """

    # -------------------------------------------------------------------------
    # SUBSECTION: __init__ — set up the tracker's initial state
    # -------------------------------------------------------------------------
    def __init__(self, max_distance: float = 30.0, area_weight: float = 0.01):
        # Cost function parameters. Higher max_distance = more permissive
        # matching (cells can move further between frames). Higher area_weight
        # = area mismatches matter more relative to position mismatches.
        self.max_distance = max_distance
        self.area_weight = area_weight

        # State we carry between calls:
        # - _next_track_id      : monotonic counter for issuing fresh track IDs
        # - _prev_frame_assignments : track_id per cell in the previous frame,
        #                              in the same order as _prev_frame_cells
        # - _prev_frame_cells   : the previous frame's cell list (for linking)
        # - tracks              : dict of track_id -> Track (grows across frames)
        #
        # The leading underscore on `_prev_frame_*` is a convention meaning
        # "internal — don't touch from outside the class." `tracks` has no
        # underscore because we DO want callers to read it (via .dump()).
        self._next_track_id = 1
        self._prev_frame_assignments: list[int] = []
        self._prev_frame_cells: list[Cell] = []
        self.tracks: dict[int, Track] = {}

    # -------------------------------------------------------------------------
    # SUBSECTION: private helpers for track bookkeeping
    # -------------------------------------------------------------------------
    def _new_track(self, frame: int, cell: Cell) -> int:
        """Allocate a fresh track ID for a cell we've never seen before."""
        tid = self._next_track_id
        self._next_track_id += 1
        self.tracks[tid] = Track(track_id=tid, points=[
            TrackPoint(
                frame=frame,
                centroid_x=float(cell["centroid_x"]),
                centroid_y=float(cell["centroid_y"]),
                area_px=int(cell["area_px"]),
            )
        ])
        return tid

    def _extend_track(self, tid: int, frame: int, cell: Cell) -> None:
        """Add a new observation to an existing track."""
        self.tracks[tid].points.append(TrackPoint(
            frame=frame,
            centroid_x=float(cell["centroid_x"]),
            centroid_y=float(cell["centroid_y"]),
            area_px=int(cell["area_px"]),
        ))

    # -------------------------------------------------------------------------
    # SUBSECTION: update() — the public entry point, called once per frame
    # -------------------------------------------------------------------------
    def update(self, frame_index: int, cells: list[Cell]) -> list[int]:
        """
        Assign every cell in this frame to either an existing track or a new
        one. Returns one track_id per input cell, in the same order.
        """
        # First frame: no previous frame to link against, so every cell
        # becomes a brand-new track. Store this frame's state and return.
        if not self._prev_frame_cells:
            assignments = [self._new_track(frame_index, c) for c in cells]
            self._prev_frame_assignments = assignments
            self._prev_frame_cells = cells
            return assignments

        # Subsequent frames: link against the previous frame using Hungarian.
        matched, new_curr, _lost = link_frames(
            self._prev_frame_cells,
            cells,
            max_distance=self.max_distance,
            area_weight=self.area_weight,
        )

        # `assignments[k]` will hold the track ID we chose for cells[k].
        # We start with None and fill in below.
        assignments: list[int | None] = [None] * len(cells)

        # For each matched pair: this current-frame cell continues the track
        # that the matched previous-frame cell belonged to.
        for prev_idx, curr_idx in matched:
            tid = self._prev_frame_assignments[prev_idx]
            self._extend_track(tid, frame_index, cells[curr_idx])
            assignments[curr_idx] = tid

        # Unmatched current-frame cells are new — issue fresh track IDs.
        for curr_idx in new_curr:
            assignments[curr_idx] = self._new_track(frame_index, cells[curr_idx])

        # NOTE ON LOST TRACKS:
        # Tracks belonging to lost_prev cells are simply not extended this
        # frame. They may resume later if a cell reappears within max_distance,
        # but our simple tracker treats reappearance as a new track. A real
        # tracker (like DeepSORT) would bridge short gaps by keeping a "recently
        # lost" pool and trying to re-match against it for a few frames. That's
        # a future improvement — see 06_hungarian_tracking.md for context.

        # Squash out any Nones (there shouldn't be any at this point; every
        # slot was filled either by a match or by a new track) and store as
        # the previous-frame state for the next call.
        final: list[int] = [a for a in assignments if a is not None]
        self._prev_frame_assignments = final
        self._prev_frame_cells = cells
        return final

    # -------------------------------------------------------------------------
    # SUBSECTION: dump() — serialize tracks for JSON output
    # -------------------------------------------------------------------------
    def dump(self) -> dict:
        """Return the full set of tracks as a JSON-serializable dict."""
        return {
            "n_tracks": len(self.tracks),
            "params": {
                "max_distance": self.max_distance,
                "area_weight": self.area_weight,
            },
            "tracks": [t.to_dict() for t in self.tracks.values()],
        }
