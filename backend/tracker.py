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
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy.optimize import linear_sum_assignment


# A "cell observation" here is the per-cell dict produced by morphology.py.
# We only require centroid_x, centroid_y, and area_px. Anything else passes
# through untouched.
Cell = dict[str, Any]


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
    cost = np.full((n_prev, n_curr), np.inf, dtype=np.float64)

    prev_xy = np.array([(c["centroid_x"], c["centroid_y"]) for c in prev_cells])
    curr_xy = np.array([(c["centroid_x"], c["centroid_y"]) for c in curr_cells])
    prev_area = np.array([c["area_px"] for c in prev_cells], dtype=np.float64)
    curr_area = np.array([c["area_px"] for c in curr_cells], dtype=np.float64)

    # Pairwise distance matrix (vectorized — much faster than Python loops).
    dx = prev_xy[:, 0, None] - curr_xy[None, :, 0]
    dy = prev_xy[:, 1, None] - curr_xy[None, :, 1]
    dist = np.sqrt(dx * dx + dy * dy)
    area_diff = np.abs(prev_area[:, None] - curr_area[None, :])

    within_range = dist <= max_distance
    cost[within_range] = dist[within_range] + area_weight * area_diff[within_range]
    return cost


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
    if not prev_cells or not curr_cells:
        return [], list(range(len(curr_cells))), list(range(len(prev_cells)))

    cost = _cost_matrix(prev_cells, curr_cells, max_distance, area_weight)

    # linear_sum_assignment cannot handle +inf, so replace with a "large but
    # finite" sentinel that's still way bigger than any legitimate cost.
    BIG = 1e9
    safe_cost = np.where(np.isinf(cost), BIG, cost)
    row_ind, col_ind = linear_sum_assignment(safe_cost)

    matched: list[tuple[int, int]] = []
    matched_prev: set[int] = set()
    matched_curr: set[int] = set()
    for r, c in zip(row_ind, col_ind):
        if cost[r, c] < np.inf:  # only keep pairs that were actually in range
            matched.append((int(r), int(c)))
            matched_prev.add(int(r))
            matched_curr.add(int(c))

    new_curr = [j for j in range(len(curr_cells)) if j not in matched_curr]
    lost_prev = [i for i in range(len(prev_cells)) if i not in matched_prev]
    return matched, new_curr, lost_prev


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
    points: list[TrackPoint] = field(default_factory=list)

    def to_dict(self) -> dict:
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

    def __init__(self, max_distance: float = 30.0, area_weight: float = 0.01):
        self.max_distance = max_distance
        self.area_weight = area_weight
        self._next_track_id = 1
        self._prev_frame_assignments: list[int] = []  # track_id per cell in prev frame
        self._prev_frame_cells: list[Cell] = []
        self.tracks: dict[int, Track] = {}

    def _new_track(self, frame: int, cell: Cell) -> int:
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
        self.tracks[tid].points.append(TrackPoint(
            frame=frame,
            centroid_x=float(cell["centroid_x"]),
            centroid_y=float(cell["centroid_y"]),
            area_px=int(cell["area_px"]),
        ))

    def update(self, frame_index: int, cells: list[Cell]) -> list[int]:
        """
        Assign every cell in this frame to either an existing track or a new
        one. Returns one track_id per input cell, in the same order.
        """
        # First frame: every cell becomes a new track.
        if not self._prev_frame_cells:
            assignments = [self._new_track(frame_index, c) for c in cells]
            self._prev_frame_assignments = assignments
            self._prev_frame_cells = cells
            return assignments

        matched, new_curr, _lost = link_frames(
            self._prev_frame_cells,
            cells,
            max_distance=self.max_distance,
            area_weight=self.area_weight,
        )

        assignments: list[int | None] = [None] * len(cells)
        for prev_idx, curr_idx in matched:
            tid = self._prev_frame_assignments[prev_idx]
            self._extend_track(tid, frame_index, cells[curr_idx])
            assignments[curr_idx] = tid

        for curr_idx in new_curr:
            assignments[curr_idx] = self._new_track(frame_index, cells[curr_idx])

        # Tracks belonging to lost_prev cells are simply not extended this frame.
        # They may resume later if a cell reappears within max_distance — but our
        # simple tracker treats reappearance as a new track. A real tracker would
        # bridge short gaps; that's a future improvement.

        final: list[int] = [a for a in assignments if a is not None]
        self._prev_frame_assignments = final
        self._prev_frame_cells = cells
        return final

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
