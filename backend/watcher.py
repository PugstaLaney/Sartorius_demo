"""
Folder-drop ingestion daemon. The second entry point into the service,
alongside main.py's HTTP endpoints.

Role in the architecture
------------------------
Layer:       Orchestrator + standalone entry point
Imported by: Nothing. This script is run directly via `py watcher.py`.
Depends on:  inference, morphology, incucyte_filename
Runs when:   Continuously in a poll loop, until Ctrl+C

Reuses the same CellSegmenter that main.py uses. The model is loaded once
at startup, then reused across every file processed. That is the same
"load-once, reuse-many" pattern that makes the HTTP service fast. Folder-drop
is just a different orchestration wrapping the same model.

WHY THIS EXISTS
---------------
Incucyte (Sartorius's live-cell imaging instrument) does not expose a public
REST API. Its v2025A software added an "Auto Archive to a designated network
location" feature, meaning the canonical integration surface for a Sartorius
instrument is:

    Instrument  --writes TIFFs-->  Network share  <--reads--  Our service

So we mirror that pattern locally. A folder called `incoming/` plays the role
of the network share. The watcher polls it, processes any new TIFF/PNG/JPEG
files, and writes JSON sidecar results into `outgoing/`. We also move
processed sources into `processed/` so they aren't re-processed on every poll.

This script can run standalone:
    py watcher.py

It loads the segmentation model once at startup (same lifecycle pattern as
the FastAPI service in main.py), then loops forever.

See learning_materials/05_layered_architecture.md. This is a second
orchestrator alongside main.py, sharing the same leaf modules.
"""

from __future__ import annotations


# =============================================================================
# IMPORTS
# =============================================================================
# argparse gives us a real CLI (`--incoming`, `--once`, etc.).
# shutil.move is how we relocate processed files across folders.
# datetime.timezone.utc is imported explicitly so we can build UTC timestamps
# without relying on the ambient system timezone.

import argparse
import json
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image

from incucyte_filename import parse as parse_incucyte    # aliased for readability
from inference import CellSegmenter
from morphology import per_cell_stats, summary_stats


# =============================================================================
# CONFIG
# =============================================================================

# File extensions we treat as candidate images. Anything else in `incoming/`
# (README files, notes, .DS_Store on Mac, etc.) is ignored.
IMAGE_EXTENSIONS = {".tif", ".tiff", ".png", ".jpg", ".jpeg"}


# =============================================================================
# HELPERS
# =============================================================================
# Small private utilities used by the main loop below.

def _utc_iso() -> str:
    """Timestamp helper — UTC, ISO-8601, second precision."""
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _process_one(
    source: Path,
    outgoing_dir: Path,
    processed_dir: Path,
    segmenter: CellSegmenter,
) -> dict:
    """Run segmentation on one file, write the JSON sidecar, move the source."""
    started_at = _utc_iso()

    # STEP 1: read the file into memory and segment it. This is the same
    # segmenter.segment(bytes) call that /segment in main.py makes — the
    # model wrapper doesn't care whether it's driven by HTTP or by a file.
    image_bytes = source.read_bytes()
    result = segmenter.segment(image_bytes)

    # STEP 2: compute per-cell morphology and aggregate summary.
    cells = per_cell_stats(result.mask)
    summary = summary_stats(cells)

    # STEP 3: try to parse the filename as Incucyte-formatted. If it doesn't
    # match (e.g. we're processing a file with a random name), incucyte_meta
    # will be None and we surface null in the JSON.
    incucyte_meta = parse_incucyte(source.name)

    # STEP 4: build the sidecar dict.
    # "Sidecar" = a small metadata file that lives ALONGSIDE the source file,
    # carrying the analysis results. This is a common pattern in scientific
    # imaging (ImageJ, CellProfiler, OMERO all use it).
    sidecar = {
        "source_file": source.name,
        "ingested_at": started_at,
        "completed_at": _utc_iso(),
        "incucyte_metadata": incucyte_meta.to_dict() if incucyte_meta else None,
        "segmentation": {
            "model": f"cellpose-{segmenter.model_type}",
            "device": result.device,
            "inference_ms": round(result.inference_ms, 2),
            "cell_count": result.cell_count,
        },
        "summary": summary,
        "per_cell": cells,
    }

    # STEP 5: write the sidecar as JSON. `source.stem` is the filename
    # without the extension, so "A172_..._1.tif" -> "A172_..._1.json".
    sidecar_path = outgoing_dir / f"{source.stem}.json"
    sidecar_path.write_text(json.dumps(sidecar, indent=2))

    # STEP 6: move the source into `processed/` so we don't process it again.
    # If we left it in `incoming/`, the next poll would see it and re-segment.
    shutil.move(str(source), str(processed_dir / source.name))

    return sidecar


def _scan(incoming: Path) -> list[Path]:
    """Return TIFF/PNG/JPEG files currently sitting in the incoming folder."""
    # Filter and sort so we process files in a stable order (roughly by name,
    # which for Incucyte-format filenames sorts by timestamp naturally).
    return sorted(
        p for p in incoming.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )


# =============================================================================
# THE MAIN LOOP
# =============================================================================
# This is the daemon body. Same "load model once, use many times" pattern
# as the FastAPI service, but driven by a polling loop instead of an
# event-driven web framework.

def run_watcher(
    incoming: Path,
    outgoing: Path,
    processed: Path,
    poll_seconds: float = 1.5,
    once: bool = False,
) -> None:
    """Main loop. Set once=True to do a single pass and exit (useful for tests)."""

    # Make sure all three folders exist. `parents=True` also creates any
    # missing parent directories; `exist_ok=True` doesn't complain if they
    # already exist.
    for folder in (incoming, outgoing, processed):
        folder.mkdir(parents=True, exist_ok=True)

    # STARTUP: load the model. This is the same ~10-second cost as when the
    # FastAPI service starts. We pay it once here.
    print(f"[{_utc_iso()}] Loading segmentation model...", flush=True)
    segmenter = CellSegmenter()
    print(f"[{_utc_iso()}] Model ready on device: {segmenter.device}", flush=True)
    print(f"[{_utc_iso()}] Watching: {incoming}", flush=True)
    print(f"[{_utc_iso()}] Writing results to: {outgoing}", flush=True)

    # POLLING LOOP: scan the folder, process anything found, sleep briefly,
    # repeat. `once=True` short-circuits after one pass — useful for tests.
    while True:
        pending = _scan(incoming)
        for source in pending:
            try:
                result = _process_one(source, outgoing, processed, segmenter)
                cells = result["segmentation"]["cell_count"]
                ms = result["segmentation"]["inference_ms"]
                print(
                    f"[{_utc_iso()}] {source.name}: {cells} cells, {ms} ms",
                    flush=True,
                )
            except Exception as exc:
                # A failure on one file should not stop the watcher. Print
                # the error to stderr, leave the file in `incoming/` for
                # inspection, and move on to the next file.
                print(
                    f"[{_utc_iso()}] FAILED {source.name}: {exc}",
                    file=sys.stderr,
                    flush=True,
                )
                # Don't move the file on failure — leave it for inspection.

        if once:
            return
        time.sleep(poll_seconds)


# =============================================================================
# CLI ENTRY POINT
# =============================================================================
# argparse turns command-line flags into Python arguments. This gives the
# watcher a real CLI so it can be run in different configurations without
# editing code:
#     py watcher.py --incoming custom/folder --poll-seconds 5
#     py watcher.py --once                          # for testing

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    project_root = Path(__file__).resolve().parent.parent
    parser.add_argument("--incoming", type=Path, default=project_root / "data" / "incoming")
    parser.add_argument("--outgoing", type=Path, default=project_root / "data" / "outgoing")
    parser.add_argument("--processed", type=Path, default=project_root / "data" / "processed")
    parser.add_argument("--poll-seconds", type=float, default=1.5)
    parser.add_argument("--once", action="store_true",
                        help="Process whatever is in incoming/ then exit. Used for tests.")
    args = parser.parse_args()

    # Wrap the main loop so Ctrl+C prints a clean "Stopped." instead of a
    # KeyboardInterrupt traceback.
    try:
        run_watcher(
            incoming=args.incoming,
            outgoing=args.outgoing,
            processed=args.processed,
            poll_seconds=args.poll_seconds,
            once=args.once,
        )
    except KeyboardInterrupt:
        print(f"\n[{_utc_iso()}] Stopped.", flush=True)
    return 0


# The classic "run this file directly" guard. When this script is IMPORTED
# by another file, `__name__` becomes "watcher". When it's RUN directly
# (e.g. `py watcher.py`), `__name__` becomes "__main__" and this block
# fires, calling main().
if __name__ == "__main__":
    sys.exit(main())
