"""
Folder watcher — the file-drop ingestion path.

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
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image

from incucyte_filename import parse as parse_incucyte
from inference import CellSegmenter
from morphology import per_cell_stats, summary_stats


# File extensions we treat as candidate images.
IMAGE_EXTENSIONS = {".tif", ".tiff", ".png", ".jpg", ".jpeg"}


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

    image_bytes = source.read_bytes()
    result = segmenter.segment(image_bytes)

    # Convert numpy types to plain Python so the dict serializes cleanly.
    cells = per_cell_stats(result.mask)
    summary = summary_stats(cells)

    incucyte_meta = parse_incucyte(source.name)

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

    sidecar_path = outgoing_dir / f"{source.stem}.json"
    sidecar_path.write_text(json.dumps(sidecar, indent=2))

    # Move the source so we don't process it again on the next poll.
    shutil.move(str(source), str(processed_dir / source.name))

    return sidecar


def _scan(incoming: Path) -> list[Path]:
    """Return TIFF/PNG/JPEG files currently sitting in the incoming folder."""
    return sorted(
        p for p in incoming.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )


def run_watcher(
    incoming: Path,
    outgoing: Path,
    processed: Path,
    poll_seconds: float = 1.5,
    once: bool = False,
) -> None:
    """Main loop. Set once=True to do a single pass and exit (useful for tests)."""
    for folder in (incoming, outgoing, processed):
        folder.mkdir(parents=True, exist_ok=True)

    print(f"[{_utc_iso()}] Loading segmentation model...", flush=True)
    segmenter = CellSegmenter()
    print(f"[{_utc_iso()}] Model ready on device: {segmenter.device}", flush=True)
    print(f"[{_utc_iso()}] Watching: {incoming}", flush=True)
    print(f"[{_utc_iso()}] Writing results to: {outgoing}", flush=True)

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
                print(
                    f"[{_utc_iso()}] FAILED {source.name}: {exc}",
                    file=sys.stderr,
                    flush=True,
                )
                # Don't move the file on failure — leave it for inspection.

        if once:
            return
        time.sleep(poll_seconds)


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


if __name__ == "__main__":
    sys.exit(main())
