"""
Drop a batch of demo image files into data/incoming/ with Incucyte-style
filenames so the folder-watcher has something to watch.

The watcher's filename parser expects:
    <CellType>_<Modality>_<Well>_<Location>_<Timestamp>_<Crop>.tif

We don't have a real Incucyte instrument writing to a network share, so this
script fakes one: it copies the existing sample image (data/sample_images/
cells_demo.png) into data/incoming/ under several plausible Incucyte filenames.
Drop a real instrument in front of the same `incoming/` directory and nothing
about the watcher would change.

Usage:
    py prepare_demo_drop.py            # default: drop 4 files
    py prepare_demo_drop.py --count 8

Why not download real LIVECell images here? Because they ship as 1.2 GB zips
on S3 — too heavy for a quick demo. The download_livecell_full.py script (NOT
written yet — see README's "Phase 2 stretch") handles that case for users
who want to fine-tune or test on actual Sartorius-published data.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

# A few sample (cell_type, well, location, timestamp) tuples drawn from the
# LIVECell paper's cell lines and Incucyte's typical well-plate addressing.
DEMO_RECIPES = [
    ("A172",   "Phase", "C7", "1", "00d00h00m", "1"),
    ("A172",   "Phase", "C7", "1", "00d12h00m", "1"),  # later timepoint, same well
    ("BV2",    "Phase", "B3", "2", "00d00h00m", "1"),
    ("MCF7",   "Phase", "D5", "1", "01d00h00m", "1"),
    ("Huh7",   "Phase", "E2", "3", "00d06h00m", "1"),
    ("SHSY5Y", "Phase", "F8", "1", "00d00h00m", "2"),
    ("SkBr3",  "Phase", "G4", "2", "00d18h00m", "1"),
    ("BT474",  "Phase", "A1", "1", "02d00h00m", "1"),
]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    project_root = Path(__file__).resolve().parent.parent
    parser.add_argument(
        "--source",
        type=Path,
        default=project_root / "data" / "sample_images" / "cells_demo.png",
        help="Image to clone into the drop folder.",
    )
    parser.add_argument(
        "--incoming",
        type=Path,
        default=project_root / "data" / "incoming",
        help="Folder the watcher monitors.",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=4,
        help=f"How many files to drop (max {len(DEMO_RECIPES)}).",
    )
    args = parser.parse_args()

    if not args.source.exists():
        print(f"Source image not found: {args.source}", file=sys.stderr)
        print("Run smoke_test.py first to download the bundled sample.", file=sys.stderr)
        return 1

    args.incoming.mkdir(parents=True, exist_ok=True)

    count = min(args.count, len(DEMO_RECIPES))
    print(f"Dropping {count} files into {args.incoming}", flush=True)

    for recipe in DEMO_RECIPES[:count]:
        cell_type, modality, well, location, timestamp, crop = recipe
        # Keep the original file extension so we don't lie about the format.
        filename = f"{cell_type}_{modality}_{well}_{location}_{timestamp}_{crop}{args.source.suffix}"
        dest = args.incoming / filename
        shutil.copy2(args.source, dest)
        print(f"  {filename}", flush=True)

    backend_dir = Path(__file__).resolve().parent.parent / "backend"
    print(f"\nIf watcher.py is already running, look at its terminal - it should")
    print(f"be processing these files within ~2 seconds.")
    print(f"\nJSON sidecars will appear in: {args.incoming.parent / 'outgoing'}")
    print(f"Processed source files move to: {args.incoming.parent / 'processed'}")
    print(f"\nIf the watcher is NOT running yet, start it with:")
    print(f"  cd {backend_dir}")
    print(f"  C:\\Users\\palla\\venvs\\sartorius-cell\\Scripts\\python.exe watcher.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
