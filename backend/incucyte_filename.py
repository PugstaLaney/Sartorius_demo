"""
Parser for Incucyte's exported TIFF filename convention.

Format observed in the LIVECell dataset (which was captured on Incucyte HD):
    <CellType>_<Modality>_<Well>_<Location>_<Timestamp>_<Crop>.tif

Example:
    A172_Phase_C7_1_03d04h00m_1.tif
      ^    ^   ^  ^   ^         ^
      |    |   |  |   |         crop index within the well
      |    |   |  |   timestamp (days/hours/minutes since acquisition start)
      |    |   |  field-of-view / location within the well
      |    |   well ID on the plate
      |    imaging modality (Phase, Fluor, etc.)
      cell type / sample name

We intentionally parse leniently — Incucyte filename conventions have varied
across software versions, so we return None instead of throwing when the
pattern does not match. Callers can treat None as "unknown source" and fall
back to processing the file with empty metadata.
"""

from __future__ import annotations


# =============================================================================
# IMPORTS
# =============================================================================

import re                                    # regular expressions for pattern matching
from dataclasses import asdict, dataclass    # dataclass + a helper to convert to dict
from pathlib import Path                     # for accepting either str or Path filenames


# =============================================================================
# THE REGEX
# =============================================================================
# Regular expressions are hard to read the first time. Two mental models for
# this one:
#   1. Six underscore-separated tokens, last token is the crop index before
#      the file extension.
#   2. The fourth-to-last token is a timestamp in the form NNdNNhNNm.
#
# We use NAMED CAPTURE GROUPS (`(?P<name>...)`) so we can pull each token out
# of the match by name later, rather than by position. Named groups also make
# the regex self-documenting.

TIMESTAMP_PATTERN = r"\d+d\d+h\d+m"   # e.g. "03d04h00m" - digits d digits h digits m

FILENAME_RE = re.compile(
    r"^(?P<cell_type>[^_]+)_"                   # e.g. "A172" (no underscores allowed inside)
    r"(?P<modality>[^_]+)_"                     # e.g. "Phase"
    r"(?P<well>[^_]+)_"                         # e.g. "C7"
    r"(?P<location>[^_]+)_"                     # e.g. "1"
    rf"(?P<timestamp>{TIMESTAMP_PATTERN})_"     # e.g. "03d04h00m"
    r"(?P<crop>\d+)"                            # e.g. "1"
    r"\.(?:tif|tiff|png|jpg|jpeg)$",            # extension check (case-insensitive below)
    re.IGNORECASE,
)


# =============================================================================
# THE PARSED METADATA CONTAINER
# =============================================================================
# Same dataclass pattern as SegmentationResult in inference.py — hold a bundle
# of named values with auto-generated __init__ and __repr__.

@dataclass
class IncucyteMetadata:
    """Parsed components of an Incucyte filename."""
    cell_type: str
    modality: str
    well: str
    location: str
    timestamp: str       # raw token, e.g. "03d04h00m"
    crop: str

    def to_dict(self) -> dict:
        # asdict() is the dataclass helper that converts self into a plain
        # dict. Useful when we need to serialize this to JSON.
        return asdict(self)


# =============================================================================
# THE PARSE FUNCTION
# =============================================================================
# Accepts either a bare filename string OR a pathlib.Path (from which we
# extract just the name portion). Returns IncucyteMetadata on success or
# None on failure — the "None means unknown" pattern is easier for the
# caller to handle than exceptions when filenames vary a lot.

def parse(filename: str | Path) -> IncucyteMetadata | None:
    """
    Parse an Incucyte filename. Returns None if the filename does not match
    the expected convention — caller should treat this as a non-Incucyte file
    and process it with empty metadata.
    """
    # Path(...).name pulls just the filename off, so callers can pass
    # either a full path ("C:\...\A172_Phase_...tif") or a bare filename.
    name = Path(filename).name

    match = FILENAME_RE.match(name)
    if not match:
        return None

    # `match.groupdict()` returns {"cell_type": "A172", "modality": "Phase", ...}
    # which we splat (`**`) into IncucyteMetadata's constructor as keyword args.
    return IncucyteMetadata(**match.groupdict())
