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

import re
from dataclasses import asdict, dataclass
from pathlib import Path


# Two ways to read this regex:
#   1. Six underscore-separated tokens, last one is the crop index before .tif
#   2. The fourth-to-last token is a timestamp in the form NNdNNhNNm
TIMESTAMP_PATTERN = r"\d+d\d+h\d+m"

FILENAME_RE = re.compile(
    r"^(?P<cell_type>[^_]+)_"
    r"(?P<modality>[^_]+)_"
    r"(?P<well>[^_]+)_"
    r"(?P<location>[^_]+)_"
    rf"(?P<timestamp>{TIMESTAMP_PATTERN})_"
    r"(?P<crop>\d+)"
    r"\.(?:tif|tiff|png|jpg|jpeg)$",
    re.IGNORECASE,
)


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
        return asdict(self)


def parse(filename: str | Path) -> IncucyteMetadata | None:
    """
    Parse an Incucyte filename. Returns None if the filename does not match
    the expected convention — caller should treat this as a non-Incucyte file
    and process it with empty metadata.
    """
    name = Path(filename).name
    match = FILENAME_RE.match(name)
    if not match:
        return None
    return IncucyteMetadata(**match.groupdict())
