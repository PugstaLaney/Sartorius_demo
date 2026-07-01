"""
GIF -> list of PNG bytes (and back, when we need to produce a result GIF).

Animated GIFs are how the WPF console sends multi-frame inputs to the backend.
We extract each frame as a PNG so the rest of the pipeline can treat them
as ordinary images.
"""

from __future__ import annotations

import io

from PIL import Image, ImageSequence


def extract_frames(gif_bytes: bytes) -> list[bytes]:
    """
    Split an animated GIF into a list of frame PNGs (as raw bytes).

    We re-encode each frame as RGB PNG rather than keep it as a palette-indexed
    GIF frame because the downstream segmenter expects standard RGB images,
    and palette artifacts confuse Cellpose.
    """
    gif = Image.open(io.BytesIO(gif_bytes))
    frames: list[bytes] = []
    for frame in ImageSequence.Iterator(gif):
        # `.convert("RGB")` flattens the palette and discards transparency.
        rgb = frame.convert("RGB")
        buf = io.BytesIO()
        rgb.save(buf, format="PNG")
        frames.append(buf.getvalue())
    return frames


def assemble_gif(frame_images: list[Image.Image], duration_ms: int = 400) -> bytes:
    """
    Pack a list of PIL Images back into a single animated GIF (raw bytes).
    Used when the backend wants to return a downloadable result animation.
    """
    if not frame_images:
        return b""
    buf = io.BytesIO()
    frame_images[0].save(
        buf,
        format="GIF",
        save_all=True,
        append_images=frame_images[1:],
        duration=duration_ms,
        loop=0,
        optimize=False,
    )
    return buf.getvalue()
