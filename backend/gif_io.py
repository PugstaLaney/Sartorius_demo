"""
GIF -> list of PNG bytes (and back, when we need to produce a result GIF).

Animated GIFs are how the WPF console sends multi-frame inputs to the backend.
We extract each frame as a PNG so the rest of the pipeline can treat them
as ordinary images.
"""

from __future__ import annotations


# =============================================================================
# IMPORTS
# =============================================================================
# PIL's ImageSequence provides an iterator over the frames of an animated
# image (GIF, animated PNG, animated WebP). We use it to walk the frames.

import io

from PIL import Image, ImageSequence


# =============================================================================
# EXTRACT: split an animated GIF into per-frame PNG bytes
# =============================================================================
# The trick here is the RGB conversion. GIF frames are palette-indexed (each
# pixel is an index into a small color table). Cellpose expects standard RGB
# arrays, so we re-encode each frame as an RGB PNG before returning it.

def extract_frames(gif_bytes: bytes) -> list[bytes]:
    """
    Split an animated GIF into a list of frame PNGs (as raw bytes).

    We re-encode each frame as RGB PNG rather than keep it as a palette-indexed
    GIF frame because the downstream segmenter expects standard RGB images,
    and palette artifacts confuse Cellpose.
    """
    # Open the GIF from the in-memory bytes.
    gif = Image.open(io.BytesIO(gif_bytes))
    frames: list[bytes] = []

    # Iterate through every frame. ImageSequence.Iterator handles seeking
    # through the GIF's timeline for us.
    for frame in ImageSequence.Iterator(gif):
        # `.convert("RGB")` flattens the palette and discards transparency.
        # After this, `rgb` is a standard 3-channel PIL image.
        rgb = frame.convert("RGB")

        # Serialize as PNG bytes. Downstream code can pass these directly to
        # `CellSegmenter.segment(bytes)` as if they came from an upload.
        buf = io.BytesIO()
        rgb.save(buf, format="PNG")
        frames.append(buf.getvalue())

    return frames


# =============================================================================
# ASSEMBLE: pack a list of PIL images back into one animated GIF
# =============================================================================
# Reverse of `extract_frames`. Not currently used by the running service
# (the WPF cache handles animation client-side), but kept here for future
# use if we ever want to return a downloadable result GIF.

def assemble_gif(frame_images: list[Image.Image], duration_ms: int = 400) -> bytes:
    """
    Pack a list of PIL Images back into a single animated GIF (raw bytes).
    Used when the backend wants to return a downloadable result animation.
    """
    if not frame_images:
        return b""

    # PIL's save-with-save_all pattern is how you write animated GIFs.
    # The FIRST image is the target of `.save()`; the rest are passed as
    # `append_images`. `duration` sets per-frame display time in ms.
    # `loop=0` means loop forever; `loop=1` would play once and stop.
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
