# How an image flows through the code

The complete lifecycle of a single image from when it's dropped onto the WPF console to when the segmentation overlay appears back in the window. Trace this alongside the actual source files ([backend/main.py](../backend/main.py), [backend/inference.py](../backend/inference.py), [backend/morphology.py](../backend/morphology.py), and [frontend_wpf/MainWindow.xaml.cs](../frontend_wpf/MainWindow.xaml.cs)) — reading them side by side is how the architecture stops feeling abstract.

---

```
STEP 1 — The image is on your hard disk.
   cells_demo.png  (~130 KB PNG on the SSD)

STEP 2 — WPF reads it into RAM as bytes.
   MainWindow.xaml.cs opens the file, reads all bytes into a
   byte[] array (still ~130 KB, just in RAM now).

STEP 3 — WPF sends the bytes over HTTP.
   The bytes travel across localhost to port 8000. From Python's
   perspective, they arrive as an UploadFile object.

STEP 4 — main.py reads the upload into a Python bytes object.
   image_bytes = await file.read()
   → image_bytes is now a Python `bytes` object holding those ~130 KB.

STEP 5 — main.py calls the segmenter.
   result = SEGMENTER.segment(image_bytes)
   → Python enters CellSegmenter.segment method.
   → Inside the method, `self` is SEGMENTER (already-built object)
     and `image_bytes` is the bytes buffer we just handed over.

STEP 6 — Inside segment(): bytes → PIL image → numpy array.
   img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
       ↑ PIL parses the PNG header + pixel data into an Image object.
       ↑ Height × Width × 3 (RGB), stored in a lightweight PIL wrapper.

   arr = np.array(img)
       ↑ Copies pixel data into a numpy ndarray of shape (H, W, 3)
         and dtype uint8. For a 512×512 image that's ~800 KB in RAM.

STEP 7 — Inside segment(): numpy array → Cellpose → mask.
   masks, _, _, _ = self.model.eval([arr], channels=[0,0], diameter=None)
       ↑ self.model is the already-loaded Cellpose model in VRAM.
       ↑ Cellpose copies the numpy array up to GPU VRAM,
         runs the U-Net forward pass on the GPU (~600 ms),
         computes flow field, clusters pixels into instances,
         copies the resulting mask back down to RAM.
       ↑ masks is now a list containing one 2D ndarray, shape (H, W),
         dtype int32 or similar, where each pixel value is either
         0 (background) or a cell ID 1..N.

STEP 8 — Inside segment(): package the result.
   return SegmentationResult(
       mask=masks[0],
       cell_count=int(masks[0].max()),
       inference_ms=elapsed_ms,
       device=self.device,
   )
       ↑ Creates a small dataclass wrapper holding references
         to the mask and the metadata. The mask itself is not
         copied; the dataclass just holds a pointer to it.

STEP 9 — Back up in main.py, we do more work with the result.
   cells = per_cell_stats(result.mask)     # morphology.py chews the mask
   summary = summary_stats(cells)
   mask_png_base64 = _mask_to_png_base64(result.mask)   # colorize + encode

STEP 10 — main.py builds a JSON response.
   return JSONResponse({
       "cell_count": result.cell_count,
       "inference_ms": ...,
       "device": ...,
       "mask_png_base64": mask_png_base64,
       "per_cell": cells,
       "summary": summary,
   })

STEP 11 — FastAPI serializes JSON, sends response bytes back over HTTP.

STEP 12 — WPF receives JSON, decodes the base64 PNG into a
   BitmapImage, and draws it on the right-hand panel.

STEP 13 — Everything the image WAS in Python (the bytes, the PIL
   Image, the numpy array, the mask, the SegmentationResult) goes
   out of scope and gets garbage-collected. RAM is reclaimed.
   The Cellpose model in VRAM stays, ready for the next request.
```

---

## Key mental-model takeaways

- **The model never moves.** Cellpose weights get pinned to GPU VRAM once at startup and stay there for the life of the process. Every request reuses the same in-memory model.
- **The image data moves and transforms constantly.** Bytes → PIL Image → numpy array → GPU tensor → mask array → dataclass → JSON. Each step is one transformation into the shape the next step expects.
- **Two languages, two processes, one workflow.** WPF (C# / .NET) and the backend (Python / FastAPI) are separate programs. Their only shared vocabulary is HTTP + JSON. Neither knows how the other is implemented.
- **`SEGMENTER` is a long-lived object.** The `SEGMENTER = CellSegmenter()` line in [backend/main.py](../backend/main.py) runs *once* when the server starts. Every request reuses that same instance — that's why `segment()` runs in ~650 ms instead of ~10 seconds.
- **Garbage collection frees per-request memory.** After each request, the temporary bytes, arrays, and masks fall out of scope and Python reclaims that RAM automatically. Only the long-lived state (`SEGMENTER`, the model in VRAM) persists.
