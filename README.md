# Sartorius Cell Segmentation

A cell segmentation and tracking service designed around Sartorius Incucyte-style live-cell imaging workflows. A PyTorch/Cellpose model is deployed behind a FastAPI HTTP service and consumed by a Windows desktop console written in C# / WPF. The project also mirrors Incucyte's file-drop integration pattern via a folder-watching daemon, and supports frame-by-frame cell tracking on time-lapse GIF input.

Explores how research ML models get productionized onto scientific instruments. Informed by hands-on use of an Incucyte in a previous research lab.

---

## What it does

Two ingest paths, one shared inference engine:

| Path | Trigger | Output |
|---|---|---|
| **HTTP** | Drop image or GIF into the WPF console → click **Run** | Overlay image, per-cell morphology table, latency + device metrics, and (for GIFs) a scrubbable frame slider |
| **File-drop watcher** | Drop TIFF/PNG files into `data/incoming/` (simulates an instrument writing to a network share) | JSON sidecar files in `data/outgoing/` with parsed Incucyte metadata, segmentation, and per-cell measurements |

For animated GIFs the service performs **cell tracking** across frames using Hungarian assignment on a cost matrix combining centroid distance and area similarity, so each cell keeps a stable ID as it moves through time.

---

## Architecture

```
┌─────────────────────────────────────┐         HTTP           ┌────────────────────────────────────┐
│  WPF desktop console (C# / .NET 8)  │  ─────────────────────► │  FastAPI service (Python 3.11)     │
│  frontend_wpf/                       │        JSON + bytes     │  backend/                          │
│  Drag-drop, slider, per-cell table  │  ◄─────────────────────  │  Cellpose + PyTorch + CUDA         │
└─────────────────────────────────────┘                          └────────────────────────────────────┘
                                                                          ▲
                                                                          │ file drop
                                                                          │
                                                     ┌────────────────────────────────────┐
                                                     │  Folder watcher                     │
                                                     │  (mirrors Incucyte v2025A          │
                                                     │   Auto-Archive-to-network-share)   │
                                                     └────────────────────────────────────┘
```

The two frontends and the file-drop watcher all share **one loaded model** — Cellpose is loaded once at startup and reused across every request. Loading the model per request would cost ~10 seconds each time; reusing it keeps inference at ~650 ms on an RTX 3060.

---

## Project structure

```
Sartorius_Cell_Segmentation/
├── backend/                        The running service
│   ├── main.py                     FastAPI app: routes, CORS, file uploads
│   ├── inference.py                Cellpose model wrapper (loaded once at startup)
│   ├── morphology.py               Per-cell measurements (area, perimeter, eccentricity, solidity, centroid)
│   ├── tracker.py                  Hungarian-assignment cell tracker for time-lapse
│   ├── track_visualization.py      Overlay rendering: cells colored by track ID + motion trails
│   ├── gif_io.py                   Split animated GIF into per-frame PNGs
│   ├── incucyte_filename.py        Regex parser for Incucyte's TIFF filename convention
│   ├── watcher.py                  Folder-drop ingestion daemon
│   └── requirements.txt            Python dependencies
│
├── frontend_wpf/                   C# / .NET 8 WPF desktop console
│   ├── MainWindow.xaml             Window layout: panels, slider, per-cell table
│   ├── MainWindow.xaml.cs          Behavior: drag-drop, HTTP calls, slider scrubbing
│   └── SegmentationConsole.csproj  .NET project file
│
├── frontend/                       Minimal HTML/JS frontend (lighter alternative to WPF)
│   ├── index.html
│   ├── app.js
│   └── style.css
│
├── dev_scripts/                    Developer utilities — NOT part of the running service
│   ├── smoke_test.py               End-to-end sanity check
│   ├── prepare_demo_drop.py        Simulate an instrument writing files to data/incoming/
│   ├── synthetic_timelapse.py      Generate a controlled-drift time-lapse from one image
│   └── process_timelapse.py        CLI version of the tracking pipeline
│
├── notebooks/
│   └── 01_cellpose_basics.ipynb    Walkthrough: tensors → model → masks → metrics
│
├── data/                           Sample images, drop folders, generated time-lapses
├── docs/
│   └── architecture.md             Design decisions and tradeoffs
│
├── Launch Sartorius Demo.cmd       Double-click launcher (opens backend + WPF window)
├── run_wpf.ps1                     What the launcher invokes
└── README.md                       This file
```

**The line between `backend/` and `dev_scripts/`**: if the running service depends on it, it belongs in `backend/`. If a developer runs it by hand to generate test data or spot-check something, it belongs in `dev_scripts/`. Nothing in `dev_scripts/` is imported by the running service.

---

## How the pieces fit together

Module dependencies inside the Python codebase. Arrows point from importer to imported.

```
                         ┌──────────────────────┐
                         │  main.py              │  FastAPI app — the entry point for HTTP traffic
                         │  (HTTP endpoints)     │
                         └────────┬──────────────┘
                                  │
                ┌─────────────────┼──────────────────┬────────────────────┬──────────────┐
                │                 │                  │                    │              │
                ▼                 ▼                  ▼                    ▼              ▼
      ┌─────────────────┐ ┌──────────────┐ ┌────────────────┐ ┌────────────────────┐ ┌──────────┐
      │  inference.py   │ │morphology.py │ │  tracker.py    │ │track_visualization │ │ gif_io.py│
      │  CellSegmenter  │ │per_cell_stats│ │  Tracker       │ │ render_overlay     │ │ frames   │
      └─────────────────┘ └──────────────┘ └────────────────┘ └────────────────────┘ └──────────┘
                ▲                 ▲                  ▲                    ▲
                │                 │                  │                    │
                │                 │                  │                    │
                │                 │                  │                    │
                └─────────────────┴──────────────────┴────────────────────┘
                                        ▲
                         ┌──────────────┴─────────┐
                         │  watcher.py            │  Standalone daemon — separate entry point
                         │  Folder-drop ingestion │
                         └────────┬───────────────┘
                                  │
                                  ▼
                         ┌────────────────────┐
                         │incucyte_filename.py│
                         │  parse()           │
                         └────────────────────┘
```

Two things worth noticing:

1. **`inference.py`, `morphology.py`, `tracker.py`, `track_visualization.py`, `gif_io.py`, `incucyte_filename.py` are all leaf modules.** None of them import from another backend module. They only depend on third-party libraries (numpy, PIL, scipy, cellpose, etc.). This means any of them can be modified or tested in isolation without ripple effects.

2. **`main.py` and `watcher.py` are the two entry points.** They each compose the leaf modules to serve one workflow. If you want to add a third ingest path (e.g. a gRPC service, a CLI, a message-queue consumer), it would be a third file at their level, reusing the same leaf modules.

`dev_scripts/` follows the same principle: each script imports what it needs from `backend/` via a small `sys.path` shim at the top. Nothing in `dev_scripts/` imports from another `dev_scripts/` file.

---

## What happens on a single request

Tracing one click of **Run segmentation** in the WPF console, end to end:

```
1. User drops cells_demo.png onto the left panel, clicks Run segmentation.

2. WPF (frontend_wpf/MainWindow.xaml.cs):
   RunButton_Click → RunSingleImage(path)
   → Http.PostAsync("http://localhost:8000/segment", multipart_form_with_image)

3. Python (backend/main.py):
   @app.post("/segment") handler runs.
   image_bytes = await file.read()
   result = SEGMENTER.segment(image_bytes)         # SEGMENTER was created at startup

4. Python (backend/inference.py):
   CellSegmenter.segment(image_bytes)
   → Image.open(BytesIO(image_bytes)).convert("RGB")
   → self.model.eval([array], channels=[0,0])     # actual PyTorch forward pass on GPU
   → returns SegmentationResult(mask, cell_count, inference_ms, device)

5. Python (backend/main.py again):
   cells = per_cell_stats(result.mask)             # scikit-image regionprops
   summary = summary_stats(cells)
   response = {mask_png_base64, cell_count, inference_ms, device, per_cell, summary}
   return JSONResponse(response)

6. WPF (frontend_wpf/MainWindow.xaml.cs):
   Deserialize JSON → SegmentResponse object
   → ApplySingleResult(result)
   → Decode base64 PNG, display on right panel
   → Update badges, summary text, per-cell DataGrid
```

Total wall clock: ~700 ms steady-state on an RTX 3060.

The time-lapse path (`POST /track_timelapse`) does the same thing per frame in a loop, feeding each frame's per-cell data to a `Tracker` instance that assigns track IDs, and returns every frame's data in a single JSON response. The WPF caches that response and the frame slider reads from the cache — scrubbing is instant and does not touch the network.

---

## Getting started

### Prerequisites

- **Windows 10/11** (WPF is Windows-only)
- **Python 3.11** — the `py -3.11` launcher must resolve to it
- **.NET 8 SDK** — [download](https://dotnet.microsoft.com/en-us/download/dotnet/8.0)
- **NVIDIA GPU with CUDA 12.4** — CPU inference works but is ~10x slower

### Install

```powershell
# Create a Python venv OUTSIDE any OneDrive-synced folder
py -3.11 -m venv C:\Users\<you>\venvs\sartorius-cell

# Activate it and install torch with the CUDA 12.4 build first (separate index)
C:\Users\<you>\venvs\sartorius-cell\Scripts\python.exe -m pip install `
    torch==2.6.0+cu124 torchvision==0.21.0+cu124 `
    --index-url https://download.pytorch.org/whl/cu124

# Install the rest
C:\Users\<you>\venvs\sartorius-cell\Scripts\python.exe -m pip install `
    -r backend\requirements.txt
```

### Run

The one-click path: **double-click `Launch Sartorius Demo.cmd`**. It starts the backend, polls health, and opens the WPF window.

Manual paths, when you want to see each piece separately:

```powershell
# Backend only
cd backend
C:\Users\<you>\venvs\sartorius-cell\Scripts\python.exe -m uvicorn main:app --port 8000

# WPF console only (backend must be running)
cd frontend_wpf
dotnet run

# Interactive API docs (with backend running)
# Open in browser: http://localhost:8000/docs
```

### Verify everything works

```powershell
# From the project root:
C:\Users\<you>\venvs\sartorius-cell\Scripts\python.exe dev_scripts\smoke_test.py
```

Expected output: model loads on `cuda`, segmentation runs on `data/sample_images/cells_demo.png`, `cell count > 0`, `Smoke test PASSED`.

---

## The two ingest paths in more detail

### HTTP path — for interactive use

`POST /segment` accepts a single image and returns segmentation + morphology.
`POST /track_timelapse` accepts an animated GIF and returns per-frame data + track assignments + tracking summary.
`GET /health` returns 200 once the model is loaded.
`GET /metrics` returns a rolling window of recent request latencies.

The WPF console uses `/segment` for single images and `/track_timelapse` for GIFs, branching on file extension.

### File-drop path — for instrument-side deployment

`backend/watcher.py` runs as a daemon that polls `data/incoming/` for new TIFF/PNG files. For each file it:

1. Parses the filename with `incucyte_filename.py` (returns `None` if not Incucyte-formatted)
2. Runs segmentation
3. Writes a JSON sidecar to `data/outgoing/` with the parsed metadata + per-cell morphology
4. Moves the source into `data/processed/` to avoid re-processing

This mirrors the integration pattern Incucyte v2025A exposes via its Auto-Archive-to-designated-network-location feature: instrument writes files to a share, downstream tools read from the share, no HTTP API needed.

To try it locally:

```powershell
# Terminal 1: start the watcher
cd backend
C:\Users\<you>\venvs\sartorius-cell\Scripts\python.exe watcher.py

# Terminal 2: simulate an instrument by copying demo files into data/incoming/
cd ..
C:\Users\<you>\venvs\sartorius-cell\Scripts\python.exe dev_scripts\prepare_demo_drop.py --count 4

# Watch Terminal 1 process the files, then look at data/outgoing/ for the JSON sidecars
```

---

## Time-lapse cell tracking

The tracker links segmentation results across frames so each cell keeps a stable identity through time. Algorithm: **Hungarian assignment** (`scipy.optimize.linear_sum_assignment`) on a cost matrix combining centroid distance and area similarity. Same approach as ImageJ's TrackMate simple tracker.

To feel it work:

```powershell
# From the project root:
C:\Users\<you>\venvs\sartorius-cell\Scripts\python.exe dev_scripts\synthetic_timelapse.py
C:\Users\<you>\venvs\sartorius-cell\Scripts\python.exe dev_scripts\process_timelapse.py

# Open the result:
# data/timelapse/synthetic_run/_results/timelapse.gif
```

You can also drag a GIF straight into the WPF console. The service processes all frames server-side, returns everything in one response, and the slider scrubs an in-memory cache — no HTTP call per scrub tick.

---

## What to study first

For a new reader (including future-you) trying to understand the code:

1. **`notebooks/01_cellpose_basics.ipynb`** — teaching artifact. Tensors, U-Net, what Cellpose actually does. The foundation for everything else.
2. **`backend/inference.py`** — 80 lines. Model wrapper. Smallest and most self-contained piece.
3. **`backend/morphology.py`** — 80 lines. Pure math on masks. No framework noise.
4. **`backend/main.py`** — 250 lines. FastAPI service. Shows how the model wrapper becomes an HTTP service.
5. **`backend/tracker.py`** — 150 lines. Hungarian assignment for time-lapse tracking. Most algorithmically interesting file.
6. **`backend/watcher.py`** — 150 lines. Folder-drop daemon. Shows how the same model is reused from a different entry point.
7. **`frontend_wpf/MainWindow.xaml.cs`** — the C# side. Read once the Python is familiar.
8. **`docs/architecture.md`** — why each decision was made.

---

## Roadmap

Ideas that are consistent with the current architecture but not yet built:

- **Docker packaging** — a Dockerfile for the backend so the whole service ships as one artifact deployable on any CUDA-capable host.
- **Real time-lapse data** — swap the synthetic generator for a Cell Tracking Challenge or LIVECell download.
- **Detectron2 backend option** — swap Cellpose for a detectron2 model pretrained on LIVECell (Sartorius's own dataset), demonstrating that the model layer is truly swappable.
- **Gap closing in the tracker** — let a cell missed for 1-2 frames re-link when it reappears.
- **Per-track CSV export** — biologists want spreadsheets.
- **Model versioning** — expose the model version in `/health` and `/metrics`.
- **ONNX export** — for smaller, more portable deployment onto instrument hardware.

---

## Notes

- **Venv location**: Python's venv lives outside any OneDrive-synced folder because OneDrive's file-sync agent breaks CUDA .dll loading if it re-touches the files during install.
- **Torch version pinning**: `torch==2.6.0+cu124` is pinned because newer torch versions error on import on this specific Windows + CUDA combination.
- **CPython bytecode cache** (`__pycache__/`): auto-generated by Python, excluded from git, safe to delete anytime.
- **CORS**: the backend allows all origins in dev. In a real deployment this would be locked down to the known frontend host.
