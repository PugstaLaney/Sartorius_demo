# Architecture Decisions

Read this when you want the **why** behind every choice in this project. Each section is something you can directly turn into an interview talking point.

## Why FastAPI instead of Streamlit

**Streamlit** is a fast way to ship a data science demo. It bundles the UI and the inference logic into one process, runs a websocket between the browser and the Python kernel, and re-runs your whole script on every interaction.

**FastAPI** decouples the model service from the UI. The model lives behind an HTTP endpoint. Any client — a browser, a mobile app, an instrument running C++, another microservice — can call it the same way.

For an *AI Deployment Engineer* role at a company that ships instruments into customer environments, the decoupled pattern is the one they live with daily. Streamlit would say "I built a prototype." FastAPI says "I built a service."

## Why load the model in a lifespan hook

```python
@asynccontextmanager
async def lifespan(app: FastAPI):
    global SEGMENTER
    SEGMENTER = CellSegmenter()
    yield
```

The model takes ~5-10 seconds to load and pin into GPU memory. If we loaded it per request, every user-facing latency would be dominated by reload time. Lifespan startup means **the first request is already fast** because the model is warm in VRAM.

This is the single most important deployment-engineering pattern in this codebase. Memorize it.

## Why a separate inference.py module

`main.py` knows about HTTP, file uploads, and JSON. `inference.py` knows about PyTorch tensors, Cellpose, and CUDA. Neither imports anything from the other except a small dataclass.

This separation means we can:
- Swap Cellpose for a custom-trained U-Net without touching the API surface
- Test inference in a notebook by importing `CellSegmenter` directly (no HTTP needed)
- Eventually move the inference to a separate process (different container, different machine) without rewriting routes

This is the **service boundary**. It's the same pattern Sartorius would use to ship a model onto an instrument and call it from a workstation.

## Why we return a base64-encoded PNG instead of raw mask data

A raw instance mask is a 2D array of integers (e.g., 1024×1024×4 bytes = 4 MB). Sending that over the network as JSON would be slow and brittle.

A PNG of the colored overlay is typically 50-200 KB and the browser can render it directly. We do the colorization server-side because we already have the mask in memory there.

Tradeoff: the client cannot recover the original cell IDs from the PNG. If we needed per-cell analytics on the client, we'd return a separate JSON array of cell metadata. We don't need that for this demo.

## Why the LATENCY_WINDOW deque

```python
LATENCY_WINDOW: deque[float] = deque(maxlen=100)
```

A rolling window of the last N latencies is the simplest possible observability surface. It costs almost nothing (constant memory, O(1) appends), and it lets `/metrics` answer "is the service fast right now?" without writing to a metrics backend.

In production at Sartorius this would be Prometheus + Grafana. The principle is the same; the surface area is bigger.

## Why no Docker yet

Docker is the next step. The reason it is not in version 1 is that getting CUDA + PyTorch + Cellpose into a working Dockerfile on Windows requires Docker Desktop + WSL2 + the NVIDIA Container Toolkit, and each of those has its own setup story. We want a working baseline first, then containerize.

When we add Docker, the Dockerfile will be ~15 lines and the talking point becomes: "I packaged the entire service — model weights, dependencies, runtime — into a single artifact that can be deployed identically on any CUDA-capable host."

## Why Cellpose instead of training a U-Net from scratch

Two reasons:

1. **Time.** Training a U-Net on the Sartorius Kaggle dataset takes ~6-12 hours of GPU time plus all the setup. With three days to a Loopback interview, that time is better spent on Loopback prep.
2. **Honesty.** Saying "I used a pretrained model trained on biological imagery and deployed it as a service" is true and impressive. Saying "I trained my own segmentation model in three days" would invite questions you cannot yet answer.

The deployment architecture is identical either way. Post-Loopback, the upgrade path is: fine-tune a U-Net on the Sartorius Kaggle competition data, save the weights, swap them into `CellSegmenter`. The API surface does not change at all.

## What this project does NOT demonstrate yet

Honest gap list. Address as time allows:

- Container packaging (Docker + docker-compose)
- Continuous integration (GitHub Actions running a test on push)
- Model versioning / a model registry pattern
- Batch inference for throughput optimization
- Memory profiling on a constrained device
- Hardware-aware model compression (quantization, ONNX export)

Each of these is a real component of "MLOps" and "deployment to embedded systems." You don't need all of them to interview — but knowing which you have and which you don't is the difference between credible and not.
