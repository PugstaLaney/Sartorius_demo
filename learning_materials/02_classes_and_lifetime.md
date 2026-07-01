# Classes, instances, and how long they live

If you're coming from data analytics, you've been using classes forever without writing them. Every time you did `df = pd.read_csv(...)` and then `df.head()`, you were using a class. This doc unpacks what's actually going on so you can read (and eventually write) code like [backend/inference.py](../backend/inference.py) without any of it feeling like magic.

---

## The class / instance / method distinction

A **class** is a recipe. An **instance** is what you get when you follow the recipe once. A **method** is a step in the recipe you can invoke on any instance.

| Term | Pandas example | Our code |
|---|---|---|
| Class | `pd.DataFrame` | `CellSegmenter` |
| Instance | `df = pd.DataFrame(data)` | `seg = CellSegmenter()` |
| Method | `df.head()` | `seg.segment(image_bytes)` |
| Attribute | `df.columns` | `seg.device` |

There is one class definition. There can be many instances of that class, each holding its own state. `df1 = pd.DataFrame(...)` and `df2 = pd.DataFrame(...)` are two separate instances that don't share data.

---

## `__init__` is just the setup function

When you write `seg = CellSegmenter()`, Python does these things invisibly:

1. Allocate memory for a fresh, empty `CellSegmenter` instance.
2. Call its `__init__` method with any arguments you passed.
3. Whatever `__init__` does (assigning to `self.model`, `self.device`, etc.) stores those values on the new instance.
4. Hand the now-set-up instance back to you and bind it to the name `seg`.

`__init__` is nothing more than **the function that runs once, automatically, when you create a new instance**. Same idea as pandas's `DataFrame.__init__` running when you write `pd.DataFrame(some_data)` — it's what turns the raw data you passed into a working DataFrame with `columns`, `index`, `dtypes`, etc.

Read the `__init__` of [inference.py](../backend/inference.py) alongside this and it should now look mechanical:

```python
def __init__(self, model_type: str = "cyto3", use_gpu: bool | None = None):
    if use_gpu is None:
        use_gpu = torch.cuda.is_available()
    self.device = "cuda" if use_gpu else "cpu"
    self.model = models.Cellpose(gpu=use_gpu, model_type=model_type)
    self.model_type = model_type
```

Translated: "when somebody builds a new CellSegmenter, decide whether to use GPU, store the device string, load the Cellpose model into memory, and store both on the new instance."

---

## `self` is just "this particular instance"

When you write `seg.segment(bytes)`, Python turns that into `CellSegmenter.segment(seg, bytes)`. The instance you called the method on becomes the first argument to the method. Inside the function definition, that argument is called `self` by universal convention:

```python
def segment(self, image_bytes: bytes) -> SegmentationResult:
    #        ^
    #        this is "the specific instance the user called .segment() on"
    ...
    masks, _, _, _ = self.model.eval([arr], channels=[0, 0])
    #                ^^^^^^^^^^
    #                the model attribute of THAT specific instance
```

If we had two different `CellSegmenter` instances (one with `cyto3` and one with `nuclei` weights, say), each would remember its own `self.model`. When you called `.segment()` on one, `self` would be that one, and `self.model` would point to *its* model, not the other's.

You could rename `self` to anything else and Python wouldn't care — it's just a variable name. Everyone uses `self` because that's what every Python developer expects.

---

## `@dataclass` is a shortcut for "container of values"

Look at `SegmentationResult` in inference.py:

```python
@dataclass
class SegmentationResult:
    mask: np.ndarray
    cell_count: int
    inference_ms: float
    device: str
```

Without `@dataclass`, you'd have to write this by hand:

```python
class SegmentationResult:
    def __init__(self, mask, cell_count, inference_ms, device):
        self.mask = mask
        self.cell_count = cell_count
        self.inference_ms = inference_ms
        self.device = device

    def __repr__(self):
        return f"SegmentationResult(mask=..., cell_count={self.cell_count}, ...)"

    def __eq__(self, other):
        return (self.mask == other.mask).all() and self.cell_count == other.cell_count and ...
```

`@dataclass` writes all that boilerplate for you at class-creation time. The `:` lines aren't variable assignments — they're **field declarations**, like columns in a table. Python reads them, generates the `__init__` and `__repr__` and `__eq__` methods automatically, and you get a clean object with autocomplete on `.mask`, `.cell_count`, etc.

**The rule of thumb**: use `@dataclass` when you would otherwise pass a dict around. Dicts have no type hints, no autocomplete, and let you typo keys. Dataclasses fix all three.

---

## The two-phase lifetime

Classes let us split a program into "expensive setup that runs once" and "cheap operations that run many times." This is the core deployment-engineering pattern in the code.

**Phase 1 — startup.** In [backend/main.py](../backend/main.py):

```python
@asynccontextmanager
async def lifespan(app: FastAPI):
    global SEGMENTER
    SEGMENTER = CellSegmenter()   # __init__ runs here. Slow (~10s).
    yield
```

This runs *once*, when the FastAPI process starts. The `SEGMENTER = CellSegmenter()` line calls `__init__`, which loads the Cellpose model into GPU VRAM. That takes ~10 seconds, but it only happens once.

**Phase 2 — every request.** Also in main.py:

```python
@app.post("/segment")
async def segment(file: UploadFile = File(...)):
    image_bytes = await file.read()
    result = SEGMENTER.segment(image_bytes)   # segment() runs here. Fast (~650ms).
    ...
```

Every HTTP POST to `/segment` calls this handler, which calls `SEGMENTER.segment(bytes)`. Because `SEGMENTER` was built during startup, the model is already in VRAM — `segment()` doesn't reload anything, it just reuses the loaded model. That's why inference is ~650 ms and not ~10 seconds.

The pattern in one sentence:

> Long-lived state (the loaded model) is stored on a module-level instance created at startup. Per-request work (parsing bytes, running inference, computing morphology) reuses that state without rebuilding it.

Every serious ML production service has some version of this pattern. Interview-level phrasing:

> "The model lifecycle is separated from the request lifecycle. Model loading is a one-time startup cost; every request is O(model-inference-time) instead of O(model-inference-time + model-load-time). That's the difference between a service that scales and one that doesn't."

---

## Where does the instance actually live?

The `SEGMENTER` variable is defined at module scope in `main.py`:

```python
SEGMENTER: CellSegmenter | None = None
```

"Module scope" means it lives inside `main.py`'s module state, which lives for the lifetime of the Python process. As long as uvicorn is running, `main.py` is loaded, `SEGMENTER` exists, and the instance it references is alive.

The instance itself is a small object in RAM holding a few attributes:

- `self.device` — a short string (`"cuda"`)
- `self.model_type` — a short string (`"cyto3"`)
- `self.model` — a reference to a big Cellpose object, which internally references a PyTorch model whose weights sit in GPU VRAM (~500 MB)

When the Python process ends (Ctrl+C in the backend terminal), Python tears everything down. The instance is garbage-collected. The GPU allocations are released. Next time you start the backend, `__init__` runs again from scratch — a fresh instance, fresh VRAM allocation.

---

## Try this — feel the lifecycle in a REPL

With the venv active, in a terminal:

```python
from inference import CellSegmenter
import torch

print(torch.cuda.memory_allocated() / 1e6, "MB used on GPU")   # ~0 MB

# Build the instance. __init__ runs. Slow.
seg = CellSegmenter()

print(seg.device)                                                # "cuda"
print(seg.model_type)                                            # "cyto3"
print(torch.cuda.memory_allocated() / 1e6, "MB used on GPU")   # ~500 MB (model loaded)

# Call the method. Fast. __init__ does NOT re-run.
with open("../data/sample_images/cells_demo.png", "rb") as f:
    result1 = seg.segment(f.read())

# Call it again — still no __init__. Still fast.
with open("../data/sample_images/cells_demo.png", "rb") as f:
    result2 = seg.segment(f.read())

print(result1.cell_count, result2.cell_count)   # both 101

# Destroy the instance. VRAM is reclaimed.
del seg
import gc; gc.collect()
torch.cuda.empty_cache()
print(torch.cuda.memory_allocated() / 1e6, "MB used on GPU")   # back to ~0
```

The fact that the second `.segment()` call is fast, without ever seeing the "Loading Cellpose model" message, is the entire point of using a class here. You *felt* the startup cost pay for itself.

---

## Related docs

- Next: [03_http_and_json_boundaries.md](03_http_and_json_boundaries.md) — how the WPF window actually talks to the Python process.
- Also useful: [01_image_journey.md](01_image_journey.md) — the trace of one image through this exact instance.
