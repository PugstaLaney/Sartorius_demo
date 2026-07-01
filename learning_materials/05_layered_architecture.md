# Layered architecture — why the code is split the way it is

Most working code doesn't look like a notebook. It looks like *this repo*: dozens of small files, each with one job, arranged so that some files import others in a specific direction. This doc explains why that pattern exists, what it costs, and how it maps to the files in front of you.

---

## The historical arc

Early software (1960s-70s) really was often one big linear script. As programs got bigger, keeping everything in one file became unmanageable. Three waves of ideas followed:

1. **Structured programming (1970s).** Break the script into named functions, each doing one thing. This is what your Jupyter notebooks look like when you're doing data analytics.

2. **Object-oriented programming (1980s-90s).** Bundle related data and functions into classes. Now `CellSegmenter` can carry its own model and expose `segment()` as a method — the data and its operations travel together.

3. **Service-oriented / layered architecture (2000s onward).** Split the program into modules that only depend on things below them, and eventually into separate services that communicate over networks. This is what your Sartorius project is.

Nobody demands you use all three all the time. The rule is: **structure has a cost, and you invest in more of it as the code gets bigger or more people work on it.** For a one-off Jupyter analysis, one file is right. For a service maintained by a team for years, layered modules are essential.

---

## The dependency direction rule

The single most important idea in layered architecture:

> **Modules should only import from modules "below" them, never from modules "above" them.**

Read the module dependency diagram from the [README](../README.md#how-the-pieces-fit-together):

```
                         ┌──────────────────────┐
                         │  main.py              │  <- entry point (top layer)
                         └────────┬──────────────┘
                                  │ imports from ↓
                ┌─────────────────┴────────────────────┐
                ▼                  ▼                    ▼
      ┌─────────────────┐  ┌────────────────┐ ┌────────────────────┐
      │  inference.py   │  │  morphology.py │ │  tracker.py        │  <- leaf layer
      └─────────────────┘  └────────────────┘ └────────────────────┘
```

`main.py` imports `inference.py`. `inference.py` does NOT import `main.py`. That's the direction rule.

Why? Because if `inference.py` imported `main.py`, you'd have a **circular dependency** — Python can't fully load either module without the other, and the two files would be forever entangled. You couldn't test `inference.py` alone. You couldn't swap `main.py` out for a CLI without also rewriting `inference.py`.

Enforcing the rule means: **the layers below can be understood, tested, and reused without knowing anything about the layers above.** `inference.py` doesn't know that FastAPI exists. `morphology.py` doesn't know there's a Cellpose model in the codebase. `tracker.py` doesn't know it's being used by a WPF window on the other side of the world.

---

## Leaf modules vs orchestrators

Look at what each file in [backend/](../backend/) imports:

| File | Imports from `backend/` | Role |
|---|---|---|
| `inference.py` | (none) | Leaf — only depends on cellpose/torch/PIL |
| `morphology.py` | (none) | Leaf — only depends on scikit-image |
| `tracker.py` | (none) | Leaf — only depends on scipy |
| `track_visualization.py` | (none) | Leaf — only depends on PIL/numpy |
| `gif_io.py` | (none) | Leaf — only depends on PIL |
| `incucyte_filename.py` | (none) | Leaf — only depends on stdlib re |
| `main.py` | inference, morphology, gif_io, tracker, track_visualization | Orchestrator |
| `watcher.py` | inference, morphology, incucyte_filename | Orchestrator |

**Six leaf modules, two orchestrators.** The orchestrators compose the leaves into workflows. The leaves are ignorant of the orchestrators.

This is the shape of most healthy layered codebases. A wide layer of small, composable, testable "does one thing" modules, with a thin layer of orchestrators on top that pick and choose leaves depending on the workflow.

---

## Why this specific split

Consider what would happen if we merged everything into one big `backend.py`:

```python
# backend.py — the anti-example
import torch, cellpose, PIL, scipy.optimize, skimage.measure, re, fastapi
from collections import deque
from contextlib import asynccontextmanager

SEGMENTER = None
LATENCY = deque(maxlen=100)
TRACKS = {}
# ... 800 more lines of mixed responsibilities ...

@app.post("/segment")
async def segment(file):
    # inline: parse bytes, run cellpose, compute regionprops, encode PNG,
    # build JSON, all in one place
    ...

@app.post("/track_timelapse")
async def track_timelapse(file):
    # inline: split GIF, run cellpose per frame, run Hungarian assignment,
    # render overlays, encode base64, build JSON, all in one place
    # (duplicating half of /segment's logic)
    ...
```

That file *works*. But:

- **You can't test the tracker without spinning up FastAPI.** The Hungarian assignment logic is buried inside a route handler.
- **You can't reuse the morphology code in the folder watcher** without either duplicating it or importing from `backend.py` (which then imports everything else transitively).
- **Any change requires reading and understanding the entire 1000-line file** to know what will break.
- **Two developers can't work on separate features without stepping on each other's file.**
- **Every import gets pulled in on every startup** — even if a specific run doesn't need `scipy.optimize`.

By contrast, in the current split:

- The tracker can be tested with fabricated cell dicts, no model needed, no HTTP needed.
- The folder watcher and the HTTP endpoint both use the same `per_cell_stats` function.
- Each file is short enough to fit on a screen.
- Someone editing `tracker.py` won't accidentally break the segmentation code.

---

## When to add a layer, when not to

Modularity has diminishing returns. Splitting a 30-line utility across four files is *worse* than leaving it in one, because now the reader has to jump between files to understand what should have been one continuous thought.

Heuristics:

- **Under ~200 lines, keep it in one file.**
- **When you notice a natural seam** (e.g., "the top half is I/O, the bottom half is math"), split.
- **When two orchestrators start duplicating the same code**, extract the duplicated code into a leaf.
- **When one file grows past 600 lines**, split — that's beyond the point where someone can hold it in their head.
- **If splitting would require creating a bidirectional dependency**, don't split. The seam is in the wrong place.

The Sartorius project follows all of these. `inference.py` is 80 lines because CellSegmenter is one cohesive concept. `main.py` is 250 lines because it has multiple endpoints. `tracker.py` is 150 lines because the Tracker class + link_frames function are conceptually one thing.

---

## Cross-language boundaries are just extra-strong layers

`frontend_wpf/` and `backend/` are in different **languages**. The dependency direction rule applies at that layer too: WPF depends on the backend's HTTP API; the backend does not know WPF exists.

This is the deepest possible form of decoupling. The two sides:

- Cannot share memory (different processes)
- Cannot share objects (different type systems)
- Can only exchange JSON text over HTTP

That means:

- The C# team and the Python team can work independently
- Either side can be rewritten without touching the other
- A second frontend (browser, mobile app, CLI) can be added trivially — it's just another HTTP client

You already have proof of this: the [frontend/](../frontend/) folder is an HTML/JS frontend that hits the same backend. Two frontends, one backend, zero code duplication.

---

## The trade-off in one paragraph

Layered architecture makes each individual piece harder to write (you have to think about what layer it belongs in, what its interface should be) but makes the *system* dramatically easier to maintain, test, and extend. The bet is: over the lifetime of the code, you'll edit and read it many more times than you'll write it. Optimizing for reading and editing beats optimizing for writing.

If you plan to throw the code away in a week, don't invest in layers. If you plan to keep it around and evolve it, invest heavily.

---

## Try this — trace an import chain

In the venv, try this:

```powershell
cd backend
py -c "import main; print('Loaded, module set:'); import sys; print([m for m in sys.modules if not m.startswith('_')][:30])"
```

You'll see the transitive chain: importing `main` pulls in `inference`, `morphology`, `tracker`, `track_visualization`, `gif_io`, plus all their third-party dependencies. Now try the reverse:

```powershell
py -c "import tracker; import sys; assert 'main' not in sys.modules; print('OK: tracker.py has no idea main.py exists')"
```

That assertion passes. `tracker.py` can be loaded in complete ignorance of `main.py`. That's the dependency direction rule in action.

---

## Interview-flavored talking points

- "The service is layered by concern: leaf modules for the model, morphology, tracker, and visualization; a thin orchestrator layer for the HTTP service and the folder watcher. Nothing at the leaf layer depends on the orchestrators, which is what lets any piece be tested or reused independently."
- "The cross-language boundary is intentional. Python owns the model. C# owns the operator UI. They exchange JSON over HTTP. Either side can be rewritten without touching the other. A different frontend (HTML/JS, mobile, CLI) can consume the same backend without any changes."
- "The rule I follow: modules depend downward, never upward. That prevents circular dependencies and keeps the leaves reusable."

---

## Related docs

- Previous: [04_arrays_tensors_and_gpus.md](04_arrays_tensors_and_gpus.md)
- Next: [06_hungarian_tracking.md](06_hungarian_tracking.md)
