# Learning materials

A curriculum of concept notes organized in reading order. Each doc is grounded in the actual code of this repo and ends with a hands-on experiment. Read the notebook first, then work through these top to bottom.

| # | File | What you'll leave with |
|---|---|---|
| 00 | [00_cellpose_basics.ipynb](00_cellpose_basics.ipynb) | What image segmentation is, PyTorch tensor basics, how to run Cellpose on one image, how to measure GPU vs CPU latency. Foundation for everything else. |
| 01 | [01_image_journey.md](01_image_journey.md) | Step-by-step trace of one image from disk → WPF → HTTP → Python → GPU → JSON → back to the WPF window. The concrete-first mental model. |
| 02 | [02_classes_and_lifetime.md](02_classes_and_lifetime.md) | What a class actually is (recipe vs. instance), what `__init__` and `self` mean, what `@dataclass` gives you, where instances live in memory, and how FastAPI's lifespan hook creates the long-lived `SEGMENTER` singleton. |
| 03 | [03_http_and_json_boundaries.md](03_http_and_json_boundaries.md) | HTTP as a text protocol, GET vs POST, status codes, JSON as the shared vocabulary across languages, why the WPF and backend can each be replaced without touching the other. |
| 04 | [04_arrays_tensors_and_gpus.md](04_arrays_tensors_and_gpus.md) | Images as 3D numpy arrays, dtypes and shapes, what a tensor is, how PyTorch moves data between CPU and GPU, and why GPUs are dramatically faster for the forward pass. |
| 05 | [05_layered_architecture.md](05_layered_architecture.md) | Why the code is split by concern, the dependency direction rule, leaf modules vs. orchestrators, and when adding a layer stops being helpful. Interview-heavy. |
| 06 | [06_hungarian_tracking.md](06_hungarian_tracking.md) | The bipartite matching problem, why greedy nearest-neighbor is broken, how the cost matrix is built, and what `scipy.optimize.linear_sum_assignment` is actually doing under the hood. |
| 07 | [07_async_and_concurrency.md](07_async_and_concurrency.md) | Python's single-threaded execution model, what `async`/`await` actually give you, where FastAPI's async is helping and where it isn't, and what would change for production concurrency. |

## How to use these

- **Read one at a time**, in order. Each builds on the last.
- **Run the experiments.** Every doc ends with something concrete you can type into a REPL, a terminal, or the WPF app.
- **Modify the code.** Change a number in `tracker.py` and re-run `process_timelapse.py`. Delete `__pycache__/` and watch it come back. The code is not fragile — poke it.
- **Come back to the doc after coding.** These are reference material, not linear novels. If you forget what a dataclass is three weeks from now, doc 02 is the answer.
