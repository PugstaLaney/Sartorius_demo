# Async, concurrency, and the request lifecycle

This is the deepest doc in the curriculum. It covers what's actually happening when many HTTP requests hit the backend at once — a topic most tutorials skip and most beginners get wrong. If you can hold this in your head, you'll read production Python at a much higher level.

---

## Python is single-threaded (usually)

Python has a mechanism called the **Global Interpreter Lock** (GIL). At any moment, exactly one line of pure Python code is executing across your entire Python process, even on a many-core CPU. Two threads that both want to run Python bytecode take turns; they don't actually run at the same time.

This is a shock to people coming from languages like Go or Java where genuine multi-threading is normal. Python's single-threaded execution is why your instinct that "if I add threads it'll be faster" often turns out to be wrong.

But there are two big exceptions:

1. **I/O releases the GIL.** When Python code waits on the network, disk, or another process, the GIL is released and other Python code can run. That's why async and threading *do* help for I/O-bound work.

2. **C extensions can release the GIL.** Numpy, PyTorch, and similar libraries release the GIL when they hand work off to their underlying C or CUDA code. That's why `arr.mean()` can run "in parallel" while another thread does something else.

---

## What `async`/`await` actually gives you

Look at [backend/main.py](../backend/main.py):

```python
@app.post("/segment")
async def segment(file: UploadFile = File(...)):
    image_bytes = await file.read()
    result = SEGMENTER.segment(image_bytes)
    ...
```

The `async` keyword marks this function as a **coroutine**. The `await` keyword inside it means "yield control back to the event loop until this thing is done, then resume from here." When Python hits `await file.read()`, it pauses this request and can service OTHER requests in the meantime.

Async does NOT give you multi-threading. It gives you **cooperative multitasking on a single thread.**

Analogy: imagine one waiter at a restaurant. They take order A, walk it to the kitchen, and *while the kitchen cooks*, they take orders from tables B and C, refill drinks at table D, and check back on table A when its food is ready. One person, many concurrent tasks, none actually happening at the same instant — but a lot happening within any given minute.

That's async. One thread, many requests interleaved, work happening while other requests wait on I/O.

---

## Where async helps and where it doesn't

**Helps a lot** — when your work is dominated by waiting:

- Reading a file from disk
- Waiting on a database query
- Waiting on a network call to another service
- Reading data off a socket

**Helps only a little** — when your work is dominated by CPU:

- Complex math on numpy arrays
- Image encoding
- Big Python loops

**Doesn't help at all** — when GPU work is dominated by a single request:

- Actual model inference

Look at what our `/segment` handler does per request:

```python
image_bytes = await file.read()               # I/O — async helps here
result = SEGMENTER.segment(image_bytes)       # CPU + GPU — async does NOT help here
cells = per_cell_stats(result.mask)           # CPU — async doesn't help
_mask_to_png_base64(result.mask)              # CPU — async doesn't help
return JSONResponse({...})                    # I/O — async helps at send time
```

So during the ~650 ms of inference, this request is blocking the async event loop. Other requests are queued behind it. FastAPI is not magically running two segmentations at once.

**That's actually fine for a single-user instrument-side service.** If only one client is calling `/segment` at a time (like our WPF console), there's no contention. Async gives you nothing to optimize.

**It's a problem at scale.** If ten browsers hit `/segment` simultaneously, they queue up and take `10 × 650 ms = 6.5 seconds` for the last one to finish. Async doesn't fix that. Only genuine parallelism does.

---

## What would we do differently at scale?

If we needed to serve many concurrent requests, we'd need actual parallelism. Options:

1. **Multiple uvicorn worker processes.** Run `uvicorn main:app --workers 4`. Now four Python processes run in parallel, each with its own SEGMENTER, each on its own CPU core. Each can serve one request at a time; together they can serve four. Drawback: each worker loads its own copy of the model into VRAM, so a 4-worker deployment needs 4× the GPU memory.

2. **Batch inference.** Accept multiple images per request. Cellpose can process a batch of images faster per image than one at a time, because the GPU stays busy. Trade off: latency per image goes up (you wait for the batch to fill).

3. **Async model inference with an executor.** Use `run_in_executor` to push the CPU/GPU work into a thread pool, freeing the event loop to accept new requests. Doesn't speed up individual requests but keeps the event loop responsive so `/health` and `/metrics` stay fast even during inference.

4. **Message queue + separate worker service.** Split the API front-end (accepts uploads, returns immediately with a job ID) from the inference workers (pull jobs off a queue, process, write results somewhere). Client polls a status endpoint. Handles arbitrary load, at the cost of a much more complex system.

For this project's stated use case (one operator at an instrument), (1) with `--workers 1` is fine.

---

## Threads, processes, and async in one table

The three main concurrency tools in Python:

| Tool | Where the work runs | Blocked by GIL? | Right for |
|---|---|---|---|
| Threads | Same process, multiple threads | Yes for pure Python; No for I/O and C extensions | I/O concurrency without async, mixing with C-heavy libs |
| Processes | Separate OS processes | No (each has its own GIL) | CPU-bound Python work, isolation |
| Async | Same process, same thread | N/A (single-threaded cooperative) | Massive I/O concurrency (thousands of connections) |

FastAPI can use all three. In this project we only use async (via the `async def` decorator). We'd add multiprocessing (via uvicorn workers) if we wanted to scale.

---

## The request lifecycle, in detail

Here's what happens when a client hits `/segment`:

```
1. Client opens TCP connection to port 8000.
   → uvicorn's event loop accepts the connection.

2. Client sends HTTP request bytes.
   → uvicorn buffers them off the socket.
   → When headers are complete, uvicorn parses the request.
   → uvicorn identifies the route: POST /segment.

3. FastAPI dispatches to the segment() coroutine.
   → Coroutine starts running.
   → Hits `await file.read()`. Event loop suspends this coroutine
     and can service other requests until file bytes are all buffered.

4. file.read() returns. Coroutine resumes.
   → Calls SEGMENTER.segment(bytes). This is a REGULAR (sync) method call.
     While it runs, the event loop is BLOCKED. Any other pending coroutines
     wait. This lasts ~650 ms.
   → GPU work happens during this window. The GIL is briefly released while
     the CUDA kernel runs, but the Python function is still "on the stack"
     and the event loop can't advance.

5. segment() returns. Response is built.
   → JSON is serialized to bytes.
   → Response headers and body are written back to the socket.
   → Coroutine finishes. Event loop moves on.
```

The single blocking moment is step 4 (the sync inference call). Everything else is async-friendly.

---

## Try this — feel the queueing

Start the backend. In one PowerShell terminal, run this script that fires ten concurrent requests:

```powershell
$img = "c:\Users\palla\OneDrive\Documents\Coding Projects\Sartorius_Cell_Segmentation\data\sample_images\cells_demo.png"
$jobs = 1..10 | ForEach-Object {
    Start-Job -ScriptBlock {
        param($img, $i)
        $start = Get-Date
        $r = Invoke-WebRequest -Uri http://localhost:8000/segment -Method Post `
             -Form @{ file = Get-Item $img } -UseBasicParsing
        $ms = ((Get-Date) - $start).TotalMilliseconds
        "[$i] latency = $($ms.ToString('0')) ms, cells = $($r.Content | Select-String -Pattern '""cell_count"":\d+' -AllMatches | ForEach-Object { $_.Matches[0].Value })"
    } -ArgumentList $img, $_
}
$jobs | Wait-Job | Receive-Job | Sort-Object
$jobs | Remove-Job
```

If the backend were truly concurrent, all 10 would finish in ~700ms. What you'll actually see is a staircase: the first finishes near 700 ms, the second near 1400 ms, the third near 2100 ms, and so on. The requests are queued.

Now stop the backend and restart it with 4 workers:

```powershell
cd backend
C:\Users\palla\venvs\sartorius-cell\Scripts\python.exe -m uvicorn main:app --port 8000 --workers 4
```

Run the same 10-request test. You should see them finish in groups of 4 — the first 4 near 700 ms, the next 4 near 1400 ms, the last 2 near 2100 ms. **True parallelism**, at the cost of 4x the VRAM.

That's the concurrency trade-off in a nutshell.

---

## Interview-grade summary

> "The service uses async endpoints so I/O like reading the upload and writing the response doesn't block the event loop. Inference itself is CPU/GPU-bound and blocks the loop for the duration of a request. That's fine for a single-operator instrument-side use case; at higher load we'd run multiple uvicorn workers to get real parallelism, trading VRAM for concurrency, or batch requests inside a single worker."

---

## What "production-grade" concurrency would add

If we needed to make this ready for real production:

- **Multiple workers behind a reverse proxy** (nginx, Caddy) with health-check based routing.
- **Request timeout** — hard-kill any request that takes more than N seconds so a slow client can't hold a worker forever.
- **Rate limiting** — cap requests per client per minute.
- **Backpressure signals** — return `503 Service Unavailable` when the queue depth exceeds a threshold, so clients back off instead of piling up.
- **Metrics/observability** — Prometheus, distributed tracing.
- **Async model executor** — run inference in a thread pool with a bounded queue so the event loop stays responsive.

None of these are hard. All of them are the "boring but essential" work that turns a demo into a production service. The current code is deliberately minimal — you can point at each of these and honestly say "this is what I'd add for production."

---

## Related docs

- Previous: [06_hungarian_tracking.md](06_hungarian_tracking.md)
- Back to the [index](README.md)

---

## Where to go from here

Once you're comfortable with all seven docs, you'll be able to:

- Read most modern Python service code without confusion
- Explain the tradeoffs behind every design choice in this project
- Modify any layer with confidence that you won't break the others
- Interview credibly for ML deployment / applied ML engineering roles

At that point, the highest-leverage next moves are:

1. **Build the same thing in a different framework.** Rewrite the WPF console in TypeScript + Electron, or in Rust + Tauri. You'll feel the boundary hold — the backend needs zero changes.
2. **Add real production plumbing.** Docker, CI, model versioning, monitoring. All small tasks individually, all real signals to a hiring manager.
3. **Improve the model layer.** Fine-tune a U-Net on LIVECell. Replace Cellpose with detectron2. Prove the model is truly swappable by doing the swap.

Each of those unlocks a new interview talking point, and each is a manageable weekend project.
