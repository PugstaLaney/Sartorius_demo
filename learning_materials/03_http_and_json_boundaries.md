# HTTP, JSON, and the boundary between languages

The WPF console is C#. The backend is Python. They exchange no code — no imports, no shared libraries, no shared runtime. The only thing they agree on is a network protocol (HTTP) and a data format (JSON). This doc explains why that's a big deal and how it works in practice.

---

## HTTP in 30 seconds

HTTP (HyperText Transfer Protocol) is the language two programs use to talk over a network. It's the same protocol your browser uses to fetch a webpage from google.com — client and server exchange messages according to a fixed shape.

Every HTTP **request** has three parts:

1. A **method** (a verb) — what kind of action you're asking for
2. A **URL** — what resource you're acting on
3. Optionally a **body** — data attached to the request

Every HTTP **response** has three parts too:

1. A **status code** — a three-digit number saying what happened
2. **Headers** — metadata like content type, size, etc.
3. Optionally a **body** — data returned to you

That's it. Everything else is layered on top.

---

## Methods (verbs) you'll see

| Verb | Meaning | Example in this project |
|---|---|---|
| **GET** | "Give me X" | `GET /health` — "are you alive?" |
| **POST** | "Here, take this X" | `POST /segment` — "here's an image, process it" |
| **PUT** | "Replace X with this" | (not used here) |
| **DELETE** | "Remove X" | (not used here) |

The rule of thumb: **use GET when you're only reading, use POST when you're sending data or triggering work.**

---

## Status codes

The response starts with a 3-digit code. The first digit tells you the category:

| Range | Meaning | Common examples |
|---|---|---|
| 2xx | Success | 200 OK, 201 Created |
| 3xx | Redirect | 301 Moved Permanently |
| 4xx | Client's fault | 400 Bad Request, 404 Not Found, 401 Unauthorized |
| 5xx | Server's fault | 500 Internal Server Error, 503 Service Unavailable |

Your `/segment` endpoint returns 200 when segmentation works, 400 when the upload is empty, 500 when Cellpose errors, and 503 when the model isn't loaded yet. Each has a specific meaning and clients (like the WPF window) can distinguish them.

---

## JSON — the shared vocabulary

C# and Python have completely different type systems. C# has `List<T>`, `Dictionary<K,V>`, `int`, `double`. Python has `list`, `dict`, `int`, `float`. They can't directly share objects — a C# `List<int>` doesn't exist in Python memory.

**JSON** solves this by being a *lingua franca*: a simple text format that every modern language can produce and consume. When you serialize an object to JSON, you get text like:

```json
{"cell_count": 101, "inference_ms": 645.2, "device": "cuda"}
```

That text is language-agnostic. C# can produce it. Python can produce it. Rust can produce it. On the other end, any of those languages can parse the text back into a native object.

So the actual flow across the boundary is:

```
Python object  →  JSON text  →  bytes on wire  →  JSON text  →  C# object
    dict          serialize        HTTP body         parse       SegmentResponse
```

Both sides write code that produces and consumes JSON. Neither side sees the other's memory.

---

## The exchange in one full round-trip

When you click **Run segmentation** in the WPF window and it processes `cells_demo.png`:

**WPF sends:**

```
POST /segment HTTP/1.1
Host: localhost:8000
Content-Type: multipart/form-data; boundary=...
Content-Length: 133962

[the binary bytes of the PNG image]
```

That's a raw text message with the PNG bytes appended. It travels across the localhost network interface (no actual network hop since same machine, but same protocol) and lands at uvicorn.

**Python responds:**

```
HTTP/1.1 200 OK
Content-Type: application/json
Content-Length: 64000

{"cell_count":101,"inference_ms":645.2,"device":"cuda","mask_png_base64":"iVBORw0K...","per_cell":[...]}
```

The response body is JSON text. WPF parses it back into a `SegmentResponse` object (see [MainWindow.xaml.cs](../frontend_wpf/MainWindow.xaml.cs) for the class definition), reads the fields, updates the UI.

Neither side had to know what language the other was written in. They only had to agree on the URL, the method, and the JSON schema of the request and response.

---

## Why this decoupling is a big deal

Because the boundary is HTTP + JSON, you can:

- **Rewrite the frontend in a different language** (Rust, TypeScript, whatever) without touching the backend
- **Rewrite the backend in a different language** (Go, Rust) without touching the frontend
- **Add a second frontend** that talks to the same backend (your project actually does this — the HTML/JS frontend and the WPF console both hit the same `/segment` endpoint)
- **Test the backend without a UI** by hitting the endpoints directly with `curl`, `httpx`, or a browser
- **Deploy the two sides on different machines** — the WPF on an instrument workstation, the backend on a shared GPU server across the network

That's why "decouple your services" is a mantra in modern architecture. The cost is small (some serialization overhead) and the flexibility is huge.

---

## FastAPI's decorator syntax

The Python side uses FastAPI's `@app.post("/segment")` decorator syntax to attach a Python function to an HTTP endpoint:

```python
@app.post("/segment")
async def segment(file: UploadFile = File(...)):
    ...
```

Read the decorator as: **"when a POST request arrives at `/segment`, call this function."** FastAPI handles the rest — parsing the multipart form, extracting the file, converting return values to JSON, setting the Content-Type header, etc.

The parameter type hints (`file: UploadFile = File(...)`) tell FastAPI what to pull out of the request body. If you added `params: SegmentParams` where `SegmentParams` is a Pydantic model, FastAPI would automatically parse a JSON body into that class. This is what makes FastAPI feel magical — it inspects your Python type hints and figures out the wire protocol from them.

---

## The interactive docs page

FastAPI ships a built-in Swagger UI at `http://localhost:8000/docs` while the backend is running. It:

- Lists every endpoint (`/health`, `/metrics`, `/segment`, `/track_timelapse`)
- Shows their methods, parameters, and response schemas
- Lets you upload files and hit the endpoints from the browser — no WPF, no `curl`, just click "Try it out"

If you want to feel HTTP viscerally, do the exercise below.

---

## Try this — hit the API without the WPF

Start the backend (double-click `Launch Sartorius Demo.cmd` or run uvicorn manually). Then, in a separate PowerShell terminal:

```powershell
# GET the health endpoint. Notice no body — GET has no body.
Invoke-WebRequest -Uri http://localhost:8000/health -UseBasicParsing
```

You'll see something like:

```
StatusCode        : 200
StatusDescription : OK
Content           : {"status":"ok","device":"cuda"}
```

Now POST an image directly:

```powershell
$img = "c:\Users\palla\OneDrive\Documents\Coding Projects\Sartorius_Cell_Segmentation\data\sample_images\cells_demo.png"
Invoke-WebRequest -Uri http://localhost:8000/segment -Method Post `
    -Form @{ file = Get-Item $img } | Select-Object -ExpandProperty Content | Select-Object -First 500
```

You'll get a wall of JSON: cell count, latency, device, base64-encoded mask, per-cell array. That's *exactly* what the WPF window is receiving. The WPF window is not doing anything magical — it's just parsing the same JSON and drawing pixels.

Then open `http://localhost:8000/docs` in a browser. Try the endpoints from the Swagger UI. Upload an image via the browser. Watch the same JSON come back.

You'll never think of HTTP as mysterious again.

---

## The interview-grade summary

> "The frontend and backend are decoupled by an HTTP + JSON contract. Neither side depends on the other's runtime. That means either can be replaced or scaled independently, and a second client (browser, CLI, different language) can consume the same backend without any changes to the service."

That sentence describes exactly what you built.

---

## Related docs

- Previous: [02_classes_and_lifetime.md](02_classes_and_lifetime.md)
- Next: [04_arrays_tensors_and_gpus.md](04_arrays_tensors_and_gpus.md)
