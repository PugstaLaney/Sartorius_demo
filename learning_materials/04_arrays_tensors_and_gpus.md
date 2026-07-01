# Arrays, tensors, and how images become GPU math

Segmentation is *just numerical computation on multi-dimensional arrays*. Understanding how an image gets represented as numbers, and how those numbers get shuffled onto and off of a GPU, is the foundation for reading any ML code. This doc walks through those layers using the code in [backend/inference.py](../backend/inference.py) as the concrete example.

---

## Digital images are 3D arrays

A color image with height `H`, width `W`, and 3 color channels (red, green, blue) is nothing more than a 3D grid of numbers with shape `(H, W, 3)`. Each cell in the grid is an integer between 0 and 255 saying "how much red / green / blue this pixel has."

- A 512×512 RGB image = `512 × 512 × 3 = 786,432` numbers.
- Each number is a `uint8` (unsigned 8-bit integer), taking 1 byte.
- Total memory footprint: ~800 KB in RAM.

When Cellpose does its work, the output — the segmentation mask — is a 2D grid of shape `(H, W)`. Each cell in that grid is an integer saying "this pixel belongs to background (0), cell 1, cell 2, ..., or cell N." Same idea, one less dimension.

---

## `numpy` in one paragraph

`numpy` is the Python library for numerical arrays. Every ML library in Python (PyTorch, TensorFlow, scikit-image) uses numpy at its edges — they accept numpy arrays as input and hand you numpy arrays back out.

Two things you'll always want to check on any numpy array:

- **`.shape`** — the size along each dimension, as a tuple. `(512, 512, 3)` means a 512-row, 512-column, 3-channel image.
- **`.dtype`** — the numeric type of the entries. `uint8` (0-255 integers), `int32` (large integers), `float32` (decimal, standard for ML), `float64` (double-precision decimal).

Reading numpy code is easier if you keep those two questions in the back of your mind: *what shape is this array, and what type are the numbers?*

---

## The transformation chain in `segment()`

Let's read the relevant piece of [inference.py](../backend/inference.py) with dtype and shape annotations:

```python
def segment(self, image_bytes: bytes) -> SegmentationResult:
    # image_bytes: raw file bytes (~130 KB for a small PNG)
    
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    # img: PIL Image object, wraps decoded pixels.
    # After .convert("RGB"), guaranteed to be 3 channels regardless of source
    # (a grayscale PNG becomes RGB by duplicating the gray channel three times).
    
    arr = np.array(img)
    # arr.shape:  (H, W, 3)
    # arr.dtype:  uint8
    # arr.nbytes: H * W * 3

    masks, _, _, _ = self.model.eval([arr], channels=[0, 0], diameter=None)
    # Cellpose accepts a LIST of arrays and returns a LIST of masks.
    # Behind the scenes:
    #   1. numpy array copied to GPU as a torch tensor
    #   2. tensor sent through U-Net forward pass on GPU
    #   3. flow field + probability map computed on GPU
    #   4. pixel clustering on CPU
    #   5. resulting mask copied back to CPU as a numpy array

    mask = masks[0]
    # mask.shape:  (H, W)
    # mask.dtype:  int32 or int16 depending on cell count
    # mask.min():  0
    # mask.max():  number of cells detected

    cell_count = int(mask.max())
    return SegmentationResult(mask=mask, cell_count=cell_count, ...)
```

The mental picture: an image comes in as a big grid of pixel values, gets converted through progressively more structured representations, gets pushed onto the GPU for the heavy math, and comes back as a different big grid of numbers where the numbers now mean cell identity instead of color.

---

## What's a tensor?

A **tensor** is essentially a numpy array with two extra powers:

1. **It can live on a GPU** (not just RAM). numpy arrays only live on the CPU.
2. **PyTorch tracks operations on it** for automatic differentiation (used during training).

For our purposes (inference only, not training), think of a tensor as *"a numpy array that PyTorch can move to the GPU."*

In PyTorch, moving an array to the GPU looks like this:

```python
import torch
import numpy as np

arr = np.zeros((512, 512, 3), dtype=np.float32)  # a numpy array in RAM
tensor_cpu = torch.from_numpy(arr)                # PyTorch tensor still in RAM
tensor_gpu = tensor_cpu.cuda()                    # copied to GPU VRAM
                                                  # or equivalently: .to("cuda")

# Every operation you do on tensor_gpu happens on the GPU
result_gpu = tensor_gpu.mean()  # GPU computes this
result_cpu = result_gpu.cpu()   # copy result back to RAM to look at it
```

Cellpose hides all of this from you. When you call `self.model.eval([arr], ...)`, the library internally converts your numpy array into a tensor, moves it to whichever device the model is on (`cuda` in our case), runs the forward pass, and copies results back — no manual `.cuda()` needed.

But it *is* doing all those steps. It's why Cellpose "warms up" slowly on the first inference (the GPU has to allocate memory pools and compile CUDA kernels) but then runs fast on subsequent calls.

---

## Why the GPU is so much faster

The neural network's forward pass is dominated by **matrix multiplications**. A single convolution layer effectively multiplies large matrices together. A CPU has ~8-16 cores that each do one multiplication at a time. An RTX 3060 has 3,584 CUDA cores that all do multiplications *in parallel*.

For image-sized matrices with millions of entries, the GPU wins by ~10-50x. On your machine, the notebook exercise `01_cellpose_basics.ipynb` measured Cellpose at:

- GPU (RTX 3060): ~200-650 ms per image
- CPU: ~3,000-6,000 ms per image

That's the entire reason ML production services care about GPUs. Not because they're smarter — because they're stupidly parallel and neural network math happens to be very stupidly parallel.

---

## Where each thing lives

For one segmentation call, memory looks like this:

| Data | Type | Where it lives | Rough size |
|---|---|---|---|
| Original image on disk | PNG file | SSD | 130 KB |
| Raw bytes buffer | Python `bytes` | RAM | 130 KB |
| PIL Image | Python object wrapping pixel data | RAM | ~800 KB |
| numpy array (`arr`) | numpy `uint8` array, shape (H,W,3) | RAM | ~800 KB |
| GPU tensor (input) | PyTorch tensor on CUDA | GPU VRAM | ~800 KB |
| Cellpose model weights | PyTorch model on CUDA | GPU VRAM | ~500 MB (persistent) |
| Intermediate activations | many temporary tensors | GPU VRAM | ~50-200 MB (freed after forward pass) |
| Mask (output) | numpy int array, shape (H,W) | RAM | ~1 MB |

The persistent ~500 MB of model weights on the GPU is what makes reuse across requests so much faster than reloading — that's the "load once, reuse many" pattern the class lifecycle enables.

---

## Try this — feel the shapes

In a REPL with the venv active:

```python
import numpy as np
from PIL import Image

path = "../data/sample_images/cells_demo.png"

# Load and inspect an image at each stage of transformation
img = Image.open(path).convert("RGB")
print(f"PIL Image size: {img.size}")           # (W, H) — note PIL swaps order!

arr = np.array(img)
print(f"numpy shape:    {arr.shape}")          # (H, W, 3)
print(f"numpy dtype:    {arr.dtype}")          # uint8
print(f"numpy nbytes:   {arr.nbytes}")         # H * W * 3
print(f"numpy min/max:  {arr.min()} / {arr.max()}")  # 0..255

# Convert to grayscale by averaging channels
gray = arr.mean(axis=2).astype(np.uint8)
print(f"gray shape:     {gray.shape}")          # (H, W)

# Now a torch tensor exercise (needs the venv with torch installed)
import torch
tensor_cpu = torch.from_numpy(arr)
print(f"tensor device:  {tensor_cpu.device}")   # cpu
tensor_gpu = tensor_cpu.cuda()
print(f"tensor device:  {tensor_gpu.device}")   # cuda:0

# Check GPU memory
print(f"GPU MB used:    {torch.cuda.memory_allocated() / 1e6:.1f}")

# Move it back
back_to_cpu = tensor_gpu.cpu().numpy()
print(f"back on cpu:    {back_to_cpu.shape}, matches original: {(back_to_cpu == arr).all()}")
```

You'll physically feel the data hopping between RAM and VRAM, and see the shape/dtype at each stage. That's the entire ML data-plumbing story in ten lines.

---

## Two things beginners get wrong

1. **Shape ordering.** PIL uses `(W, H)`. numpy image arrays use `(H, W, 3)`. PyTorch model tensors often use `(batch, channels, H, W)`. Convert carefully or you'll get "shape mismatch" errors that look confusing. Always print `.shape` when in doubt.

2. **dtype and value range.** numpy image arrays are `uint8` with values 0-255. Neural networks want `float32` normalized to 0-1 or -1 to 1. Cellpose handles this conversion for you, but if you write your own model code you'll need to do:
   ```python
   x = arr.astype(np.float32) / 255.0
   ```

---

## Related docs

- Previous: [03_http_and_json_boundaries.md](03_http_and_json_boundaries.md)
- Next: [05_layered_architecture.md](05_layered_architecture.md)
- Also useful: the notebook [../notebooks/01_cellpose_basics.ipynb](../notebooks/01_cellpose_basics.ipynb) has hands-on tensor experiments.
