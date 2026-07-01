// Frontend client for the segmentation service.
// Deliberately framework-free so you can read every line in one sitting.

const API_BASE = "http://localhost:8000";

const fileInput = document.getElementById("fileInput");
const runBtn = document.getElementById("runBtn");
const originalCanvas = document.getElementById("originalCanvas");
const overlayCanvas = document.getElementById("overlayCanvas");
const deviceBadge = document.getElementById("deviceBadge");
const latencyMetric = document.getElementById("latencyMetric");
const countMetric = document.getElementById("countMetric");

let currentFile = null;
let currentImageBitmap = null;

// Check the service is up on page load and report the device it is using.
async function checkHealth() {
  try {
    const r = await fetch(`${API_BASE}/health`);
    const data = await r.json();
    deviceBadge.textContent = `device: ${data.device}`;
    deviceBadge.classList.add("ok");
  } catch (e) {
    deviceBadge.textContent = "device: backend offline";
    deviceBadge.classList.add("err");
  }
}

fileInput.addEventListener("change", async (e) => {
  currentFile = e.target.files[0];
  if (!currentFile) return;

  currentImageBitmap = await createImageBitmap(currentFile);
  drawImageOnCanvas(originalCanvas, currentImageBitmap);
  // Clear the overlay canvas until we run inference.
  const ctx = overlayCanvas.getContext("2d");
  overlayCanvas.width = currentImageBitmap.width;
  overlayCanvas.height = currentImageBitmap.height;
  ctx.clearRect(0, 0, overlayCanvas.width, overlayCanvas.height);
  runBtn.disabled = false;
});

runBtn.addEventListener("click", async () => {
  if (!currentFile) return;
  runBtn.disabled = true;
  runBtn.textContent = "Running...";

  try {
    const form = new FormData();
    form.append("file", currentFile);

    const t0 = performance.now();
    const r = await fetch(`${API_BASE}/segment`, { method: "POST", body: form });
    const data = await r.json();
    const totalMs = performance.now() - t0;

    if (!r.ok) throw new Error(data.detail || "request failed");

    latencyMetric.textContent =
      `latency: ${data.inference_ms} ms model / ${totalMs.toFixed(0)} ms total`;
    countMetric.textContent = `cells: ${data.cell_count}`;

    await drawOverlay(data.mask_png_base64);
  } catch (e) {
    alert(`Inference failed: ${e.message}`);
  } finally {
    runBtn.disabled = false;
    runBtn.textContent = "Run segmentation";
  }
});

function drawImageOnCanvas(canvas, bitmap) {
  canvas.width = bitmap.width;
  canvas.height = bitmap.height;
  canvas.getContext("2d").drawImage(bitmap, 0, 0);
}

async function drawOverlay(maskBase64) {
  // Composite: original image, then translucent mask on top.
  const ctx = overlayCanvas.getContext("2d");
  overlayCanvas.width = currentImageBitmap.width;
  overlayCanvas.height = currentImageBitmap.height;
  ctx.drawImage(currentImageBitmap, 0, 0);

  const maskImg = new Image();
  maskImg.src = `data:image/png;base64,${maskBase64}`;
  await new Promise((resolve) => (maskImg.onload = resolve));
  ctx.drawImage(maskImg, 0, 0);
}

checkHealth();
