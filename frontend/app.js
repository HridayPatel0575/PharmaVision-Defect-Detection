const uploadTab = document.getElementById("uploadTab");
const cameraTab = document.getElementById("cameraTab");
const uploadMode = document.getElementById("uploadMode");
const cameraMode = document.getElementById("cameraMode");

const imageInput = document.getElementById("imageInput");
const apiEndpointInput = document.getElementById("apiEndpoint");
const confidenceInput = document.getElementById("confidenceInput");
const strictModeInput = document.getElementById("strictModeInput");

const cameraView = document.getElementById("cameraView");
const previewImage = document.getElementById("previewImage");
const overlayCanvas = document.getElementById("overlayCanvas");

const startCameraBtn = document.getElementById("startCameraBtn");
const captureBtn = document.getElementById("captureBtn");
const stopCameraBtn = document.getElementById("stopCameraBtn");
const detectBtn = document.getElementById("detectBtn");

const statusText = document.getElementById("statusText");
const resultsOutput = document.getElementById("resultsOutput");
const toggleMetricsBtn = document.getElementById("toggleMetricsBtn");
const classMetricsPanel = document.getElementById("classMetricsPanel");

let currentMode = "upload";
let metricsVisible = false;
let currentImageBlob = null;
let currentImageObjectUrl = "";
let cameraStream = null;
let latestDetectionData = null;

function setStatus(message) {
  statusText.textContent = message;
}

function setMode(mode) {
  currentMode = mode;

  const isUpload = mode === "upload";
  uploadTab.classList.toggle("active", isUpload);
  cameraTab.classList.toggle("active", !isUpload);

  uploadTab.setAttribute("aria-selected", String(isUpload));
  cameraTab.setAttribute("aria-selected", String(!isUpload));

  uploadMode.classList.toggle("active-mode", isUpload);
  cameraMode.classList.toggle("active-mode", !isUpload);

  if (isUpload) {
    cameraView.classList.remove("visible");
  }

  clearOverlay();
}

function clearOverlay() {
  const ctx = overlayCanvas.getContext("2d");
  if (!ctx) return;
  ctx.clearRect(0, 0, overlayCanvas.width, overlayCanvas.height);
}

function syncOverlaySize() {
  const target = previewImage.classList.contains("visible") ? previewImage : cameraView;
  const rect = target.getBoundingClientRect();
  overlayCanvas.width = Math.max(1, Math.floor(rect.width));
  overlayCanvas.height = Math.max(1, Math.floor(rect.height));
}

function setPreviewFromObjectUrl(url) {
  if (currentImageObjectUrl) {
    URL.revokeObjectURL(currentImageObjectUrl);
  }
  currentImageObjectUrl = url;

  previewImage.src = url;
  previewImage.classList.add("visible");
  cameraView.classList.remove("visible");

  previewImage.onload = () => {
    syncOverlaySize();
    clearOverlay();
  };
}

async function startCamera() {
  if (cameraStream) return;

  try {
    cameraStream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
    cameraView.srcObject = cameraStream;

    cameraView.onloadedmetadata = () => {
      cameraView.classList.add("visible");
      previewImage.classList.remove("visible");
      syncOverlaySize();
      clearOverlay();
    };

    captureBtn.disabled = false;
    stopCameraBtn.disabled = false;
    setStatus("Camera started. Capture a frame, then run detection.");
  } catch (error) {
    setStatus("Unable to access camera. Check permissions and try again.");
    console.error(error);
  }
}

function stopCamera() {
  if (!cameraStream) return;

  for (const track of cameraStream.getTracks()) {
    track.stop();
  }
  cameraStream = null;

  cameraView.srcObject = null;
  cameraView.classList.remove("visible");

  captureBtn.disabled = true;
  stopCameraBtn.disabled = true;
}

function captureFrame() {
  if (!cameraStream) {
    setStatus("Start camera first.");
    return;
  }

  const w = cameraView.videoWidth;
  const h = cameraView.videoHeight;
  if (!w || !h) {
    setStatus("Camera is not ready yet.");
    return;
  }

  const tempCanvas = document.createElement("canvas");
  tempCanvas.width = w;
  tempCanvas.height = h;

  const ctx = tempCanvas.getContext("2d");
  ctx.drawImage(cameraView, 0, 0, w, h);

  tempCanvas.toBlob((blob) => {
    if (!blob) {
      setStatus("Capture failed. Try again.");
      return;
    }

    currentImageBlob = blob;
    const objectUrl = URL.createObjectURL(blob);
    setPreviewFromObjectUrl(objectUrl);
    setStatus("Frame captured. Ready for detection.");
  }, "image/jpeg", 0.95);
}

imageInput.addEventListener("change", () => {
  const file = imageInput.files && imageInput.files[0];
  if (!file) {
    return;
  }

  currentImageBlob = file;
  const objectUrl = URL.createObjectURL(file);
  setPreviewFromObjectUrl(objectUrl);
  setStatus(`Loaded ${file.name}. Ready for detection.`);
});

uploadTab.addEventListener("click", () => setMode("upload"));
cameraTab.addEventListener("click", () => setMode("camera"));
startCameraBtn.addEventListener("click", startCamera);
stopCameraBtn.addEventListener("click", () => {
  stopCamera();
  setStatus("Camera stopped.");
});
captureBtn.addEventListener("click", captureFrame);

toggleMetricsBtn.addEventListener("click", () => {
  metricsVisible = !metricsVisible;
  classMetricsPanel.style.display = metricsVisible ? "block" : "none";
  toggleMetricsBtn.textContent = metricsVisible ? "Hide Class Metrics" : "Show Class Metrics";
});

function normalizePredictions(data) {
  if (!data || typeof data !== "object") return [];

  if (Array.isArray(data.predictions)) return data.predictions;
  if (Array.isArray(data.detections)) return data.detections;
  return [];
}

function drawPredictions(predictions, data = null) {
  syncOverlaySize();

  const ctx = overlayCanvas.getContext("2d");
  if (!ctx) return;

  ctx.clearRect(0, 0, overlayCanvas.width, overlayCanvas.height);
  if (!predictions.length) return;

  ctx.lineWidth = 2;
  ctx.font = "14px Space Grotesk";

  const srcW = data && typeof data.image_width === "number" ? data.image_width : null;
  const srcH = data && typeof data.image_height === "number" ? data.image_height : null;

  let scaleX = 1;
  let scaleY = 1;
  if (srcW && srcH) {
    scaleX = overlayCanvas.width / srcW;
    scaleY = overlayCanvas.height / srcH;
  }

  for (const p of predictions) {
    let x = p.x;
    let y = p.y;
    let w = p.width;
    let h = p.height;

    // Fallback if API returns x1,y1,x2,y2 format.
    if (["x1", "y1", "x2", "y2"].every((k) => typeof p[k] === "number")) {
      x = p.x1;
      y = p.y1;
      w = p.x2 - p.x1;
      h = p.y2 - p.y1;
    }

    if ([x, y, w, h].some((v) => typeof v !== "number")) {
      continue;
    }

    const drawX = x * scaleX;
    const drawY = y * scaleY;
    const drawW = w * scaleX;
    const drawH = h * scaleY;

    ctx.strokeStyle = "#27c6a8";
    ctx.fillStyle = "rgba(0, 0, 0, 0.65)";
    ctx.strokeRect(drawX, drawY, drawW, drawH);

    const label = `${p.class || p.label || "defect"} ${((p.confidence || 0) * 100).toFixed(1)}%`;
    const textWidth = ctx.measureText(label).width + 12;

    ctx.fillRect(drawX, Math.max(0, drawY - 22), textWidth, 20);
    ctx.fillStyle = "#ecfffb";
    ctx.fillText(label, drawX + 6, Math.max(14, drawY - 7));
  }
}

async function runDetection() {
  if (!currentImageBlob) {
    setStatus("Select an image or capture a frame first.");
    return;
  }

  const endpoint = apiEndpointInput.value.trim();
  const confidence = Number(confidenceInput.value || 0.75);
  const strictMode = Boolean(strictModeInput.checked);
  if (!endpoint) {
    setStatus("Please enter an API endpoint.");
    return;
  }

  if (Number.isNaN(confidence) || confidence < 0.1 || confidence > 0.99) {
    setStatus("Confidence must be between 0.10 and 0.99.");
    return;
  }

  detectBtn.disabled = true;
  setStatus("Running detection...");

  try {
    const formData = new FormData();
    formData.append("file", currentImageBlob, "input.jpg");
    formData.append("conf", String(confidence));
    formData.append("strict_mode", strictMode ? "true" : "false");

    const response = await fetch(endpoint, {
      method: "POST",
      body: formData
    });

    const text = await response.text();
    let data = null;

    try {
      data = JSON.parse(text);
    } catch {
      data = { raw: text };
    }

    if (!response.ok) {
      throw new Error(data.raw || `Request failed with status ${response.status}`);
    }

    latestDetectionData = data;

    resultsOutput.textContent = JSON.stringify(data, null, 2);

    const predictions = normalizePredictions(data);
    drawPredictions(predictions, data);

    setStatus(predictions.length ? `Detection complete: ${predictions.length} object(s).` : "Detection complete: no objects detected.");
  } catch (error) {
    console.error(error);
    resultsOutput.textContent = String(error.message || error);
    clearOverlay();
    setStatus("Detection failed. Check endpoint and backend logs.");
  } finally {
    detectBtn.disabled = false;
  }
}

detectBtn.addEventListener("click", runDetection);
window.addEventListener("resize", () => {
  syncOverlaySize();
  if (!latestDetectionData) return;
  drawPredictions(normalizePredictions(latestDetectionData), latestDetectionData);
});
window.addEventListener("beforeunload", stopCamera);

setMode("upload");
setStatus("Ready.");
