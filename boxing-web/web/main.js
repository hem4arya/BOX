// Check WebGPU support
if (!navigator.gpu) {
  document.getElementById("error-msg").style.display = "block";
  document.getElementById("game-canvas").style.display = "none";
  throw new Error("WebGPU not supported");
}

import { PoseLandmarker, FilesetResolver } from "@mediapipe/tasks-vision";
import init, {
  init as wasmInit,
  render_frame,
  physics_tick,
  calibrate_depth,
  apply_mediapipe_correction,
  set_frame_metrics,
  set_mediapipe_latency,
  set_physics_time,
} from "../pkg/boxing_web.js";

// ============================================================================
// STATS MONITOR HUD
// ============================================================================

class StatsMonitor {
  constructor() {
    this.fps = 0;
    this.frameCount = 0;
    this.lastFpsUpdate = 0;
    this.startTime = 0;

    // Timing metrics
    this.mediapipeMs = 0;
    this.physicsMs = 0;
    this.frameMs = 0;

    // Create HUD overlay
    this.div = document.createElement("div");
    this.div.style.cssText = `
      position: fixed;
      top: 10px;
      right: 10px;
      color: #00ff00;
      font-family: 'Consolas', 'Monaco', monospace;
      font-size: 13px;
      background: rgba(0, 0, 0, 0.8);
      padding: 10px 14px;
      border-radius: 6px;
      z-index: 9999;
      min-width: 180px;
      line-height: 1.5;
      border: 1px solid #333;
    `;
    document.body.appendChild(this.div);
  }

  begin() {
    this.startTime = performance.now();
  }

  end() {
    const now = performance.now();
    this.frameMs = now - this.startTime;

    this.frameCount++;
    if (now - this.lastFpsUpdate >= 500) {
      this.fps = Math.round(
        (this.frameCount * 1000) / (now - this.lastFpsUpdate)
      );
      this.frameCount = 0;
      this.lastFpsUpdate = now;
      this.updateDisplay();
    }
  }

  setMediapipe(ms) {
    this.mediapipeMs = this.mediapipeMs * 0.8 + ms * 0.2;
  }
  setPhysics(ms) {
    this.physicsMs = this.physicsMs * 0.8 + ms * 0.2;
  }

  updateDisplay() {
    const frameColor =
      this.frameMs > 16 ? "#ff4444" : this.frameMs > 8 ? "#ffaa00" : "#00ff00";
    const mpColor =
      this.mediapipeMs > 40
        ? "#ff4444"
        : this.mediapipeMs > 25
        ? "#ffaa00"
        : "#00ff00";

    this.div.innerHTML = `
      <b>FPS:</b> ${this.fps}<br>
      <b>Frame:</b> <span style="color:${frameColor}">${this.frameMs.toFixed(
      1
    )}ms</span><br>
      <b>MediaPipe:</b> <span style="color:${mpColor}">${this.mediapipeMs.toFixed(
      0
    )}ms</span><br>
      <b>Physics:</b> ${this.physicsMs.toFixed(2)}ms<br>
      <span style="color:#888">Punch detection: Rust</span>
    `;
  }
}

// ============================================================================
// GLOBALS
// ============================================================================

const status = document.getElementById("status");
const video = document.getElementById("camera-video");
const stats = new StatsMonitor();

let lastRenderTime = 0;

// Calibration
let calibrationCountdown = 0;
let calibrationInterval = null;

function startCalibrationCountdown() {
  if (calibrationInterval) return;
  calibrationCountdown = 5;
  status.textContent = `📐 T-POSE in ${calibrationCountdown}...`;
  calibrationInterval = setInterval(() => {
    calibrationCountdown--;
    if (calibrationCountdown > 0) {
      status.textContent = `📐 T-POSE in ${calibrationCountdown}...`;
    } else {
      clearInterval(calibrationInterval);
      calibrationInterval = null;
      calibrate_depth();
      status.textContent = "✅ Calibrated! Punch to detect.";
    }
  }, 1000);
}

async function main() {
  status.textContent = "Loading WASM...";
  await init();
  await wasmInit();
  console.log("🥊 Boxing Web WASM loaded");

  // ONNX DISABLED - Using physics-based punch detection in Rust
  console.log("ℹ️ Punch detection: Physics-based (Rust)");

  status.textContent = "Requesting camera...";
  let stream;
  try {
    stream = await navigator.mediaDevices.getUserMedia({
      video: {
        width: { ideal: 640 },
        height: { ideal: 480 },
        facingMode: "user",
      },
    });
  } catch (err) {
    status.textContent = "❌ Camera access denied";
    return;
  }

  video.srcObject = stream;
  await video.play();

  status.textContent = "Loading MediaPipe...";
  const vision = await FilesetResolver.forVisionTasks(
    "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/wasm"
  );

  const poseLandmarker = await PoseLandmarker.createFromOptions(vision, {
    baseOptions: {
      modelAssetPath:
        "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task",
      delegate: "GPU",
    },
    runningMode: "VIDEO",
    numPoses: 1,
  });

  console.log("🤖 MediaPipe ready");
  status.textContent = "✅ Ready - Press C to calibrate";

  document.addEventListener("keydown", (e) => {
    if (e.key === "c" || e.key === "C") startCalibrationCountdown();
  });

  // ========================================================================
  // RENDER LOOP (120Hz) - Independent, never waits for MediaPipe
  // ========================================================================
  function renderLoop(timestamp) {
    stats.begin();

    const frameTime = timestamp - lastRenderTime;
    lastRenderTime = timestamp;

    // Physics tick every frame (120Hz)
    const dt = Math.min(frameTime / 1000.0, 0.1);
    if (dt > 0.001) {
      const physicsStart = performance.now();
      physics_tick(dt);
      stats.setPhysics(performance.now() - physicsStart);
    }

    render_frame();

    stats.end();
    requestAnimationFrame(renderLoop);
  }

  // ========================================================================
  // DETECTION LOOP (async) - Runs independently
  // ========================================================================
  async function detectionLoop() {
    while (true) {
      const mpStart = performance.now();
      const results = poseLandmarker.detectForVideo(video, performance.now());
      const mpTime = performance.now() - mpStart;

      stats.setMediapipe(mpTime);
      set_mediapipe_latency(mpTime);

      if (results.landmarks && results.landmarks[0]) {
        const landmarks = results.landmarks[0];
        const flatArray = new Float32Array(landmarks.length * 3);
        landmarks.forEach((lm, i) => {
          flatArray[i * 3] = lm.x;
          flatArray[i * 3 + 1] = lm.y;
          flatArray[i * 3 + 2] = lm.z;
        });

        // Punch detection happens in Rust (apply_mediapipe_correction)
        apply_mediapipe_correction(flatArray);
      }

      await new Promise((r) => setTimeout(r, 0));
    }
  }

  // Start both loops
  video.addEventListener("loadeddata", () => {
    lastRenderTime = performance.now();
    requestAnimationFrame(renderLoop);
    detectionLoop();
  });

  if (video.readyState >= 2) {
    lastRenderTime = performance.now();
    requestAnimationFrame(renderLoop);
    detectionLoop();
  }
}

main().catch((err) => {
  console.error("❌ Fatal error:", err);
  status.textContent = "❌ Error: " + err.message;
});
