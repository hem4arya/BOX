// Check WebGPU support
if (!navigator.gpu) {
  document.getElementById("error-msg").style.display = "block";
  document.getElementById("game-canvas").style.display = "none";
  throw new Error("WebGPU not supported");
}

import { HandLandmarker, FilesetResolver } from "@mediapipe/tasks-vision";
import init, {
  init as wasmInit,
  render_frame,
  physics_tick,
  calibrate_hand,
  apply_hand_landmarks,
  set_frame_metrics,
  set_mediapipe_latency,
  set_physics_time,
  get_debug_overlay_text,
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
    this.mediapipeMs = 0;
    this.physicsMs = 0;
    this.frameMs = 0;
    this.cameraLatency = 0;
    this.armInfo = "";

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
      min-width: 200px;
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
  setCameraLatency(ms) {
    this.cameraLatency = this.cameraLatency * 0.9 + ms * 0.1;
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

    // Get debug info from WASM
    try {
      this.armInfo = get_debug_overlay_text();
    } catch (e) {
      this.armInfo = "";
    }

    this.div.innerHTML = `
      <b>FPS:</b> ${this.fps}<br>
      <b>Frame:</b> <span style="color:${frameColor}">${this.frameMs.toFixed(
      1
    )}ms</span><br>
      <b>MediaPipe:</b> <span style="color:${mpColor}">${this.mediapipeMs.toFixed(
      0
    )}ms</span><br>
      <b>Physics:</b> ${this.physicsMs.toFixed(2)}ms<br>
      <span style="color:#0ff">${this.armInfo.replace(/\n/g, "<br>")}</span><br>
      <span style="color:#888">🖐️ Hand Tracking Mode</span>
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
let handLandmarker = null;

// Calibration (not used for hands but kept for compatibility)
let calibrationCountdown = 0;
let calibrationInterval = null;

function startCalibrationCountdown() {
  if (calibrationInterval) return;
  calibrationCountdown = 3;
  status.textContent = `✋ Hold hands OPEN & FLAT in ${calibrationCountdown}...`;
  calibrationInterval = setInterval(() => {
    calibrationCountdown--;
    if (calibrationCountdown > 0) {
      status.textContent = `✋ Hold hands OPEN & FLAT in ${calibrationCountdown}...`;
    } else {
      clearInterval(calibrationInterval);
      calibrationInterval = null;
      calibrate_hand(); // Calibrate hand bone lengths
      status.textContent = "✅ Hand Calibrated! Finger lengths locked.";
    }
  }, 1000);
}

async function main() {
  status.textContent = "Loading WASM...";
  await init();
  await wasmInit();
  console.log("🖐️ Hand Tracking WASM loaded");

  // ========================================================================
  // LOW-LATENCY CAMERA
  // ========================================================================
  status.textContent = "Requesting camera...";
  let stream;
  try {
    stream = await navigator.mediaDevices.getUserMedia({
      video: {
        width: { ideal: 640, max: 640 },
        height: { ideal: 360, max: 360 },
        frameRate: { ideal: 30, max: 30 },
        facingMode: "user",
      },
      audio: false,
    });

    console.log("📷 Camera ready");
  } catch (err) {
    status.textContent = "❌ Camera access denied";
    console.error("Camera error:", err);
    return;
  }

  video.srcObject = stream;
  await video.play();

  // ========================================================================
  // HAND LANDMARKER (21 landmarks per hand, up to 2 hands)
  // ========================================================================
  status.textContent = "Loading MediaPipe Hand Tracking...";
  const vision = await FilesetResolver.forVisionTasks(
    "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/wasm"
  );

  handLandmarker = await HandLandmarker.createFromOptions(vision, {
    baseOptions: {
      modelAssetPath:
        "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
      delegate: "GPU",
    },
    runningMode: "VIDEO",
    numHands: 2,
  });

  console.log("🖐️ MediaPipe Hand Tracking ready");
  status.textContent = "✅ Ready - Show your hands!";

  document.addEventListener("keydown", (e) => {
    if (e.key === "c" || e.key === "C") startCalibrationCountdown();
  });

  // ========================================================================
  // RENDER LOOP (120Hz) - Independent
  // ========================================================================
  function renderLoop(timestamp) {
    stats.begin();

    const frameTime = timestamp - lastRenderTime;
    lastRenderTime = timestamp;

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
  // DETECTION LOOP (requestVideoFrameCallback for smooth sync)
  // ========================================================================
  function onVideoFrame(now, metadata) {
    if (!video.videoWidth) {
      video.requestVideoFrameCallback(onVideoFrame);
      return;
    }

    const mpStart = performance.now();
    const results = handLandmarker.detectForVideo(video, mpStart);
    const mpTime = performance.now() - mpStart;

    stats.setMediapipe(mpTime);
    set_mediapipe_latency(mpTime);

    // Process each detected hand
    // Hand 0 = Left hand (from camera view, appears on right side)
    // Hand 1 = Right hand (from camera view, appears on left side)
    if (results.landmarks && results.landmarks.length > 0) {
      // Flatten all hand landmarks: [hand0_lm0_x, y, z, hand0_lm1_x, y, z, ..., hand1_lm0_x, ...]
      // 21 landmarks per hand, 3 floats per landmark = 63 floats per hand
      // Up to 2 hands = 126 floats max
      const numHands = results.landmarks.length;
      const flatArray = new Float32Array(numHands * 21 * 3);

      for (let h = 0; h < numHands; h++) {
        const hand = results.landmarks[h];
        for (let i = 0; i < 21; i++) {
          const lm = hand[i];
          const base = h * 21 * 3 + i * 3;
          flatArray[base] = lm.x;
          flatArray[base + 1] = lm.y;
          flatArray[base + 2] = lm.z;
        }
      }

      apply_hand_landmarks(flatArray, numHands);
    }

    video.requestVideoFrameCallback(onVideoFrame);
  }

  video.addEventListener("loadeddata", () => {
    lastRenderTime = performance.now();
    requestAnimationFrame(renderLoop);
    video.requestVideoFrameCallback(onVideoFrame);
  });

  if (video.readyState >= 2) {
    lastRenderTime = performance.now();
    requestAnimationFrame(renderLoop);
    video.requestVideoFrameCallback(onVideoFrame);
  }
}

main().catch((err) => {
  console.error("❌ Fatal error:", err);
  status.textContent = "❌ Error: " + err.message;
});
