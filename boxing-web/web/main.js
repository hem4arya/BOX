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
    const camColor = this.cameraLatency > 50 ? "#ff4444" : "#00ff00";

    // Get arm debug info from WASM
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
      <b>Camera:</b> <span style="color:${camColor}">${this.cameraLatency.toFixed(
      0
    )}ms</span><br>
      <b>MediaPipe:</b> <span style="color:${mpColor}">${this.mediapipeMs.toFixed(
      0
    )}ms</span><br>
      <b>Physics:</b> ${this.physicsMs.toFixed(2)}ms<br>
      <span style="color:#0ff">${this.armInfo.replace(/\n/g, "<br>")}</span><br>
      <span style="color:#888">VideoFrame API ✓</span>
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
let poseLandmarker = null;

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

  // ========================================================================
  // TIER 1 FIX #1: LOW-LATENCY CAMERA CONSTRAINTS
  // ========================================================================
  status.textContent = "Requesting camera (low-latency mode)...";
  let stream;
  try {
    stream = await navigator.mediaDevices.getUserMedia({
      video: {
        width: { ideal: 640, max: 640 },
        height: { ideal: 360, max: 360 }, // Lower res = faster
        frameRate: { ideal: 30, max: 30 },
        facingMode: "user",
        // Low-latency hints (browser may ignore)
        latency: { ideal: 0 },
        resizeMode: "none",
      },
      audio: false,
    });

    // Disable auto-adjustments that cause delays
    const track = stream.getVideoTracks()[0];
    const capabilities = track.getCapabilities?.() || {};
    const constraints = {};

    if (capabilities.exposureMode?.includes("manual")) {
      constraints.exposureMode = "manual";
    }
    if (capabilities.focusMode?.includes("manual")) {
      constraints.focusMode = "manual";
    }
    if (capabilities.whiteBalanceMode?.includes("manual")) {
      constraints.whiteBalanceMode = "manual";
    }

    if (Object.keys(constraints).length > 0) {
      await track.applyConstraints({ advanced: [constraints] });
      console.log("📷 Camera: manual mode applied", constraints);
    }

    console.log("📷 Camera: low-latency constraints applied");
  } catch (err) {
    status.textContent = "❌ Camera access denied";
    console.error("Camera error:", err);
    return;
  }

  // Keep video element for fallback/preview
  video.srcObject = stream;
  await video.play();

  status.textContent = "Loading MediaPipe...";
  const vision = await FilesetResolver.forVisionTasks(
    "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/wasm"
  );

  poseLandmarker = await PoseLandmarker.createFromOptions(vision, {
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
  // TIER 1 FIX #2: VIDEOFRAME API (Zero-buffering)
  // ========================================================================
  const videoTrack = stream.getVideoTracks()[0];

  // Check if VideoFrame API is supported
  if (typeof MediaStreamTrackProcessor !== "undefined") {
    console.log("🎬 Using VideoFrame API (zero-buffering)");

    const trackProcessor = new MediaStreamTrackProcessor({ track: videoTrack });
    const reader = trackProcessor.readable.getReader();

    async function processVideoFrames() {
      while (true) {
        try {
          const { value: frame, done } = await reader.read();
          if (done) break;

          // Calculate camera-to-process latency
          const captureTime = frame.timestamp / 1000; // microseconds to ms
          const now = performance.now();
          stats.setCameraLatency(now - captureTime);

          // Run MediaPipe on raw frame (no video element buffering!)
          const mpStart = performance.now();
          const results = poseLandmarker.detectForVideo(frame, now);
          stats.setMediapipe(performance.now() - mpStart);
          set_mediapipe_latency(performance.now() - mpStart);

          // CRITICAL: Close frame to release memory
          frame.close();

          if (results.landmarks && results.landmarks[0]) {
            const landmarks = results.landmarks[0];
            const flatArray = new Float32Array(landmarks.length * 3);
            landmarks.forEach((lm, i) => {
              flatArray[i * 3] = lm.x;
              flatArray[i * 3 + 1] = lm.y;
              flatArray[i * 3 + 2] = lm.z;
            });
            apply_mediapipe_correction(flatArray);
          }
        } catch (err) {
          console.error("VideoFrame error:", err);
          break;
        }
      }
    }

    processVideoFrames();
    lastRenderTime = performance.now();
    requestAnimationFrame(renderLoop);
  } else {
    // Fallback: Use video element (older browsers)
    console.log("⚠️ VideoFrame API not supported, using <video> fallback");

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
          apply_mediapipe_correction(flatArray);
        }

        await new Promise((r) => setTimeout(r, 0));
      }
    }

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
}

main().catch((err) => {
  console.error("❌ Fatal error:", err);
  status.textContent = "❌ Error: " + err.message;
});
