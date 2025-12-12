// 1. Check WebGPU support BEFORE loading anything
if (!navigator.gpu) {
  document.getElementById("error-msg").style.display = "block";
  document.getElementById("game-canvas").style.display = "none";
  throw new Error("WebGPU not supported");
}

import { PoseLandmarker, FilesetResolver } from "@mediapipe/tasks-vision";
import * as ort from "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.17.0/dist/esm/ort.min.js";
import init, {
  init as wasmInit,
  update_landmarks,
  render_frame,
  physics_tick,
  calibrate_depth,
  set_classifier_ready,
  get_classification_buffer,
  is_buffer_ready,
  set_punch_result,
  set_frame_metrics,
  set_mediapipe_latency,
  set_onnx_latency,
  set_physics_time,
  get_debug_overlay_text,
} from "../pkg/boxing_web.js";

const status = document.getElementById("status");
const video = document.getElementById("camera-video");
const debugOverlay = document.getElementById("debug-overlay");

// Timing
let lastPhysicsTime = 0;
let lastFrameTime = 0;
let fps = 0;
let frameCount = 0;
let lastFpsUpdate = 0;

// ONNX model
let onnxSession = null;
const CONFIDENCE_THRESHOLD = 0.5;
const PUNCH_NAMES = ["JAB", "CROSS", "HOOK", "UPPERCUT", "IDLE"];

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

  // Load ONNX model
  status.textContent = "Loading ML model...";
  try {
    ort.env.wasm.wasmPaths =
      "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.17.0/dist/";
    onnxSession = await ort.InferenceSession.create(
      "/assets/punch_classifier.onnx"
    );
    set_classifier_ready();
    console.log("🧠 Punch classifier loaded");
  } catch (err) {
    console.warn("⚠️ Could not load classifier:", err.message);
  }

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

  let lastDetectionTime = 0;
  let lastClassifyTime = 0;

  async function runClassification() {
    if (!onnxSession || !is_buffer_ready()) return;
    const now = performance.now();
    if (now - lastClassifyTime < 100) return;
    lastClassifyTime = now;

    try {
      const onnxStart = performance.now();
      const bufferData = get_classification_buffer();
      if (!bufferData) return;

      const inputTensor = new ort.Tensor("float32", bufferData, [1, 30, 10]);
      const results = await onnxSession.run({ input: inputTensor });
      const output = results.output.data;

      set_onnx_latency(performance.now() - onnxStart);

      const maxLogit = Math.max(...output);
      const expSum = output.reduce((sum, x) => sum + Math.exp(x - maxLogit), 0);
      const probs = output.map((x) => Math.exp(x - maxLogit) / expSum);

      let maxIdx = 0,
        maxProb = probs[0];
      for (let i = 1; i < probs.length; i++) {
        if (probs[i] > maxProb) {
          maxIdx = i;
          maxProb = probs[i];
        }
      }

      set_punch_result(maxIdx, maxProb);
      if (maxProb > CONFIDENCE_THRESHOLD && maxIdx < 4) {
        console.log(
          `🥊 ${PUNCH_NAMES[maxIdx]} (${(maxProb * 100).toFixed(0)}%)`
        );
      }
    } catch (err) {
      console.error("Classification error:", err);
    }
  }

  function gameLoop(timestamp) {
    // FPS calculation
    const frameTime = timestamp - lastFrameTime;
    lastFrameTime = timestamp;
    frameCount++;

    if (timestamp - lastFpsUpdate >= 500) {
      fps = (frameCount * 1000) / (timestamp - lastFpsUpdate);
      set_frame_metrics(fps, frameTime);
      frameCount = 0;
      lastFpsUpdate = timestamp;

      // Update debug overlay every 500ms
      debugOverlay.textContent = get_debug_overlay_text();
    }

    // Physics tick ~120Hz
    const physicsElapsed = timestamp - lastPhysicsTime;
    if (physicsElapsed >= 8) {
      const physicsStart = performance.now();
      physics_tick(physicsElapsed / 1000.0);
      set_physics_time(performance.now() - physicsStart);
      lastPhysicsTime = timestamp;
    }

    // MediaPipe ~30Hz
    const detectionElapsed = timestamp - lastDetectionTime;
    if (detectionElapsed >= 33) {
      lastDetectionTime = timestamp;

      const mpStart = performance.now();
      const results = poseLandmarker.detectForVideo(video, timestamp);
      set_mediapipe_latency(performance.now() - mpStart);

      if (results.landmarks && results.landmarks[0]) {
        const landmarks = results.landmarks[0];
        const flatArray = new Float32Array(landmarks.length * 3);
        landmarks.forEach((lm, i) => {
          flatArray[i * 3] = lm.x;
          flatArray[i * 3 + 1] = lm.y;
          flatArray[i * 3 + 2] = lm.z;
        });
        update_landmarks(flatArray);
        runClassification();
      }
    }

    render_frame();
    requestAnimationFrame(gameLoop);
  }

  video.addEventListener("loadeddata", () => {
    lastPhysicsTime = performance.now();
    lastFrameTime = performance.now();
    lastFpsUpdate = performance.now();
    requestAnimationFrame(gameLoop);
  });

  if (video.readyState >= 2) {
    lastPhysicsTime = performance.now();
    lastFrameTime = performance.now();
    lastFpsUpdate = performance.now();
    requestAnimationFrame(gameLoop);
  }
}

main().catch((err) => {
  console.error("❌ Fatal error:", err);
  status.textContent = "❌ Error: " + err.message;
});
