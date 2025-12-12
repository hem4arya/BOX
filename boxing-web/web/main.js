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
} from "../pkg/boxing_web.js";

const status = document.getElementById("status");
const video = document.getElementById("camera-video");

// Physics timing
let lastPhysicsTime = 0;

// ONNX model session
let onnxSession = null;

// Punch detection state
const CONFIDENCE_THRESHOLD = 0.5;
const PUNCH_NAMES = ["JAB", "CROSS", "HOOK", "UPPERCUT", "IDLE"];

async function main() {
  status.textContent = "Loading WASM...";

  // Initialize WASM module
  await init();
  await wasmInit();
  console.log("🥊 Boxing Web WASM loaded");

  // Try to load ONNX model
  status.textContent = "Loading ML model...";
  try {
    ort.env.wasm.wasmPaths =
      "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.17.0/dist/";
    onnxSession = await ort.InferenceSession.create(
      "/assets/punch_classifier.onnx"
    );
    set_classifier_ready();
    console.log("🧠 Punch classifier loaded via onnxruntime-web");
  } catch (err) {
    console.warn("⚠️ Could not load classifier:", err.message);
    console.warn(
      "   Run: python convert_to_onnx.py in the boxing-web directory"
    );
    console.warn("   Then copy punch_classifier.onnx to assets/");
  }

  status.textContent = "Requesting camera...";

  // 2. Get camera stream
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
    console.error("Camera error:", err);
    return;
  }

  video.srcObject = stream;
  await video.play();
  console.log("📹 Camera started");

  status.textContent = "Loading MediaPipe...";

  // 3. Initialize MediaPipe Pose Landmarker
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

  console.log("🤖 MediaPipe Pose Landmarker ready");
  status.textContent = "✅ Ready - Press C to calibrate (T-pose)";

  // Keyboard handler for calibration
  document.addEventListener("keydown", (e) => {
    if (e.key === "c" || e.key === "C") {
      calibrate_depth();
      status.textContent = "✅ Calibrated! Punch to detect.";
    }
  });

  // 4. Detection + Render loop
  let lastDetectionTime = 0;
  let frameCount = 0;
  let lastClassifyTime = 0;

  async function runClassification() {
    if (!onnxSession || !is_buffer_ready()) return;

    // Only classify every 100ms (10Hz) to avoid overload
    const now = performance.now();
    if (now - lastClassifyTime < 100) return;
    lastClassifyTime = now;

    try {
      const bufferData = get_classification_buffer();
      if (!bufferData) return;

      // Create input tensor: (1, 30, 10)
      const inputTensor = new ort.Tensor("float32", bufferData, [1, 30, 10]);

      // Run inference
      const results = await onnxSession.run({ input: inputTensor });
      const output = results.output.data;

      // Softmax + argmax
      const maxLogit = Math.max(...output);
      const expSum = output.reduce((sum, x) => sum + Math.exp(x - maxLogit), 0);
      const probs = output.map((x) => Math.exp(x - maxLogit) / expSum);

      let maxIdx = 0;
      let maxProb = probs[0];
      for (let i = 1; i < probs.length; i++) {
        if (probs[i] > maxProb) {
          maxIdx = i;
          maxProb = probs[i];
        }
      }

      // Send result to Rust
      set_punch_result(maxIdx, maxProb);

      // Log punch detection
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
    // Physics tick at ~120Hz
    const physicsElapsed = timestamp - lastPhysicsTime;
    if (physicsElapsed >= 8) {
      physics_tick(physicsElapsed / 1000.0);
      lastPhysicsTime = timestamp;
    }

    // MediaPipe detection at ~30Hz
    const detectionElapsed = timestamp - lastDetectionTime;
    if (detectionElapsed >= 33) {
      lastDetectionTime = timestamp;

      const results = poseLandmarker.detectForVideo(video, timestamp);

      if (results.landmarks && results.landmarks[0]) {
        const landmarks = results.landmarks[0];
        const flatArray = new Float32Array(landmarks.length * 3);

        landmarks.forEach((lm, i) => {
          flatArray[i * 3] = lm.x;
          flatArray[i * 3 + 1] = lm.y;
          flatArray[i * 3 + 2] = lm.z;
        });

        // Send to Rust (triggers Kalman UPDATE + Feature extraction)
        update_landmarks(flatArray);

        // Run classification (async)
        runClassification();

        frameCount++;
      }
    }

    // Render every frame
    render_frame();

    requestAnimationFrame(gameLoop);
  }

  // Start game loop
  video.addEventListener("loadeddata", () => {
    console.log("🎬 Starting game loop");
    lastPhysicsTime = performance.now();
    requestAnimationFrame(gameLoop);
  });

  if (video.readyState >= 2) {
    console.log("🎬 Video already ready");
    lastPhysicsTime = performance.now();
    requestAnimationFrame(gameLoop);
  }
}

main().catch((err) => {
  console.error("❌ Fatal error:", err);
  status.textContent = "❌ Error: " + err.message;
});
