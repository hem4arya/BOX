// 1. Check WebGPU support BEFORE loading anything
if (!navigator.gpu) {
  document.getElementById("error-msg").style.display = "block";
  document.getElementById("game-canvas").style.display = "none";
  throw new Error("WebGPU not supported");
}

import { PoseLandmarker, FilesetResolver } from "@mediapipe/tasks-vision";
import init, {
  init as wasmInit,
  update_landmarks,
  render_frame,
  physics_tick,
  calibrate_depth,
} from "../pkg/boxing_web.js";

const status = document.getElementById("status");
const video = document.getElementById("camera-video");

// Physics timing
const PHYSICS_DT = 1.0 / 120.0; // 120Hz physics
let lastPhysicsTime = 0;

async function main() {
  status.textContent = "Loading WASM...";

  // Initialize WASM module
  await init();
  await wasmInit();
  console.log("🥊 Boxing Web WASM loaded");

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
      status.textContent =
        "✅ Calibrated! Move your arms fast to see prediction.";
    }
  });

  // 4. Detection + Render loop
  let lastDetectionTime = 0;
  let frameCount = 0;

  function gameLoop(timestamp) {
    // Physics tick at ~120Hz
    const physicsElapsed = timestamp - lastPhysicsTime;
    if (physicsElapsed >= 8) {
      // ~120fps
      physics_tick(physicsElapsed / 1000.0);
      lastPhysicsTime = timestamp;
    }

    // MediaPipe detection at ~30Hz
    const detectionElapsed = timestamp - lastDetectionTime;
    if (detectionElapsed >= 33) {
      // ~30fps
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

        // Send to Rust (triggers Kalman UPDATE at 30Hz)
        update_landmarks(flatArray);

        // Debug log every 2 seconds
        frameCount++;
        if (frameCount % 60 === 0) {
          const rightWrist = landmarks[16];
          console.log(
            `🎯 Wrist: (${rightWrist.x.toFixed(2)}, ${rightWrist.y.toFixed(2)})`
          );
        }
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
