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
} from "../pkg/boxing_web.js";

const status = document.getElementById("status");
const video = document.getElementById("camera-video");

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
  status.textContent = "✅ Ready - Move your arms!";

  // 4. Detection + Render loop
  let lastTime = 0;
  let frameCount = 0;

  function detectFrame(timestamp) {
    // Throttle to ~30fps for detection (MediaPipe is heavy)
    if (timestamp - lastTime < 33) {
      requestAnimationFrame(detectFrame);
      return;
    }
    lastTime = timestamp;

    // Run pose detection
    const results = poseLandmarker.detectForVideo(video, timestamp);

    if (results.landmarks && results.landmarks[0]) {
      // Convert to flat Float32Array for Rust (33 landmarks * 3 coords = 99 floats)
      const landmarks = results.landmarks[0];
      const flatArray = new Float32Array(landmarks.length * 3);

      landmarks.forEach((lm, i) => {
        flatArray[i * 3] = lm.x; // 0-1 normalized X
        flatArray[i * 3 + 1] = lm.y; // 0-1 normalized Y
        flatArray[i * 3 + 2] = lm.z; // Relative depth
      });

      // Send landmarks to Rust
      update_landmarks(flatArray);

      // Debug: log every 60 frames
      frameCount++;
      if (frameCount % 60 === 0) {
        const rightWrist = landmarks[16];
        console.log(
          `🎯 Right wrist: (${rightWrist.x.toFixed(2)}, ${rightWrist.y.toFixed(
            2
          )})`
        );
      }
    }

    // Render frame with current landmarks
    render_frame();

    requestAnimationFrame(detectFrame);
  }

  // Start detection when video is ready
  video.addEventListener("loadeddata", () => {
    console.log("🎬 Video ready, starting detection loop");
    requestAnimationFrame(detectFrame);
  });

  // If video already loaded
  if (video.readyState >= 2) {
    console.log("🎬 Video already ready");
    requestAnimationFrame(detectFrame);
  }
}

main().catch((err) => {
  console.error("❌ Fatal error:", err);
  status.textContent = "❌ Error: " + err.message;
});
