# BOXING GAME - WEB ARCHITECTURE (FINAL)

## 🚀 PIVOT: Python Desktop → Web (Rust + WASM + WebGPU)

**Status:** LOCKED 🔒  
**Target:** High-End Web (Chrome/Edge/Safari)  
**Constraint 1:** NO WEBGL FALLBACK. Strict WebGPU requirement.  
**Constraint 2:** PREDICTIVE PHYSICS. Decoupled 120Hz Logic / 30Hz AI.

---

## I. THE "TIME TRAVEL" PROBLEM & SOLUTION

### The Problem: Latency Mismatch

- MediaPipe (AI) runs at ~30 FPS (33ms)
- Render loop runs at 120 FPS (8ms)
- **If we use MediaPipe coordinates directly, the hitbox is always ~25-40ms in the past**

### The Fix: Client-Side Prediction (Kalman Filter)

We treat the hand as a **physics object** in the Rust engine:

| Step        | Rate                    | Action                                              |
| ----------- | ----------------------- | --------------------------------------------------- |
| **Predict** | 120Hz (every frame)     | `Position += Velocity * dt` (instant hitbox update) |
| **Correct** | 30Hz (every ~4th frame) | Kalman update nudges estimate toward MediaPipe data |

**Result:** Hitbox aligns with user's proprioception, not camera lag.

---

## II. ARCHITECTURE: Zero-Copy Predictive Pipeline

```
┌─────────────────────────────────────────────────────────────────┐
│                        CAMERA FEED                               │
│                    (GPUExternalTexture)                          │
└───────────────────────────┬─────────────────────────────────────┘
                            │
            ┌───────────────┴───────────────┐
            ▼                               ▼
┌───────────────────────────┐   ┌───────────────────────────────┐
│ PATH A: The "Sensor"      │   │ PATH B: The "Visuals"          │
│ (Async AI - 30Hz)         │   │ (Sync GPU - 120Hz)             │
│                           │   │                                 │
│ • MediaPipe WASM          │   │ • WGSL Compute Shaders         │
│ • Extract 33 landmarks    │   │ • Motion blur                  │
│ • JS→WASM Bridge          │   │ • Velocity field               │
│ • Role: "Correction Data" │   │ • Impact particles             │
└─────────────┬─────────────┘   └─────────────┬───────────────────┘
              │                               │
              ▼                               │
┌───────────────────────────────────────────┐ │
│ PATH C: The "Brain" (Predictive Physics)  │ │
│                                           │ │
│ • Rust WASM + nalgebra                    │ │
│ • KalmanFilter struct                     │ │
│ • 120Hz: position += velocity * dt        │ │
│ • 30Hz: Kalman correction from Path A     │ │
│ • Real-time hitbox (predicted coords)     │ │
└─────────────┬─────────────────────────────┘ │
              │                               │
              └───────────────┬───────────────┘
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    WebGPU RENDER PASS                            │
│                                                                  │
│ • Draw skeleton using PREDICTED coordinates (green)             │
│ • Layer visual effects from Path B                              │
│ • 120Hz locked to display                                        │
└─────────────────────────────────────────────────────────────────┘
```

---

## III. EXECUTION PATHS

### Path A: The "Sensor" (Asynchronous AI)

| Property | Value                                          |
| -------- | ---------------------------------------------- |
| Tech     | MediaPipe (WASM/JS)                            |
| Rate     | ~30Hz (Variable)                               |
| Role     | **Correction Data only** - NOT source of truth |
| Output   | 33 Landmarks (x, y, z)                         |

### Path B: The "Visuals" (Synchronous GPU)

| Property   | Value                                             |
| ---------- | ------------------------------------------------- |
| Tech       | Rust wgpu + WGSL Compute Shaders                  |
| Rate       | 120Hz (Locked to display)                         |
| Role       | Visual feedback / "Juice"                         |
| Key Shader | `velocity_field.wgsl` - Punch force visualization |

### Path C: The "Brain" (Predictive Rust Logic)

| Property | Value                                 |
| -------- | ------------------------------------- |
| Tech     | Rust WASM + nalgebra (linear algebra) |
| Role     | Physics Simulation                    |

```rust
// Runs every 8ms (120Hz)
struct HandState {
    position: Vec3,
    velocity: Vec3,
    acceleration: Vec3,
    covariance: Mat3, // Uncertainty matrix
}

fn update(&mut self, dt: f32) {
    // 1. Physics Step (Prediction)
    self.position += self.velocity * dt;
    self.velocity += self.acceleration * dt;

    // 2. Decay/Friction (Prevents infinite drift)
    self.velocity *= 0.98;
}

fn on_mediapipe_data(&mut self, measurement: Vec3) {
    // 3. Correction Step (Kalman Update)
    let k_gain = self.calculate_kalman_gain();
    self.position = self.position + k_gain * (measurement - self.position);
}
```

---

## IV. TECH STACK (Strict)

| Component | Technology    | Reasoning                               |
| --------- | ------------- | --------------------------------------- |
| Language  | **Rust**      | Complex math (Kalman) without GC pauses |
| Math      | **nalgebra**  | Robust linear algebra for matrices      |
| Graphics  | **wgpu**      | Native WebGPU access                    |
| Shaders   | **WGSL**      | Compute shaders for effects             |
| AI        | **MediaPipe** | Reliable "Sensor" data                  |

---

## V. BROWSER COMPATIBILITY (No Fallbacks)

| Browser              | Status          | Action                          |
| -------------------- | --------------- | ------------------------------- |
| Chrome / Edge (113+) | ✅ Supported    | Run Game                        |
| Safari (18+)         | ✅ Supported    | Run Game                        |
| Firefox              | ❌ Experimental | Block: "Please use Chrome/Edge" |
| Mobile Chrome        | ✅ Supported    | Run Game (optimized)            |

---

## VI. DEVELOPMENT PHASES

### Phase 1: The Foundation (WebGPU Setup) ⬜

- [ ] Initialize Rust + wgpu
- [ ] Implement GPUExternalTexture camera feed
- [ ] Create "Hardware Check" screen (Blocks non-WebGPU users)

### Phase 2: The "Sensor" Bridge ⬜

- [ ] Integrate MediaPipe JS
- [ ] Create Rust-to-JS bridge for raw landmarks
- [ ] Debug: Draw raw MediaPipe (🔴 Red dots) vs camera - note the lag

### Phase 3: The Predictor (Physics Engine) ⬜

- [ ] Implement `KalmanFilter` struct in Rust
- [ ] Tune covariance matrices (Process Noise vs Measurement Noise)
- [ ] Debug: Draw Predicted (🟢 Green dots) - Goal: Green leads Red during fast movement

### Phase 4: Gameplay Logic ⬜

- [ ] Hitbox detection using Predicted (Green) coordinates
- [ ] Port Python scoring/combo logic

### Phase 5: The "Juice" (Compute Shaders) ⬜

- [ ] `motion_blur.wgsl` - Visual speed effect
- [ ] `impact_particles.wgsl` - Spawn particles on hit

---

## VII. FILE STRUCTURE

```
boxing-web/
├── Cargo.toml              # Add: nalgebra, wgpu
├── src/
│   ├── lib.rs              # Entry point
│   ├── engine.rs           # WGPU Loop
│   ├── physics/
│   │   ├── kalman.rs       # The Prediction Logic (The "Fix")
│   │   └── hitbox.rs       # Collision math
│   ├── bridge/
│   │   └── mediapipe.rs    # JS Data receiver
│   └── shaders/
│       ├── compute_velocity.wgsl
│       └── render_particles.wgsl
├── web/
│   └── main.js             # MediaPipe init + WASM glue
└── pkg/                    # WASM build output
```

---

## VIII. DEBUG VISUALIZATION

During development, display both coordinate sources:

| Marker        | Source           | Meaning                              |
| ------------- | ---------------- | ------------------------------------ |
| 🔴 Red dots   | Raw MediaPipe    | Where AI _thinks_ hand is (lagged)   |
| 🟢 Green dots | Kalman Predicted | Where hand _actually_ is (real-time) |

**Goal:** Green dots should **lead** red dots during fast movement.

---

## IX. EXPECTED PERFORMANCE

| Metric          | Value                     |
| --------------- | ------------------------- |
| Render FPS      | 120 (locked)              |
| Physics FPS     | 120 (locked)              |
| AI FPS          | ~30 (async)               |
| Total Latency   | **<10ms** (vs 40ms naive) |
| Hitbox Accuracy | **±5ms** (vs ±40ms naive) |

---

## X. PYTHON PROTOTYPE REFERENCE

The Python desktop prototype is complete:

- ✅ MediaPipe skeleton tracking
- ✅ Hybrid AI + heuristic punch detection
- ✅ Combo system + scoring
- ✅ Visual effects

**Location:** `d:/CLEAN/AUTOBOT/python-tracker/skeleton_hit_detector.py`

Port game logic from Python → Rust WASM for Phase 4.
