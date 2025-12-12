# Python Tracker Project Documentation

## 1. Project Overview

The **Python Tracker Project** is a sophisticated computer vision system designed for real-time body tracking, specifically tailored for a shadow-boxing game. Unlike basic hand tracking demonstrations, this system employs a multi-layered sensor fusion approach to achieve robustness against high-speed motion blur, occlusion, and depth ambiguity.

The system integrates **MediaPipe** (for high-fidelity landmarks), **CV2 Optical Flow** (for bridging tracking gaps), **Kalman/OneEuro Filters** (for signal denoising), and custom **Physics Engines** (Ballistic extrapolation) to create a "lossless" tracking experience.

## 2. System Architecture

The project is built on a modular "Master Vision" architecture where specialized components handle distinct responsibilities:

- **Master Vision (`master_vision.py`)**: The central hub that coordinates input, processing, and state management.
- **Tracking Layer**: Uses `hand_tracker_invincible.py` and `motion_tracker.py` to maintain 2D landmark positions even when AI detection fails.
- **Depth Layer**: `depth_estimator.py` and `depth_manager.py` convert 2D camera inputs into 3D spatial coordinates using geometric constraints (Anchor Method).
- **Validation Layer**: `flow_gated_validator.py` and `fusion_detector.py` weigh different sensor inputs to reject false positives.
- **Logic Layer**: `punch_fsm.py` and `skeleton_hit_detector.py` interpret raw motion data into semantic game events (e.g., "JAB", "HOOK").

---

## 3. Module Breakdown

### 3.1. Core Application

#### `boxing_game.py`

- **Purpose**: The main entry point for the game application.
- **Logic**: Manages the Pygame event loop, rendering, and game state (score, health, round timer).
- **Key Components**:
  - `PunchingBag`: Interactive target with collision physics.
  - `Glove`: Visual representation of hand tracking data.
  - `HandTracker`: A simplified tracker instance (or wrapper) for the game loop.

### 3.2. Vision & Tracking

#### `master_vision.py`

- **Purpose**: The V2 unified vision system.
- **Logic**: Integrates Pose landmarks (shoulder, elbow, wrist) for stability and Hand landmarks for gesture details. Calculates kinematic chains to derive arm velocity and length.
- **Math**:
  - Kinematic Chain: $L_{arm} = ||P_{shoulder} - P_{elbow}|| + ||P_{elbow} - P_{wrist}||$
  - Velocity: $\vec{v} = \frac{P_t - P_{t-1}}{\Delta t}$

#### `hand_tracker_invincible.py`

- **Purpose**: A robust, specific tracker implementation designed to "never lose".
- **Logic**: Implements a 4-layer fallback hierarchy:
  1.  **MediaPipe**: High precision.
  2.  **YOLO**: Robust detection on blurred images.
  3.  **Optical Flow**: Pixel-level tracking when AI fails.
  4.  **Motion Energy**: Last-resort blob tracking.
- **Features**: Includes "Pose Locking" to freeze gesture state during rapid punches to prevent flickering.

#### `motion_tracker.py`

- **Purpose**: Dedicated optical flow and physics tracking.
- **Logic**: Uses Lucas-Kanade Optical Flow to track feature points on the hand. Implements "drift prevention" by constraining the flow ROI (Region of Interest) and applying velocity limits ($V_{max} = 150px/frame$). Includes physics "coasting" (inertia) when visual tracking is totally lost.

#### `hybrid_detector.py`

- **Purpose**: Fuses different detection neural networks.
- **Logic**: Primary extraction via MediaPipe; if confidence drops, switches to YOLOv8-Nano to re-acquire the bounding box, then runs MediaPipe on the cropped region (ROI) for efficiency.

#### `occlusion_handler.py`

- **Purpose**: Maintains state when hands cross or leave the frame.
- **Logic**: Uses a velocity-based prediction model ($P_{new} = P_{old} + \vec{v} \cdot \Delta t$) to estimate positions during short occlusion windows (< 0.5s).

### 3.3. Depth & Geometry

#### `depth_estimator.py`

- **Purpose**: Calculates "True" geometric depth without a depth camera.
- **Algorithm**: **Anchor Method**.
  - Uses the shoulder midpoint as an "Anchor".
  - Normalizes arm length against "Shoulder Width" ($W_{shoulders}$) to be distance-invariant.
- **Math**:
  - Scale Factor $S = ||P_{L\_Shoulder} - P_{R\_Shoulder}||$
  - Normalized Reach $R = \frac{||P_{wrist} - P_{shoulder}||}{S}$
  - **Arm Extension Coefficient (AEC)**: A derived percentage (0-100%) indicating how close the arm is to full anatomical extension.

#### `depth_manager.py`

- **Purpose**: Manages per-user calibration.
- **Logic**: Stores "Chin" (retracted) and "Reach" (extended) Z-values for each hand. Normalizes current Z input to a linear 0-1 range based on these calibration set points.

#### `ik_solver.py`

- **Purpose**: Inverse Kinematics for hand gesture recognition.
- **Logic**: Calculates angles between finger bone vectors to determine "curl".
- **Math**:
  - Vector Angle: $\theta = \arccos(\frac{\vec{u} \cdot \vec{v}}{||\vec{u}|| ||\vec{v}||})$
  - Detects `FIST`, `OPEN`, `PEACE`, etc., based on finger curl thresholds.

### 3.4. Signal Processing & Validation

#### `flow_gated_validator.py`

- **Purpose**: Sensor fusion gating.
- **Logic**: Decides which signal to trust.
  - IF `MediaPipe_Conf` > High: Use MediaPipe.
  - ELSE IF `Flow_Quality` > High: Use Optical Flow (Projected).
  - ELSE: Use Ballistic Prediction.

#### `fusion_detector.py`

- **Purpose**: Weighted voting system for punch confirmation.
- **Logic**: Aggregates scores from Neural Nets, Flow, and Physics to output a single `Confidence` score [0.0 - 1.0].
- **Features**: Includes `EnergyFilter` to reject random motion spikes that don't match the velocity profile of a punch (smooth acceleration curve).

#### `signal_processor.py`

- **Purpose**: Signal smoothing.
- **Algorithm**: **OneEuro Filter**.
  - An adaptive low-pass filter.
  - **Static State**: High smoothing (removes jitter).
  - **Dynamic State**: Low smoothing (minimizes latency during punches).
- **Math**: Adjusts cutoff frequency $f_c$ relative to current velocity magnitude $|\vec{v}|$.

### 3.5. Game Logic & State Machines

#### `punch_fsm.py`

- **Purpose**: Finite State Machine for punch lifecycle.
- **States**: `IDLE` $\to$ `ACCELERATION` $\to$ `EXTENSION` $\to$ `RETRACTION` $\to$ `COOLDOWN`.
- **Integration**: Transitions are driven by voters (Skeleton Velocity, Flow Magnitude, Geometry/Reach).

#### `skeleton_hit_detector.py`

- **Purpose**: High-level hit validation.
- **Logic**: Checks for "Physically Valid" hits.
  - **Sustained Velocity**: Rolling average > threshold.
  - **Straight Arm**: Elbow angle > $150^\circ$.
  - **Kinetic Energy**: $E_k = \frac{1}{2}mv^2$ proxy check.
  - **Trajectory Phase**: Must be in `EXTENSION` or `PEAK` phase.

#### `ballistic_engine.py`

- **Purpose**: Simulates projectile physics for the hands.
- **Logic**: When a punch is detected, it can spawn a "Ballistic Object" that travels independently, decoupling the visual hand from the tracking input to ensure the punch completes visually even if the camera loses the fast-moving hand.

### 3.6. Utilities & Legacy

- `hand_renderer.py`: Visualizes the skeleton, including "Ghost" hands for debug visualization of optical flow / prediction states.
- `telemetry.py`: Data logger (CSV) for recording punch metrics (accel, vel, pos) for analysis/training.
- `lstm_predictor.py`: A PyTorch LSTM implementation to predict future landmarks ($t+1, t+2$) based on history, intended to reduce perceived latency.
- `onnx_detector.py`: A wrapper for running ONNX models (e.g., YOLO) on GPU via `onnxruntime-gpu`.
- `hand_tracker.py` & `hand_tracker_mt.py`: Older or alternative threading implementations of the base tracker.

---

## 4. Key Mathematical Concepts Used

1.  **Vector Mathematics**:

    - Used extensively for calculating distances, relative positions, and angles.
    - **Dot Product**: For calculating the angle between the upper arm and forearm (Elbow Angle).
    - **Euclidean Distance**: For skeleton bone lengths and spatial reach.

2.  **Signal Filtering**:

    - **OneEuro Filter**: $x_i = \alpha x_i + (1-\alpha)x_{i-1}$, where $\alpha$ varies with velocity.
    - **Kalman Filter** (referenced in `skeleton_hit_detector`): Used for optimal state estimation of position/velocity from noisy measurements.

3.  **Kinematics**:

    - **Forward Kinematics**: Enforcing bone length constraints.
    - **Inverse Kinematics (IK)**: Simplified analytical IK to deduce geometric configurations (finger curl, arm extension) from joint positions.

4.  **Optical Flow**:

    - **Lucas-Kanade Method**: Solves for $\vec{v} = (u,v)$ by minimizing the error $\sum (I(x,y) - J(x+u, y+v))^2$ in a local window.

5.  **State Machines**:
    - Deterministic transition logic based on multi-variable thresholds (Voting System).

---

## 5. Conclusion

The Python Tracker project represents a highly optimized "V2" approach to webcam-based boxing. It moves beyond simple API calls by adding layers of physics simulation, geometric validation, and signal processing to solve the specific challenges of tracking high-speed combat movements.
