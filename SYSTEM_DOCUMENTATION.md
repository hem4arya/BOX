# 🥊 Punch Detection System - Complete Documentation

## System Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           PUNCH DETECTION SYSTEM                            │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌───────────────┐    ┌────────────────────┐    ┌─────────────────────┐    │
│  │   CAMERA      │───▶│   MEDIAPIPE POSE   │───▶│  SKELETON TRACKER   │    │
│  │   INPUT       │    │   (33 Landmarks)   │    │  (Shoulder-Elbow-   │    │
│  │   (640x480)   │    │                    │    │   Wrist)            │    │
│  └───────────────┘    └────────────────────┘    └──────────┬──────────┘    │
│                                                            │               │
│                       ┌────────────────────────────────────┼───────────────┤
│                       ▼                                    ▼               │
│  ┌─────────────────────────────┐    ┌──────────────────────────────────┐  │
│  │   PHYSICS-BASED DEPTH       │    │   AI-BASED CLASSIFICATION       │  │
│  │   ─────────────────────     │    │   ────────────────────────      │  │
│  │   • Anthropometric Depth    │    │   • 1D-CNN Classifier           │  │
│  │   • IK Solver (Forearm)     │    │   • 30 frames × 10 features     │  │
│  │   • Elbow Angle Gating      │    │   • JAB/CROSS/HOOK/UPPERCUT     │  │
│  └──────────────┬──────────────┘    └──────────────┬───────────────────┘  │
│                 │                                   │                      │
│                 └───────────────┬───────────────────┘                      │
│                                 ▼                                          │
│  ┌─────────────────────────────────────────────────────────────────────┐  │
│  │                      HIT DETECTION LOGIC                             │  │
│  │   ─────────────────────────────────────────                          │  │
│  │   HYBRID: (AI confidence > 50%) OR (velocity > 0.08 + depth > 30%)  │  │
│  └─────────────────────────────────┬───────────────────────────────────┘  │
│                                    │                                       │
│                                    ▼                                       │
│  ┌─────────────────────────────────────────────────────────────────────┐  │
│  │                      GAME EFFECTS ENGINE                             │  │
│  │   ─────────────────────────────────────                              │  │
│  │   • Hit Flash (color = punch type)                                   │  │
│  │   • Punch Type Popup (shrinking animation)                           │  │
│  │   • Combo Counter (x2, x3, x4...)                                    │  │
│  │   • Score Display (base + combo bonus)                               │  │
│  └─────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## File Breakdown

### Core Files (In Use)

| File                         | Purpose                                              | Lines |
| ---------------------------- | ---------------------------------------------------- | ----- |
| `skeleton_hit_detector.py`   | Main application - punch detection with game effects | 1842  |
| `train_classifier.py`        | Train 1D-CNN on synthetic data                       | 296   |
| `generate_synthetic_data.py` | Create 12,500 synthetic punch trajectories           | 531   |
| `punch_classifier.pth`       | Trained PyTorch model weights                        | -     |
| `synthetic_punches.npz`      | Training dataset (12,500 samples)                    | -     |

---

# FILE 1: skeleton_hit_detector.py

## Purpose

The main application that runs the punch detection game. It captures webcam input, detects body pose using MediaPipe, calculates punch metrics (velocity, depth, angle), classifies punch types using AI, and displays gamified visual feedback.

## Key Components

### 1. PunchClassifierCNN (Lines 35-81)

A 1D Convolutional Neural Network for punch type classification.

**Architecture:**

```
Input: (batch, 30, 10)  →  30 frames × 10 features per frame

Layer 1: Conv1d(10→32) + BatchNorm + ReLU + MaxPool2  →  (batch, 32, 15)
Layer 2: Conv1d(32→64) + BatchNorm + ReLU + MaxPool2  →  (batch, 64, 7)
Layer 3: Conv1d(64→128) + BatchNorm + ReLU + AdaptiveAvgPool1  →  (batch, 128, 1)

Classifier: Dropout(0.3) → Linear(128→64) → ReLU → Dropout(0.2) → Linear(64→5)

Output: (batch, 5)  →  Probabilities for [jab, cross, hook, uppercut, idle]
```

**Math - 1D Convolution:**

```
y[i] = Σ(k=0 to K-1) x[i+k] × w[k] + b

Where:
- x = input sequence
- w = kernel weights
- K = kernel size (3)
- b = bias
```

---

### 2. TemporalPunchClassifier (Lines 112-266)

A rule-based fallback classifier using biomechanical patterns.

**Features Tracked:**

- Velocity: Speed of wrist movement
- Depth Change: Forward/backward motion ratio
- Elbow Angle: Arm extension (90° = bent, 180° = straight)
- Direction Consistency: How straight is the punch path

**Classification Rules:**

```
JAB:     velocity > 0.12 AND depth > 50% AND elbow > 150° AND quick recovery
CROSS:   velocity > 0.15 AND depth > 60% AND elbow > 160° AND body rotation
HOOK:    horizontal_velocity > vertical_velocity AND arcing path
UPPERCUT: vertical_velocity > horizontal_velocity AND upward motion
```

---

### 3. ArmIKSolver (Lines 272-375)

Inverse Kinematics solver using forearm-only geometry for 3D depth estimation.

**Why Forearm-Only?**

- Fewer joints = less error propagation
- Forearm length more consistent than full arm
- Simpler geometric calculation
- Less sensitive to shoulder/elbow detection noise

**Math - Pythagoras Theorem for Depth:**

```
Given:
- L = Calibrated forearm length (from T-pose)
- P = Current 2D projected forearm length
- Z = Depth (what we want to find)

Since L² = P² + Z² (3D to 2D projection)

Therefore:
Z = √(L² - P²)

Depth Percentage:
depth_percent = (Z / L) × 100%
```

**Calibration Process:**

1. User performs T-pose (arms extended sideways)
2. In T-pose, Z ≈ 0, so 2D length ≈ 3D length
3. Store this as `calibrated_forearm_length`
4. All future measurements compare against this

---

### 4. KalmanStateEstimator (Lines 376-536)

A Kalman Filter for smoothing noisy position data and estimating velocity/acceleration.

**State Vector:**

```
x = [x, y, vx, vy, ax, ay]ᵀ

Where:
- (x, y) = position
- (vx, vy) = velocity
- (ax, ay) = acceleration
```

**Prediction Step (Physics Model):**

```
x(t+dt) = x(t) + vx(t)×dt + 0.5×ax(t)×dt²
vx(t+dt) = vx(t) + ax(t)×dt
ax(t+dt) = ax(t)  [assume constant]
```

**Update Step (Measurement Correction):**

```
K = P × Hᵀ × (H × P × Hᵀ + R)⁻¹   [Kalman Gain]
x = x + K × (z - H × x)            [State Update]
P = (I - K × H) × P                [Covariance Update]

Where:
- P = state covariance
- H = measurement matrix [1,0,0,0,0,0; 0,1,0,0,0,0]
- R = measurement noise covariance
- z = measurement (observed position)
```

---

### 5. SkeletonHitDetector (Lines 618-1516)

The main detector class that orchestrates all components.

**Key Parameters:**
| Parameter | Default | Description |
|-----------|---------|-------------|
| velocity_threshold | 0.17 | Minimum velocity for hit detection |
| cooldown_frames | 15 | Frames between consecutive hits |
| AI_CONFIDENCE_THRESHOLD | 0.5 | Minimum AI confidence for punch |

**Anthropometric Depth Calculation (Lines 749-839):**

```python
def calculate_anthropometric_depth(pose_landmarks, hand):
    # Get skeleton points
    shoulder, elbow, wrist = get_arm_landmarks(hand)

    # Calculate 2D arm segments
    upper_arm_2d = distance(shoulder, elbow)
    forearm_2d = distance(elbow, wrist)
    total_arm_2d = upper_arm_2d + forearm_2d

    # Compare to calibrated length
    if calibrated:
        depth = sqrt(calibrated_arm_length² - total_arm_2d²)
        depth_percent = (depth / calibrated_arm_length) × 100

    return depth, depth_percent
```

**Elbow Angle Calculation (Lines 1140-1152):**

```python
def calculate_elbow_angle(shoulder, elbow, wrist):
    # Vector from elbow to shoulder
    v1 = shoulder - elbow

    # Vector from elbow to wrist
    v2 = wrist - elbow

    # Angle using dot product
    cos_angle = (v1 · v2) / (|v1| × |v2|)
    angle = arccos(cos_angle) × (180/π)

    return angle  # 90° = fully bent, 180° = fully straight
```

**Velocity Calculation (Lines 1227-1250):**

```python
# 3-frame averaging to filter jitter
VELOCITY_FRAMES = 3
DEAD_ZONE = 0.02

if len(position_history) >= VELOCITY_FRAMES + 1:
    old_pos = position_history[-VELOCITY_FRAMES-1]
    distance = |current_pos - old_pos|
    raw_velocity = distance / VELOCITY_FRAMES

    # Apply dead zone
    velocity = 0 if raw_velocity < DEAD_ZONE else raw_velocity
```

**AI Feature Extraction (Lines 1033-1077):**

```python
def extract_ai_features(pose_landmarks, hand, velocity, velocity_vector):
    """Extract 10 features per frame for AI classifier."""
    features = [
        wrist.x, wrist.y,      # 1-2: Wrist position
        elbow.x, elbow.y,      # 3-4: Elbow position
        shoulder.x, shoulder.y, # 5-6: Shoulder position
        elbow_angle / 180.0,   # 7: Normalized elbow angle
        min(velocity * 2, 1.0), # 8: Scaled velocity
        direction_x, direction_y # 9-10: Movement direction
    ]
    return features  # 10 features total
```

**Hybrid Hit Detection Logic (Lines 1462-1493):**

```python
# Method 1: AI Classification
is_punch = ai_punch_type in ['jab', 'cross', 'hook', 'uppercut']
is_confident = ai_confidence > 0.5

# Method 2: Velocity Fallback
is_fast = velocity > 0.08
is_straight = elbow_angle >= 130
is_forward = effective_depth > 30
velocity_fallback = is_fast and is_straight and is_forward

# Combined Decision
if (is_punch and is_confident) or velocity_fallback:
    if not in_cooldown:
        register_hit()
```

---

### 6. Game Effects Engine (Lines 1562-1786)

**Hit Flash Effect:**

```python
if hit_flash_timer > 0:
    flash_intensity = hit_flash_timer / HIT_FLASH_DURATION
    overlay = frame.copy()
    cv2.rectangle(overlay, (0,0), (w,h), punch_color, -1)
    frame = cv2.addWeighted(overlay, flash_intensity * 0.25, frame, 1, 0)
```

**Scoring System:**

```python
PUNCH_SCORES = {
    'jab': 10,
    'cross': 15,
    'hook': 20,
    'uppercut': 25
}

# Combo bonus: +5 points per combo level
total_score = base_score + (combo_count - 1) * 5
```

**Combo System:**

```python
COMBO_TIMEOUT = 45  # frames (~1.5 seconds at 30 FPS)

on_hit():
    combo_count += 1
    combo_timer = COMBO_TIMEOUT

each_frame():
    if combo_timer > 0:
        combo_timer -= 1
    else:
        combo_count = 0  # Reset on timeout
```

---

# FILE 2: train_classifier.py

## Purpose

Trains the 1D-CNN model on synthetic punch data.

## Training Pipeline

```
1. Load Data
   synthetic_punches.npz → X(12500, 30, 10), y(12500,)

2. Split Data
   Train: 80% (10,000 samples)
   Test: 20% (2,500 samples)

3. Create Model
   PunchClassifierCNN (40,933 parameters)

4. Train (30 epochs)
   Optimizer: Adam (lr=0.001)
   Loss: CrossEntropyLoss
   Scheduler: StepLR (step=10, gamma=0.5)

5. Evaluate
   Per-class accuracy
   Overall accuracy

6. Save
   punch_classifier.pth
```

## Key Functions

**load_data():**

- Loads `synthetic_punches.npz`
- Shuffles and splits into train/test sets
- Returns TensorDataset objects

**train_model():**

- Runs training loop for specified epochs
- Tracks best accuracy and saves best weights
- Uses learning rate scheduler

**evaluate_model():**

- Computes per-class accuracy
- Reports overall accuracy

---

# FILE 3: generate_synthetic_data.py

## Purpose

Generates 12,500 biomechanically accurate punch trajectories for training the classifier.

## Generation Process

```
For each punch class (jab, cross, hook, uppercut, idle):
    Generate 2,500 samples
    Each sample = 30 frames × 10 features

Total: 5 classes × 2,500 = 12,500 samples
```

## Biomechanical Models

### JAB Generator (Lines 105-167)

```
Characteristics:
- Lead hand (left for orthodox, right for southpaw)
- Quick, straight extension
- Duration: 8-15 frames

Trajectory:
frame 0-5:   Rapid forward extension
frame 5-10:  Peak extension (elbow ~170°)
frame 10-15: Quick retraction to guard
```

### CROSS Generator (Lines 170-229)

```
Characteristics:
- Rear hand power punch
- Longer wind-up, hip rotation
- Duration: 10-18 frames

Trajectory:
frame 0-3:   Wind-up (slight retraction)
frame 3-10:  Power extension with rotation
frame 10-18: Retraction with hip reset
```

### HOOK Generator (Lines 232-306)

```
Characteristics:
- Curved lateral motion
- Elbow stays bent (~90°)
- Horizontal velocity > vertical
- Duration: 12-20 frames

Trajectory:
frame 0-5:   Arm moves outward
frame 5-12:  Arc toward center target
frame 12-20: Return to guard
```

### UPPERCUT Generator (Lines 309-369)

```
Characteristics:
- Upward motion
- Vertical velocity > horizontal
- Elbow opens during ascent
- Duration: 10-16 frames

Trajectory:
frame 0-4:   Slight dip (loading)
frame 4-10:  Upward explosion
frame 10-16: Descent and return
```

### IDLE Generator (Lines 372-425)

```
Characteristics:
- Random fluctuations around guard
- No consistent direction
- Slow velocity
- Duration: 30 frames (full window)
```

## Feature Format

Each frame contains 10 features:

```
[0] wrist_x      - Wrist X position (0-1)
[1] wrist_y      - Wrist Y position (0-1)
[2] elbow_x      - Elbow X position (0-1)
[3] elbow_y      - Elbow Y position (0-1)
[4] shoulder_x   - Shoulder X position (0-1)
[5] shoulder_y   - Shoulder Y position (0-1)
[6] elbow_angle  - Angle at elbow (0-1, normalized from 0-180°)
[7] velocity     - Speed of wrist movement (0-1)
[8] direction_x  - X component of movement direction (-1 to 1)
[9] direction_y  - Y component of movement direction (-1 to 1)
```

---

# Mathematical Foundations

## 1. Pythagorean Depth Estimation

When an arm extends toward the camera, its 2D projection shortens:

```
3D Length (L) = Constant (calibrated in T-pose)
2D Projection (P) = What we see in camera

Depth (Z) = √(L² - P²)

                    Camera View
                        ↓
           L ─────────────────→
           │╲
           │ ╲
           │  ╲
         Z │   ╲ Arm
           │    ╲
           │     ╲
           ↓──────•
              P
```

## 2. Elbow Angle via Dot Product

```
Given vectors:
  v1 = shoulder - elbow  (upper arm)
  v2 = wrist - elbow     (forearm)

cos(θ) = (v1 · v2) / (|v1| × |v2|)

θ = arccos(cos(θ))

Where:
  v1 · v2 = v1x×v2x + v1y×v2y  (dot product)
  |v| = √(vx² + vy²)           (magnitude)
```

## 3. Kalman Filter Update Equations

**Predict:**

```
x̂ₖ|ₖ₋₁ = F × x̂ₖ₋₁|ₖ₋₁
Pₖ|ₖ₋₁ = F × Pₖ₋₁|ₖ₋₁ × Fᵀ + Q
```

**Update:**

```
Kₖ = Pₖ|ₖ₋₁ × Hᵀ × (H × Pₖ|ₖ₋₁ × Hᵀ + R)⁻¹
x̂ₖ|ₖ = x̂ₖ|ₖ₋₁ + Kₖ × (zₖ - H × x̂ₖ|ₖ₋₁)
Pₖ|ₖ = (I - Kₖ × H) × Pₖ|ₖ₋₁
```

## 4. 1D-CNN for Temporal Classification

**Convolution Operation:**

```
y[t] = Σᵢ Σₖ x[t+k, i] × W[k, i] + b

Where:
- t = time step
- i = feature channel
- k = kernel position
- W = learned weights
```

**Softmax Classification:**

```
P(class_i) = exp(z_i) / Σⱼ exp(z_j)

Where z is the raw output logits
```

---

# Data Flow Summary

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                            DATA FLOW                                         │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  1. TRAINING PHASE (One-time)                                               │
│  ─────────────────────────────                                              │
│                                                                              │
│  generate_synthetic_data.py                                                 │
│         │                                                                    │
│         ▼                                                                    │
│  synthetic_punches.npz (12,500 samples × 30 frames × 10 features)           │
│         │                                                                    │
│         ▼                                                                    │
│  train_classifier.py                                                        │
│         │                                                                    │
│         ▼                                                                    │
│  punch_classifier.pth (trained weights)                                     │
│                                                                              │
│                                                                              │
│  2. INFERENCE PHASE (Real-time)                                             │
│  ─────────────────────────────                                              │
│                                                                              │
│  Camera Frame (640×480)                                                     │
│         │                                                                    │
│         ▼                                                                    │
│  MediaPipe Pose (33 landmarks)                                              │
│         │                                                                    │
│         ├───────────────────────────────────────────┐                        │
│         ▼                                           ▼                        │
│  Physics Engine                              AI Classifier                   │
│  (Depth, Angle, Velocity)                   (30 frames buffer)              │
│         │                                           │                        │
│         ├───────────────────────────────────────────┘                        │
│         ▼                                                                    │
│  Hybrid Hit Detection                                                       │
│         │                                                                    │
│         ▼                                                                    │
│  Game Effects (Flash, Popup, Combo, Score)                                  │
│         │                                                                    │
│         ▼                                                                    │
│  Display Output                                                             │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

# Dependencies

```
opencv-python    # Camera capture and display
mediapipe        # Pose detection (33 body landmarks)
numpy            # Numerical operations
torch            # 1D-CNN classifier
time             # FPS calculation
```

---

# Controls

| Key | Action                                        |
| --- | --------------------------------------------- |
| C   | Start 5-second calibration countdown (T-pose) |
| R   | Reset score and combo                         |
| Q   | Quit application                              |

---

# Configuration

Key parameters in `skeleton_hit_detector.py`:

```python
# Detection Sensitivity
velocity_threshold = 0.17      # Minimum velocity for fallback detection
AI_CONFIDENCE_THRESHOLD = 0.5  # Minimum AI confidence

# Timing
cooldown_frames = 15           # Frames between hits
COMBO_TIMEOUT = 45             # Frames before combo resets (~1.5s)

# Visual
HIT_FLASH_DURATION = 12        # Frames for flash effect
VELOCITY_FRAMES = 3            # Frames for velocity averaging
DEAD_ZONE = 0.02               # Velocity below this = 0

# Scoring
PUNCH_SCORES = {'jab': 10, 'cross': 15, 'hook': 20, 'uppercut': 25}
```

---

# Performance

| Metric            | Value                     |
| ----------------- | ------------------------- |
| Target FPS        | 30                        |
| MediaPipe Model   | model_complexity=0 (Lite) |
| Resolution        | 640×480                   |
| AI Inference      | ~5ms on GPU               |
| AI Model Size     | 40,933 parameters         |
| Training Time     | ~24 seconds               |
| Training Accuracy | 100% on synthetic data    |

---

_Generated: December 12, 2025_
