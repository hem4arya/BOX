//! Landmark storage, physics integration, and JS bridge
//! 
//! Receives MediaPipe landmarks from JavaScript, runs Kalman prediction,
//! and stores both raw and predicted positions for rendering.

use wasm_bindgen::prelude::*;
use std::cell::RefCell;
use crate::physics::{
    KalmanFilter, DepthEstimator, DepthResult, VelocityTracker, 
    calculate_elbow_angle, OneEuroFilter2D, KinematicConstraints,
    PunchDetector, PunchType, Extrapolator, ConfidenceGate, ArmIK,
    clamp_velocity, reject_outlier, clamp_elbow_angle,
};
use crate::renderer::update_arm_metrics;

// ============================================================================
// LANDMARK INDICES (MediaPipe Pose - 33 total)
// ============================================================================

pub const NOSE: usize = 0;
pub const LEFT_SHOULDER: usize = 11;
pub const RIGHT_SHOULDER: usize = 12;
pub const LEFT_ELBOW: usize = 13;
pub const RIGHT_ELBOW: usize = 14;
pub const LEFT_WRIST: usize = 15;
pub const RIGHT_WRIST: usize = 16;
pub const LEFT_HIP: usize = 23;
pub const RIGHT_HIP: usize = 24;

/// Skeleton connections for arms (pairs of landmark indices)
pub const ARM_SKELETON: [(usize, usize); 4] = [
    (LEFT_SHOULDER, LEFT_ELBOW),
    (LEFT_ELBOW, LEFT_WRIST),
    (RIGHT_SHOULDER, RIGHT_ELBOW),
    (RIGHT_ELBOW, RIGHT_WRIST),
];

/// Key landmarks to draw as dots (boxing-relevant)
pub const KEY_LANDMARKS: [usize; 8] = [
    NOSE,
    LEFT_SHOULDER, RIGHT_SHOULDER,
    LEFT_ELBOW, RIGHT_ELBOW,
    LEFT_WRIST, RIGHT_WRIST,
    LEFT_HIP,
];

// ============================================================================
// LANDMARK DATA STRUCTURES
// ============================================================================

/// A single 3D landmark point (normalized coordinates)
#[derive(Clone, Copy, Default)]
pub struct Landmark {
    pub x: f32,
    pub y: f32,
    pub z: f32,
}

/// Physics state for one hand
struct HandPhysics {
    kalman: KalmanFilter,
    velocity: VelocityTracker,
    one_euro: OneEuroFilter2D,
    elbow_gate: ConfidenceGate,
    wrist_gate: ConfidenceGate,
    last_wrist: (f32, f32),
    initialized: bool,
}

impl Default for HandPhysics {
    fn default() -> Self {
        Self {
            kalman: KalmanFilter::new(),
            velocity: VelocityTracker::new(),
            one_euro: OneEuroFilter2D::new(),
            elbow_gate: ConfidenceGate::new(),
            wrist_gate: ConfidenceGate::new(),
            last_wrist: (0.5, 0.5),
            initialized: false,
        }
    }
}

/// Full state including raw landmarks, predictions, and physics
struct PhysicsState {
    raw_landmarks: [Landmark; 33],
    predicted_wrists: [(f32, f32); 2],
    smoothed_wrists: [(f32, f32); 2],
    left_hand: HandPhysics,
    right_hand: HandPhysics,
    depth_estimator: DepthEstimator,
    left_constraints: KinematicConstraints,
    right_constraints: KinematicConstraints,
    left_ik: ArmIK,
    right_ik: ArmIK,
    punch_detector: PunchDetector,
    left_extrapolator: Extrapolator,
    right_extrapolator: Extrapolator,
    last_punch: PunchType,
    last_punch_confidence: f32,
    has_data: bool,
    shoulder_width: f32,
    left_elbow_angle: f32,
    right_elbow_angle: f32,
    left_depth: f32,
    right_depth: f32,
    left_depth_valid: bool,
    right_depth_valid: bool,
    left_direction_angle: f32,
    right_direction_angle: f32,
    left_velocity: f32,
    right_velocity: f32,
    frame_count: u32,
    timestamp: f64,
}

impl Default for PhysicsState {
    fn default() -> Self {
        Self {
            raw_landmarks: [Landmark::default(); 33],
            predicted_wrists: [(0.0, 0.0); 2],
            smoothed_wrists: [(0.0, 0.0); 2],
            left_hand: HandPhysics::default(),
            right_hand: HandPhysics::default(),
            depth_estimator: DepthEstimator::new(),
            left_constraints: KinematicConstraints::new(),
            right_constraints: KinematicConstraints::new(),
            left_ik: ArmIK::new(false),  // Left arm
            right_ik: ArmIK::new(true),  // Right arm
            punch_detector: PunchDetector::new(),
            left_extrapolator: Extrapolator::new(),
            right_extrapolator: Extrapolator::new(),
            last_punch: PunchType::Idle,
            last_punch_confidence: 0.0,
            has_data: false,
            shoulder_width: 0.0,
            left_elbow_angle: 180.0,
            right_elbow_angle: 180.0,
            left_depth: 0.0,
            right_depth: 0.0,
            left_depth_valid: false,
            right_depth_valid: false,
            left_direction_angle: 0.0,
            right_direction_angle: 0.0,
            left_velocity: 0.0,
            right_velocity: 0.0,
            frame_count: 0,
            timestamp: 0.0,
        }
    }
}

// Thread-local storage (WASM is single-threaded)
thread_local! {
    static STATE: RefCell<PhysicsState> = RefCell::new(PhysicsState::default());
}

// ============================================================================
// WASM-BINDGEN ENTRY POINTS
// ============================================================================

/// Called from JavaScript with flat Float32Array of 99 values
/// (33 landmarks × 3 coordinates: x, y, z)
/// 
/// This runs at ~30Hz (MediaPipe rate)
#[wasm_bindgen]
pub fn update_landmarks(data: &[f32]) {
    if data.len() != 99 {
        web_sys::console::warn_1(
            &format!("Invalid landmark data length: {} (expected 99)", data.len()).into()
        );
        return;
    }
    
    STATE.with(|state_cell| {
        let mut state = state_cell.borrow_mut();
        
        // Parse raw landmarks
        for i in 0..33 {
            state.raw_landmarks[i] = Landmark {
                x: data[i * 3],
                y: data[i * 3 + 1],
                z: data[i * 3 + 2],
            };
        }
        state.has_data = true;
        
        // Extract key points
        let left_wrist = state.raw_landmarks[LEFT_WRIST];
        let right_wrist = state.raw_landmarks[RIGHT_WRIST];
        let left_shoulder = state.raw_landmarks[LEFT_SHOULDER];
        let right_shoulder = state.raw_landmarks[RIGHT_SHOULDER];
        let left_elbow = state.raw_landmarks[LEFT_ELBOW];
        let right_elbow = state.raw_landmarks[RIGHT_ELBOW];
        
        // Initialize Kalman filters on first data
        if !state.left_hand.initialized {
            state.left_hand.kalman.initialize(left_wrist.x, left_wrist.y);
            state.left_hand.initialized = true;
        }
        if !state.right_hand.initialized {
            state.right_hand.kalman.initialize(right_wrist.x, right_wrist.y);
            state.right_hand.initialized = true;
        }
        
        // Kalman UPDATE step (30Hz correction)
        state.left_hand.kalman.update(left_wrist.x, left_wrist.y);
        state.right_hand.kalman.update(right_wrist.x, right_wrist.y);
        
        // Update velocity trackers
        state.left_hand.velocity.update((left_wrist.x, left_wrist.y));
        state.right_hand.velocity.update((right_wrist.x, right_wrist.y));
        
        // Calculate elbow angles
        state.left_elbow_angle = calculate_elbow_angle(
            (left_shoulder.x, left_shoulder.y),
            (left_elbow.x, left_elbow.y),
            (left_wrist.x, left_wrist.y),
        );
        state.right_elbow_angle = calculate_elbow_angle(
            (right_shoulder.x, right_shoulder.y),
            (right_elbow.x, right_elbow.y),
            (right_wrist.x, right_wrist.y),
        );
        
        // Calculate GATED depth (elbow + direction gates)
        let left_result = state.depth_estimator.calculate_gated(
            (left_wrist.x, left_wrist.y),
            (left_shoulder.x, left_shoulder.y),
            state.left_elbow_angle,
        );
        state.left_depth = left_result.gated_percent;
        state.left_depth_valid = left_result.is_valid;
        state.left_direction_angle = left_result.direction_angle;
        
        let right_result = state.depth_estimator.calculate_gated(
            (right_wrist.x, right_wrist.y),
            (right_shoulder.x, right_shoulder.y),
            state.right_elbow_angle,
        );
        state.right_depth = right_result.gated_percent;
        state.right_depth_valid = right_result.is_valid;
        state.right_direction_angle = right_result.direction_angle;
        
        // Store predicted positions
        state.predicted_wrists[0] = state.left_hand.kalman.position();
        state.predicted_wrists[1] = state.right_hand.kalman.position();
        
        // Get velocity magnitudes for classification and store for debug
        let left_vel = state.left_hand.velocity.update((left_wrist.x, left_wrist.y));
        let right_vel = state.right_hand.velocity.update((right_wrist.x, right_wrist.y));
        state.left_velocity = left_vel;
        state.right_velocity = right_vel;
        
        // Run classification (if model is loaded)
        super::classifier_integration::process_classification_frame(
            &state.raw_landmarks,
            left_vel,
            right_vel,
        );
    });
}

/// Called every render frame (~120Hz) to run physics prediction
#[wasm_bindgen]
pub fn physics_tick(dt: f32) {
    STATE.with(|state_cell| {
        let mut state = state_cell.borrow_mut();
        
        if !state.has_data {
            return;
        }
        
        // Update timestamp
        state.timestamp += dt as f64;
        let t = state.timestamp;
        
        // 120Hz Physics tick (fast path - pure x += vx*dt)
        state.left_hand.kalman.tick(dt);
        state.right_hand.kalman.tick(dt);
        
        // Get predicted positions
        let left_pred = state.left_hand.kalman.position();
        let right_pred = state.right_hand.kalman.position();
        state.predicted_wrists[0] = left_pred;
        state.predicted_wrists[1] = right_pred;
        
        // Apply One Euro Filter for jitter reduction
        state.smoothed_wrists[0] = state.left_hand.one_euro.filter(t, left_pred);
        state.smoothed_wrists[1] = state.right_hand.one_euro.filter(t, right_pred);
        
        state.frame_count += 1;
    });
}

/// Apply soft MediaPipe correction (~30Hz) - doesn't replace, just nudges physics
#[wasm_bindgen]
pub fn apply_mediapipe_correction(data: &[f32]) {
    if data.len() != 99 {
        return;
    }
    
    STATE.with(|state_cell| {
        let mut state = state_cell.borrow_mut();
        
        // Extract raw wrist positions
        let raw_left_wrist = (data[LEFT_WRIST * 3], data[LEFT_WRIST * 3 + 1]);
        let raw_right_wrist = (data[RIGHT_WRIST * 3], data[RIGHT_WRIST * 3 + 1]);
        
        // Get predicted positions from Kalman
        let pred_left = state.left_hand.kalman.position();
        let pred_right = state.right_hand.kalman.position();
        
        // Get previous positions (smoothed wrists from last frame)
        let prev_left = state.smoothed_wrists[0];
        let prev_right = state.smoothed_wrists[1];
        
        // ====== LAYER 4: OUTLIER REJECTION ======
        let (left_wrist, _left_rejected) = reject_outlier(raw_left_wrist, pred_left, prev_left);
        let (right_wrist, _right_rejected) = reject_outlier(raw_right_wrist, pred_right, prev_right);
        
        // ====== LAYER 3: VELOCITY CLAMPING ======
        let left_wrist = clamp_velocity(left_wrist, prev_left);
        let right_wrist = clamp_velocity(right_wrist, prev_right);
        
        // Soft Kalman correction with constrained position
        state.left_hand.kalman.correct(left_wrist.0, left_wrist.1);
        state.right_hand.kalman.correct(right_wrist.0, right_wrist.1);
        
        // Store raw landmarks for angle/depth calculation
        for i in 0..33 {
            state.raw_landmarks[i] = Landmark {
                x: data[i * 3],
                y: data[i * 3 + 1],
                z: data[i * 3 + 2],
            };
        }
        state.has_data = true;
        
        // Update angles and depth (from raw landmarks)
        let left_shoulder = state.raw_landmarks[LEFT_SHOULDER];
        let left_elbow = state.raw_landmarks[LEFT_ELBOW];
        let left_wrist = state.raw_landmarks[LEFT_WRIST];
        let right_shoulder = state.raw_landmarks[RIGHT_SHOULDER];
        let right_elbow = state.raw_landmarks[RIGHT_ELBOW];
        let right_wrist = state.raw_landmarks[RIGHT_WRIST];
        
        // Calculate elbow angles
        state.left_elbow_angle = calculate_elbow_angle(
            (left_shoulder.x, left_shoulder.y),
            (left_elbow.x, left_elbow.y),
            (left_wrist.x, left_wrist.y),
        );
        state.right_elbow_angle = calculate_elbow_angle(
            (right_shoulder.x, right_shoulder.y),
            (right_elbow.x, right_elbow.y),
            (right_wrist.x, right_wrist.y),
        );
        
        // ====== LAYER 5: ONE EURO FILTER (smooth wrist jitter) ======
        // Apply to RAW positions first (before bone fix)
        let timestamp = js_sys::Date::now() / 1000.0;  // Convert to seconds
        let smooth_left_wrist = state.left_hand.one_euro.filter(timestamp, (left_wrist.x, left_wrist.y));
        let smooth_right_wrist = state.right_hand.one_euro.filter(timestamp, (right_wrist.x, right_wrist.y));
        let smooth_left_elbow = (left_elbow.x, left_elbow.y); // Elbows don't need smoothing (less jitter)
        let smooth_right_elbow = (right_elbow.x, right_elbow.y);
        
        // ====== LAYER 6: KALMAN CORRECTION (final smooth + prediction) ======
        state.left_hand.kalman.correct(smooth_left_wrist.0, smooth_left_wrist.1);
        state.right_hand.kalman.correct(smooth_right_wrist.0, smooth_right_wrist.1);
        
        // Use Kalman-smoothed positions
        let kalman_left_wrist = state.left_hand.kalman.position();
        let kalman_right_wrist = state.right_hand.kalman.position();
        
        // ====== LAYER 7: BONE LENGTH FIX (FINAL - after all filtering) ======
        // NOW apply bone constraints to the filtered positions
        // This ensures bone lengths are ALWAYS exact, even after smoothing
        let (final_left_elbow, final_left_wrist) = if state.left_ik.is_calibrated() {
            state.left_ik.solve_fk(
                (left_shoulder.x, left_shoulder.y),
                smooth_left_elbow,
                kalman_left_wrist,
            )
        } else {
            (smooth_left_elbow, kalman_left_wrist)
        };
        
        let (final_right_elbow, final_right_wrist) = if state.right_ik.is_calibrated() {
            state.right_ik.solve_fk(
                (right_shoulder.x, right_shoulder.y),
                smooth_right_elbow,
                kalman_right_wrist,
            )
        } else {
            (smooth_right_elbow, kalman_right_wrist)
        };
        
        // Store FINAL positions (bone-constrained) for rendering
        state.raw_landmarks[LEFT_ELBOW].x = final_left_elbow.0;
        state.raw_landmarks[LEFT_ELBOW].y = final_left_elbow.1;
        state.raw_landmarks[LEFT_WRIST].x = final_left_wrist.0;
        state.raw_landmarks[LEFT_WRIST].y = final_left_wrist.1;
        state.raw_landmarks[RIGHT_ELBOW].x = final_right_elbow.0;
        state.raw_landmarks[RIGHT_ELBOW].y = final_right_elbow.1;
        state.raw_landmarks[RIGHT_WRIST].x = final_right_wrist.0;
        state.raw_landmarks[RIGHT_WRIST].y = final_right_wrist.1;
        
        // Calculate gated depth (using final wrist positions)
        let left_result = state.depth_estimator.calculate_gated(
            final_left_wrist,
            (left_shoulder.x, left_shoulder.y),
            state.left_elbow_angle,
        );
        state.left_depth = left_result.gated_percent;
        state.left_depth_valid = left_result.is_valid;
        
        let right_result = state.depth_estimator.calculate_gated(
            final_right_wrist,
            (right_shoulder.x, right_shoulder.y),
            state.right_elbow_angle,
        );
        state.right_depth = right_result.gated_percent;
        state.right_depth_valid = right_result.is_valid;
        
        // Update velocity trackers (using final wrist positions)
        state.left_velocity = state.left_hand.velocity.update(final_left_wrist);
        state.right_velocity = state.right_hand.velocity.update(final_right_wrist);
        
        // Update extrapolators for latency compensation
        let now = js_sys::Date::now();
        state.left_extrapolator.update(final_left_wrist, now);
        state.right_extrapolator.update(final_right_wrist, now);
        
        // Store final wrists for smoothed rendering
        state.smoothed_wrists = [final_left_wrist, final_right_wrist];
        
        // ====== RECALCULATE ELBOW ANGLES (using FINAL bone-constrained positions) ======
        // This ensures the debug overlay shows the correct angles for the rendered skeleton
        state.left_elbow_angle = calculate_elbow_angle(
            (left_shoulder.x, left_shoulder.y),
            final_left_elbow,
            final_left_wrist,
        );
        state.right_elbow_angle = calculate_elbow_angle(
            (right_shoulder.x, right_shoulder.y),
            final_right_elbow,
            final_right_wrist,
        );
        
        // Physics-based punch detection (replaces ONNX)
        // Copy values to avoid borrow conflict
        let left_vel = state.left_velocity;
        let right_vel = state.right_velocity;
        let left_d = state.left_depth;
        let right_d = state.right_depth;
        let left_valid = state.left_depth_valid;
        let right_valid = state.right_depth_valid;
        let right_dir_y = right_wrist.y - state.predicted_wrists[1].1;
        
        let (punch, confidence) = state.punch_detector.detect(
            left_vel,
            right_vel,
            left_d,
            right_d,
            left_valid,
            right_valid,
            right_dir_y,
        );
        
        if confidence > 0.5 && punch != PunchType::Idle {
            state.last_punch = punch;
            state.last_punch_confidence = confidence;
            web_sys::console::log_1(&format!("🥊 {} ({:.0}%)", punch.name(), confidence * 100.0).into());
        }
        
        // Update debug overlay with FINAL physics values (after all filtering and constraints)
        update_arm_metrics(
            left_d, state.left_elbow_angle, left_vel, left_valid,
            right_d, state.right_elbow_angle, right_vel, right_valid,
        );
        
        // Push frame to classification buffer (kept for compatibility)
        let left_v = state.left_velocity;
        let right_v = state.right_velocity;
        let raw_lm = state.raw_landmarks.clone();
        super::classifier_integration::process_classification_frame(
            &raw_lm,
            left_v,
            right_v,
        );
    });
}

/// Calibrate depth estimator and kinematic constraints (call during T-pose)
#[wasm_bindgen]
pub fn calibrate_depth() {
    STATE.with(|state_cell| {
        let mut state = state_cell.borrow_mut();
        
        if !state.has_data {
            return;
        }
        
        let left_shoulder = (state.raw_landmarks[LEFT_SHOULDER].x, state.raw_landmarks[LEFT_SHOULDER].y);
        let right_shoulder = (state.raw_landmarks[RIGHT_SHOULDER].x, state.raw_landmarks[RIGHT_SHOULDER].y);
        let left_wrist = (state.raw_landmarks[LEFT_WRIST].x, state.raw_landmarks[LEFT_WRIST].y);
        let right_wrist = (state.raw_landmarks[RIGHT_WRIST].x, state.raw_landmarks[RIGHT_WRIST].y);
        let left_elbow = (state.raw_landmarks[LEFT_ELBOW].x, state.raw_landmarks[LEFT_ELBOW].y);
        let right_elbow = (state.raw_landmarks[RIGHT_ELBOW].x, state.raw_landmarks[RIGHT_ELBOW].y);
        
        // Calculate shoulder width
        let dx = right_shoulder.0 - left_shoulder.0;
        let dy = right_shoulder.1 - left_shoulder.1;
        let shoulder_width = (dx * dx + dy * dy).sqrt();
        state.shoulder_width = shoulder_width;
        
        // Calibrate depth estimator (with full context)
        state.depth_estimator.calibrate(left_shoulder, right_shoulder, left_wrist, left_shoulder);
        
        // Calibrate kinematic constraints for both arms (FK fallback)
        state.left_constraints.calibrate(left_shoulder, left_elbow, left_wrist, shoulder_width);
        state.right_constraints.calibrate(right_shoulder, right_elbow, right_wrist, shoulder_width);
        
        // Calibrate IK solvers - these use only bone lengths from T-pose
        state.left_ik.calibrate(left_shoulder, left_elbow, left_wrist);
        state.right_ik.calibrate(right_shoulder, right_elbow, right_wrist);
        
        web_sys::console::log_1(&format!(
            "✅ Calibrated: shoulder_width={:.3}", shoulder_width
        ).into());
    });
}

// ============================================================================
// INTERNAL API (no wasm_bindgen)
// ============================================================================

/// Get all current RAW landmarks (for renderer)
pub fn get_all_landmarks() -> Option<[Landmark; 33]> {
    STATE.with(|state_cell| {
        let state = state_cell.borrow();
        if state.has_data {
            Some(state.raw_landmarks)
        } else {
            None
        }
    })
}

/// Get predicted wrist positions [left, right]
pub fn get_predicted_wrists() -> Option<[(f32, f32); 2]> {
    STATE.with(|state_cell| {
        let state = state_cell.borrow();
        if state.has_data {
            Some(state.predicted_wrists)
        } else {
            None
        }
    })
}

/// Get debug info for overlay (angles and depths)
pub fn get_debug_info() -> (f32, f32, f32, f32, f32, f32) {
    STATE.with(|state_cell| {
        let state = state_cell.borrow();
        (
            state.left_elbow_angle,
            state.right_elbow_angle,
            state.left_depth,
            state.right_depth,
            state.left_velocity,
            state.right_velocity,
        )
    })
}

/// Get extended debug info (depth validity and direction angles)
pub fn get_depth_validity() -> (bool, bool, f32, f32) {
    STATE.with(|state_cell| {
        let state = state_cell.borrow();
        (
            state.left_depth_valid,
            state.right_depth_valid,
            state.left_direction_angle,
            state.right_direction_angle,
        )
    })
}

/// Get smoothed wrist positions (after One Euro filter) [left, right]
pub fn get_smoothed_wrists() -> Option<[(f32, f32); 2]> {
    STATE.with(|state_cell| {
        let state = state_cell.borrow();
        if state.has_data {
            Some(state.smoothed_wrists)
        } else {
            None
        }
    })
}

/// Get extrapolated wrist positions (predicted ahead to compensate for latency)
/// Returns [left_x, left_y, right_x, right_y]
#[wasm_bindgen]
pub fn get_extrapolated_wrists() -> Vec<f32> {
    STATE.with(|state_cell| {
        let state = state_cell.borrow();
        let now = js_sys::Date::now();
        
        let left = state.left_extrapolator.predict(now);
        let right = state.right_extrapolator.predict(now);
        
        vec![left.0, left.1, right.0, right.1]
    })
}

/// Get raw wrist positions (no extrapolation, for comparison)
/// Returns [left_x, left_y, right_x, right_y]
#[wasm_bindgen]
pub fn get_raw_wrists() -> Vec<f32> {
    STATE.with(|state_cell| {
        let state = state_cell.borrow();
        
        let left = state.left_extrapolator.raw_position();
        let right = state.right_extrapolator.raw_position();
        
        vec![left.0, left.1, right.0, right.1]
    })
}

/// Set extrapolation parameters (for tuning)
#[wasm_bindgen]
pub fn set_extrapolation_params(latency_ms: f32, overshoot: f32) {
    STATE.with(|state_cell| {
        let mut state = state_cell.borrow_mut();
        state.left_extrapolator.set_latency(latency_ms);
        state.left_extrapolator.set_overshoot(overshoot);
        state.right_extrapolator.set_latency(latency_ms);
        state.right_extrapolator.set_overshoot(overshoot);
        web_sys::console::log_1(&format!("⚡ Extrapolation: latency={}ms, overshoot={}x", latency_ms, overshoot).into());
    });
}

