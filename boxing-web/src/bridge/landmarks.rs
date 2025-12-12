//! Landmark storage, physics integration, and JS bridge
//! 
//! Receives MediaPipe landmarks from JavaScript, runs Kalman prediction,
//! and stores both raw and predicted positions for rendering.

use wasm_bindgen::prelude::*;
use std::cell::RefCell;
use crate::physics::{
    KalmanFilter, DepthEstimator, DepthResult, VelocityTracker, 
    calculate_elbow_angle, OneEuroFilter2D, KinematicConstraints
};

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
    initialized: bool,
}

impl Default for HandPhysics {
    fn default() -> Self {
        Self {
            kalman: KalmanFilter::new(),
            velocity: VelocityTracker::new(),
            one_euro: OneEuroFilter2D::new(),
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
        
        // Calculate depth with verticality filter
        if let Some(result) = state.depth_estimator.calculate_with_filter(
            (left_wrist.x, left_wrist.y),
            (left_shoulder.x, left_shoulder.y),
            state.left_elbow_angle,
        ) {
            state.left_depth = result.depth_percent;
            state.left_depth_valid = result.is_valid_punch_vector;
            state.left_direction_angle = result.direction_angle_deg;
        }
        if let Some(result) = state.depth_estimator.calculate_with_filter(
            (right_wrist.x, right_wrist.y),
            (right_shoulder.x, right_shoulder.y),
            state.right_elbow_angle,
        ) {
            state.right_depth = result.depth_percent;
            state.right_depth_valid = result.is_valid_punch_vector;
            state.right_direction_angle = result.direction_angle_deg;
        }
        
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
        
        // Kalman PREDICT step (120Hz physics)
        state.left_hand.kalman.predict(dt);
        state.right_hand.kalman.predict(dt);
        
        // Get Kalman predicted positions
        let left_pred = state.left_hand.kalman.position();
        let right_pred = state.right_hand.kalman.position();
        state.predicted_wrists[0] = left_pred;
        state.predicted_wrists[1] = right_pred;
        
        // Apply One Euro Filter for jitter reduction (AFTER Kalman)
        state.smoothed_wrists[0] = state.left_hand.one_euro.filter(t, left_pred);
        state.smoothed_wrists[1] = state.right_hand.one_euro.filter(t, right_pred);
        
        state.frame_count += 1;
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
        
        // Calibrate kinematic constraints for both arms
        state.left_constraints.calibrate(left_shoulder, left_elbow, left_wrist, shoulder_width);
        state.right_constraints.calibrate(right_shoulder, right_elbow, right_wrist, shoulder_width);
        
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
