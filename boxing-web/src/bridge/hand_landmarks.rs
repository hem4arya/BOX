//! Hand Landmark storage with BONE LENGTH + PHYSICAL CONSTRAINTS
//! 
//! Features:
//! - Wrist-only physics (Kalman + One Euro + Predictor)
//! - Bone length calibration (press C to calibrate)
//! - Forward Kinematics (enforces calibrated lengths)
//! - Joint angle constraints (prevents impossible poses)

use wasm_bindgen::prelude::*;
use std::cell::RefCell;
use std::f32::consts::PI;
use crate::physics::{KalmanFilter, OneEuroFilter2D, LinearPredictor};

// ============================================================================
// HAND LANDMARK INDICES
// ============================================================================

pub const WRIST: usize = 0;
pub const THUMB_CMC: usize = 1;
pub const THUMB_MCP: usize = 2;
pub const THUMB_IP: usize = 3;
pub const THUMB_TIP: usize = 4;
pub const INDEX_MCP: usize = 5;
pub const INDEX_PIP: usize = 6;
pub const INDEX_DIP: usize = 7;
pub const INDEX_TIP: usize = 8;
pub const MIDDLE_MCP: usize = 9;
pub const MIDDLE_PIP: usize = 10;
pub const MIDDLE_DIP: usize = 11;
pub const MIDDLE_TIP: usize = 12;
pub const RING_MCP: usize = 13;
pub const RING_PIP: usize = 14;
pub const RING_DIP: usize = 15;
pub const RING_TIP: usize = 16;
pub const PINKY_MCP: usize = 17;
pub const PINKY_PIP: usize = 18;
pub const PINKY_DIP: usize = 19;
pub const PINKY_TIP: usize = 20;

/// Hand skeleton connections for rendering
pub const HAND_SKELETON: [(usize, usize); 21] = [
    (WRIST, THUMB_CMC), (THUMB_CMC, THUMB_MCP), (THUMB_MCP, THUMB_IP), (THUMB_IP, THUMB_TIP),
    (WRIST, INDEX_MCP), (INDEX_MCP, INDEX_PIP), (INDEX_PIP, INDEX_DIP), (INDEX_DIP, INDEX_TIP),
    (WRIST, MIDDLE_MCP), (MIDDLE_MCP, MIDDLE_PIP), (MIDDLE_PIP, MIDDLE_DIP), (MIDDLE_DIP, MIDDLE_TIP),
    (WRIST, RING_MCP), (RING_MCP, RING_PIP), (RING_PIP, RING_DIP), (RING_DIP, RING_TIP),
    (WRIST, PINKY_MCP), (PINKY_MCP, PINKY_PIP), (PINKY_PIP, PINKY_DIP), (PINKY_DIP, PINKY_TIP),
    (INDEX_MCP, MIDDLE_MCP),
];

/// Bones for calibration (parent -> child)
const CALIBRATION_BONES: [(usize, usize); 20] = [
    (WRIST, THUMB_CMC), (THUMB_CMC, THUMB_MCP), (THUMB_MCP, THUMB_IP), (THUMB_IP, THUMB_TIP),
    (WRIST, INDEX_MCP), (INDEX_MCP, INDEX_PIP), (INDEX_PIP, INDEX_DIP), (INDEX_DIP, INDEX_TIP),
    (WRIST, MIDDLE_MCP), (MIDDLE_MCP, MIDDLE_PIP), (MIDDLE_PIP, MIDDLE_DIP), (MIDDLE_DIP, MIDDLE_TIP),
    (WRIST, RING_MCP), (RING_MCP, RING_PIP), (RING_PIP, RING_DIP), (RING_DIP, RING_TIP),
    (WRIST, PINKY_MCP), (PINKY_MCP, PINKY_PIP), (PINKY_PIP, PINKY_DIP), (PINKY_DIP, PINKY_TIP),
];

// ============================================================================
// JOINT ANGLE CONSTRAINTS (Anatomical limits in radians)
// ============================================================================

/// Joint constraint: (parent, joint, child, min_angle, max_angle)
/// Angles are relative to the straight line (0 = straight, positive = flexion)
const JOINT_CONSTRAINTS: [(usize, usize, usize, f32, f32); 14] = [
    // Thumb (more flexible)
    (THUMB_CMC, THUMB_MCP, THUMB_IP, -0.3, 1.2),  // MCP: -17° to 70°
    (THUMB_MCP, THUMB_IP, THUMB_TIP, -0.2, 1.4),  // IP: -11° to 80°
    
    // Index finger
    (WRIST, INDEX_MCP, INDEX_PIP, -0.5, 1.6),     // MCP: -30° to 90°
    (INDEX_MCP, INDEX_PIP, INDEX_DIP, 0.0, 1.8),  // PIP: 0° to 100° (can't extend past straight)
    (INDEX_PIP, INDEX_DIP, INDEX_TIP, 0.0, 1.4),  // DIP: 0° to 80°
    
    // Middle finger
    (WRIST, MIDDLE_MCP, MIDDLE_PIP, -0.5, 1.6),
    (MIDDLE_MCP, MIDDLE_PIP, MIDDLE_DIP, 0.0, 1.8),
    (MIDDLE_PIP, MIDDLE_DIP, MIDDLE_TIP, 0.0, 1.4),
    
    // Ring finger
    (WRIST, RING_MCP, RING_PIP, -0.5, 1.6),
    (RING_MCP, RING_PIP, RING_DIP, 0.0, 1.8),
    (RING_PIP, RING_DIP, RING_TIP, 0.0, 1.4),
    
    // Pinky finger
    (WRIST, PINKY_MCP, PINKY_PIP, -0.5, 1.6),
    (PINKY_MCP, PINKY_PIP, PINKY_DIP, 0.0, 1.8),
    (PINKY_PIP, PINKY_DIP, PINKY_TIP, 0.0, 1.4),
];

// ============================================================================
// DATA STRUCTURES
// ============================================================================

#[derive(Clone, Copy, Default)]
pub struct HandLandmark {
    pub x: f32,
    pub y: f32,
    pub z: f32,
}

#[derive(Clone)]
pub struct HandData {
    pub landmarks: [HandLandmark; 21],
    pub valid: bool,
}

impl Default for HandData {
    fn default() -> Self {
        Self {
            landmarks: [HandLandmark::default(); 21],
            valid: false,
        }
    }
}

/// Stores calibrated bone lengths for one hand
#[derive(Clone)]
struct HandCalibration {
    bone_lengths: [f32; 20],
    is_calibrated: bool,
}

impl Default for HandCalibration {
    fn default() -> Self {
        Self {
            bone_lengths: [0.0; 20],
            is_calibrated: false,
        }
    }
}

impl HandCalibration {
    fn calibrate(&mut self, landmarks: &[HandLandmark; 21]) {
        for (i, (parent, child)) in CALIBRATION_BONES.iter().enumerate() {
            let p = &landmarks[*parent];
            let c = &landmarks[*child];
            let dx = c.x - p.x;
            let dy = c.y - p.y;
            self.bone_lengths[i] = (dx * dx + dy * dy).sqrt();
        }
        self.is_calibrated = true;
        web_sys::console::log_1(&"🖐️ Hand calibrated with bone lengths + joint constraints".into());
    }
    
    fn apply_fk(&self, raw_landmarks: &[HandLandmark; 21]) -> [HandLandmark; 21] {
        if !self.is_calibrated {
            return *raw_landmarks;
        }
        
        let mut result = *raw_landmarks;
        
        for (i, (parent_idx, child_idx)) in CALIBRATION_BONES.iter().enumerate() {
            let parent = result[*parent_idx];
            let child = raw_landmarks[*child_idx];
            
            let dx = child.x - parent.x;
            let dy = child.y - parent.y;
            let len = (dx * dx + dy * dy).sqrt();
            
            if len > 0.0001 {
                let scale = self.bone_lengths[i] / len;
                result[*child_idx] = HandLandmark {
                    x: parent.x + dx * scale,
                    y: parent.y + dy * scale,
                    z: child.z,
                };
            }
        }
        
        result
    }
}

// ============================================================================
// PHYSICAL CONSTRAINT FUNCTIONS
// ============================================================================

/// Calculate angle at joint (in radians)
/// Returns the flexion angle (0 = straight, positive = bent)
fn calculate_joint_angle(parent: HandLandmark, joint: HandLandmark, child: HandLandmark) -> f32 {
    let v1x = parent.x - joint.x;
    let v1y = parent.y - joint.y;
    let v2x = child.x - joint.x;
    let v2y = child.y - joint.y;
    
    let dot = v1x * v2x + v1y * v2y;
    let cross = v1x * v2y - v1y * v2x;
    
    cross.atan2(dot).abs() // Return absolute angle (flexion)
}

/// Apply joint angle constraints to prevent impossible poses
fn apply_joint_constraints(landmarks: &mut [HandLandmark; 21], calibration: &HandCalibration) {
    if !calibration.is_calibrated {
        return;
    }
    
    for constraint in JOINT_CONSTRAINTS.iter() {
        let (parent_idx, joint_idx, child_idx, min_angle, max_angle) = *constraint;
        
        let parent = landmarks[parent_idx];
        let joint = landmarks[joint_idx];
        let child = landmarks[child_idx];
        
        // Skip if any landmark is invalid
        if parent.x < 0.001 && parent.y < 0.001 { continue; }
        if joint.x < 0.001 && joint.y < 0.001 { continue; }
        if child.x < 0.001 && child.y < 0.001 { continue; }
        
        let current_angle = calculate_joint_angle(parent, joint, child);
        
        // Check if angle is outside limits
        let clamped_angle = current_angle.clamp(min_angle, max_angle);
        
        if (current_angle - clamped_angle).abs() > 0.01 {
            // Need to adjust child position to respect constraint
            let p_to_j_x = joint.x - parent.x;
            let p_to_j_y = joint.y - parent.y;
            let p_to_j_len = (p_to_j_x * p_to_j_x + p_to_j_y * p_to_j_y).sqrt();
            
            if p_to_j_len > 0.0001 {
                let base_angle = p_to_j_y.atan2(p_to_j_x);
                let constrained_angle = base_angle + PI - clamped_angle;
                
                // Get bone length for this segment
                let bone_idx = CALIBRATION_BONES.iter().position(|(p, c)| *p == joint_idx && *c == child_idx);
                let bone_len = if let Some(idx) = bone_idx {
                    calibration.bone_lengths[idx]
                } else {
                    let dx = child.x - joint.x;
                    let dy = child.y - joint.y;
                    (dx * dx + dy * dy).sqrt()
                };
                
                landmarks[child_idx] = HandLandmark {
                    x: joint.x + constrained_angle.cos() * bone_len,
                    y: joint.y + constrained_angle.sin() * bone_len,
                    z: child.z,
                };
            }
        }
    }
}

// ============================================================================
// PHYSICS
// ============================================================================

struct WristPhysics {
    kalman: KalmanFilter,
    one_euro: OneEuroFilter2D,
    predictor: LinearPredictor,
}

impl WristPhysics {
    fn new() -> Self {
        Self {
            kalman: KalmanFilter::new(),
            one_euro: OneEuroFilter2D::new(),
            predictor: LinearPredictor::new(),
        }
    }
    
    fn apply(&mut self, x: f32, y: f32, t: f64) -> (f32, f32) {
        let (smooth_x, smooth_y) = self.one_euro.filter(t, (x, y));
        self.kalman.correct(smooth_x, smooth_y);
        let (kalman_x, kalman_y) = self.kalman.position();
        self.predictor.update((kalman_x, kalman_y))
    }
}

struct HandState {
    hands: [HandData; 2],
    raw_hands: [HandData; 2],
    wrist_physics: [WristPhysics; 2],
    calibration: [HandCalibration; 2],
    num_hands: usize,
    has_data: bool,
    last_valid_hands: [HandData; 2],
}

impl Default for HandState {
    fn default() -> Self {
        Self {
            hands: [HandData::default(), HandData::default()],
            raw_hands: [HandData::default(), HandData::default()],
            wrist_physics: [WristPhysics::new(), WristPhysics::new()],
            calibration: [HandCalibration::default(), HandCalibration::default()],
            num_hands: 0,
            has_data: false,
            last_valid_hands: [HandData::default(), HandData::default()],
        }
    }
}

thread_local! {
    static HAND_STATE: RefCell<HandState> = RefCell::new(HandState::default());
}

// ============================================================================
// WASM API
// ============================================================================

#[wasm_bindgen]
pub fn calibrate_hand() {
    HAND_STATE.with(|state_cell| {
        let mut state = state_cell.borrow_mut();
        for h in 0..2 {
            if state.raw_hands[h].valid {
                let landmarks = state.raw_hands[h].landmarks;
                state.calibration[h].calibrate(&landmarks);
            }
        }
    });
}

#[wasm_bindgen]
pub fn apply_hand_landmarks(flat_data: &[f32], num_hands: usize) {
    let now = js_sys::Date::now() / 1000.0;
    
    HAND_STATE.with(|state_cell| {
        let mut state = state_cell.borrow_mut();
        
        let actual_hands = num_hands.min(2);
        state.num_hands = actual_hands;
        state.has_data = actual_hands > 0;
        
        for h in 0..2 {
            if h < actual_hands {
                // Parse raw landmarks
                let mut raw_landmarks = [HandLandmark::default(); 21];
                for i in 0..21 {
                    let base = h * 21 * 3 + i * 3;
                    if base + 2 < flat_data.len() {
                        raw_landmarks[i] = HandLandmark {
                            x: flat_data[base],
                            y: flat_data[base + 1],
                            z: flat_data[base + 2],
                        };
                    }
                }
                
                // Store raw for calibration
                state.raw_hands[h].landmarks = raw_landmarks;
                state.raw_hands[h].valid = true;
                
                // Apply wrist physics
                let (wrist_x, wrist_y) = state.wrist_physics[h].apply(
                    raw_landmarks[WRIST].x,
                    raw_landmarks[WRIST].y,
                    now
                );
                raw_landmarks[WRIST].x = wrist_x;
                raw_landmarks[WRIST].y = wrist_y;
                
                // Apply FK (bone length enforcement)
                let mut final_landmarks = state.calibration[h].apply_fk(&raw_landmarks);
                
                // Apply joint angle constraints (physical limits)
                apply_joint_constraints(&mut final_landmarks, &state.calibration[h]);
                
                state.hands[h].landmarks = final_landmarks;
                state.hands[h].valid = true;
                state.last_valid_hands[h] = state.hands[h].clone();
            } else {
                state.hands[h] = state.last_valid_hands[h].clone();
                state.hands[h].valid = false;
                state.raw_hands[h].valid = false;
            }
        }
    });
}

// ============================================================================
// INTERNAL API
// ============================================================================

pub fn get_hand_data() -> Option<([HandData; 2], usize)> {
    HAND_STATE.with(|state_cell| {
        let state = state_cell.borrow();
        if state.has_data || state.last_valid_hands[0].valid || state.last_valid_hands[1].valid {
            Some((state.hands.clone(), state.num_hands.max(
                if state.last_valid_hands[0].valid { 1 } else { 0 } +
                if state.last_valid_hands[1].valid { 1 } else { 0 }
            )))
        } else {
            None
        }
    })
}

pub fn is_hand_calibrated() -> (bool, bool) {
    HAND_STATE.with(|state_cell| {
        let state = state_cell.borrow();
        (state.calibration[0].is_calibrated, state.calibration[1].is_calibrated)
    })
}
