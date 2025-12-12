//! Classifier integration - connects ML classifier with landmark data
//!
//! Manages frame buffer, feature extraction, and exports data for JS inference.
//! The actual ONNX inference runs in JavaScript using onnxruntime-web.

use wasm_bindgen::prelude::*;
use std::cell::RefCell;
use crate::classifier::{FrameBuffer, extract_features};
use super::landmarks::{
    LEFT_WRIST, RIGHT_WRIST, LEFT_ELBOW, RIGHT_ELBOW, LEFT_SHOULDER, RIGHT_SHOULDER
};

/// Minimum confidence for punch detection
const CONFIDENCE_THRESHOLD: f32 = 0.5;

/// Classifier state
struct ClassifierState {
    /// Frame buffer for right hand (primary punch hand)
    right_buffer: FrameBuffer,
    
    /// Previous position for direction calculation
    prev_right_pos: Option<(f32, f32)>,
    
    /// Whether ONNX model is loaded (set by JS)
    model_ready: bool,
}

impl Default for ClassifierState {
    fn default() -> Self {
        Self {
            right_buffer: FrameBuffer::new(),
            prev_right_pos: None,
            model_ready: false,
        }
    }
}

thread_local! {
    static CLASSIFIER_STATE: RefCell<ClassifierState> = RefCell::new(ClassifierState::default());
}

/// Called from JS when ONNX model is loaded
#[wasm_bindgen]
pub fn set_classifier_ready() {
    CLASSIFIER_STATE.with(|state_cell| {
        state_cell.borrow_mut().model_ready = true;
    });
    web_sys::console::log_1(&"✅ Punch classifier ready".into());
}

/// Check if classifier is ready
pub fn is_classifier_ready() -> bool {
    CLASSIFIER_STATE.with(|state_cell| {
        state_cell.borrow().model_ready
    })
}

/// Get buffer data for JS to run inference on
/// Returns None if buffer isn't ready
#[wasm_bindgen]
pub fn get_classification_buffer() -> Option<Vec<f32>> {
    CLASSIFIER_STATE.with(|state_cell| {
        let state = state_cell.borrow();
        if state.right_buffer.is_ready() && state.model_ready {
            Some(state.right_buffer.as_flat())
        } else {
            None
        }
    })
}

/// Check if buffer is ready for classification
#[wasm_bindgen]
pub fn is_buffer_ready() -> bool {
    CLASSIFIER_STATE.with(|state_cell| {
        let state = state_cell.borrow();
        state.right_buffer.is_ready() && state.model_ready
    })
}

/// Process frame for classification (called from update_landmarks)
/// Extracts features and adds to buffer. JS handles the actual inference.
pub fn process_classification_frame(
    landmarks: &[super::landmarks::Landmark; 33],
    _left_velocity: f32,
    right_velocity: f32,
) {
    CLASSIFIER_STATE.with(|state_cell| {
        let mut state = state_cell.borrow_mut();
        
        // Extract landmark positions
        let right_wrist = (landmarks[RIGHT_WRIST].x, landmarks[RIGHT_WRIST].y);
        let right_elbow = (landmarks[RIGHT_ELBOW].x, landmarks[RIGHT_ELBOW].y);
        let right_shoulder = (landmarks[RIGHT_SHOULDER].x, landmarks[RIGHT_SHOULDER].y);
        
        // Calculate velocity direction
        let right_dir = if let Some(prev) = state.prev_right_pos {
            (right_wrist.0 - prev.0, right_wrist.1 - prev.1)
        } else {
            (0.0, 0.0)
        };
        
        // Update previous position
        state.prev_right_pos = Some(right_wrist);
        
        // Extract features for right hand (primary punch hand)
        let features = extract_features(
            right_wrist,
            right_elbow,
            right_shoulder,
            right_velocity,
            right_dir,
        );
        
        // Push to buffer
        state.right_buffer.push(features);
    });
}

/// Get last detected punch (for UI display) - bridges to model.rs
pub fn get_last_punch() -> Option<(crate::classifier::PunchType, f32)> {
    crate::classifier::get_last_punch()
}

/// Placeholder for init_classifier (actual loading happens in JS)
#[wasm_bindgen]
pub fn init_classifier(_onnx_bytes: &[u8]) -> Result<(), JsValue> {
    // Note: ONNX loading happens in JavaScript using onnxruntime-web
    // This function is a placeholder for backwards compatibility
    web_sys::console::log_1(&"ℹ️ ONNX loading delegated to JavaScript".into());
    Ok(())
}
