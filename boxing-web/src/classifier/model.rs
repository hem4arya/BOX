//! ONNX model loading and inference
//!
//! NOTE: Due to WASM compatibility issues with tract-onnx, the actual ONNX
//! inference runs in JavaScript using onnxruntime-web. This module provides
//! the data structures and stub implementation.
//!
//! The JavaScript code calls classify_punch_js() with the prediction result.

use wasm_bindgen::prelude::*;
use std::cell::RefCell;

/// Punch type labels (order matches training)
pub const PUNCH_TYPES: [&str; 5] = ["jab", "cross", "hook", "uppercut", "idle"];

/// Punch type enum for type-safe handling
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PunchType {
    Jab,
    Cross,
    Hook,
    Uppercut,
    Idle,
}

impl PunchType {
    pub fn from_index(idx: usize) -> Self {
        match idx {
            0 => PunchType::Jab,
            1 => PunchType::Cross,
            2 => PunchType::Hook,
            3 => PunchType::Uppercut,
            _ => PunchType::Idle,
        }
    }
    
    pub fn as_str(&self) -> &'static str {
        match self {
            PunchType::Jab => "jab",
            PunchType::Cross => "cross",
            PunchType::Hook => "hook",
            PunchType::Uppercut => "uppercut",
            PunchType::Idle => "idle",
        }
    }
    
    /// Score value for each punch type
    pub fn score(&self) -> u32 {
        match self {
            PunchType::Jab => 10,
            PunchType::Cross => 15,
            PunchType::Hook => 20,
            PunchType::Uppercut => 25,
            PunchType::Idle => 0,
        }
    }
}

/// Stub classifier - actual inference happens in JS
pub struct PunchClassifier;

impl PunchClassifier {
    pub fn new() -> Self {
        Self
    }
}

impl Default for PunchClassifier {
    fn default() -> Self {
        Self::new()
    }
}

// Thread-local storage for last punch result (set by JS)
thread_local! {
    static LAST_PUNCH: RefCell<Option<(PunchType, f32)>> = RefCell::new(None);
}

/// Called from JavaScript with classification result
#[wasm_bindgen]
pub fn set_punch_result(punch_index: usize, confidence: f32) {
    let punch_type = PunchType::from_index(punch_index);
    LAST_PUNCH.with(|p| {
        *p.borrow_mut() = Some((punch_type, confidence));
    });
}

/// Get last punch result (for Rust code to read)
pub fn get_last_punch() -> Option<(PunchType, f32)> {
    LAST_PUNCH.with(|p| *p.borrow())
}

/// Clear last punch result
pub fn clear_last_punch() {
    LAST_PUNCH.with(|p| {
        *p.borrow_mut() = None;
    });
}
