//! Hand Landmark storage and processing
//! 
//! Receives MediaPipe Hand landmarks (21 per hand, up to 2 hands)
//! and stores them for rendering.

use wasm_bindgen::prelude::*;
use std::cell::RefCell;

// ============================================================================
// HAND LANDMARK INDICES (MediaPipe Hand - 21 total per hand)
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

/// Hand skeleton connections (pairs of landmark indices)
pub const HAND_SKELETON: [(usize, usize); 21] = [
    // Thumb
    (WRIST, THUMB_CMC),
    (THUMB_CMC, THUMB_MCP),
    (THUMB_MCP, THUMB_IP),
    (THUMB_IP, THUMB_TIP),
    // Index
    (WRIST, INDEX_MCP),
    (INDEX_MCP, INDEX_PIP),
    (INDEX_PIP, INDEX_DIP),
    (INDEX_DIP, INDEX_TIP),
    // Middle
    (WRIST, MIDDLE_MCP),
    (MIDDLE_MCP, MIDDLE_PIP),
    (MIDDLE_PIP, MIDDLE_DIP),
    (MIDDLE_DIP, MIDDLE_TIP),
    // Ring
    (WRIST, RING_MCP),
    (RING_MCP, RING_PIP),
    (RING_PIP, RING_DIP),
    (RING_DIP, RING_TIP),
    // Pinky
    (WRIST, PINKY_MCP),
    (PINKY_MCP, PINKY_PIP),
    (PINKY_PIP, PINKY_DIP),
    (PINKY_DIP, PINKY_TIP),
    // Palm connections (knuckle line)
    (INDEX_MCP, MIDDLE_MCP),
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

#[derive(Clone, Default)]
pub struct HandData {
    pub landmarks: [HandLandmark; 21],
    pub valid: bool,
}

struct HandState {
    hands: [HandData; 2], // Up to 2 hands
    num_hands: usize,
    has_data: bool,
}

impl Default for HandState {
    fn default() -> Self {
        Self {
            hands: [HandData::default(), HandData::default()],
            num_hands: 0,
            has_data: false,
        }
    }
}

thread_local! {
    static HAND_STATE: RefCell<HandState> = RefCell::new(HandState::default());
}

// ============================================================================
// WASM API
// ============================================================================

/// Apply hand landmarks from MediaPipe
/// flat_data: [hand0_lm0_x, y, z, hand0_lm1_x, y, z, ..., hand1_...]
/// 21 landmarks per hand, 3 floats per landmark = 63 floats per hand
#[wasm_bindgen]
pub fn apply_hand_landmarks(flat_data: &[f32], num_hands: usize) {
    HAND_STATE.with(|state_cell| {
        let mut state = state_cell.borrow_mut();
        
        let actual_hands = num_hands.min(2);
        state.num_hands = actual_hands;
        state.has_data = actual_hands > 0;
        
        for h in 0..2 {
            if h < actual_hands {
                state.hands[h].valid = true;
                for i in 0..21 {
                    let base = h * 21 * 3 + i * 3;
                    if base + 2 < flat_data.len() {
                        state.hands[h].landmarks[i] = HandLandmark {
                            x: flat_data[base],
                            y: flat_data[base + 1],
                            z: flat_data[base + 2],
                        };
                    }
                }
            } else {
                state.hands[h].valid = false;
            }
        }
    });
}

// ============================================================================
// INTERNAL API (for renderer)
// ============================================================================

/// Get all hand data for rendering
pub fn get_hand_data() -> Option<([HandData; 2], usize)> {
    HAND_STATE.with(|state_cell| {
        let state = state_cell.borrow();
        if state.has_data {
            Some((state.hands.clone(), state.num_hands))
        } else {
            None
        }
    })
}
