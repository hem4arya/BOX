//! Landmark storage and JS bridge
//! 
//! Receives MediaPipe landmarks from JavaScript and stores them
//! for the renderer and physics systems to read.

use wasm_bindgen::prelude::*;
use std::cell::RefCell;

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
// LANDMARK DATA STRUCTURE
// ============================================================================

/// A single 3D landmark point (normalized coordinates)
#[derive(Clone, Copy, Default)]
pub struct Landmark {
    pub x: f32,  // 0-1 normalized
    pub y: f32,  // 0-1 normalized
    pub z: f32,  // Relative depth
}

/// Internal storage for current frame's landmarks
struct LandmarkStore {
    landmarks: [Landmark; 33],
    has_data: bool,
}

impl Default for LandmarkStore {
    fn default() -> Self {
        Self {
            landmarks: [Landmark::default(); 33],
            has_data: false,
        }
    }
}

// Thread-local storage (WASM is single-threaded)
thread_local! {
    static LANDMARKS: RefCell<LandmarkStore> = RefCell::new(LandmarkStore::default());
}

// ============================================================================
// WASM-BINDGEN ENTRY POINTS
// ============================================================================

/// Called from JavaScript with flat Float32Array of 99 values
/// (33 landmarks × 3 coordinates: x, y, z)
#[wasm_bindgen]
pub fn update_landmarks(data: &[f32]) {
    if data.len() != 99 {
        web_sys::console::warn_1(
            &format!("Invalid landmark data length: {} (expected 99)", data.len()).into()
        );
        return;
    }
    
    LANDMARKS.with(|store_cell| {
        let mut store = store_cell.borrow_mut();
        
        for i in 0..33 {
            store.landmarks[i] = Landmark {
                x: data[i * 3],
                y: data[i * 3 + 1],
                z: data[i * 3 + 2],
            };
        }
        store.has_data = true;
    });
}

// ============================================================================
// INTERNAL API (no wasm_bindgen)
// ============================================================================

/// Get all current landmarks (for renderer/physics)
pub fn get_all_landmarks() -> Option<[Landmark; 33]> {
    LANDMARKS.with(|store_cell| {
        let store = store_cell.borrow();
        if store.has_data {
            Some(store.landmarks)
        } else {
            None
        }
    })
}

/// Get a specific landmark by index
#[allow(dead_code)]
pub fn get_landmark(index: usize) -> Option<Landmark> {
    LANDMARKS.with(|store_cell| {
        let store = store_cell.borrow();
        if store.has_data && index < 33 {
            Some(store.landmarks[index])
        } else {
            None
        }
    })
}

/// Check if we have valid landmark data
#[allow(dead_code)]
pub fn has_landmarks() -> bool {
    LANDMARKS.with(|store_cell| store_cell.borrow().has_data)
}
