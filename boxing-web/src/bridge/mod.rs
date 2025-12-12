//! Bridge module - JS ↔ Rust communication
//! 
//! All #[wasm_bindgen] entry points live here.
//! Re-exports only in mod.rs, logic in submodules.

mod landmarks;

pub use landmarks::{
    update_landmarks, 
    get_all_landmarks, 
    Landmark,
    // Constants for landmark indices
    NOSE, LEFT_SHOULDER, RIGHT_SHOULDER,
    LEFT_ELBOW, RIGHT_ELBOW,
    LEFT_WRIST, RIGHT_WRIST,
    ARM_SKELETON, KEY_LANDMARKS,
};
