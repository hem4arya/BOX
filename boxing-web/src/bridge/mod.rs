//! Bridge module - JS ↔ Rust communication
//! 
//! All #[wasm_bindgen] entry points live here.
//! Re-exports only in mod.rs, logic in submodules.

mod landmarks;

pub use landmarks::{
    // WASM entry points
    update_landmarks, 
    physics_tick,
    calibrate_depth,
    // Internal API
    get_all_landmarks, 
    get_predicted_wrists,
    get_debug_info,
    Landmark,
    // Constants
    NOSE, LEFT_SHOULDER, RIGHT_SHOULDER,
    LEFT_ELBOW, RIGHT_ELBOW,
    LEFT_WRIST, RIGHT_WRIST,
    ARM_SKELETON, KEY_LANDMARKS,
};
