//! Bridge module - JS ↔ Rust communication
//! 
//! All #[wasm_bindgen] entry points live here.
//! Re-exports only in mod.rs, logic in submodules.

mod landmarks;
mod classifier_integration;
mod hand_landmarks;

pub use landmarks::{
    // WASM entry points
    update_landmarks, 
    physics_tick,
    calibrate_depth,
    apply_mediapipe_correction,
    get_extrapolated_wrists,
    get_raw_wrists,
    set_extrapolation_params,
    // Internal API
    get_all_landmarks, 
    get_predicted_wrists,
    get_smoothed_wrists,
    get_debug_info,
    get_depth_validity,
    Landmark,
    // Constants
    NOSE, LEFT_SHOULDER, RIGHT_SHOULDER,
    LEFT_ELBOW, RIGHT_ELBOW,
    LEFT_WRIST, RIGHT_WRIST,
    ARM_SKELETON, KEY_LANDMARKS,
};

pub use classifier_integration::{
    init_classifier, 
    set_classifier_ready,
    get_classification_buffer,
    is_buffer_ready,
    get_last_punch, 
    is_classifier_ready
};

pub use hand_landmarks::{
    apply_hand_landmarks,
    calibrate_hand,
    get_hand_data,
    is_hand_calibrated,
    HandData,
    HandLandmark,
    HAND_SKELETON,
};
