//! Boxing Web - WebGPU Punch Detection Game
//! 
//! Entry point for WASM module. Only contains:
//! - Module declarations
//! - wasm_bindgen entry points that delegate to submodules

mod bridge;
mod classifier;
mod physics;
mod renderer;

use wasm_bindgen::prelude::*;

// Re-export wasm_bindgen functions for JS access
pub use bridge::{
    update_landmarks, 
    physics_tick, 
    calibrate_depth, 
    apply_mediapipe_correction,
    apply_hand_landmarks,
    calibrate_hand,
    get_extrapolated_wrists,
    get_raw_wrists,
    set_extrapolation_params,
    init_classifier,
    set_classifier_ready,
    get_classification_buffer,
    is_buffer_ready,
};
pub use classifier::set_punch_result;
pub use renderer::{
    set_frame_metrics, set_mediapipe_latency, set_onnx_latency,
    set_physics_time, get_debug_overlay_text,
};

// ============================================================================
// CONSOLE LOGGING
// ============================================================================

#[wasm_bindgen]
extern "C" {
    #[wasm_bindgen(js_namespace = console)]
    fn log(s: &str);
}

macro_rules! console_log {
    ($($t:tt)*) => (log(&format_args!($($t)*).to_string()))
}

// ============================================================================
// WASM ENTRY POINTS
// ============================================================================

/// Called automatically when WASM module loads
#[wasm_bindgen(start)]
pub fn init_panic_hook() {
    console_error_panic_hook::set_once();
}

/// Initialize WebGPU - must be called before render_frame
#[wasm_bindgen]
pub async fn init() -> Result<(), JsValue> {
    renderer::initialize_gpu().await?;
    console_log!("✅ WebGPU initialized with landmark rendering");
    Ok(())
}

/// Render one frame with current landmarks
#[wasm_bindgen]
pub fn render_frame() {
    renderer::render_frame();
}
