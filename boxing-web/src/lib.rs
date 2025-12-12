//! Boxing Web - WebGPU Punch Detection Game
//! 
//! Entry point for WASM module. Only contains:
//! - Module declarations
//! - wasm_bindgen entry points that delegate to submodules

mod bridge;
mod renderer;

use wasm_bindgen::prelude::*;

// Re-export wasm_bindgen functions for JS access
pub use bridge::update_landmarks;

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
