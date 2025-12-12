//! Renderer module - WebGPU state and drawing
//!
//! Re-exports only. All logic in submodules.

mod state;
mod shapes;
mod skeleton;
mod debug_ui;

pub use state::{initialize_gpu, GpuStateError};
pub use skeleton::render_frame;
pub use debug_ui::{
    set_frame_metrics, set_mediapipe_latency, set_onnx_latency,
    set_physics_time, get_debug_overlay_text, update_arm_metrics
};
pub use shapes::{Vertex, create_circle_vertices, create_line_vertices};
