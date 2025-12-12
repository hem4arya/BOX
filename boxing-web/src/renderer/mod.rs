//! Renderer module - WebGPU rendering for landmarks and effects
//! 
//! Re-exports only. All logic in submodules.

mod state;
mod skeleton;
mod shapes;

pub use state::{initialize_gpu, GpuStateError};
pub use skeleton::render_frame;
pub use shapes::{Vertex, create_circle_vertices, create_line_vertices};
