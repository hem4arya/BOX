//! Classifier module - 1D-CNN punch classification
//!
//! Note: ONNX inference runs in JavaScript using onnxruntime-web.
//! Rust handles feature extraction and frame buffering.

mod buffer;
mod features;
mod model;

pub use buffer::{FrameBuffer, BUFFER_SIZE, FEATURE_COUNT};
pub use features::extract_features;
pub use model::{PunchType, PUNCH_TYPES, set_punch_result, get_last_punch};
