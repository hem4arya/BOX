//! Physics module - Kalman filter prediction and biomechanics
//!
//! Re-exports only. All logic in submodules.

mod state;
mod kalman;
mod depth;
mod angles;
mod velocity;

pub use state::HandState;
pub use kalman::KalmanFilter;
pub use depth::DepthEstimator;
pub use angles::calculate_elbow_angle;
pub use velocity::{VelocityTracker, VELOCITY_FRAMES, DEAD_ZONE};
