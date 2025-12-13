//! Physics module - Kalman filter prediction and biomechanics
//!
//! Re-exports only. All logic in submodules.

mod state;
mod kalman;
mod depth;
mod angles;
mod velocity;
mod one_euro;
mod kinematic_constraints;
mod detection;

pub use state::HandState;
pub use kalman::KalmanFilter;
pub use depth::{DepthEstimator, DepthResult};
pub use angles::calculate_elbow_angle;
pub use velocity::{VelocityTracker, VELOCITY_FRAMES, DEAD_ZONE};
pub use one_euro::{OneEuroFilter, OneEuroFilter2D};
pub use kinematic_constraints::{KinematicConstraints, clamp_velocity, reject_outlier, clamp_elbow_angle};
pub use detection::{PunchDetector, PunchType};
