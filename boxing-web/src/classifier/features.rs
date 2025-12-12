//! Feature extraction for punch classification
//!
//! Extracts 10 features per frame matching the Python training data format.

use crate::physics::calculate_elbow_angle;

/// Extract 10 features from current frame
/// 
/// Features (matches Python exactly):
/// - 0: wrist_x (0-1 normalized)
/// - 1: wrist_y (0-1 normalized)
/// - 2: elbow_x (0-1 normalized)
/// - 3: elbow_y (0-1 normalized)
/// - 4: shoulder_x (0-1 normalized)
/// - 5: shoulder_y (0-1 normalized)
/// - 6: elbow_angle (0-1, normalized from 0-180°)
/// - 7: velocity (0-1, scaled and capped)
/// - 8: direction_x (-1 to 1)
/// - 9: direction_y (-1 to 1)
pub fn extract_features(
    wrist: (f32, f32),
    elbow: (f32, f32),
    shoulder: (f32, f32),
    velocity: f32,
    velocity_dir: (f32, f32),
) -> [f32; 10] {
    // Calculate elbow angle (90° = bent, 180° = straight)
    let elbow_angle = calculate_elbow_angle(shoulder, elbow, wrist);
    
    // Normalize direction vector
    let dir_mag = (velocity_dir.0 * velocity_dir.0 + velocity_dir.1 * velocity_dir.1).sqrt();
    let (dir_x, dir_y) = if dir_mag > 0.001 {
        (velocity_dir.0 / dir_mag, velocity_dir.1 / dir_mag)
    } else {
        (0.0, 0.0)
    };
    
    [
        wrist.0,                          // 0: wrist_x
        wrist.1,                          // 1: wrist_y
        elbow.0,                          // 2: elbow_x
        elbow.1,                          // 3: elbow_y
        shoulder.0,                       // 4: shoulder_x
        shoulder.1,                       // 5: shoulder_y
        elbow_angle / 180.0,              // 6: normalized elbow angle
        (velocity * 2.0).min(1.0),        // 7: scaled velocity (capped at 1.0)
        dir_x,                            // 8: direction_x
        dir_y,                            // 9: direction_y
    ]
}
