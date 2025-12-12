//! Elbow angle calculation using dot product
//!
//! Calculates the angle at the elbow joint using vectors from
//! shoulder→elbow (upper arm) and elbow→wrist (forearm).

/// Calculate elbow angle in degrees
/// 
/// Uses dot product formula: cos(θ) = (v1 · v2) / (|v1| × |v2|)
/// 
/// Returns angle in degrees:
/// - 90° = fully bent (fist near shoulder)
/// - 180° = fully straight (arm extended)
pub fn calculate_elbow_angle(
    shoulder: (f32, f32),
    elbow: (f32, f32),
    wrist: (f32, f32),
) -> f32 {
    // Vector from elbow to shoulder (upper arm)
    let v1 = (shoulder.0 - elbow.0, shoulder.1 - elbow.1);
    
    // Vector from elbow to wrist (forearm)
    let v2 = (wrist.0 - elbow.0, wrist.1 - elbow.1);
    
    // Dot product
    let dot = v1.0 * v2.0 + v1.1 * v2.1;
    
    // Magnitudes
    let mag1 = (v1.0 * v1.0 + v1.1 * v1.1).sqrt();
    let mag2 = (v2.0 * v2.0 + v2.1 * v2.1).sqrt();
    
    // Handle degenerate case
    if mag1 < 0.0001 || mag2 < 0.0001 {
        return 180.0; // Assume straight if we can't calculate
    }
    
    // cos(angle) = dot / (mag1 * mag2)
    let cos_angle = (dot / (mag1 * mag2)).clamp(-1.0, 1.0);
    
    // Convert to degrees
    cos_angle.acos().to_degrees()
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_straight_arm() {
        // Arm in a straight line
        let shoulder = (0.0, 0.0);
        let elbow = (0.5, 0.0);
        let wrist = (1.0, 0.0);
        let angle = calculate_elbow_angle(shoulder, elbow, wrist);
        assert!((angle - 180.0).abs() < 1.0);
    }
    
    #[test]
    fn test_bent_arm() {
        // Arm bent at 90 degrees
        let shoulder = (0.0, 0.0);
        let elbow = (0.5, 0.0);
        let wrist = (0.5, 0.5);
        let angle = calculate_elbow_angle(shoulder, elbow, wrist);
        assert!((angle - 90.0).abs() < 1.0);
    }
}
