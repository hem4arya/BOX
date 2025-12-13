//! Depth estimation with gated output
//!
//! Uses elbow angle + direction gates to prevent false depth on vertical movement.

use std::f32::consts::PI;

/// Result of depth calculation with gating
#[derive(Clone, Copy, Debug, Default)]
pub struct DepthResult {
    /// Raw Pythagorean depth (always calculated)
    pub raw_percent: f32,
    /// Gated depth (use for hit detection)
    pub gated_percent: f32,
    /// True if all gates passed (straight arm + horizontal)
    pub is_valid: bool,
    /// Direction angle in degrees (-180 to 180)
    pub direction_angle: f32,
}

/// Pythagorean depth estimator with gating
pub struct DepthEstimator {
    calibrated_arm_length: Option<f32>,
    calibrated_shoulder_width: Option<f32>,
}

impl DepthEstimator {
    pub fn new() -> Self {
        Self {
            calibrated_arm_length: None,
            calibrated_shoulder_width: None,
        }
    }
    
    /// Calibrate during T-pose
    pub fn calibrate(
        &mut self,
        left_shoulder: (f32, f32),
        right_shoulder: (f32, f32),
        wrist: (f32, f32),
        shoulder: (f32, f32),
    ) {
        let dx = right_shoulder.0 - left_shoulder.0;
        let dy = right_shoulder.1 - left_shoulder.1;
        self.calibrated_shoulder_width = Some((dx * dx + dy * dy).sqrt());
        
        let dx = wrist.0 - shoulder.0;
        let dy = wrist.1 - shoulder.1;
        self.calibrated_arm_length = Some((dx * dx + dy * dy).sqrt());
    }
    
    pub fn is_calibrated(&self) -> bool {
        self.calibrated_arm_length.is_some()
    }
    
    /// Calculate GATED depth with elbow + direction gates
    pub fn calculate_gated(
        &self,
        wrist: (f32, f32),
        shoulder: (f32, f32),
        elbow_angle: f32,
    ) -> DepthResult {
        let calibrated_arm = match self.calibrated_arm_length {
            Some(v) => v,
            None => return DepthResult::default(),
        };
        
        let dx = wrist.0 - shoulder.0;
        let dy = wrist.1 - shoulder.1;
        
        // 1. Raw Pythagorean depth
        let projected = (dx * dx + dy * dy).sqrt();
        let raw_depth = if projected < calibrated_arm {
            ((calibrated_arm.powi(2) - projected.powi(2)).sqrt() / calibrated_arm) * 100.0
        } else {
            0.0
        };
        
        // 2. Direction angle test
        let direction_angle = dy.atan2(dx) * 180.0 / PI;
        let abs_angle = direction_angle.abs();
        let is_vertical = abs_angle > 60.0 && abs_angle < 120.0;
        
        // 3. Elbow gate (arm must be straight: >150°)
        let is_straight = elbow_angle > 150.0;
        
        // 4. GATED DEPTH
        let gated_depth = if is_vertical {
            0.0  // Force zero on vertical movement
        } else if !is_straight {
            raw_depth * 0.3  // Heavy penalty if arm is bent
        } else {
            raw_depth  // Full depth only when arm straight + horizontal
        };
        
        DepthResult {
            raw_percent: raw_depth,
            gated_percent: gated_depth,
            is_valid: !is_vertical && is_straight,
            direction_angle,
        }
    }
}

impl Default for DepthEstimator {
    fn default() -> Self {
        Self::new()
    }
}
