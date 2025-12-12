//! Depth estimation with verticality filter
//!
//! Uses Pythagorean depth + cylindrical anchor filter to reject vertical movements.

use std::f32::consts::PI;

/// Result of depth calculation with validation
#[derive(Clone, Copy, Debug)]
pub struct DepthResult {
    /// Depth percentage (0-100)
    pub depth_percent: f32,
    /// True if movement is horizontal (valid punch vector)
    pub is_valid_punch_vector: bool,
    /// Direction angle in degrees (-180 to 180)
    pub direction_angle_deg: f32,
}

/// Pythagorean depth estimator with verticality filter
pub struct DepthEstimator {
    /// Calibrated arm length (shoulder to wrist in T-pose)
    calibrated_arm_length: Option<f32>,
    /// Calibrated shoulder width (for normalization)
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
        // Store shoulder width for normalization
        let dx = right_shoulder.0 - left_shoulder.0;
        let dy = right_shoulder.1 - left_shoulder.1;
        self.calibrated_shoulder_width = Some((dx * dx + dy * dy).sqrt());
        
        // Store arm length
        let dx = wrist.0 - shoulder.0;
        let dy = wrist.1 - shoulder.1;
        self.calibrated_arm_length = Some((dx * dx + dy * dy).sqrt());
    }
    
    /// Simple calibrate (legacy, uses only one arm)
    pub fn calibrate_simple(&mut self, shoulder: (f32, f32), wrist: (f32, f32)) {
        let dx = wrist.0 - shoulder.0;
        let dy = wrist.1 - shoulder.1;
        self.calibrated_arm_length = Some((dx * dx + dy * dy).sqrt());
        // Use arm length as approximation for missing shoulder width
        self.calibrated_shoulder_width = self.calibrated_arm_length;
    }
    
    /// Calculate depth WITH verticality filter
    pub fn calculate_with_filter(
        &self,
        wrist: (f32, f32),
        shoulder: (f32, f32),
        elbow_angle_deg: f32,
    ) -> Option<DepthResult> {
        let calibrated_arm = self.calibrated_arm_length?;
        
        // Calculate delta from shoulder to wrist
        let delta_x = wrist.0 - shoulder.0;
        let delta_y = wrist.1 - shoulder.1;
        
        // 2D reach distance
        let reach_2d = (delta_x * delta_x + delta_y * delta_y).sqrt();
        
        // Direction angle (0° = right, 90° = down, -90° = up)
        let direction_angle = delta_y.atan2(delta_x) * 180.0 / PI;
        
        // VERTICALITY TEST: Punch cone is ±60° from horizontal
        // Horizontal = angles near 0° or ±180°
        // Vertical = angles near ±90°
        let abs_angle = direction_angle.abs();
        let is_horizontal = abs_angle < 60.0 || abs_angle > 120.0;
        
        // Also require arm to be somewhat extended
        let is_arm_extended = elbow_angle_deg > 130.0;
        
        // Valid punch = horizontal movement + arm extended
        let is_valid = is_horizontal && is_arm_extended;
        
        // Calculate Pythagorean depth
        let depth_percent = if reach_2d < calibrated_arm {
            let depth_raw = (calibrated_arm * calibrated_arm - reach_2d * reach_2d).sqrt();
            (depth_raw / calibrated_arm) * 100.0
        } else {
            0.0
        };
        
        Some(DepthResult {
            depth_percent,
            is_valid_punch_vector: is_valid,
            direction_angle_deg: direction_angle,
        })
    }
    
    /// Legacy calculate (for compatibility)
    pub fn calculate(&self, shoulder: (f32, f32), wrist: (f32, f32)) -> Option<f32> {
        let calibrated = self.calibrated_arm_length?;
        let projected = ((wrist.0 - shoulder.0).powi(2) + (wrist.1 - shoulder.1).powi(2)).sqrt();
        
        if projected >= calibrated {
            return Some(0.0);
        }
        
        let depth_squared = calibrated.powi(2) - projected.powi(2);
        let depth = depth_squared.sqrt();
        Some((depth / calibrated) * 100.0)
    }
    
    pub fn is_calibrated(&self) -> bool {
        self.calibrated_arm_length.is_some()
    }
}

impl Default for DepthEstimator {
    fn default() -> Self {
        Self::new()
    }
}
