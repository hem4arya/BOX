//! Pythagorean depth estimation
//!
//! Uses Pythagorean theorem to estimate how far forward the arm is extended.
//! Calibrated against T-pose where arm is fully extended sideways (depth = 0).

/// Depth estimator using Pythagorean theorem
/// 
/// Math: Given calibrated arm length L and current 2D projected length P,
/// the depth Z = √(L² - P²), and depth_percent = (Z/L) × 100
pub struct DepthEstimator {
    /// Calibrated arm length from T-pose (2D distance shoulder→wrist)
    calibrated_arm_length: Option<f32>,
}

impl DepthEstimator {
    pub fn new() -> Self {
        Self {
            calibrated_arm_length: None,
        }
    }
    
    /// Check if calibration has been done
    pub fn is_calibrated(&self) -> bool {
        self.calibrated_arm_length.is_some()
    }
    
    /// Calibrate during T-pose (arms extended sideways)
    /// 
    /// In T-pose, the arm is at maximum 2D extension (depth ≈ 0),
    /// so we use this as the reference "full length".
    pub fn calibrate(&mut self, shoulder: (f32, f32), wrist: (f32, f32)) {
        let dx = wrist.0 - shoulder.0;
        let dy = wrist.1 - shoulder.1;
        let length = (dx * dx + dy * dy).sqrt();
        
        // Sanity check: arm should have some length
        if length > 0.05 {
            self.calibrated_arm_length = Some(length);
        }
    }
    
    /// Calculate depth percentage (0 = arm at side, 100+ = extended forward)
    /// 
    /// Returns None if not calibrated or calculation fails.
    pub fn calculate(&self, shoulder: (f32, f32), wrist: (f32, f32)) -> Option<f32> {
        let calibrated = self.calibrated_arm_length?;
        
        let dx = wrist.0 - shoulder.0;
        let dy = wrist.1 - shoulder.1;
        let projected = (dx * dx + dy * dy).sqrt();
        
        // If projected >= calibrated, arm is not reaching forward
        if projected >= calibrated {
            return Some(0.0);
        }
        
        // Pythagorean: Z = √(L² - P²)
        let depth_squared = calibrated * calibrated - projected * projected;
        let depth = depth_squared.sqrt();
        
        // Convert to percentage
        Some((depth / calibrated) * 100.0)
    }
    
    /// Get the calibrated arm length (for debugging)
    pub fn calibrated_length(&self) -> Option<f32> {
        self.calibrated_arm_length
    }
}

impl Default for DepthEstimator {
    fn default() -> Self {
        Self::new()
    }
}
