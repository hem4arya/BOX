//! Kinematic constraints - fixed bone lengths to eliminate stretch/shrink jitter
//!
//! Enforces calibrated bone lengths by adjusting joint positions while
//! preserving the direction from MediaPipe detection.

/// Enforces fixed bone lengths to eliminate jitter
pub struct KinematicConstraints {
    /// Upper arm length normalized by shoulder width
    upper_arm_length: Option<f32>,
    /// Forearm length normalized by shoulder width
    forearm_length: Option<f32>,
}

impl KinematicConstraints {
    pub fn new() -> Self {
        Self {
            upper_arm_length: None,
            forearm_length: None,
        }
    }
    
    /// Calibrate during T-pose
    pub fn calibrate(
        &mut self,
        shoulder: (f32, f32),
        elbow: (f32, f32),
        wrist: (f32, f32),
        shoulder_width: f32,
    ) {
        if shoulder_width < 0.001 {
            return;
        }
        
        let upper_arm = Self::distance(shoulder, elbow) / shoulder_width;
        let forearm = Self::distance(elbow, wrist) / shoulder_width;
        
        self.upper_arm_length = Some(upper_arm);
        self.forearm_length = Some(forearm);
    }
    
    /// Check if calibrated
    pub fn is_calibrated(&self) -> bool {
        self.upper_arm_length.is_some() && self.forearm_length.is_some()
    }
    
    /// Apply constraints to enforce fixed bone lengths
    /// 
    /// Shoulder is fixed anchor. Elbow and wrist positions are adjusted
    /// to maintain calibrated bone lengths while preserving direction.
    /// 
    /// Returns: (corrected_elbow, corrected_wrist)
    pub fn apply(
        &self,
        shoulder: (f32, f32),
        raw_elbow: (f32, f32),
        raw_wrist: (f32, f32),
        shoulder_width: f32,
    ) -> ((f32, f32), (f32, f32)) {
        let upper_len = match self.upper_arm_length {
            Some(l) => l * shoulder_width,
            None => return (raw_elbow, raw_wrist),
        };
        let fore_len = match self.forearm_length {
            Some(l) => l * shoulder_width,
            None => return (raw_elbow, raw_wrist),
        };
        
        // Step 1: Correct elbow - keep direction, force length
        let elbow_dir = Self::normalize(
            raw_elbow.0 - shoulder.0,
            raw_elbow.1 - shoulder.1,
        );
        let corrected_elbow = (
            shoulder.0 + elbow_dir.0 * upper_len,
            shoulder.1 + elbow_dir.1 * upper_len,
        );
        
        // Step 2: Correct wrist - keep direction from corrected elbow, force length
        let wrist_dir = Self::normalize(
            raw_wrist.0 - corrected_elbow.0,
            raw_wrist.1 - corrected_elbow.1,
        );
        let corrected_wrist = (
            corrected_elbow.0 + wrist_dir.0 * fore_len,
            corrected_elbow.1 + wrist_dir.1 * fore_len,
        );
        
        (corrected_elbow, corrected_wrist)
    }
    
    fn distance(a: (f32, f32), b: (f32, f32)) -> f32 {
        let dx = b.0 - a.0;
        let dy = b.1 - a.1;
        (dx * dx + dy * dy).sqrt()
    }
    
    fn normalize(x: f32, y: f32) -> (f32, f32) {
        let mag = (x * x + y * y).sqrt();
        if mag < 0.0001 {
            (0.0, 0.0)
        } else {
            (x / mag, y / mag)
        }
    }
}

impl Default for KinematicConstraints {
    fn default() -> Self {
        Self::new()
    }
}
