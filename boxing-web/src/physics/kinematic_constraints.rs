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
        
        // Calculate current bone lengths (for comparison in debug log)
        let current_upper = Self::distance(shoulder, raw_elbow);
        let current_fore = Self::distance(raw_elbow, raw_wrist);
        
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
        
        // Verify corrected lengths
        let corrected_upper = Self::distance(shoulder, corrected_elbow);
        let corrected_fore = Self::distance(corrected_elbow, corrected_wrist);
        
        // DEBUG: Log BOTH input and output bone lengths
        static mut FRAME: u32 = 0;
        unsafe {
            FRAME += 1;
            if FRAME % 60 == 0 {
                web_sys::console::log_1(&format!(
                    "🦴 Bones: target={:.3}/{:.3} | raw={:.3}/{:.3} | corrected={:.3}/{:.3}",
                    upper_len, fore_len, 
                    current_upper, current_fore,
                    corrected_upper, corrected_fore
                ).into());
            }
        }
        
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

// ============================================================================
// LAYER 3: VELOCITY CLAMPING
// ============================================================================

/// Maximum allowed movement per frame (30% of screen at 30Hz)
/// Olympic boxer punch: ~15 m/s ≈ 0.3 normalized units per frame
const MAX_VELOCITY: f32 = 0.3;

/// Clamp position to maximum human velocity
/// If movement exceeds MAX_VELOCITY, limit it while preserving direction.
pub fn clamp_velocity(current: (f32, f32), previous: (f32, f32)) -> (f32, f32) {
    let dx = current.0 - previous.0;
    let dy = current.1 - previous.1;
    let speed = (dx * dx + dy * dy).sqrt();
    
    if speed <= MAX_VELOCITY {
        current  // Valid velocity
    } else {
        // Clamp to max velocity in same direction
        let ratio = MAX_VELOCITY / speed;
        (
            previous.0 + dx * ratio,
            previous.1 + dy * ratio,
        )
    }
}

// ============================================================================
// LAYER 4: OUTLIER REJECTION
// ============================================================================

/// Maximum allowed jump distance from prediction (15% of screen)
const MAX_JUMP: f32 = 0.15;

/// Reject outlier measurements that teleport too far from prediction.
/// Returns the measurement if valid, or falls back to predicted position.
pub fn reject_outlier(
    measured: (f32, f32), 
    predicted: (f32, f32),
    previous: (f32, f32),
) -> ((f32, f32), bool) {
    let dx = measured.0 - predicted.0;
    let dy = measured.1 - predicted.1;
    let jump = (dx * dx + dy * dy).sqrt();
    
    if jump > MAX_JUMP {
        // Outlier! Use smooth interpolation toward prediction
        // Don't snap - blend toward predicted
        let blend = 0.3;
        let blended = (
            previous.0 + (predicted.0 - previous.0) * blend,
            previous.1 + (predicted.1 - previous.1) * blend,
        );
        (blended, true)  // true = outlier was rejected
    } else {
        (measured, false)  // Accept measurement
    }
}

// ============================================================================
// LAYER 2: JOINT ANGLE LIMITS
// ============================================================================

/// Clamp elbow angle to human range [30°, 180°]
/// Returns adjusted wrist position if angle was out of range.
pub fn clamp_elbow_angle(
    shoulder: (f32, f32),
    elbow: (f32, f32),
    wrist: (f32, f32),
) -> ((f32, f32), bool) {
    // Calculate current angle
    let v1 = (shoulder.0 - elbow.0, shoulder.1 - elbow.1);
    let v2 = (wrist.0 - elbow.0, wrist.1 - elbow.1);
    
    let dot = v1.0 * v2.0 + v1.1 * v2.1;
    let mag1 = (v1.0 * v1.0 + v1.1 * v1.1).sqrt();
    let mag2 = (v2.0 * v2.0 + v2.1 * v2.1).sqrt();
    
    if mag1 < 0.0001 || mag2 < 0.0001 {
        return (wrist, false);
    }
    
    let cos_angle = (dot / (mag1 * mag2)).clamp(-1.0, 1.0);
    let angle = cos_angle.acos().to_degrees();
    
    // Valid human elbow range
    const MIN_ANGLE: f32 = 30.0;
    const MAX_ANGLE: f32 = 180.0;
    
    if angle >= MIN_ANGLE && angle <= MAX_ANGLE {
        return (wrist, false);  // Valid angle
    }
    
    // Clamp to nearest valid angle
    let target_angle = angle.clamp(MIN_ANGLE, MAX_ANGLE);
    let target_rad = target_angle.to_radians();
    
    // Rotate wrist around elbow
    let upper_angle = v1.1.atan2(v1.0);
    let new_angle = upper_angle + std::f32::consts::PI - target_rad;
    
    let new_wrist = (
        elbow.0 + new_angle.cos() * mag2,
        elbow.1 + new_angle.sin() * mag2,
    );
    
    (new_wrist, true)  // true = angle was clamped
}

