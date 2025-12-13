//! 2-Bone Analytical Inverse Kinematics Solver
//!
//! Mathematically solves for elbow position given shoulder (anchor) and wrist (target).
//! Ignores MediaPipe elbow data entirely - derives elbow from circle-circle intersection.

/// 2-Bone Analytical IK Solver for arm
pub struct ArmIK {
    /// Calibrated upper arm length (shoulder to elbow)
    upper_arm: f32,
    /// Calibrated forearm length (elbow to wrist)
    forearm: f32,
    /// Is this the right arm? (determines elbow bend direction)
    is_right_arm: bool,
}

impl ArmIK {
    pub fn new(is_right: bool) -> Self {
        Self {
            upper_arm: 0.0,
            forearm: 0.0,
            is_right_arm: is_right,
        }
    }
    
    /// Calibrate during T-pose (one time)
    pub fn calibrate(&mut self, shoulder: (f32, f32), elbow: (f32, f32), wrist: (f32, f32)) {
        self.upper_arm = Self::distance(shoulder, elbow);
        self.forearm = Self::distance(elbow, wrist);
        
        web_sys::console::log_1(&format!(
            "🦴 IK Calibrated {}: upper={:.3} fore={:.3}",
            if self.is_right_arm { "R" } else { "L" },
            self.upper_arm, self.forearm
        ).into());
    }
    
    /// Check if calibrated
    pub fn is_calibrated(&self) -> bool {
        self.upper_arm > 0.0 && self.forearm > 0.0
    }
    
    /// Solve for elbow given shoulder anchor and target wrist
    /// 
    /// Returns (solved_elbow, clamped_wrist)
    /// 
    /// The elbow MUST lie at the intersection of two circles:
    /// - Circle centered at shoulder with radius = upper arm length
    /// - Circle centered at wrist with radius = forearm length
    /// Solve for elbow given shoulder anchor and target wrist
    /// 
    /// Returns (solved_elbow, clamped_wrist)
    /// 
    /// The elbow MUST lie at the intersection of two circles:
    /// - Circle centered at shoulder with radius = upper arm length
    /// - Circle centered at wrist with radius = forearm length
    ///
    /// Uses raw_elbow_hint to choose which intersection point is correct (up/down).
    pub fn solve(&self, shoulder: (f32, f32), target_wrist: (f32, f32), raw_elbow_hint: (f32, f32)) -> ((f32, f32), (f32, f32)) {
        if !self.is_calibrated() {
            return (raw_elbow_hint, target_wrist);
        }
        
        let max_reach = self.upper_arm + self.forearm;
        let min_reach = (self.upper_arm - self.forearm).abs();
        
        // Vector from shoulder to target wrist
        let dx = target_wrist.0 - shoulder.0;
        let dy = target_wrist.1 - shoulder.1;
        let dist = (dx * dx + dy * dy).sqrt();
        
        // Clamp wrist to reachable range
        let clamped_wrist = if dist > max_reach * 0.999 {
            // Too far - clamp to max reach
            let scale = (max_reach * 0.999) / dist;
            (shoulder.0 + dx * scale, shoulder.1 + dy * scale)
        } else if dist < min_reach * 1.001 {
            // Too close - push out to min reach
            let scale = (min_reach * 1.001) / dist.max(0.001);
            (shoulder.0 + dx * scale, shoulder.1 + dy * scale)
        } else {
            target_wrist
        };
        
        // Recalculate distance after clamping
        let dx = clamped_wrist.0 - shoulder.0;
        let dy = clamped_wrist.1 - shoulder.1;
        let dist = (dx * dx + dy * dy).sqrt();
        
        // === CIRCLE-CIRCLE INTERSECTION (Law of Cosines) ===
        // Find angle at shoulder using: c² = a² + b² - 2ab*cos(C)
        let cos_angle = (self.upper_arm.powi(2) + dist.powi(2) - self.forearm.powi(2))
            / (2.0 * self.upper_arm * dist);
        let angle = cos_angle.clamp(-1.0, 1.0).acos();
        
        // Base angle (shoulder to wrist direction)
        let base_angle = dy.atan2(dx);
        
        // Calculate both possible elbow positions
        let elbow1 = (
            shoulder.0 + self.upper_arm * (base_angle + angle).cos(),
            shoulder.1 + self.upper_arm * (base_angle + angle).sin(),
        );
        
        let elbow2 = (
            shoulder.0 + self.upper_arm * (base_angle - angle).cos(),
            shoulder.1 + self.upper_arm * (base_angle - angle).sin(),
        );
        
        // Choose the solution closer to the raw elbow hint
        let d1 = Self::distance(elbow1, raw_elbow_hint);
        let d2 = Self::distance(elbow2, raw_elbow_hint);
        
        let elbow = if d1 < d2 { elbow1 } else { elbow2 };
        
        (elbow, clamped_wrist)
    }
    
    /// Solve using "Bone Length Fix with Original Direction" (FK)
    /// 
    /// Instead of calculating where the elbow *should* be (IK),
    /// this trusts the MediaPipe direction but enforces the calibrated length.
    /// 
    /// Algorithm:
    /// 1. Elbow = Shoulder + normalize(RawElbow - Shoulder) * UpperLength
    /// 2. Wrist = CorrectedElbow + normalize(RawWrist - CorrectedElbow) * ForearmLength
    pub fn solve_fk(&self, shoulder: (f32, f32), raw_elbow: (f32, f32), raw_wrist: (f32, f32)) -> ((f32, f32), (f32, f32)) {
        if !self.is_calibrated() {
            return (raw_elbow, raw_wrist);
        }

        // --- Step 1: Fix the Elbow ---
        // Get direction from Shoulder to Raw Elbow
        let dx1 = raw_elbow.0 - shoulder.0;
        let dy1 = raw_elbow.1 - shoulder.1;
        let len1 = (dx1 * dx1 + dy1 * dy1).sqrt();
        
        let (elbow_dx, elbow_dy) = if len1 > 0.001 {
            (dx1 / len1, dy1 / len1)
        } else {
            (0.0, 0.0)
        };

        // Force the Elbow to be at the exact "Upper Arm Length" distance
        let clean_elbow = (
            shoulder.0 + elbow_dx * self.upper_arm,
            shoulder.1 + elbow_dy * self.upper_arm,
        );

        // --- Step 2: Fix the Wrist ---
        // Get direction from Clean Elbow to Raw Wrist
        let dx2 = raw_wrist.0 - clean_elbow.0;
        let dy2 = raw_wrist.1 - clean_elbow.1;
        let len2 = (dx2 * dx2 + dy2 * dy2).sqrt();
        
        let (wrist_dx, wrist_dy) = if len2 > 0.001 {
            (dx2 / len2, dy2 / len2)
        } else {
            (0.0, 0.0)
        };

        // Force the Wrist to be at the exact "Forearm Length" distance
        let clean_wrist = (
            clean_elbow.0 + wrist_dx * self.forearm,
            clean_elbow.1 + wrist_dy * self.forearm,
        );

        (clean_elbow, clean_wrist)
    }
    
    /// Get calibrated lengths
    pub fn get_lengths(&self) -> (f32, f32) {
        (self.upper_arm, self.forearm)
    }
    
    fn distance(a: (f32, f32), b: (f32, f32)) -> f32 {
        let dx = b.0 - a.0;
        let dy = b.1 - a.1;
        (dx * dx + dy * dy).sqrt()
    }
}

impl Default for ArmIK {
    fn default() -> Self {
        Self::new(false)
    }
}
