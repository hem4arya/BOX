//! Physics-based punch detection
//!
//! Replaces ONNX ML with simple physics rules for sub-1ms detection.

/// Detected punch types
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PunchType {
    Jab,
    Cross,
    Hook,
    Uppercut,
    Idle,
}

impl PunchType {
    pub fn name(&self) -> &'static str {
        match self {
            PunchType::Jab => "JAB",
            PunchType::Cross => "CROSS",
            PunchType::Hook => "HOOK",
            PunchType::Uppercut => "UPPERCUT",
            PunchType::Idle => "IDLE",
        }
    }
}

/// Punch detector with physics-based rules
pub struct PunchDetector {
    /// Minimum velocity to trigger detection
    velocity_threshold: f32,
    /// Minimum depth percentage
    depth_threshold: f32,
    /// Cooldown frames between detections
    cooldown: u32,
    /// Current cooldown counter
    cooldown_counter: u32,
    /// Last detected punch
    last_punch: PunchType,
    /// Confidence of last detection
    last_confidence: f32,
}

impl PunchDetector {
    pub fn new() -> Self {
        Self {
            velocity_threshold: 0.08,
            depth_threshold: 25.0,
            cooldown: 15, // ~0.5s at 30Hz
            cooldown_counter: 0,
            last_punch: PunchType::Idle,
            last_confidence: 0.0,
        }
    }
    
    /// Detect punch from physics data
    /// Returns (PunchType, confidence)
    pub fn detect(
        &mut self,
        left_velocity: f32,
        right_velocity: f32,
        left_depth: f32,
        right_depth: f32,
        left_valid: bool,
        right_valid: bool,
        right_direction_y: f32, // For uppercut detection
    ) -> (PunchType, f32) {
        // Cooldown prevents spam
        if self.cooldown_counter > 0 {
            self.cooldown_counter -= 1;
            return (self.last_punch, self.last_confidence * 0.9);
        }
        
        let mut punch = PunchType::Idle;
        let mut confidence: f32 = 0.0;
        
        // JAB: Left hand forward punch
        if left_valid && left_velocity > self.velocity_threshold && left_depth > self.depth_threshold {
            let vel_score = (left_velocity / 0.2).min(1.0);
            let depth_score = (left_depth / 60.0).min(1.0);
            let jab_conf = vel_score * 0.5 + depth_score * 0.5;
            if jab_conf > confidence {
                punch = PunchType::Jab;
                confidence = jab_conf;
            }
        }
        
        // CROSS: Right hand forward punch (stronger)
        if right_valid && right_velocity > self.velocity_threshold * 1.2 && right_depth > self.depth_threshold * 1.3 {
            let vel_score = (right_velocity / 0.25).min(1.0);
            let depth_score = (right_depth / 70.0).min(1.0);
            let cross_conf = vel_score * 0.5 + depth_score * 0.5;
            if cross_conf > confidence {
                punch = PunchType::Cross;
                confidence = cross_conf;
            }
        }
        
        // UPPERCUT: Right hand moving upward
        if right_velocity > self.velocity_threshold && right_direction_y < -0.3 {
            let vel_score = (right_velocity / 0.15).min(1.0);
            let up_score = (-right_direction_y / 0.5).min(1.0);
            let upper_conf = vel_score * 0.5 + up_score * 0.5;
            if upper_conf > confidence {
                punch = PunchType::Uppercut;
                confidence = upper_conf;
            }
        }
        
        // HOOK: Right hand lateral (not forward, but fast)
        if right_velocity > self.velocity_threshold && !right_valid && right_depth < 20.0 {
            let vel_score = (right_velocity / 0.15).min(1.0);
            let hook_conf = vel_score * 0.8;
            if hook_conf > confidence {
                punch = PunchType::Hook;
                confidence = hook_conf;
            }
        }
        
        // Apply cooldown if punch detected
        if punch != PunchType::Idle && confidence > 0.5 {
            self.cooldown_counter = self.cooldown;
            self.last_punch = punch;
            self.last_confidence = confidence;
        }
        
        (punch, confidence)
    }
    
    /// Get last detected punch
    pub fn last_result(&self) -> (PunchType, f32) {
        (self.last_punch, self.last_confidence)
    }
    
    /// Reset detector state
    pub fn reset(&mut self) {
        self.cooldown_counter = 0;
        self.last_punch = PunchType::Idle;
        self.last_confidence = 0.0;
    }
}

impl Default for PunchDetector {
    fn default() -> Self {
        Self::new()
    }
}
