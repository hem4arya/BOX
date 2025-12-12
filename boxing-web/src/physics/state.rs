//! Hand state - shared data structure for physics calculations

/// Represents the physical state of a hand (wrist position)
#[derive(Clone, Default)]
pub struct HandState {
    /// Position in normalized coordinates (0-1)
    pub position: (f32, f32),
    
    /// Velocity in normalized units per frame
    pub velocity: (f32, f32),
    
    /// Depth percentage (0 = arm at side, 100+ = extended forward)
    pub depth_percent: f32,
    
    /// Elbow angle in degrees (90° = bent, 180° = straight)
    pub elbow_angle: f32,
    
    /// Is this the predicted state or raw measurement?
    pub is_predicted: bool,
}

impl HandState {
    pub fn new() -> Self {
        Self::default()
    }
    
    /// Speed magnitude of the hand
    pub fn speed(&self) -> f32 {
        (self.velocity.0 * self.velocity.0 + self.velocity.1 * self.velocity.1).sqrt()
    }
}
