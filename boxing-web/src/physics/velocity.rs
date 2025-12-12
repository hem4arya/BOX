//! Velocity tracking with dead zone
//!
//! Tracks position history and calculates smoothed velocity.
//! Applies dead zone to filter out jitter.

use std::collections::VecDeque;

/// Number of frames to average for velocity calculation
pub const VELOCITY_FRAMES: usize = 3;

/// Velocity below this threshold is treated as zero (filters jitter)
pub const DEAD_ZONE: f32 = 0.02;

/// Velocity tracker with position history
pub struct VelocityTracker {
    /// Ring buffer of recent positions
    history: VecDeque<(f32, f32)>,
}

impl VelocityTracker {
    pub fn new() -> Self {
        Self {
            history: VecDeque::with_capacity(VELOCITY_FRAMES + 2),
        }
    }
    
    /// Update with new position, returns smoothed velocity magnitude
    /// 
    /// Uses 3-frame averaging to filter jitter.
    /// Applies dead zone to suppress small movements.
    pub fn update(&mut self, pos: (f32, f32)) -> f32 {
        self.history.push_back(pos);
        
        // Keep only enough history
        if self.history.len() > VELOCITY_FRAMES + 1 {
            self.history.pop_front();
        }
        
        // Need enough frames to calculate
        if self.history.len() < VELOCITY_FRAMES + 1 {
            return 0.0;
        }
        
        // Calculate velocity from oldest to newest over N frames
        let old = self.history.front().unwrap();
        let dx = pos.0 - old.0;
        let dy = pos.1 - old.1;
        let raw_velocity = (dx * dx + dy * dy).sqrt() / VELOCITY_FRAMES as f32;
        
        // Apply dead zone
        if raw_velocity < DEAD_ZONE {
            0.0
        } else {
            raw_velocity
        }
    }
    
    /// Get velocity vector (direction and magnitude)
    pub fn velocity_vector(&self) -> (f32, f32) {
        if self.history.len() < VELOCITY_FRAMES + 1 {
            return (0.0, 0.0);
        }
        
        let old = self.history.front().unwrap();
        let new = self.history.back().unwrap();
        
        let dx = (new.0 - old.0) / VELOCITY_FRAMES as f32;
        let dy = (new.1 - old.1) / VELOCITY_FRAMES as f32;
        
        let mag = (dx * dx + dy * dy).sqrt();
        if mag < DEAD_ZONE {
            (0.0, 0.0)
        } else {
            (dx, dy)
        }
    }
    
    /// Clear history (useful on calibration reset)
    pub fn clear(&mut self) {
        self.history.clear();
    }
}

impl Default for VelocityTracker {
    fn default() -> Self {
        Self::new()
    }
}
