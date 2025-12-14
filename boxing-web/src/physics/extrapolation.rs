//! Position extrapolation for sub-20ms perceived latency
//!
//! Predicts where the hand IS NOW based on velocity and system latency.

/// System latency components (in milliseconds)
const CAMERA_LATENCY: f32 = 33.0;   // Camera capture time
const PIPELINE_LATENCY: f32 = 20.0; // USB/Browser overhead
const MEDIAPIPE_LATENCY: f32 = 25.0; // Inference time
const TOTAL_SYSTEM_LATENCY: f32 = 90.0; // Tuned for "Aggressive" latency compensation

/// Overshoot factor (1.2 = predict 20% further for snappier feel)
const OVERSHOOT: f32 = 1.2;

/// Velocity smoothing factor (0.4 = 40% new, 60% old - better stability)
const VELOCITY_ALPHA: f32 = 0.4;

/// Extrapolator predicts position ahead based on velocity
pub struct Extrapolator {
    /// Last known position
    last_pos: (f32, f32),
    /// Smoothed velocity (units per ms)
    velocity: (f32, f32),
    /// Timestamp of last update (ms)
    last_time: f64,
    /// Configurable system latency
    system_latency: f32,
    /// Configurable overshoot
    overshoot: f32,
}

impl Extrapolator {
    pub fn new() -> Self {
        Self {
            last_pos: (0.5, 0.5),
            velocity: (0.0, 0.0),
            last_time: 0.0,
            system_latency: TOTAL_SYSTEM_LATENCY,
            overshoot: OVERSHOOT,
        }
    }
    
    /// Update with new measurement
    pub fn update(&mut self, pos: (f32, f32), time_ms: f64) {
        let dt = time_ms - self.last_time;
        
        // Only update velocity for reasonable frame times
        if dt > 0.0 && dt < 200.0 {
            let dt_f = dt as f32;
            let new_vx = (pos.0 - self.last_pos.0) / dt_f;
            let new_vy = (pos.1 - self.last_pos.1) / dt_f;
            
            // Exponential smoothing
            self.velocity.0 = self.velocity.0 * (1.0 - VELOCITY_ALPHA) + new_vx * VELOCITY_ALPHA;
            self.velocity.1 = self.velocity.1 * (1.0 - VELOCITY_ALPHA) + new_vy * VELOCITY_ALPHA;
        }
        
        self.last_pos = pos;
        self.last_time = time_ms;
    }
    
    /// Predict position at current time (compensates for latency)
    pub fn predict(&self, now_ms: f64) -> (f32, f32) {
        // How late is the last known position?
        let time_since_capture = (now_ms - self.last_time) as f32;
        
        // Total latency = time since capture + system latency
        let total_latency = time_since_capture + self.system_latency;
        
        // Predict ahead with overshoot
        (
            self.last_pos.0 + self.velocity.0 * total_latency * self.overshoot,
            self.last_pos.1 + self.velocity.1 * total_latency * self.overshoot,
        )
    }
    
    /// Get raw (non-predicted) position
    pub fn raw_position(&self) -> (f32, f32) {
        self.last_pos
    }
    
    /// Get current velocity
    pub fn velocity(&self) -> (f32, f32) {
        self.velocity
    }
    
    /// Get velocity magnitude (for punch detection)
    pub fn velocity_magnitude(&self) -> f32 {
        (self.velocity.0 * self.velocity.0 + self.velocity.1 * self.velocity.1).sqrt()
    }
    
    /// Tune system latency
    pub fn set_latency(&mut self, latency_ms: f32) {
        self.system_latency = latency_ms;
    }
    
    /// Tune overshoot
    pub fn set_overshoot(&mut self, overshoot: f32) {
        self.overshoot = overshoot;
    }
}

impl Default for Extrapolator {
    fn default() -> Self {
        Self::new()
    }
}
