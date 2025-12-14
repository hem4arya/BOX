//! One Euro Filter - adaptive low-pass filter for jitter reduction
//!
//! Smooth when slow (reduces jitter), responsive when fast (tracks punches).
//! Applied as post-processing after Kalman filter.

use std::f32::consts::PI;

/// Adaptive low-pass filter: smooth at rest, responsive during motion
pub struct OneEuroFilter {
    /// Minimum cutoff frequency (Hz) - lower = smoother at rest
    min_cutoff: f32,
    /// Speed coefficient - higher = less lag during fast motion
    beta: f32,
    /// Derivative cutoff frequency (Hz)
    d_cutoff: f32,
    
    // State
    x_prev: f32,
    dx_prev: f32,
    t_prev: f64,
    initialized: bool,
}

impl OneEuroFilter {
    pub fn new(min_cutoff: f32, beta: f32) -> Self {
        Self {
            min_cutoff,
            beta,
            d_cutoff: 1.0,
            x_prev: 0.0,
            dx_prev: 0.0,
            t_prev: 0.0,
            initialized: false,
        }
    }
    
    /// Boxing-tuned preset - good balance of smoothing and responsiveness
    pub fn for_boxing() -> Self {
        Self::new(1.0, 0.15)
    }
    
    /// Calculate smoothing factor alpha
    fn smoothing_factor(t_e: f32, cutoff: f32) -> f32 {
        let r = 2.0 * PI * cutoff * t_e;
        r / (r + 1.0)
    }
    
    /// Filter a single value
    /// 
    /// - `t`: timestamp in seconds
    /// - `x`: raw input value
    /// Returns: filtered value
    pub fn filter(&mut self, t: f64, x: f32) -> f32 {
        if !self.initialized {
            self.x_prev = x;
            self.t_prev = t;
            self.initialized = true;
            return x;
        }
        
        let t_e = (t - self.t_prev) as f32;
        if t_e <= 0.0 { 
            return self.x_prev; 
        }
        
        // 1. Estimate derivative (velocity)
        let a_d = Self::smoothing_factor(t_e, self.d_cutoff);
        let dx = (x - self.x_prev) / t_e;
        let dx_hat = a_d * dx + (1.0 - a_d) * self.dx_prev;
        
        // 2. Adaptive cutoff: more smoothing when slow, less when fast
        let cutoff = self.min_cutoff + self.beta * dx_hat.abs();
        let a = Self::smoothing_factor(t_e, cutoff);
        
        // 3. Apply filter
        let x_hat = a * x + (1.0 - a) * self.x_prev;
        
        // Update state
        self.x_prev = x_hat;
        self.dx_prev = dx_hat;
        self.t_prev = t;
        
        x_hat
    }
    
    /// Reset filter state
    pub fn reset(&mut self) {
        self.initialized = false;
    }
}

impl Default for OneEuroFilter {
    fn default() -> Self {
        Self::for_boxing()
    }
}

/// Pair of One Euro Filters for 2D position
pub struct OneEuroFilter2D {
    pub x: OneEuroFilter,
    pub y: OneEuroFilter,
}

impl OneEuroFilter2D {
    pub fn new() -> Self {
        Self {
            x: OneEuroFilter::for_boxing(),
            y: OneEuroFilter::for_boxing(),
        }
    }
    
    pub fn filter(&mut self, t: f64, pos: (f32, f32)) -> (f32, f32) {
        (self.x.filter(t, pos.0), self.y.filter(t, pos.1))
    }
    
    pub fn reset(&mut self) {
        self.x.reset();
        self.y.reset();
    }
}

impl Default for OneEuroFilter2D {
    fn default() -> Self {
        Self::new()
    }
}
