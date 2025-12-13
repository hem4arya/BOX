//! Confidence Gate - Layer 1 of stability stack
//!
//! When landmark confidence < threshold, lock joint to parent position.
//! This prevents jitter from low-quality detections.

/// Confidence gate for a single joint
pub struct ConfidenceGate {
    /// Last valid offset from parent (saved when confidence was good)
    last_valid_offset: (f32, f32),
    /// Minimum confidence to accept raw data
    threshold: f32,
}

impl ConfidenceGate {
    pub fn new() -> Self {
        Self {
            last_valid_offset: (0.0, 0.0),
            threshold: 0.5,
        }
    }
    
    /// Set confidence threshold
    pub fn set_threshold(&mut self, threshold: f32) {
        self.threshold = threshold;
    }
    
    /// Apply confidence gating
    /// 
    /// If confidence >= threshold: use raw position and save offset
    /// If confidence < threshold: lock to parent using saved offset
    pub fn apply(
        &mut self,
        raw_pos: (f32, f32),
        parent_pos: (f32, f32),
        confidence: f32,
    ) -> (f32, f32) {
        if confidence >= self.threshold {
            // Good data - save offset relative to parent
            self.last_valid_offset = (
                raw_pos.0 - parent_pos.0,
                raw_pos.1 - parent_pos.1,
            );
            raw_pos
        } else {
            // Bad data - lock to parent with saved offset
            (
                parent_pos.0 + self.last_valid_offset.0,
                parent_pos.1 + self.last_valid_offset.1,
            )
        }
    }
    
    /// Get last valid offset
    pub fn last_offset(&self) -> (f32, f32) {
        self.last_valid_offset
    }
}

impl Default for ConfidenceGate {
    fn default() -> Self {
        Self::new()
    }
}
