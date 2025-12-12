//! Rolling frame buffer for 30-frame sequence storage
//!
//! Stores 30 frames of 10 features each for CNN classification.

/// Number of frames in the classification window
pub const BUFFER_SIZE: usize = 30;

/// Number of features per frame
pub const FEATURE_COUNT: usize = 10;

/// Rolling buffer that maintains last 30 frames in chronological order
pub struct FrameBuffer {
    /// Circular buffer data: [frame][feature]
    data: [[f32; FEATURE_COUNT]; BUFFER_SIZE],
    
    /// Current write position (points to next slot to write)
    write_index: usize,
    
    /// Whether buffer has been filled at least once
    filled: bool,
}

impl FrameBuffer {
    pub fn new() -> Self {
        Self {
            data: [[0.0; FEATURE_COUNT]; BUFFER_SIZE],
            write_index: 0,
            filled: false,
        }
    }
    
    /// Push a new frame of features into the buffer
    pub fn push(&mut self, features: [f32; FEATURE_COUNT]) {
        self.data[self.write_index] = features;
        self.write_index = (self.write_index + 1) % BUFFER_SIZE;
        
        // Mark as filled when we wrap around
        if self.write_index == 0 {
            self.filled = true;
        }
    }
    
    /// Check if buffer has 30 frames (ready for classification)
    pub fn is_ready(&self) -> bool {
        self.filled
    }
    
    /// Get frame count (for debugging)
    pub fn frame_count(&self) -> usize {
        if self.filled {
            BUFFER_SIZE
        } else {
            self.write_index
        }
    }
    
    /// Returns data in chronological order as flat array
    /// Shape: [frame0.f0, frame0.f1, ..., frame29.f9] = 300 floats
    pub fn as_flat(&self) -> Vec<f32> {
        let mut result = Vec::with_capacity(BUFFER_SIZE * FEATURE_COUNT);
        
        // Start from oldest frame and go to newest
        for i in 0..BUFFER_SIZE {
            let idx = (self.write_index + i) % BUFFER_SIZE;
            result.extend_from_slice(&self.data[idx]);
        }
        
        result
    }
    
    /// Clear the buffer (useful on reset)
    pub fn clear(&mut self) {
        self.data = [[0.0; FEATURE_COUNT]; BUFFER_SIZE];
        self.write_index = 0;
        self.filled = false;
    }
}

impl Default for FrameBuffer {
    fn default() -> Self {
        Self::new()
    }
}
