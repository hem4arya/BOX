use crate::physics::velocity::VelocityTracker;

pub struct LinearPredictor {
    pub raw_velocity: VelocityTracker,
    pub predicted_pos: (f32, f32),
    pub prediction_factor_frames: f32, // 1.5 frames = 50ms at 30fps
    pub last_velocity_mag: f32,
}

impl LinearPredictor {
    pub fn new() -> Self {
        Self {
            raw_velocity: VelocityTracker::new(),
            predicted_pos: (0.0, 0.0),
            prediction_factor_frames: 15.0, // 15 frames ahead (~500ms) - "The Edge"
            last_velocity_mag: 0.0,
        }
    }

    pub fn update(&mut self, current_pos: (f32, f32)) -> (f32, f32) {
        // 1. Update tracker and Get Velocity per Frame
        self.raw_velocity.update(current_pos);
        let (vx, vy) = self.raw_velocity.velocity_vector();
        
        // 2. Detect Abrupt Stop
        let curr_mag = (vx*vx + vy*vy).sqrt();
        let is_stopping = curr_mag < self.last_velocity_mag * 0.5;
        
        // 3. Calculate Target Prediction (15 frames ahead)
        let target_x = current_pos.0 + vx * self.prediction_factor_frames;
        let target_y = current_pos.1 + vy * self.prediction_factor_frames;
        
        // 4. Smoothing / Lerp
        let (final_x, final_y) = if is_stopping {
            (
                self.predicted_pos.0 + (target_x - self.predicted_pos.0) * 0.7, // Harder braking for high speed
                self.predicted_pos.1 + (target_y - self.predicted_pos.1) * 0.7
            )
        } else {
            (target_x, target_y)
        };
        
        self.predicted_pos = (final_x, final_y);
        self.last_velocity_mag = curr_mag;
        
        self.predicted_pos
    }
}
