use crate::physics::velocity::VelocityTracker;

pub struct LinearPredictor {
    pub raw_velocity: VelocityTracker,
    pub predicted_pos: (f32, f32),
    pub prediction_factor_frames: f32, // Reduced to 2.0 (~66ms)
    pub last_velocity_mag: f32,
    pub smoothed_velocity: (f32, f32), // EMA for velocity
}

impl LinearPredictor {
    pub fn new() -> Self {
        Self {
            raw_velocity: VelocityTracker::new(),
            predicted_pos: (0.0, 0.0),
            prediction_factor_frames: 2.0, // 2.0 frames = Conservative & Stable
            last_velocity_mag: 0.0,
            smoothed_velocity: (0.0, 0.0),
        }
    }

    pub fn update(&mut self, current_pos: (f32, f32)) -> (f32, f32) {
        // 1. Update tracker and Get Velocity per Frame
        self.raw_velocity.update(current_pos);
        let (raw_vx, raw_vy) = self.raw_velocity.velocity_vector();
        
        // 2. Smooth Velocity (EMA) to reduce "Sensitivity"
        // alpha 0.5 = 50% current, 50% history.
        let alpha = 0.5;
        let vx = raw_vx * alpha + self.smoothed_velocity.0 * (1.0 - alpha);
        let vy = raw_vy * alpha + self.smoothed_velocity.1 * (1.0 - alpha);
        self.smoothed_velocity = (vx, vy);
        
        // 3. Detect Abrupt Stop
        let curr_mag = (vx*vx + vy*vy).sqrt();
        let is_stopping = curr_mag < self.last_velocity_mag * 0.6;
        
        // 4. Calculate Target Prediction (2 frames ahead)
        // With only 2 frames, this stays very close to the raw hand.
        let target_x = current_pos.0 + vx * self.prediction_factor_frames;
        let target_y = current_pos.1 + vy * self.prediction_factor_frames;
        
        // 5. Smoothing / Lerp
        let (final_x, final_y) = if is_stopping {
            (
                self.predicted_pos.0 + (target_x - self.predicted_pos.0) * 0.4, // Soft braking
                self.predicted_pos.1 + (target_y - self.predicted_pos.1) * 0.4
            )
        } else {
            (target_x, target_y)
        };
        
        self.predicted_pos = (final_x, final_y);
        self.last_velocity_mag = curr_mag;
        
        self.predicted_pos
    }
}
