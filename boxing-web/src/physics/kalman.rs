//! Kalman Filter for position prediction
//!
//! State vector: [x, y, vx, vy, ax, ay]ᵀ (6 elements)
//! Prediction at 120Hz (every frame), correction at 30Hz (MediaPipe rate)

use nalgebra::{SMatrix, SVector};

/// 6-element state vector type
type State = SVector<f32, 6>;
/// 6x6 matrix type
type Matrix6 = SMatrix<f32, 6, 6>;
/// 2x6 matrix type (observation)
type Matrix2x6 = SMatrix<f32, 2, 6>;
/// 6x2 matrix type (Kalman gain)
type Matrix6x2 = SMatrix<f32, 6, 2>;
/// 2x2 matrix type
type Matrix2 = SMatrix<f32, 2, 2>;
/// 2-element vector type
type Vector2 = SVector<f32, 2>;

/// Kalman Filter for hand position tracking
/// 
/// Implements 6-state predictor with position, velocity, and acceleration.
/// Runs predict() at 120Hz, update() at 30Hz.
pub struct KalmanFilter {
    /// State: [x, y, vx, vy, ax, ay]
    state: State,
    
    /// State covariance matrix (uncertainty)
    covariance: Matrix6,
    
    /// Process noise (how much we trust physics model)
    process_noise: Matrix6,
    
    /// Measurement noise covariance
    measurement_noise: Matrix2,
}

impl KalmanFilter {
    /// Create a new Kalman filter with default noise parameters
    pub fn new() -> Self {
        let state = State::zeros();
        
        // Initial covariance - moderate uncertainty
        let covariance = Matrix6::identity() * 0.1;
        
        // Process noise - trust physics prediction somewhat
        // Higher values = more responsive, lower = smoother
        let mut process_noise = Matrix6::zeros();
        process_noise[(0, 0)] = 0.001;  // x position
        process_noise[(1, 1)] = 0.001;  // y position
        process_noise[(2, 2)] = 0.01;   // x velocity
        process_noise[(3, 3)] = 0.01;   // y velocity
        process_noise[(4, 4)] = 0.1;    // x acceleration
        process_noise[(5, 5)] = 0.1;    // y acceleration
        
        // Measurement noise - MediaPipe has some jitter
        let measurement_noise = Matrix2::identity() * 0.005;
        
        Self {
            state,
            covariance,
            process_noise,
            measurement_noise,
        }
    }
    
    /// Build transition matrix F for given timestep
    /// 
    /// ```text
    /// | 1  0  dt  0  0.5dt²  0      |
    /// | 0  1  0   dt 0       0.5dt² |
    /// | 0  0  1   0  dt      0      |
    /// | 0  0  0   1  0       dt     |
    /// | 0  0  0   0  1       0      |
    /// | 0  0  0   0  0       1      |
    /// ```
    fn transition_matrix(dt: f32) -> Matrix6 {
        let dt2 = 0.5 * dt * dt;
        Matrix6::new(
            1.0, 0.0, dt,  0.0, dt2, 0.0,
            0.0, 1.0, 0.0, dt,  0.0, dt2,
            0.0, 0.0, 1.0, 0.0, dt,  0.0,
            0.0, 0.0, 0.0, 1.0, 0.0, dt,
            0.0, 0.0, 0.0, 0.0, 1.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 1.0,
        )
    }
    
    /// Observation matrix H (we only measure x, y)
    fn observation_matrix() -> Matrix2x6 {
        Matrix2x6::new(
            1.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 1.0, 0.0, 0.0, 0.0, 0.0,
        )
    }
    
    /// Predict step - called at 120Hz
    /// 
    /// Uses physics model: x += v*dt + 0.5*a*dt²
    pub fn predict(&mut self, dt: f32) {
        let f = Self::transition_matrix(dt);
        
        // State prediction: x = F * x
        self.state = f * self.state;
        
        // Covariance prediction: P = F * P * Fᵀ + Q
        self.covariance = f * self.covariance * f.transpose() + self.process_noise;
        
        // Apply velocity decay to prevent infinite drift
        self.state[2] *= 0.98;  // vx decay
        self.state[3] *= 0.98;  // vy decay
        self.state[4] *= 0.95;  // ax decay
        self.state[5] *= 0.95;  // ay decay
    }
    
    /// Update step - called at 30Hz with MediaPipe measurement
    /// 
    /// Corrects prediction based on actual measurement
    pub fn update(&mut self, measured_x: f32, measured_y: f32) {
        let h = Self::observation_matrix();
        let z = Vector2::new(measured_x, measured_y);
        
        // Innovation: y = z - H * x
        let predicted_measurement = h * self.state;
        let innovation = z - predicted_measurement;
        
        // Innovation covariance: S = H * P * Hᵀ + R
        let s = h * self.covariance * h.transpose() + self.measurement_noise;
        
        // Kalman gain: K = P * Hᵀ * S⁻¹
        let s_inv = s.try_inverse().unwrap_or(Matrix2::identity());
        let k: Matrix6x2 = self.covariance * h.transpose() * s_inv;
        
        // State update: x = x + K * y
        self.state = self.state + k * innovation;
        
        // Covariance update: P = (I - K * H) * P
        let i = Matrix6::identity();
        self.covariance = (i - k * h) * self.covariance;
    }
    
    /// Get predicted position
    pub fn position(&self) -> (f32, f32) {
        (self.state[0], self.state[1])
    }
    
    /// Get estimated velocity
    pub fn velocity(&self) -> (f32, f32) {
        (self.state[2], self.state[3])
    }
    
    /// Get estimated acceleration
    pub fn acceleration(&self) -> (f32, f32) {
        (self.state[4], self.state[5])
    }
    
    /// Initialize filter with first measurement
    pub fn initialize(&mut self, x: f32, y: f32) {
        self.state = State::zeros();
        self.state[0] = x;
        self.state[1] = y;
    }
}

impl Default for KalmanFilter {
    fn default() -> Self {
        Self::new()
    }
}
