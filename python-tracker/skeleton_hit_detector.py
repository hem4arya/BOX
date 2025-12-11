"""
Skeleton-Based Hit Detector with Advanced Signal Processing
============================================================
PHYSICS-BASED detection using:
1. MediaPipe Pose skeleton (shoulder → elbow → wrist)
2. Anthropometric depth (Pythagoras: Z = sqrt(L² - P²))
3. Elbow angle gating (bent arm = guard, straight = punch)

QUICK WINS:
4. Direction consistency (dot product of velocity vectors)
5. Kinetic energy profile (KE = 0.5 × v²)
6. Trajectory phase detection (ACCEL → PEAK → DECEL → IDLE)

MEDIUM EFFORT:
7. Lucas-Kanade optical flow for motion validation
8. Kalman Filter for state estimation [x, y, vx, vy, ax, ay]
"""

import numpy as np
import cv2
from collections import deque
import time


# ════════════════════════════════════════════════════════════════════════════
# MEDIUM EFFORT #2: Kalman Filter State Estimator
# ════════════════════════════════════════════════════════════════════════════
class KalmanStateEstimator:
    """
    Kalman Filter for hand state estimation.
    
    State vector: [x, y, vx, vy, ax, ay]
    - x, y: Position (normalized)
    - vx, vy: Velocity
    - ax, ay: Acceleration
    
    Fuses skeleton measurements with physics-based motion model.
    """
    
    def __init__(self, process_noise=0.01, measurement_noise=0.1):
        # State: [x, y, vx, vy, ax, ay]
        self.state = np.zeros(6)
        
        # State transition matrix (constant acceleration model)
        # dt will be applied each update
        self.F = np.eye(6)
        
        # Measurement matrix (we observe x, y)
        self.H = np.zeros((2, 6))
        self.H[0, 0] = 1  # x
        self.H[1, 1] = 1  # y
        
        # Covariance matrix
        self.P = np.eye(6) * 0.1
        
        # Process noise (model uncertainty)
        self.Q = np.eye(6) * process_noise
        
        # Measurement noise (sensor uncertainty)
        self.R = np.eye(2) * measurement_noise
        
        self.initialized = False
        self.last_time = None
    
    def predict(self, dt):
        """Predict next state based on motion model."""
        # Update state transition matrix with dt
        self.F = np.array([
            [1, 0, dt, 0, 0.5*dt**2, 0],
            [0, 1, 0, dt, 0, 0.5*dt**2],
            [0, 0, 1, 0, dt, 0],
            [0, 0, 0, 1, 0, dt],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1]
        ])
        
        # Predict state
        self.state = self.F @ self.state
        
        # Predict covariance
        self.P = self.F @ self.P @ self.F.T + self.Q
    
    def update(self, measurement, timestamp=None):
        """
        Update state with new measurement.
        
        Args:
            measurement: [x, y] position observation
            timestamp: Optional timestamp for dt calculation
        """
        if not self.initialized:
            self.state[0] = measurement[0]
            self.state[1] = measurement[1]
            self.initialized = True
            self.last_time = timestamp or time.time()
            return self.state.copy()
        
        # Calculate dt
        current_time = timestamp or time.time()
        dt = max(0.001, current_time - self.last_time) if self.last_time else 0.033
        self.last_time = current_time
        
        # Predict step
        self.predict(dt)
        
        # Measurement residual
        z = np.array(measurement)
        y = z - self.H @ self.state
        
        # Kalman gain
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        
        # Update state
        self.state = self.state + K @ y
        
        # Update covariance
        I = np.eye(6)
        self.P = (I - K @ self.H) @ self.P
        
        return self.state.copy()
    
    def get_velocity(self):
        """Get estimated velocity [vx, vy]."""
        return self.state[2:4].copy()
    
    def get_acceleration(self):
        """Get estimated acceleration [ax, ay]."""
        return self.state[4:6].copy()
    
    def get_speed(self):
        """Get velocity magnitude."""
        return np.linalg.norm(self.state[2:4])
    
    def reset(self):
        """Reset filter state."""
        self.state = np.zeros(6)
        self.P = np.eye(6) * 0.1
        self.initialized = False
        self.last_time = None


# ════════════════════════════════════════════════════════════════════════════
# MEDIUM EFFORT #1: Optical Flow Tracker
# ════════════════════════════════════════════════════════════════════════════
class OpticalFlowTracker:
    """
    Lucas-Kanade optical flow for motion validation.
    
    Tracks specific points (wrists) using sparse optical flow.
    Provides motion validation independent of skeleton detection.
    """
    
    def __init__(self):
        # Lucas-Kanade parameters
        self.lk_params = dict(
            winSize=(21, 21),
            maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
        )
        
        self.prev_gray = None
        self.prev_points = {'Left': None, 'Right': None}
        self.flow_velocity = {'Left': np.zeros(2), 'Right': np.zeros(2)}
    
    def update(self, gray_frame, skeleton_points):
        """
        Update optical flow with new frame.
        
        Args:
            gray_frame: Grayscale frame (H, W)
            skeleton_points: Dict with 'Left' and 'Right' wrist positions as (x, y) pixels
        
        Returns:
            Dict with flow velocities for each hand
        """
        results = {'Left': np.zeros(2), 'Right': np.zeros(2)}
        
        if self.prev_gray is None:
            self.prev_gray = gray_frame.copy()
            for hand in ['Left', 'Right']:
                if hand in skeleton_points and skeleton_points[hand] is not None:
                    pt = skeleton_points[hand]
                    self.prev_points[hand] = np.array([[pt]], dtype=np.float32)
            return results
        
        for hand in ['Left', 'Right']:
            if hand not in skeleton_points or skeleton_points[hand] is None:
                continue
            
            current_pt = skeleton_points[hand]
            
            if self.prev_points[hand] is not None:
                # Track previous point forward
                next_pts, status, _ = cv2.calcOpticalFlowPyrLK(
                    self.prev_gray, gray_frame,
                    self.prev_points[hand], None,
                    **self.lk_params
                )
                
                if status[0][0] == 1:
                    # Calculate flow velocity
                    flow = next_pts[0][0] - self.prev_points[hand][0][0]
                    results[hand] = flow
                    self.flow_velocity[hand] = flow
            
            # Update tracking point from skeleton
            self.prev_points[hand] = np.array([[current_pt]], dtype=np.float32)
        
        self.prev_gray = gray_frame.copy()
        return results
    
    def get_flow_magnitude(self, hand):
        """Get magnitude of optical flow for hand."""
        return np.linalg.norm(self.flow_velocity[hand])
    
    def validate_skeleton_movement(self, hand, skeleton_velocity, threshold=0.5):
        """
        Validate skeleton-based velocity with optical flow.
        
        Returns True if optical flow agrees with skeleton movement.
        """
        flow_vel = np.linalg.norm(self.flow_velocity[hand])
        skel_vel = np.linalg.norm(skeleton_velocity)
        
        if skel_vel < 0.01:  # No skeleton movement
            return True
        
        # Check if flow velocity is in same ballpark as skeleton velocity
        # (allowing for scale differences)
        ratio = flow_vel / (skel_vel * 1000 + 0.001)  # Scale normalization
        return ratio > threshold
    
    def reset(self):
        """Reset optical flow state."""
        self.prev_gray = None
        self.prev_points = {'Left': None, 'Right': None}
        self.flow_velocity = {'Left': np.zeros(2), 'Right': np.zeros(2)}

class SkeletonHitDetector:
    """
    Velocity-based hit detection with depth awareness.
    
    Detects:
    - FORWARD: Hand moving toward camera (Z decreasing, or X/Y converging to center)
    - SIDEWAYS: Hand moving horizontally
    - IDLE: No significant movement
    """
    
    # MediaPipe Pose landmark indices
    WRIST_LEFT = 15
    WRIST_RIGHT = 16
    SHOULDER_LEFT = 11
    SHOULDER_RIGHT = 12
    ELBOW_LEFT = 13
    ELBOW_RIGHT = 14
    
    def __init__(self, velocity_threshold=0.10, cooldown_frames=8):
        """
        Args:
            velocity_threshold: Minimum normalized velocity to register hit
            cooldown_frames: Frames to wait between hits for same hand
        """
        self.velocity_threshold = velocity_threshold
        self.cooldown_frames = cooldown_frames
        
        # Position history for velocity and depth calculation
        self.position_history = {
            'Left': deque(maxlen=8),  # Increased for velocity vector tracking
            'Right': deque(maxlen=8)
        }
        
        # Velocity vector history for direction consistency
        self.velocity_vectors = {
            'Left': deque(maxlen=5),
            'Right': deque(maxlen=5)
        }
        
        # Cooldown tracking
        self.last_hit_frame = {'Left': -100, 'Right': -100}
        self.frame_count = 0
        
        # Hit statistics
        self.hit_count = {'Left': 0, 'Right': 0}
        
        # Debug values
        self.last_velocity = {'Left': 0, 'Right': 0}
        self.last_direction = {'Left': 'IDLE', 'Right': 'IDLE'}
        self.last_depth_change = {'Left': 0, 'Right': 0}
        self.last_depth_percent = {'Left': 0, 'Right': 0}
        self.last_anthropometric_depth = {'Left': 0, 'Right': 0}
        
        # ════════════════════════════════════════════════════════════════
        # QUICK WINS: Advanced Signal Processing
        # ════════════════════════════════════════════════════════════════
        
        # 1. Direction Consistency - dot product of velocity vectors
        self.last_direction_consistency = {'Left': 0, 'Right': 0}
        
        # 2. Kinetic Energy Profile - KE = 0.5 * m * v²
        self.last_kinetic_energy = {'Left': 0, 'Right': 0}
        self.peak_kinetic_energy = {'Left': 0, 'Right': 0}  # Track peak for ballistic profile
        
        # 3. Trajectory Phase - acceleration/deceleration detection
        self.trajectory_phase = {'Left': 'IDLE', 'Right': 'IDLE'}  # ACCEL, PEAK, DECEL, IDLE
        self.last_acceleration = {'Left': 0, 'Right': 0}
        
        # ════════════════════════════════════════════════════════════════
        # MEDIUM EFFORT: Optical Flow + Kalman Filter
        # ════════════════════════════════════════════════════════════════
        
        # Kalman Filters for each hand (state: [x, y, vx, vy, ax, ay])
        self.kalman = {
            'Left': KalmanStateEstimator(process_noise=0.005, measurement_noise=0.05),
            'Right': KalmanStateEstimator(process_noise=0.005, measurement_noise=0.05)
        }
        
        # Optical Flow Tracker (validates skeleton movement)
        self.optical_flow = OpticalFlowTracker()
        
        # Kalman-filtered values for debug
        self.kalman_velocity = {'Left': np.zeros(2), 'Right': np.zeros(2)}
        self.kalman_accel = {'Left': np.zeros(2), 'Right': np.zeros(2)}
        self.flow_validated = {'Left': False, 'Right': False}
        
        # ═══════════════════════════════════════════════════════════════
        # ANTHROPOMETRIC DEPTH CALIBRATION
        # ═══════════════════════════════════════════════════════════════
        # Human arm length is FIXED - we use this for physics-based depth!
        # When 2D projection shrinks, arm MUST be going into depth.
        # Z = sqrt(L² - P²) where L = calibrated length, P = projected 2D
        self.calibrated = False
        self.calibrated_arm_length = {
            'Left': 0.0,   # 2D arm length when extended SIDEWAYS (max visible)
            'Right': 0.0
        }
        
        # Auto-calibration: track max observed arm length as fallback
        self.max_observed_arm = {'Left': 0.5, 'Right': 0.5}
    
    def calibrate_arm_length(self, pose_landmarks):
        """
        Calibrate arm length with arm extended SIDEWAYS (T-pose).
        
        Press 'C' with arms extended horizontally to calibrate.
        This captures the maximum visible 2D arm length.
        """
        for hand in ['Left', 'Right']:
            if hand == 'Left':
                shoulder = pose_landmarks.landmark[self.SHOULDER_LEFT]
                wrist = pose_landmarks.landmark[self.WRIST_LEFT]
            else:
                shoulder = pose_landmarks.landmark[self.SHOULDER_RIGHT]
                wrist = pose_landmarks.landmark[self.WRIST_RIGHT]
            
            # Calculate 2D arm length (shoulder to wrist)
            dx = wrist.x - shoulder.x
            dy = wrist.y - shoulder.y
            arm_length_2d = np.sqrt(dx**2 + dy**2)
            
            self.calibrated_arm_length[hand] = arm_length_2d
            print(f"[CALIBRATE] {hand} arm length = {arm_length_2d:.4f}")
        
        self.calibrated = True
        print("[CALIBRATE] ✓ Ready! Arm length calibrated. Forward punches will now calculate depth.")
    
    def calculate_anthropometric_depth(self, pose_landmarks, hand):
        """
        Calculate REAL depth using anthropometric principle.
        
        Physics: Real arm length L is CONSTANT.
        When 2D projection P shrinks, arm MUST be extending into depth Z.
        
        Pythagoras: L² = P² + Z²
        Therefore: Z = sqrt(L² - P²)
        
        Returns:
            depth: The calculated forward extension (0 = sideways, max = toward camera)
            depth_percent: 0-100% of forward extension
        """
        if hand == 'Left':
            shoulder = pose_landmarks.landmark[self.SHOULDER_LEFT]
            wrist = pose_landmarks.landmark[self.WRIST_LEFT]
        else:
            shoulder = pose_landmarks.landmark[self.SHOULDER_RIGHT]
            wrist = pose_landmarks.landmark[self.WRIST_RIGHT]
        
        # Current 2D projection length
        dx = wrist.x - shoulder.x
        dy = wrist.y - shoulder.y
        projected_2d = np.sqrt(dx**2 + dy**2)
        
        # Track max observed for auto-calibration fallback
        if projected_2d > self.max_observed_arm[hand]:
            self.max_observed_arm[hand] = 0.95 * self.max_observed_arm[hand] + 0.05 * projected_2d
        
        # Use calibrated length if available, otherwise use max observed
        if self.calibrated and self.calibrated_arm_length[hand] > 0.01:
            L = self.calibrated_arm_length[hand]
        else:
            L = self.max_observed_arm[hand]
        
        # If projection >= calibrated length, arm is sideways (depth ≈ 0)
        if projected_2d >= L:
            return 0.0, 0.0
        
        # Calculate depth using Pythagoras: Z = sqrt(L² - P²)
        depth = np.sqrt(L**2 - projected_2d**2)
        
        # Calculate depth percentage (0% = sideways, 100% = fully forward)
        # Max possible depth is when P = 0, so max_depth = L
        depth_percent = (depth / L) * 100.0 if L > 0.01 else 0.0
        depth_percent = min(100.0, depth_percent)  # Cap at 100%
        
        return depth, depth_percent
    
    # ════════════════════════════════════════════════════════════════════
    # QUICK WIN #1: Direction Consistency
    # ════════════════════════════════════════════════════════════════════
    def calculate_direction_consistency(self, hand):
        """
        Check if velocity vectors are pointing in consistent direction.
        
        Uses dot product of consecutive velocity vectors.
        High consistency (close to 1.0) = hand moving in one direction
        Low consistency (close to 0 or negative) = erratic movement
        
        Returns: 0.0 to 1.0 (1.0 = perfectly consistent direction)
        """
        vectors = list(self.velocity_vectors[hand])
        if len(vectors) < 3:
            return 0.0
        
        # Calculate average dot product between consecutive vectors
        total_consistency = 0.0
        count = 0
        
        for i in range(1, len(vectors)):
            v1 = vectors[i-1]
            v2 = vectors[i]
            
            # Normalize vectors
            mag1 = np.linalg.norm(v1)
            mag2 = np.linalg.norm(v2)
            
            if mag1 > 0.001 and mag2 > 0.001:
                dot = np.dot(v1, v2) / (mag1 * mag2)
                total_consistency += max(0, dot)  # Only count positive consistency
                count += 1
        
        if count == 0:
            return 0.0
        
        consistency = total_consistency / count
        self.last_direction_consistency[hand] = consistency
        return consistency
    
    # ════════════════════════════════════════════════════════════════════
    # QUICK WIN #2: Kinetic Energy Profile
    # ════════════════════════════════════════════════════════════════════
    def calculate_kinetic_energy(self, velocity, hand):
        """
        Calculate kinetic energy proxy: KE = 0.5 * v²
        
        High KE indicates forceful strike intent.
        We use velocity squared as a proxy (mass is constant).
        
        Returns: Normalized kinetic energy (0.0 to 1.0)
        """
        # KE proportional to v² (we don't know actual mass)
        ke = 0.5 * velocity**2
        
        # Normalize to 0-1 range (threshold velocity = 0.1, max expected ~0.3)
        ke_normalized = min(1.0, ke / 0.045)  # 0.5 * 0.3² = 0.045
        
        self.last_kinetic_energy[hand] = ke_normalized
        
        # Track peak for ballistic profile detection
        if ke_normalized > self.peak_kinetic_energy[hand]:
            self.peak_kinetic_energy[hand] = ke_normalized
        else:
            # Decay peak slowly
            self.peak_kinetic_energy[hand] *= 0.95
        
        return ke_normalized
    
    # ════════════════════════════════════════════════════════════════════
    # QUICK WIN #3: Trajectory Phase Detection
    # ════════════════════════════════════════════════════════════════════
    def detect_trajectory_phase(self, velocity, prev_velocity, hand):
        """
        Detect if hand is in ACCEL, PEAK, DECEL, or IDLE phase.
        
        Ballistic punch profile:
        1. ACCEL: Velocity increasing rapidly
        2. PEAK: Maximum velocity reached
        3. DECEL: Velocity decreasing (contact/retraction)
        4. IDLE: Low velocity, no significant movement
        
        Returns: 'ACCEL', 'PEAK', 'DECEL', or 'IDLE'
        """
        # Calculate acceleration (change in velocity)
        acceleration = velocity - prev_velocity
        self.last_acceleration[hand] = acceleration
        
        # Thresholds
        IDLE_VELOCITY = 0.03
        ACCEL_THRESHOLD = 0.015
        DECEL_THRESHOLD = -0.015
        
        prev_phase = self.trajectory_phase[hand]
        
        if velocity < IDLE_VELOCITY:
            phase = 'IDLE'
            # Reset peak tracking when idle
            self.peak_kinetic_energy[hand] = 0
        elif acceleration > ACCEL_THRESHOLD:
            phase = 'ACCEL'
        elif acceleration < DECEL_THRESHOLD:
            phase = 'DECEL'
        elif prev_phase == 'ACCEL' and acceleration < ACCEL_THRESHOLD:
            phase = 'PEAK'  # Just passed peak velocity
        else:
            phase = prev_phase  # Maintain current phase
        
        self.trajectory_phase[hand] = phase
        return phase
    
    def _get_arm_geometry(self, pose_landmarks, hand):
        """
        Calculate full arm geometry using shoulder-elbow-wrist skeleton.
        
        Returns:
            arm_length: 2D distance from shoulder to wrist (normalized)
            elbow_angle: Angle at elbow (180 = straight, 90 = bent)
            wrist_z: Z coordinate of wrist
        """
        if hand == 'Left':
            shoulder = pose_landmarks.landmark[self.SHOULDER_LEFT]
            elbow = pose_landmarks.landmark[self.ELBOW_LEFT]
            wrist = pose_landmarks.landmark[self.WRIST_LEFT]
        else:
            shoulder = pose_landmarks.landmark[self.SHOULDER_RIGHT]
            elbow = pose_landmarks.landmark[self.ELBOW_RIGHT]
            wrist = pose_landmarks.landmark[self.WRIST_RIGHT]
        
        # Calculate 2D arm length (shoulder to wrist)
        arm_length = np.sqrt(
            (wrist.x - shoulder.x)**2 + 
            (wrist.y - shoulder.y)**2
        )
        
        # Calculate elbow angle (for additional info)
        # Upper arm vector: shoulder → elbow
        upper_arm = np.array([elbow.x - shoulder.x, elbow.y - shoulder.y])
        # Forearm vector: elbow → wrist
        forearm = np.array([wrist.x - elbow.x, wrist.y - elbow.y])
        
        # Angle between vectors
        dot = np.dot(upper_arm, forearm)
        mag_u = np.linalg.norm(upper_arm)
        mag_f = np.linalg.norm(forearm)
        
        if mag_u > 0.001 and mag_f > 0.001:
            cos_angle = np.clip(dot / (mag_u * mag_f), -1.0, 1.0)
            elbow_angle = np.degrees(np.arccos(cos_angle))
        else:
            elbow_angle = 90.0
        
        return arm_length, elbow_angle, wrist.z
    
    def _calculate_depth_percent(self, arm_length, hand):
        """
        Calculate depth percentage (0-100%) based on arm extension.
        
        0% = arm fully bent (guard)
        100% = arm fully extended (punch)
        
        Auto-calibrates min/max during use.
        """
        # Auto-calibrate: gradually adjust min/max
        if arm_length < self.min_arm_length[hand] * 0.95:
            self.min_arm_length[hand] = 0.9 * self.min_arm_length[hand] + 0.1 * arm_length
        if arm_length > self.max_arm_length[hand] * 1.05:
            self.max_arm_length[hand] = 0.9 * self.max_arm_length[hand] + 0.1 * arm_length
        
        # Calculate percentage
        arm_range = self.max_arm_length[hand] - self.min_arm_length[hand]
        if arm_range < 0.1:
            arm_range = 0.1  # Safety minimum
        
        depth_percent = (arm_length - self.min_arm_length[hand]) / arm_range * 100
        depth_percent = max(0, min(100, depth_percent))
        
        return depth_percent
    
    def _get_arm_length(self, pose_landmarks, hand):
        """Calculate arm length (shoulder to wrist) - increases when punching forward."""
        if hand == 'Left':
            shoulder = pose_landmarks.landmark[self.SHOULDER_LEFT]
            elbow = pose_landmarks.landmark[self.ELBOW_LEFT]
            wrist = pose_landmarks.landmark[self.WRIST_LEFT]
        else:
            shoulder = pose_landmarks.landmark[self.SHOULDER_RIGHT]
            elbow = pose_landmarks.landmark[self.ELBOW_RIGHT]
            wrist = pose_landmarks.landmark[self.WRIST_RIGHT]
        
        # Calculate 2D arm length (shoulder to wrist)
        arm_length = np.sqrt(
            (wrist.x - shoulder.x)**2 + 
            (wrist.y - shoulder.y)**2
        )
        return arm_length, wrist.z
    
    def _detect_direction(self, history, current_pos, current_z, arm_length):
        """
        Detect movement direction based on position and depth changes.
        
        Returns: 'FORWARD', 'SIDEWAYS', or 'IDLE'
        """
        if len(history) < 3:
            return 'IDLE', 0
        
        # Get previous frame data
        prev = history[-2]
        prev_pos = prev['pos']
        prev_z = prev.get('z', 0)
        prev_arm = prev.get('arm_length', 0)
        
        # Calculate changes
        dx = current_pos[0] - prev_pos[0]  # Horizontal movement
        dy = current_pos[1] - prev_pos[1]  # Vertical movement
        dz = current_z - prev_z             # Z change (negative = toward camera)
        d_arm = arm_length - prev_arm       # Arm length change (positive = extending)
        
        # Movement magnitude
        movement_2d = np.sqrt(dx**2 + dy**2)
        
        # Direction classification
        # FORWARD: arm extending OR Z decreasing OR moving toward frame center
        # SIDEWAYS: horizontal movement dominates, arm not extending much
        # IDLE: very little movement
        
        if movement_2d < 0.02:
            direction = 'IDLE'
            depth_change = 0
        elif d_arm > 0.01 or dz < -0.01:
            # Arm extending or Z decreasing = FORWARD punch
            direction = 'FORWARD'
            depth_change = d_arm + abs(dz if dz < 0 else 0)
        elif abs(dx) > abs(dy) * 1.5:
            # Horizontal movement dominates
            direction = 'SIDEWAYS'
            depth_change = 0
        else:
            direction = 'IDLE'
            depth_change = 0
        
        return direction, depth_change
    
    def update(self, pose_landmarks, timestamp=None):
        """
        Process pose landmarks and detect hits based on velocity + direction.
        
        Args:
            pose_landmarks: MediaPipe pose landmarks
            timestamp: Optional timing info
            
        Returns:
            dict with hit events, velocity, direction, depth, elbow angle, and quick wins
        """
        self.frame_count += 1
        base_result = {
            'hit': False, 'velocity': 0, 'direction': 'IDLE', 'depth_change': 0, 
            'depth_percent': 0, 'elbow_angle': 90, 'is_straight': False,
            'direction_consistency': 0, 'kinetic_energy': 0, 'trajectory_phase': 'IDLE'
        }
        results = {
            'Left': base_result.copy(),
            'Right': base_result.copy()
        }
        
        if pose_landmarks is None:
            return results
        
        # Get shoulder width for normalization
        l_sh = pose_landmarks.landmark[self.SHOULDER_LEFT]
        r_sh = pose_landmarks.landmark[self.SHOULDER_RIGHT]
        shoulder_width = np.sqrt((l_sh.x - r_sh.x)**2 + (l_sh.y - r_sh.y)**2)
        
        if shoulder_width < 0.01:
            return results
        
        # Process each hand
        for hand in ['Left', 'Right']:
            wrist_idx = self.WRIST_LEFT if hand == 'Left' else self.WRIST_RIGHT
            wrist = pose_landmarks.landmark[wrist_idx]
            
            # Skip if not visible
            if wrist.visibility < 0.5:
                continue
            
            # Get full arm geometry using skeleton
            arm_length, elbow_angle, wrist_z = self._get_arm_geometry(pose_landmarks, hand)
            
            # Normalize arm length by shoulder width
            normalized_arm = arm_length / shoulder_width
            
            # ════════════════════════════════════════════════════════════
            # ANTHROPOMETRIC DEPTH CALCULATION (Physics-based!)
            # ════════════════════════════════════════════════════════════
            anthro_depth, anthro_depth_percent = self.calculate_anthropometric_depth(pose_landmarks, hand)
            self.last_anthropometric_depth[hand] = anthro_depth_percent
            self.last_depth_percent[hand] = anthro_depth_percent
            
            # Current position (normalized by shoulder width)
            current_pos = np.array([wrist.x, wrist.y]) / shoulder_width
            
            # Add to history with Z and arm length
            self.position_history[hand].append({
                'pos': current_pos,
                'z': wrist_z,
                'arm_length': normalized_arm,
                'time': timestamp or time.time()
            })
            
            # Need at least 2 frames to calculate velocity
            if len(self.position_history[hand]) < 2:
                continue
            
            # Calculate velocity vector and magnitude
            prev_pos = self.position_history[hand][-2]['pos']
            velocity_vector = current_pos - prev_pos
            raw_velocity = np.linalg.norm(velocity_vector)
            
            # ════════════════════════════════════════════════════════════
            # MEDIUM EFFORT: Kalman Filter Integration
            # ════════════════════════════════════════════════════════════
            
            # Update Kalman filter with position measurement
            current_time = timestamp or time.time()
            kalman_state = self.kalman[hand].update([wrist.x, wrist.y], current_time)
            
            # Get Kalman-smoothed velocity and acceleration
            kalman_vel = self.kalman[hand].get_velocity()
            kalman_accel = self.kalman[hand].get_acceleration()
            kalman_speed = self.kalman[hand].get_speed()
            
            # Store for debug
            self.kalman_velocity[hand] = kalman_vel
            self.kalman_accel[hand] = kalman_accel
            
            # Use Kalman-smoothed speed for hit detection (more stable)
            velocity = kalman_speed * 10  # Scale to match raw velocity range
            
            # ════════════════════════════════════════════════════════════
            # QUICK WINS INTEGRATION
            # ════════════════════════════════════════════════════════════
            
            # Store velocity vector for direction consistency
            self.velocity_vectors[hand].append(velocity_vector)
            
            # Quick Win #1: Direction Consistency
            direction_consistency = self.calculate_direction_consistency(hand)
            
            # Quick Win #2: Kinetic Energy (using Kalman velocity for smoothness)
            kinetic_energy = self.calculate_kinetic_energy(velocity, hand)
            
            # Quick Win #3: Trajectory Phase
            prev_velocity = self.last_velocity.get(hand, 0)
            trajectory_phase = self.detect_trajectory_phase(velocity, prev_velocity, hand)
            
            # Detect direction (legacy)
            direction, depth_change = self._detect_direction(
                self.position_history[hand],
                current_pos,
                wrist_z,
                normalized_arm
            )
            
            # Store for debug
            self.last_velocity[hand] = velocity
            self.last_direction[hand] = direction
            self.last_depth_change[hand] = depth_change
            
            # ════════════════════════════════════════════════════════════
            # ELBOW ANGLE GATING
            # ════════════════════════════════════════════════════════════
            ELBOW_THRESHOLD = 130  # Degrees - below this is considered "bent"
            
            if elbow_angle < ELBOW_THRESHOLD:
                effective_depth = 0.0
            else:
                straightness = (elbow_angle - 90) / 90
                straightness = max(0, min(1, straightness))
                effective_depth = anthro_depth_percent * straightness
            
            # Store the gated depth
            self.last_depth_percent[hand] = effective_depth
            
            # Populate results with all metrics
            results[hand]['velocity'] = velocity
            results[hand]['direction'] = direction
            results[hand]['depth_change'] = depth_change
            results[hand]['depth_percent'] = effective_depth
            results[hand]['elbow_angle'] = elbow_angle
            results[hand]['is_straight'] = elbow_angle >= ELBOW_THRESHOLD
            
            # Quick win results
            results[hand]['direction_consistency'] = direction_consistency
            results[hand]['kinetic_energy'] = kinetic_energy
            results[hand]['trajectory_phase'] = trajectory_phase
            
            # ════════════════════════════════════════════════════════════
            # ENHANCED HIT DETECTION (with Quick Wins!)
            # ════════════════════════════════════════════════════════════
            # PUNCH = fast + straight + forward depth + consistent direction
            # + ballistic trajectory (ACCEL or PEAK phase)
            
            in_cooldown = (self.frame_count - self.last_hit_frame[hand]) < self.cooldown_frames
            
            is_fast = velocity > self.velocity_threshold
            is_straight = elbow_angle >= ELBOW_THRESHOLD
            is_forward = effective_depth > 40
            is_consistent = direction_consistency > 0.5  # Movement in one direction
            is_ballistic = trajectory_phase in ['ACCEL', 'PEAK']  # Accelerating or at peak
            
            # Enhanced hit detection: require more conditions for higher confidence
            # Fast + Straight + Forward = basic hit
            # + Consistent + Ballistic = confirmed punch
            
            basic_hit = is_fast and is_straight and is_forward
            high_confidence = basic_hit and (is_consistent or is_ballistic)
            
            if high_confidence and not in_cooldown:
                # HIGH CONFIDENCE PUNCH DETECTED!
                results[hand]['hit'] = True
                self.last_hit_frame[hand] = self.frame_count
                self.hit_count[hand] += 1
        
        return results
    
    def get_debug_string(self, hand='Right'):
        """Get debug string showing velocity."""
        vel = self.last_velocity[hand]
        threshold = self.velocity_threshold
        over = ">>>" if vel > threshold else "   "
        return f"{hand[0]}: V={vel:.3f} {over} (T={threshold:.2f})"
    
    def get_stats(self):
        """Get hit statistics."""
        return {
            'left_hits': self.hit_count['Left'],
            'right_hits': self.hit_count['Right'],
            'total_frames': self.frame_count
        }
    
    def reset(self):
        """Reset all counters."""
        self.hit_count = {'Left': 0, 'Right': 0}
        self.frame_count = 0
        self.last_hit_frame = {'Left': -100, 'Right': -100}
        for hand in ['Left', 'Right']:
            self.position_history[hand].clear()


# Standalone test
if __name__ == "__main__":
    import cv2
    import mediapipe as mp
    
    print("=" * 60)
    print("SKELETON HIT DETECTOR - Velocity Based")
    print("=" * 60)
    print("SKELETON HIT DETECTOR - Anthropometric Depth")
    print("=" * 60)
    print("Move your hands FAST to register hits!")
    print("Press C to start calibration (5 sec countdown)")
    print("Press Q to quit, R to reset")
    print("=" * 60)
    
    # Initialize
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(
        model_complexity=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    mp_draw = mp.solutions.drawing_utils
    
    detector = SkeletonHitDetector(velocity_threshold=0.12)
    
    # Calibration countdown timer
    calibration_countdown = 0  # Seconds remaining (0 = not counting)
    calibration_start_time = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Mirror the frame for natural interaction
        frame = cv2.flip(frame, 1)
        
        # Process with MediaPipe
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb)
        
        # Detect hits
        if results.pose_landmarks:
            # Draw skeleton
            mp_draw.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            
            # Check for hits
            hit_results = detector.update(results.pose_landmarks)
            
            # Draw velocity bars and hit indicators
            h, w = frame.shape[:2]
            
            for i, hand in enumerate(['Left', 'Right']):
                vel = hit_results[hand]['velocity']
                hit = hit_results[hand]['hit']
                direction = hit_results[hand]['direction']
                depth_pct = hit_results[hand]['depth_percent']
                elbow = hit_results[hand]['elbow_angle']
                
                # Bar positions
                bar_x = 50 if hand == 'Left' else w - 250
                bar_w = 50
                
                # Get elbow angle for display
                elbow = hit_results[hand]['elbow_angle']
                is_straight = hit_results[hand].get('is_straight', False)
                
                # Velocity bar (left side of pair)
                vel_h = int(min(vel / 0.3, 1.0) * 200)
                
                # Velocity bar color: GREEN if fast, RED if slow
                vel_color = (0, 255, 0) if vel > detector.velocity_threshold else (0, 0, 255)
                cv2.rectangle(frame, (bar_x, 400 - vel_h), (bar_x + bar_w, 400), vel_color, -1)
                cv2.putText(frame, "VEL", (bar_x, 420), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                # Depth bar (right side of pair)
                depth_x = bar_x + bar_w + 10
                depth_h = int(min(depth_pct / 100, 1.0) * 200)
                
                # Depth bar color: CYAN if straight arm, GRAY if bent
                if is_straight:
                    depth_color = (255, 200, 0)  # Cyan - arm straight, depth valid
                else:
                    depth_color = (100, 100, 100)  # Gray - arm bent, depth gated to 0
                    
                cv2.rectangle(frame, (depth_x, 400 - depth_h), (depth_x + bar_w, 400), depth_color, -1)
                cv2.putText(frame, "DEPTH", (depth_x - 10, 420), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                # Threshold line on velocity bar
                threshold_y = 400 - int((detector.velocity_threshold / 0.3) * 200)
                cv2.line(frame, (bar_x, threshold_y), (bar_x + bar_w, threshold_y), (255, 255, 255), 2)
                
                # Elbow angle indicator
                elbow_color = (0, 255, 0) if is_straight else (0, 0, 255)
                elbow_label = "STRAIGHT" if is_straight else "BENT"
                
                # Get quick win values
                dir_cons = hit_results[hand].get('direction_consistency', 0)
                traj_phase = hit_results[hand].get('trajectory_phase', 'IDLE')
                ke = hit_results[hand].get('kinetic_energy', 0)
                
                # Phase color
                phase_colors = {'ACCEL': (0, 255, 0), 'PEAK': (0, 255, 255), 'DECEL': (255, 165, 0), 'IDLE': (100, 100, 100)}
                phase_color = phase_colors.get(traj_phase, (100, 100, 100))
                
                # Labels
                cv2.putText(frame, f"{hand}", (bar_x, 440), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(frame, f"D={depth_pct:.0f}%", (bar_x, 460), cv2.FONT_HERSHEY_SIMPLEX, 0.5, depth_color, 2)
                cv2.putText(frame, f"E={elbow:.0f}°", (bar_x, 478), cv2.FONT_HERSHEY_SIMPLEX, 0.4, elbow_color, 1)
                cv2.putText(frame, f"{traj_phase}", (bar_x + 60, 478), cv2.FONT_HERSHEY_SIMPLEX, 0.4, phase_color, 1)
                
                # HIT indicator
                if hit:
                    hit_x = 200 if hand == 'Left' else w - 300
                    cv2.putText(frame, ">>> PUNCH! <<<", (hit_x, 100), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 3)
                    print(f">>> {hand} PUNCH! D={depth_pct:.0f}% {traj_phase} KE={ke:.2f} (#{detector.hit_count[hand]})")
            
            # Stats overlay
            stats = detector.get_stats()
            cv2.putText(frame, f"L: {stats['left_hits']}  R: {stats['right_hits']}", 
                       (w//2 - 50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            # Calibration status
            if detector.calibrated:
                cv2.putText(frame, "CALIBRATED", (w//2 - 60, 80), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            else:
                cv2.putText(frame, "Press C to calibrate (T-pose)", (w//2 - 140, 80), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        # ════════════════════════════════════════════════════════════
        # CALIBRATION COUNTDOWN DISPLAY
        # ════════════════════════════════════════════════════════════
        if calibration_countdown > 0:
            elapsed = time.time() - calibration_start_time
            remaining = 5 - elapsed
            
            if remaining > 0:
                # Show big countdown on screen
                cv2.putText(frame, f"CALIBRATING IN: {int(remaining)+1}", (w//2 - 180, h//2), 
                           cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 255), 4)
                cv2.putText(frame, "GET INTO T-POSE!", (w//2 - 150, h//2 + 60), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            else:
                # Countdown finished - CALIBRATE NOW!
                if results.pose_landmarks:
                    detector.calibrate_arm_length(results.pose_landmarks)
                calibration_countdown = 0
        
        # Show frame
        cv2.imshow("Skeleton Hit Detector", frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            detector.reset()
            print("[RESET] Counters cleared")
        elif key == ord('c'):
            # Start 5-second calibration countdown
            if calibration_countdown == 0:
                calibration_countdown = 5
                calibration_start_time = time.time()
                print("[CALIBRATE] Starting 5 second countdown... GET INTO T-POSE!")
    
    cap.release()
    cv2.destroyAllWindows()
    
    print("\n" + "=" * 60)
    print("FINAL STATS")
    print("=" * 60)
    stats = detector.get_stats()
    print(f"Left hits: {stats['left_hits']}")
    print(f"Right hits: {stats['right_hits']}")
    print(f"Total frames: {stats['total_frames']}")
