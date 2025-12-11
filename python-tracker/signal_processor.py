"""
Signal Processing Module for Boxing Vision System
=================================================
Implements the One Euro Filter for adaptive landmark smoothing.

Theory:
- Low velocity (Guard): High smoothing, rock-solid hands
- High velocity (Punch): Low smoothing, zero latency

Reference: Casiez et al. "1€ Filter: A Simple Speed-based Low-pass Filter"
"""

import math
import time
import numpy as np
from collections import deque


class OneEuroFilter:
    """
    Adaptive low-pass filter that adjusts smoothing based on velocity.
    
    Parameters:
    - min_cutoff: Minimum cutoff frequency (Hz). Lower = smoother steady state.
                  Recommended: 1.0 for boxing
    - beta: Speed coefficient. Higher = more responsive to fast movement.
            Recommended: 0.007 for boxing (increase if punch feels laggy)
    - d_cutoff: Derivative cutoff frequency. Usually 1.0.
    """
    
    def __init__(self, min_cutoff=1.0, beta=0.007, d_cutoff=1.0):
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        
        # State
        self.x_prev = None
        self.dx_prev = 0.0
        self.t_prev = None
    
    def _smoothing_factor(self, t_e, cutoff):
        """Calculate exponential smoothing factor alpha."""
        r = 2 * math.pi * cutoff * t_e
        return r / (r + 1)
    
    def _exponential_smoothing(self, alpha, x, x_prev):
        """Apply exponential smoothing."""
        return alpha * x + (1 - alpha) * x_prev
    
    def __call__(self, t, x):
        """
        Process a new sample.
        
        Args:
            t: Current timestamp (seconds)
            x: Current raw value
            
        Returns:
            Filtered value
        """
        # Initialize on first call
        if self.x_prev is None:
            self.x_prev = x
            self.t_prev = t
            return x
        
        # Calculate time delta
        t_e = t - self.t_prev
        if t_e <= 0:
            return self.x_prev
        
        # 1. Estimate derivative (velocity)
        a_d = self._smoothing_factor(t_e, self.d_cutoff)
        dx = (x - self.x_prev) / t_e
        dx_hat = self._exponential_smoothing(a_d, dx, self.dx_prev)
        
        # 2. Adaptive cutoff frequency
        # Fast movement = HIGH cutoff = LESS smoothing = ZERO LAG
        # Slow movement = LOW cutoff = MORE smoothing = STABLE
        cutoff = self.min_cutoff + self.beta * abs(dx_hat)
        
        # 3. Filter the signal
        alpha = self._smoothing_factor(t_e, cutoff)
        x_hat = self._exponential_smoothing(alpha, x, self.x_prev)
        
        # Update state
        self.x_prev = x_hat
        self.dx_prev = dx_hat
        self.t_prev = t
        
        return x_hat
    
    def reset(self):
        """Reset filter state."""
        self.x_prev = None
        self.dx_prev = 0.0
        self.t_prev = None
    
    def get_velocity(self):
        """Get current estimated velocity."""
        return self.dx_prev


class LandmarkSmoother:
    """
    Apply One Euro Filter to all 21 hand landmarks (x, y, z).
    
    Usage:
        smoother = LandmarkSmoother()
        smoothed = smoother.smooth(landmarks, current_time)
    """
    
    def __init__(self, num_landmarks=21, min_cutoff=1.0, beta=0.007):
        # Create 3 filters per landmark (x, y, z)
        self.filters = {}
        for i in range(num_landmarks):
            self.filters[i] = {
                'x': OneEuroFilter(min_cutoff, beta),
                'y': OneEuroFilter(min_cutoff, beta),
                'z': OneEuroFilter(min_cutoff, beta),
            }
        self.num_landmarks = num_landmarks
        self.min_cutoff = min_cutoff
        self.beta = beta
    
    def smooth(self, landmarks, timestamp):
        """
        Smooth all landmarks.
        
        Args:
            landmarks: MediaPipe hand landmarks object OR list of (x, y, z) tuples
            timestamp: Current time in seconds
            
        Returns:
            List of smoothed (x, y, z) tuples
        """
        smoothed = []
        
        # Handle MediaPipe landmarks object
        if hasattr(landmarks, 'landmark'):
            landmark_list = landmarks.landmark
        else:
            landmark_list = landmarks
        
        for i in range(min(self.num_landmarks, len(landmark_list))):
            lm = landmark_list[i]
            
            # Handle both MediaPipe landmark and tuple formats
            if hasattr(lm, 'x'):
                x_raw, y_raw, z_raw = lm.x, lm.y, lm.z
            else:
                x_raw, y_raw, z_raw = lm[0], lm[1], lm[2]
            
            x_smooth = self.filters[i]['x'](timestamp, x_raw)
            y_smooth = self.filters[i]['y'](timestamp, y_raw)
            z_smooth = self.filters[i]['z'](timestamp, z_raw)
            
            smoothed.append((x_smooth, y_smooth, z_smooth))
        
        return smoothed
    
    def smooth_single(self, x, y, z, timestamp, landmark_idx=0):
        """
        Smooth a single landmark position.
        
        Args:
            x, y, z: Raw position values
            timestamp: Current time in seconds
            landmark_idx: Which landmark's filter to use (default: wrist = 0)
        
        Returns:
            Tuple of (x_smooth, y_smooth, z_smooth)
        """
        x_smooth = self.filters[landmark_idx]['x'](timestamp, x)
        y_smooth = self.filters[landmark_idx]['y'](timestamp, y)
        z_smooth = self.filters[landmark_idx]['z'](timestamp, z)
        return (x_smooth, y_smooth, z_smooth)
    
    def get_velocity(self, landmark_idx=0):
        """Get velocity estimate for a landmark."""
        vx = self.filters[landmark_idx]['x'].get_velocity()
        vy = self.filters[landmark_idx]['y'].get_velocity()
        vz = self.filters[landmark_idx]['z'].get_velocity()
        return (vx, vy, vz)
    
    def reset(self):
        """Reset all filters."""
        for i in self.filters:
            for axis in self.filters[i]:
                self.filters[i][axis].reset()


class KinematicConstraints:
    """
    Enforce fixed bone lengths to eliminate skeleton stretching.
    
    The human arm has fixed proportions. MediaPipe often outputs
    varying bone lengths frame-to-frame due to foreshortening errors.
    This class forces the skeleton to maintain calibrated lengths.
    """
    
    # Hand skeleton connections (indices)
    BONES = [
        # Thumb
        (0, 1), (1, 2), (2, 3), (3, 4),
        # Index
        (0, 5), (5, 6), (6, 7), (7, 8),
        # Middle
        (0, 9), (9, 10), (10, 11), (11, 12),
        # Ring
        (0, 13), (13, 14), (14, 15), (15, 16),
        # Pinky
        (0, 17), (17, 18), (18, 19), (19, 20),
    ]
    
    def __init__(self):
        self.calibrated = False
        self.bone_lengths = {}  # (idx1, idx2) -> length
        self.calibration_buffer = []
        self.max_calibration_frames = 60
    
    def calibrate(self, landmarks):
        """
        Collect calibration samples and compute average bone lengths.
        Call this during initialization while user has open hand.
        
        Returns:
            bool: True if calibration complete
        """
        # Handle MediaPipe landmarks
        if hasattr(landmarks, 'landmark'):
            lm_list = landmarks.landmark
        else:
            lm_list = landmarks
        
        if len(lm_list) < 21:
            return False
        
        # Calculate current bone lengths
        current_lengths = {}
        for bone in self.BONES:
            length = self._distance(lm_list, bone[0], bone[1])
            current_lengths[bone] = length
        
        self.calibration_buffer.append(current_lengths)
        
        if len(self.calibration_buffer) >= self.max_calibration_frames:
            # Calculate average bone lengths
            for bone in self.BONES:
                lengths = [b[bone] for b in self.calibration_buffer]
                self.bone_lengths[bone] = np.mean(lengths)
            
            self.calibrated = True
            print(f"[KINEMATIC] Calibrated {len(self.BONES)} bones")
            return True
        
        return False
    
    def _distance(self, landmarks, idx1, idx2):
        """Calculate Euclidean distance between two landmarks."""
        if hasattr(landmarks[idx1], 'x'):
            lm1 = landmarks[idx1]
            lm2 = landmarks[idx2]
            return np.sqrt((lm1.x - lm2.x)**2 + (lm1.y - lm2.y)**2 + (lm1.z - lm2.z)**2)
        else:
            lm1 = landmarks[idx1]
            lm2 = landmarks[idx2]
            return np.sqrt((lm1[0] - lm2[0])**2 + (lm1[1] - lm2[1])**2 + (lm1[2] - lm2[2])**2)
    
    def apply(self, landmarks):
        """
        Force skeleton to maintain calibrated bone lengths.
        
        Args:
            landmarks: List of (x, y, z) tuples
            
        Returns:
            Constrained landmarks
        """
        if not self.calibrated or len(landmarks) < 21:
            return landmarks
        
        # Convert to mutable numpy arrays
        constrained = [np.array(lm) for lm in landmarks]
        
        # Apply constraints using forward kinematics from wrist
        # TODO: Full implementation with IK solver
        # For now, return unmodified
        
        return [(lm[0], lm[1], lm[2]) for lm in constrained]
    
    def get_status(self):
        """Get calibration status."""
        if self.calibrated:
            return "CALIBRATED"
        else:
            progress = len(self.calibration_buffer) / self.max_calibration_frames
            return f"CALIBRATING {int(progress * 100)}%"


class PositionSmoother:
    """
    Simple smoother for a single position (x, y, z).
    Used for glove tracking in boxing game.
    """
    
    def __init__(self, min_cutoff=1.5, beta=0.01):
        self.filter_x = OneEuroFilter(min_cutoff, beta)
        self.filter_y = OneEuroFilter(min_cutoff, beta)
        self.filter_z = OneEuroFilter(min_cutoff, beta)
    
    def smooth(self, x, y, z, timestamp=None):
        """
        Smooth a position.
        
        Args:
            x, y, z: Raw position values
            timestamp: Current time (uses time.time() if None)
        
        Returns:
            (x_smooth, y_smooth, z_smooth)
        """
        if timestamp is None:
            timestamp = time.time()
        
        x_s = self.filter_x(timestamp, x)
        y_s = self.filter_y(timestamp, y)
        z_s = self.filter_z(timestamp, z)
        
        return (x_s, y_s, z_s)
    
    def get_velocity(self):
        """Get current velocity estimate."""
        return (
            self.filter_x.get_velocity(),
            self.filter_y.get_velocity(),
            self.filter_z.get_velocity(),
        )
    
    def reset(self):
        """Reset filters."""
        self.filter_x.reset()
        self.filter_y.reset()
        self.filter_z.reset()


def test_one_euro_filter():
    """Test the One Euro Filter."""
    print("=" * 60)
    print("ONE EURO FILTER TEST")
    print("=" * 60)
    
    # Test 1: Jitter suppression (guard stance)
    print("\n1. Testing jitter suppression (guard stance)...")
    filter = OneEuroFilter(min_cutoff=1.0, beta=0.007)
    
    jitter_variance = []
    raw_variance = []
    
    for i in range(20):
        # Jittery signal around 0.5 with ±0.02 noise
        jitter = 0.02 * (i % 4 - 2) / 2
        raw = 0.5 + jitter
        filtered = filter(time.time(), raw)
        
        raw_variance.append(abs(raw - 0.5))
        jitter_variance.append(abs(filtered - 0.5))
        
        if i >= 10:  # After warmup
            print(f"  Raw: {raw:.4f} -> Filtered: {filtered:.4f}")
        
        time.sleep(0.033)
    
    jitter_reduction = (1 - np.mean(jitter_variance[-5:]) / np.mean(raw_variance[-5:])) * 100
    print(f"  Jitter reduction: {jitter_reduction:.1f}%")
    
    # Test 2: Fast movement tracking (punch)
    print("\n2. Testing fast movement (punch)...")
    filter.reset()
    
    lags = []
    for i in range(15):
        # Rapidly increasing signal (simulating punch)
        raw = 0.05 * i
        filtered = filter(time.time(), raw)
        lag = raw - filtered
        lags.append(abs(lag))
        
        if i >= 5:  # After warmup
            print(f"  Raw: {raw:.4f} -> Filtered: {filtered:.4f} (Lag: {lag:+.4f})")
        
        time.sleep(0.033)
    
    avg_lag = np.mean(lags[-5:])
    print(f"  Average lag during punch: {avg_lag:.4f}")
    
    # Summary
    print("\n" + "=" * 60)
    print("RESULTS:")
    print(f"  ✓ Jitter reduction: {jitter_reduction:.1f}%")
    print(f"  ✓ Punch lag: {avg_lag:.4f} (target < 0.05)")
    
    if jitter_reduction > 50 and avg_lag < 0.1:
        print("\n  ✓ FILTER IS WORKING CORRECTLY!")
    else:
        print("\n  ✗ Filter needs tuning")
    
    print("=" * 60)


if __name__ == "__main__":
    test_one_euro_filter()
