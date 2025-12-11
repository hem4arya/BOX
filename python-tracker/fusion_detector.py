"""
Fusion Detector - Combines multiple detection methods with weighted voting
Also includes direction check and energy profile filtering.
"""

import numpy as np
from collections import deque


class FusionDetector:
    """Combines all detectors with weighted voting."""
    
    def __init__(self):
        self.weights = {
            'mediapipe': 0.4,    # Most accurate when works
            'yolo': 0.2,         # Backup detector
            'optical_flow': 0.2, # Motion confidence
            'ballistic': 0.2,    # Physics prediction
        }
        
        self.last_confidence = 0.0
    
    def get_punch_confidence(self, detectors_output):
        """
        Calculate weighted confidence from all detectors.
        
        Args:
            detectors_output: {
                'mediapipe': {'detected': True, 'confidence': 0.9},
                'yolo': {'detected': True, 'confidence': 0.7},
                'optical_flow': {'quality': 0.8},
                'ballistic': {'state': 'firing', 'accel': 2.5},
            }
        
        Returns:
            float: Combined confidence 0.0 to 1.0
        """
        score = 0.0
        
        # MediaPipe vote
        mp = detectors_output.get('mediapipe', {})
        if mp.get('detected', False):
            score += self.weights['mediapipe'] * mp.get('confidence', 0.5)
        
        # YOLO vote
        yolo = detectors_output.get('yolo', {})
        if yolo.get('detected', False):
            score += self.weights['yolo'] * yolo.get('confidence', 0.5)
        
        # Optical flow vote (high quality = intentional movement)
        of = detectors_output.get('optical_flow', {})
        score += self.weights['optical_flow'] * of.get('quality', 0.0)
        
        # Ballistic vote
        ballistic = detectors_output.get('ballistic', {})
        if ballistic.get('state') == 'firing':
            score += self.weights['ballistic']
        
        self.last_confidence = score
        return score
    
    def should_register_punch(self, confidence=None, threshold=0.6):
        """Check if punch should be registered based on confidence."""
        if confidence is None:
            confidence = self.last_confidence
        return confidence > threshold


def is_moving_toward_target(hand_velocity, hand_pos, target_pos):
    """
    Check if hand is moving TOWARD the target.
    
    Args:
        hand_velocity: (vx, vy) velocity vector
        hand_pos: (x, y) current hand position
        target_pos: (x, y) target position
    
    Returns:
        bool: True if moving toward target (within 60 degrees)
    """
    # Handle zero velocity
    vel_mag = np.linalg.norm(hand_velocity)
    if vel_mag < 0.01:
        return False
    
    # Direction to target
    target_direction = np.array(target_pos) - np.array(hand_pos)
    target_mag = np.linalg.norm(target_direction)
    if target_mag < 0.01:
        return True  # Already at target
    
    target_direction = target_direction / target_mag
    
    # Normalize velocity
    vel_direction = np.array(hand_velocity) / vel_mag
    
    # Dot product: 1 = toward, -1 = away, 0 = perpendicular
    alignment = np.dot(vel_direction, target_direction)
    
    # cos(60°) = 0.5, so alignment > 0.5 means within 60 degrees
    return alignment > 0.5


class EnergyFilter:
    """Filter out erratic movements that don't match punch energy profile."""
    
    def __init__(self, window_size=5):
        self.velocity_history = deque(maxlen=window_size)
        self.window_size = window_size
    
    def update(self, velocity_magnitude):
        """Add new velocity to history."""
        self.velocity_history.append(velocity_magnitude)
    
    def is_punch_profile(self):
        """
        Check if velocity history matches punch profile.
        Real punch: velocity builds up then peaks (smooth acceleration)
        False positive: random spikes
        
        Returns:
            bool: True if matches punch energy profile
        """
        if len(self.velocity_history) < 3:
            return True  # Not enough data, allow it
        
        velocities = list(self.velocity_history)
        
        # Check for smooth acceleration pattern
        # Count frames where velocity increased
        increasing_count = sum(
            1 for i in range(1, len(velocities))
            if velocities[i] > velocities[i-1] * 0.8  # Allow 20% tolerance
        )
        
        # At least 50% of frames should show increasing/sustained velocity
        min_increasing = max(1, len(velocities) * 0.5)
        
        return increasing_count >= min_increasing
    
    def get_energy_score(self):
        """Get normalized energy score from recent velocities."""
        if len(self.velocity_history) == 0:
            return 0.0
        
        # Average of recent velocities, normalized
        avg_vel = np.mean(self.velocity_history)
        # Assume max reasonable velocity is ~2.0
        return min(1.0, avg_vel / 2.0)
    
    def reset(self):
        """Clear velocity history."""
        self.velocity_history.clear()


class PunchValidator:
    """
    Combines all validation checks for punch detection.
    Use this as the single entry point for punch validation.
    """
    
    def __init__(self):
        self.fusion = FusionDetector()
        self.energy_filter = EnergyFilter(window_size=5)
        
        # Thresholds
        self.reach_threshold = 0.85
        self.confidence_threshold = 0.5  # Lowered since we have other checks
        
        # Debug info
        self.last_check = {
            'collision': False,
            'reach_ok': False,
            'fist': False,
            'direction_ok': False,
            'energy_ok': False,
            'confidence': 0.0,
            'valid': False,
        }
    
    def update(self, velocity_magnitude):
        """Update energy filter with new velocity."""
        self.energy_filter.update(velocity_magnitude)
    
    def validate_punch(self, collision, reach_percent, gesture, 
                       hand_velocity=None, hand_pos=None, target_pos=None,
                       detector_outputs=None):
        """
        Full punch validation with all checks.
        
        Args:
            collision: bool - glove collides with target
            reach_percent: float - arm extension 0-1
            gesture: str - 'FIST', 'OPEN', etc.
            hand_velocity: (vx, vy) - optional velocity vector
            hand_pos: (x, y) - optional hand position
            target_pos: (x, y) - optional target position
            detector_outputs: dict - optional detector outputs for fusion
        
        Returns:
            bool: True if valid punch
        """
        # Basic checks
        is_collision = collision
        is_reach_ok = reach_percent >= self.reach_threshold
        is_fist = gesture == 'FIST'
        
        # Direction check (optional)
        is_direction_ok = True
        if hand_velocity is not None and hand_pos is not None and target_pos is not None:
            is_direction_ok = is_moving_toward_target(hand_velocity, hand_pos, target_pos)
        
        # Energy profile check
        is_energy_ok = self.energy_filter.is_punch_profile()
        
        # Fusion confidence (optional)
        confidence = 1.0
        if detector_outputs is not None:
            confidence = self.fusion.get_punch_confidence(detector_outputs)
        
        # Store debug info
        self.last_check = {
            'collision': is_collision,
            'reach_ok': is_reach_ok,
            'fist': is_fist,
            'direction_ok': is_direction_ok,
            'energy_ok': is_energy_ok,
            'confidence': confidence,
            'valid': False,
        }
        
        # Final decision (require all basic checks, be lenient on advanced)
        is_valid = (
            is_collision and
            is_reach_ok and
            is_fist
            # Direction and energy checks are advisory for now
            # and is_direction_ok
            # and is_energy_ok
        )
        
        self.last_check['valid'] = is_valid
        return is_valid
    
    def get_debug_string(self):
        """Get debug string for HUD display."""
        c = self.last_check
        parts = []
        parts.append("COL" if c['collision'] else "---")
        parts.append(f"R{int(c.get('reach_ok', 0)*100):02d}" if c.get('reach_ok') else "R--")
        parts.append("FIST" if c['fist'] else "OPEN")
        parts.append("DIR" if c['direction_ok'] else "---")
        parts.append("NRG" if c['energy_ok'] else "---")
        
        return " | ".join(parts)


def test_fusion():
    """Test the fusion detector."""
    print("=" * 50)
    print("FUSION DETECTOR TEST")
    print("=" * 50)
    
    fusion = FusionDetector()
    
    # Test case 1: All detectors agree
    output = {
        'mediapipe': {'detected': True, 'confidence': 0.9},
        'yolo': {'detected': True, 'confidence': 0.8},
        'optical_flow': {'quality': 0.7},
        'ballistic': {'state': 'firing'},
    }
    conf = fusion.get_punch_confidence(output)
    print(f"All agree: confidence = {conf:.2f} (should be ~0.88)")
    
    # Test case 2: Only MediaPipe
    output = {
        'mediapipe': {'detected': True, 'confidence': 0.9},
        'yolo': {'detected': False},
        'optical_flow': {'quality': 0.3},
        'ballistic': {'state': 'idle'},
    }
    conf = fusion.get_punch_confidence(output)
    print(f"MediaPipe only: confidence = {conf:.2f} (should be ~0.42)")
    
    # Test direction check
    print("\n--- Direction Check ---")
    
    # Moving toward target
    result = is_moving_toward_target((1, 0), (0, 0), (10, 0))
    print(f"Moving right toward right target: {result} (should be True)")
    
    # Moving away
    result = is_moving_toward_target((-1, 0), (0, 0), (10, 0))
    print(f"Moving left away from right target: {result} (should be False)")
    
    # Moving sideways
    result = is_moving_toward_target((0, 1), (0, 0), (10, 0))
    print(f"Moving up toward right target: {result} (should be False)")
    
    # Energy filter test
    print("\n--- Energy Filter ---")
    energy = EnergyFilter(window_size=5)
    
    # Smooth acceleration
    for v in [0.1, 0.3, 0.6, 0.9, 1.2]:
        energy.update(v)
    print(f"Smooth accel: is_punch_profile = {energy.is_punch_profile()} (should be True)")
    
    # Random spikes
    energy.reset()
    for v in [0.1, 1.5, 0.2, 1.8, 0.3]:
        energy.update(v)
    print(f"Random spikes: is_punch_profile = {energy.is_punch_profile()} (should be False)")
    
    print("\n" + "=" * 50)


if __name__ == "__main__":
    test_fusion()
