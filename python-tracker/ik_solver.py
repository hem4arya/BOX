"""
IK Solver - Finger Angle Calculation from 3D Landmarks
Calculates joint angles and detects hand poses (fist, open palm, etc.)
"""

import numpy as np


class SimpleIKSolver:
    """
    Calculate finger joint angles from MediaPipe world landmarks.
    Uses 3D coordinates for accurate angle calculation.
    """
    
    # Finger chains: [MCP, PIP, DIP, TIP]
    FINGER_CHAINS = {
        'thumb': [1, 2, 3, 4],
        'index': [5, 6, 7, 8],
        'middle': [9, 10, 11, 12],
        'ring': [13, 14, 15, 16],
        'pinky': [17, 18, 19, 20],
    }
    
    # Finger indices for tip detection
    FINGERTIP_INDICES = {
        'thumb': 4,
        'index': 8,
        'middle': 12,
        'ring': 16,
        'pinky': 20,
    }
    
    def __init__(self):
        self.last_angles = None
        self.smoothing_factor = 0.3  # For temporal smoothing
    
    def calculate_angles(self, world_landmarks):
        """
        Calculate bend angle for each finger.
        
        Args:
            world_landmarks: MediaPipe hand_world_landmarks
        
        Returns:
            dict with finger names as keys, containing angles
        """
        if world_landmarks is None:
            return None
        
        angles = {}
        
        for finger_name, indices in self.FINGER_CHAINS.items():
            try:
                # Get 3D positions
                joints = [world_landmarks.landmark[i] for i in indices]
                
                # Calculate vectors between joints
                v1 = self._vector(joints[0], joints[1])
                v2 = self._vector(joints[1], joints[2])
                v3 = self._vector(joints[2], joints[3])
                
                # Calculate angles between vectors
                angle1 = self._angle_between(v1, v2)  # MCP bend
                angle2 = self._angle_between(v2, v3)  # PIP bend
                
                # Total curl (normalized to 0-100%)
                total_curl = angle1 + angle2
                curl_percent = min(100, (total_curl / 180.0) * 100)
                
                angles[finger_name] = {
                    'mcp_angle': angle1,
                    'pip_angle': angle2,
                    'total_curl': total_curl,
                    'curl_percent': curl_percent
                }
            except Exception as e:
                angles[finger_name] = {
                    'mcp_angle': 0,
                    'pip_angle': 0,
                    'total_curl': 0,
                    'curl_percent': 0
                }
        
        # Apply temporal smoothing
        if self.last_angles is not None:
            for finger in angles:
                for key in ['curl_percent']:
                    angles[finger][key] = (
                        self.smoothing_factor * angles[finger][key] +
                        (1 - self.smoothing_factor) * self.last_angles[finger][key]
                    )
        
        self.last_angles = angles
        return angles
    
    def _vector(self, p1, p2):
        """Create vector from p1 to p2."""
        return np.array([p2.x - p1.x, p2.y - p1.y, p2.z - p1.z])
    
    def _angle_between(self, v1, v2):
        """Calculate angle between two vectors in degrees."""
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)
        
        if norm1 < 1e-6 or norm2 < 1e-6:
            return 0.0
        
        cos_angle = np.dot(v1, v2) / (norm1 * norm2)
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        return np.degrees(np.arccos(cos_angle))
    
    def is_fist(self, angles, threshold=70):
        """
        Check if hand is making a fist.
        All main fingers should be curled.
        """
        if angles is None:
            return False
        
        curls = [angles[f]['curl_percent'] for f in ['index', 'middle', 'ring', 'pinky']]
        return all(c > threshold for c in curls)
    
    def is_open_palm(self, angles, threshold=30):
        """
        Check if hand is open palm.
        All main fingers should be extended.
        """
        if angles is None:
            return False
        
        curls = [angles[f]['curl_percent'] for f in ['index', 'middle', 'ring', 'pinky']]
        return all(c < threshold for c in curls)
    
    def is_pointing(self, angles, threshold_extended=30, threshold_curled=60):
        """
        Check if pointing gesture (index extended, others curled).
        """
        if angles is None:
            return False
        
        index_extended = angles['index']['curl_percent'] < threshold_extended
        others_curled = all(
            angles[f]['curl_percent'] > threshold_curled 
            for f in ['middle', 'ring', 'pinky']
        )
        return index_extended and others_curled
    
    def is_peace_sign(self, angles, threshold_extended=35, threshold_curled=60):
        """
        Check if peace sign (index + middle extended, others curled).
        """
        if angles is None:
            return False
        
        extended = (
            angles['index']['curl_percent'] < threshold_extended and
            angles['middle']['curl_percent'] < threshold_extended
        )
        curled = all(
            angles[f]['curl_percent'] > threshold_curled 
            for f in ['ring', 'pinky']
        )
        return extended and curled
    
    def is_thumbs_up(self, angles, world_landmarks):
        """
        Check if thumbs up (thumb extended upward, others curled).
        """
        if angles is None or world_landmarks is None:
            return False
        
        # Check other fingers are curled
        others_curled = all(
            angles[f]['curl_percent'] > 60 
            for f in ['index', 'middle', 'ring', 'pinky']
        )
        
        # Check thumb is pointing upward (negative y in world coords)
        thumb_tip = world_landmarks.landmark[4]
        thumb_base = world_landmarks.landmark[2]
        thumb_up = (thumb_base.y - thumb_tip.y) > 0.02  # tip above base
        
        return others_curled and thumb_up
    
    def detect_gesture(self, angles, world_landmarks=None):
        """
        Detect the current hand gesture.
        
        Returns:
            str: Gesture name ('FIST', 'OPEN', 'POINTING', 'PEACE', 'THUMBS_UP', 'UNKNOWN')
        """
        if angles is None:
            return 'UNKNOWN'
        
        if self.is_fist(angles):
            return 'FIST'
        elif self.is_open_palm(angles):
            return 'OPEN'
        elif self.is_pointing(angles):
            return 'POINTING'
        elif self.is_peace_sign(angles):
            return 'PEACE'
        elif world_landmarks and self.is_thumbs_up(angles, world_landmarks):
            return 'THUMBS_UP'
        else:
            return 'UNKNOWN'
    
    def get_finger_info(self, angles):
        """
        Get a summary of finger curl percentages.
        
        Returns:
            dict with finger curl percentages
        """
        if angles is None:
            return None
        
        return {
            'thumb': int(angles['thumb']['curl_percent']),
            'index': int(angles['index']['curl_percent']),
            'middle': int(angles['middle']['curl_percent']),
            'ring': int(angles['ring']['curl_percent']),
            'pinky': int(angles['pinky']['curl_percent']),
        }
    
    def get_3d_positions(self, world_landmarks):
        """
        Extract 3D positions of key landmarks.
        
        Returns:
            dict with landmark positions in meters
        """
        if world_landmarks is None:
            return None
        
        wrist = world_landmarks.landmark[0]
        
        return {
            'wrist': {'x': wrist.x, 'y': wrist.y, 'z': wrist.z},
            'thumb_tip': self._landmark_to_dict(world_landmarks.landmark[4]),
            'index_tip': self._landmark_to_dict(world_landmarks.landmark[8]),
            'middle_tip': self._landmark_to_dict(world_landmarks.landmark[12]),
            'ring_tip': self._landmark_to_dict(world_landmarks.landmark[16]),
            'pinky_tip': self._landmark_to_dict(world_landmarks.landmark[20]),
        }
    
    def _landmark_to_dict(self, landmark):
        """Convert landmark to dict."""
        return {'x': landmark.x, 'y': landmark.y, 'z': landmark.z}


def test_ik_solver():
    """Test IK solver with mock data."""
    print("=" * 60)
    print("IK SOLVER TEST")
    print("=" * 60)
    
    solver = SimpleIKSolver()
    
    # Mock test (would need real landmarks to test properly)
    print("\nIK Solver initialized successfully!")
    print("Finger chains:", list(solver.FINGER_CHAINS.keys()))
    print("\nTo test with real data, run hand_tracker_invincible.py")
    print("=" * 60)


if __name__ == "__main__":
    test_ik_solver()
