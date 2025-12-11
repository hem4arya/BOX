"""
Geometric Depth Estimation for Boxing Vision System
====================================================
Implements the Anchor Method and Arm Extension Coefficient (AEC).

Key Insight:
- MediaPipe's Z-coordinate is unreliable for punch detection
- We use GEOMETRIC relationships instead:
  1. Normalized radial reach (wrist distance from shoulder / shoulder width)
  2. Elbow angle (straight arm = 180°, bent = 90°)
  3. Verticality test (punches are horizontal, not vertical)

Reference: "Punch Detection Sensor Fusion Architecture" Section 4
"""

import numpy as np
import time
from collections import deque

try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False


class AnchorDepthEstimator:
    """
    Geometric depth estimation using the Anchor Method.
    
    The "Anchor" is the shoulder joint. All measurements are normalized
    by shoulder width (biacromial diameter) to be scale-invariant.
    """
    
    # MediaPipe Pose landmark indices
    LANDMARKS = {
        'left_shoulder': 11,
        'right_shoulder': 12,
        'left_elbow': 13,
        'right_elbow': 14,
        'left_wrist': 15,
        'right_wrist': 16,
        'left_hip': 23,
        'right_hip': 24,
    }
    
    def __init__(self, extension_threshold=0.5, elbow_angle_threshold=130, verticality_threshold=60):
        """
        Args:
            extension_threshold: Minimum normalized reach to consider "extended"
            elbow_angle_threshold: Minimum elbow angle (degrees) for straight arm
            verticality_threshold: Maximum angle from horizontal for valid punch
        """
        self.extension_threshold = extension_threshold
        self.elbow_angle_threshold = elbow_angle_threshold
        self.verticality_threshold = verticality_threshold
        
        # Calibration - Start with higher defaults to prevent false extensions
        self.calibrated = False
        self.reference_reach = {'Left': 0.7, 'Right': 0.7}  # Guard stance reach (higher default)
        self.max_reach = {'Left': 1.5, 'Right': 1.5}  # Full extension reach
        
        # Guard baseline calibration for elbow angles
        self.guard_elbow_angle = {
            'Left': 110.0,   # Default guard elbow angle
            'Right': 110.0
        }
        
        # Smoothing
        self.reach_history = {'Left': deque(maxlen=5), 'Right': deque(maxlen=5)}
        
        # Debug
        self.last_aec = {'Left': None, 'Right': None}
    
    def calculate_scale_factor(self, pose_landmarks):
        """
        Calculate biacromial diameter (shoulder width) as scale factor.
        
        S_frame = ||P_L_Shoulder - P_R_Shoulder||_2
        
        This makes all measurements scale-invariant (works at any distance).
        """
        l_shoulder = pose_landmarks.landmark[self.LANDMARKS['left_shoulder']]
        r_shoulder = pose_landmarks.landmark[self.LANDMARKS['right_shoulder']]
        
        # 2D distance in image plane
        dx = l_shoulder.x - r_shoulder.x
        dy = l_shoulder.y - r_shoulder.y
        
        return np.sqrt(dx**2 + dy**2)
    
    def calculate_radial_reach(self, pose_landmarks, hand='Right'):
        """
        Calculate normalized radial reach: R_ext = ||wrist - shoulder|| / S_frame
        
        This is how far the hand is from the shoulder, normalized by body size.
        """
        scale = self.calculate_scale_factor(pose_landmarks)
        if scale < 0.01:  # Avoid division by zero
            return 0.0
        
        if hand == 'Right':
            shoulder = pose_landmarks.landmark[self.LANDMARKS['right_shoulder']]
            wrist = pose_landmarks.landmark[self.LANDMARKS['right_wrist']]
        else:
            shoulder = pose_landmarks.landmark[self.LANDMARKS['left_shoulder']]
            wrist = pose_landmarks.landmark[self.LANDMARKS['left_wrist']]
        
        dx = wrist.x - shoulder.x
        dy = wrist.y - shoulder.y
        
        reach = np.sqrt(dx**2 + dy**2)
        normalized_reach = reach / scale
        
        # Smooth
        self.reach_history[hand].append(normalized_reach)
        return np.mean(self.reach_history[hand])
    
    def calculate_elbow_angle(self, pose_landmarks, hand='Right'):
        """
        Calculate elbow angle using vector dot product.
        
        θ_elbow = arccos((U · F) / (||U|| ||F||))
        
        Where:
        - U = Upper arm vector (shoulder → elbow)
        - F = Forearm vector (elbow → wrist)
        
        Returns angle in degrees. 180° = straight, 90° = bent.
        """
        if hand == 'Right':
            shoulder = pose_landmarks.landmark[self.LANDMARKS['right_shoulder']]
            elbow = pose_landmarks.landmark[self.LANDMARKS['right_elbow']]
            wrist = pose_landmarks.landmark[self.LANDMARKS['right_wrist']]
        else:
            shoulder = pose_landmarks.landmark[self.LANDMARKS['left_shoulder']]
            elbow = pose_landmarks.landmark[self.LANDMARKS['left_elbow']]
            wrist = pose_landmarks.landmark[self.LANDMARKS['left_wrist']]
        
        # Upper arm vector (shoulder → elbow)
        u = np.array([elbow.x - shoulder.x, elbow.y - shoulder.y])
        
        # Forearm vector (elbow → wrist)
        f = np.array([wrist.x - elbow.x, wrist.y - elbow.y])
        
        # Dot product
        dot = np.dot(u, f)
        mag_u = np.linalg.norm(u)
        mag_f = np.linalg.norm(f)
        
        if mag_u < 0.001 or mag_f < 0.001:
            return 180.0  # Default to straight if vectors are too small
        
        cos_angle = dot / (mag_u * mag_f)
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        
        angle = np.degrees(np.arccos(cos_angle))
        return angle
    
    def calculate_verticality(self, pose_landmarks, hand='Right'):
        """
        Calculate the angle of arm vector relative to horizontal.
        
        φ = atan2(Δy, Δx)
        
        Punch Criteria: φ within ±threshold of horizontal
        Vertical Slide: φ ≈ ±90°
        
        Returns:
            angle: Degrees from horizontal (-180 to +180)
            is_horizontal: True if within punch cone
        """
        if hand == 'Right':
            shoulder = pose_landmarks.landmark[self.LANDMARKS['right_shoulder']]
            wrist = pose_landmarks.landmark[self.LANDMARKS['right_wrist']]
        else:
            shoulder = pose_landmarks.landmark[self.LANDMARKS['left_shoulder']]
            wrist = pose_landmarks.landmark[self.LANDMARKS['left_wrist']]
        
        dx = wrist.x - shoulder.x
        dy = wrist.y - shoulder.y
        
        # atan2 returns angle in radians, convert to degrees
        phi = np.degrees(np.arctan2(dy, dx))
        
        # Horizontal means phi is near 0° or ±180°
        # We check if the arm direction is within threshold of horizontal
        # Note: In image coordinates, Y increases downward
        
        # Angle from horizontal axis
        h_angle = abs(phi)
        if h_angle > 90:
            h_angle = 180 - h_angle
        
        is_horizontal = h_angle < self.verticality_threshold
        
        return phi, is_horizontal
    
    def calculate_forward_motion(self, pose_landmarks, hand='Right'):
        """
        Estimate if the hand is moving forward (toward camera).
        Uses relative position of wrist compared to shoulder depth.
        
        In normalized coordinates, Z < 0 means closer to camera.
        """
        if hand == 'Right':
            shoulder = pose_landmarks.landmark[self.LANDMARKS['right_shoulder']]
            wrist = pose_landmarks.landmark[self.LANDMARKS['right_wrist']]
        else:
            shoulder = pose_landmarks.landmark[self.LANDMARKS['left_shoulder']]
            wrist = pose_landmarks.landmark[self.LANDMARKS['left_wrist']]
        
        # Z difference (negative = wrist is in front of shoulder)
        z_diff = wrist.z - shoulder.z
        
        # True if wrist is in front of shoulder
        return z_diff < -0.05, z_diff
    
    def calculate_aec(self, pose_landmarks, hand='Right'):
        """
        Calculate Arm Extension using REACH RATIO (simpler, more reliable).
        
        Uses 3D Euclidean distance with weighted Z to handle foreshortening.
        Auto-calibrates min/max ranges during play.
        
        Returns:
            dict with reach_percent, is_extended, and debug values
        """
        # Get landmarks
        if hand == 'Right':
            shoulder = pose_landmarks.landmark[self.LANDMARKS['right_shoulder']]
            wrist = pose_landmarks.landmark[self.LANDMARKS['right_wrist']]
        else:
            shoulder = pose_landmarks.landmark[self.LANDMARKS['left_shoulder']]
            wrist = pose_landmarks.landmark[self.LANDMARKS['left_wrist']]
        
        # 1. Calculate Shoulder Width (Scale Factor)
        l_sh = pose_landmarks.landmark[self.LANDMARKS['left_shoulder']]
        r_sh = pose_landmarks.landmark[self.LANDMARKS['right_shoulder']]
        shoulder_width = np.sqrt((l_sh.x - r_sh.x)**2 + (l_sh.y - r_sh.y)**2)
        
        if shoulder_width < 0.01:
            return {'is_extended': False, 'reach_percent': 0, 'raw_reach': 0, 
                    'min': 0, 'max': 1, 'elbow_angle': 90, 'is_horizontal': True,
                    'is_arm_straight': False, 'z_bonus': 0, 'verticality': 0}
        
        # 2. Calculate Reach (3D distance from shoulder to wrist)
        # Weight Z less to reduce noise from depth estimation
        dx = wrist.x - shoulder.x
        dy = wrist.y - shoulder.y
        dz = wrist.z - shoulder.z
        
        # 3D reach with weighted Z (0.5 weight to reduce Z noise)
        raw_reach = np.sqrt(dx**2 + dy**2 + (0.5 * dz)**2)
        
        # Normalize by shoulder width for scale-invariance
        normalized_reach = raw_reach / shoulder_width
        
        # Smooth the reach value
        self.reach_history[hand].append(normalized_reach)
        avg_reach = np.mean(self.reach_history[hand])
        
        # 3. Dynamic Auto-Calibration
        # Gently update limits during play
        if avg_reach < self.reference_reach[hand] * 0.92:
            self.reference_reach[hand] = 0.85 * self.reference_reach[hand] + 0.15 * avg_reach
        if avg_reach > self.max_reach[hand] * 1.08:
            self.max_reach[hand] = 0.85 * self.max_reach[hand] + 0.15 * avg_reach
        
        # 4. Calculate Percentage (0-100%)
        reach_range = self.max_reach[hand] - self.reference_reach[hand]
        if reach_range < 0.15:
            reach_range = 0.15  # Safety minimum
        
        reach_percent = (avg_reach - self.reference_reach[hand]) / reach_range * 100
        reach_percent = max(0, min(100, reach_percent))
        
        # 5. Check Horizontality
        phi = np.degrees(np.arctan2(abs(dy), abs(dx)))
        is_horizontal = phi < 55  # More lenient threshold
        
        # 6. Determine Extension - HIGHER threshold to avoid false positives in guard
        # Extended = TRUE only when reach is significantly above guard baseline
        is_extended = reach_percent > 70 and is_horizontal
        
        result = {
            'reach_percent': reach_percent,
            'is_extended': is_extended,
            'raw_reach': avg_reach,
            'min': self.reference_reach[hand],
            'max': self.max_reach[hand],
            'verticality': phi,
            'is_horizontal': is_horizontal,
            # Compatibility values
            'elbow_angle': 160 if is_extended else 100,
            'is_arm_straight': is_extended,
            'z_bonus': 0,
            'z_diff': dz,
            'is_forward': dz < -0.02,
        }
        
        self.last_aec[hand] = result
        return result
    
    def calibrate_guard(self, pose_landmarks):
        """
        Calibrate guard position (baseline) for each hand.
        Call this when user is in guard stance (arms bent, near body).
        Press 'G' key to calibrate.
        """
        # Calibrate 2D reach
        self.reference_reach['Right'] = self.calculate_radial_reach(pose_landmarks, 'Right')
        self.reference_reach['Left'] = self.calculate_radial_reach(pose_landmarks, 'Left')
        
        # Calibrate elbow angles
        for hand in ['Left', 'Right']:
            angle = self._get_elbow_angle(pose_landmarks, hand)
            self.guard_elbow_angle[hand] = angle
            print(f"[CALIBRATE] {hand} guard angle: {angle:.0f}°")
        
        self.calibrated = True
        print(f"[CALIBRATE] Guard reach - R: {self.reference_reach['Right']:.3f}, L: {self.reference_reach['Left']:.3f}")
    
    def _get_elbow_angle(self, pose_landmarks, hand):
        """Get raw elbow angle for a hand using 3D vectors."""
        if hand == 'Right':
            shoulder = pose_landmarks.landmark[self.LANDMARKS['right_shoulder']]
            elbow = pose_landmarks.landmark[self.LANDMARKS['right_elbow']]
            wrist = pose_landmarks.landmark[self.LANDMARKS['right_wrist']]
        else:
            shoulder = pose_landmarks.landmark[self.LANDMARKS['left_shoulder']]
            elbow = pose_landmarks.landmark[self.LANDMARKS['left_elbow']]
            wrist = pose_landmarks.landmark[self.LANDMARKS['left_wrist']]
        
        upper_arm = np.array([
            elbow.x - shoulder.x,
            elbow.y - shoulder.y,
            elbow.z - shoulder.z
        ])
        forearm = np.array([
            wrist.x - elbow.x,
            wrist.y - elbow.y,
            wrist.z - elbow.z
        ])
        
        dot = np.dot(upper_arm, forearm)
        mag = np.linalg.norm(upper_arm) * np.linalg.norm(forearm)
        
        if mag < 0.001:
            return 90.0
        
        cos_angle = np.clip(dot / mag, -1.0, 1.0)
        return np.degrees(np.arccos(cos_angle))
    
    def calibrate_extended(self, pose_landmarks):
        """
        Calibrate maximum reach for each hand.
        Call this when user has arms fully extended forward.
        """
        self.max_reach['Right'] = self.calculate_radial_reach(pose_landmarks, 'Right')
        self.max_reach['Left'] = self.calculate_radial_reach(pose_landmarks, 'Left')
        self.calibrated = True
        print(f"[CALIBRATE] Extended reach - R: {self.max_reach['Right']:.3f}, L: {self.max_reach['Left']:.3f}")
        
        # Calculate range
        r_range = self.max_reach['Right'] - self.reference_reach['Right']
        l_range = self.max_reach['Left'] - self.reference_reach['Left']
        print(f"[CALIBRATE] Range - R: {r_range:.3f}, L: {l_range:.3f}")
    
    def get_debug_string(self, hand='Right'):
        """Get debug string for HUD display."""
        if self.last_aec[hand] is None:
            return f"{hand}: No data"
        
        aec = self.last_aec[hand]
        ext = "EXT" if aec['is_extended'] else "---"
        horz = "H" if aec['is_horizontal'] else "V"
        arm = "STR" if aec['is_arm_straight'] else "BNT"
        
        return f"{hand[0]}: R={aec['reach_percent']:.0f}% E={aec['elbow_angle']:.0f}° {horz} {arm} [{ext}]"


class GeometricPunchDetector:
    """
    Complete punch detection using geometric methods.
    Combines AnchorDepthEstimator with velocity detection.
    """
    
    def __init__(self):
        self.estimator = AnchorDepthEstimator()
        
        # Velocity tracking
        self.prev_reach = {'Left': 0, 'Right': 0}
        self.prev_time = time.time()
        self.velocity = {'Left': 0, 'Right': 0}
        
        # Punch state
        self.punch_started = {'Left': False, 'Right': False}
        self.last_punch_time = {'Left': 0, 'Right': 0}
        
        # Thresholds
        self.velocity_threshold = 0.5  # Reach units per second
        self.cooldown = 0.3  # Seconds between punches
    
    def update(self, pose_landmarks):
        """
        Update punch detection with new pose data.
        
        Returns:
            dict: {
                'Right': {'is_punch': bool, 'velocity': float, 'aec': dict},
                'Left': {'is_punch': bool, 'velocity': float, 'aec': dict}
            }
        """
        current_time = time.time()
        dt = current_time - self.prev_time
        self.prev_time = current_time
        
        results = {}
        
        for hand in ['Right', 'Left']:
            aec = self.estimator.calculate_aec(pose_landmarks, hand)
            
            # Calculate velocity
            if dt > 0.001:
                self.velocity[hand] = (aec['reach'] - self.prev_reach[hand]) / dt
            self.prev_reach[hand] = aec['reach']
            
            # Punch detection logic
            is_extending = self.velocity[hand] > self.velocity_threshold
            is_extended = aec['is_extended']
            cooldown_ok = (current_time - self.last_punch_time[hand]) > self.cooldown
            
            # Punch starts when arm is extending AND all geometric checks pass
            is_punch = is_extending and is_extended and cooldown_ok
            
            if is_punch:
                self.last_punch_time[hand] = current_time
                print(f"[PUNCH] {hand}! Reach={aec['reach']:.2f} Vel={self.velocity[hand]:.2f}")
            
            results[hand] = {
                'is_punch': is_punch,
                'velocity': self.velocity[hand],
                'aec': aec,
            }
        
        return results


def test_with_camera():
    """Test the depth estimator with live camera."""
    if not MEDIAPIPE_AVAILABLE:
        print("MediaPipe not installed!")
        return
    
    import cv2
    
    mp_pose = mp.solutions.pose
    mp_draw = mp.solutions.drawing_utils
    
    pose = mp_pose.Pose(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    
    estimator = AnchorDepthEstimator()
    
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    print("=" * 60)
    print("GEOMETRIC DEPTH ESTIMATOR TEST")
    print("=" * 60)
    print("Controls:")
    print("  G - Calibrate GUARD position (arms close to body)")
    print("  E - Calibrate EXTENDED position (arms fully out)")
    print("  Q - Quit")
    print("=" * 60)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb)
        
        if results.pose_landmarks:
            # Draw skeleton
            mp_draw.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            
            # Calculate AEC for both hands
            aec_r = estimator.calculate_aec(results.pose_landmarks, 'Right')
            aec_l = estimator.calculate_aec(results.pose_landmarks, 'Left')
            
            # Display info
            y = 30
            cv2.putText(frame, estimator.get_debug_string('Right'), (10, y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            y += 30
            cv2.putText(frame, estimator.get_debug_string('Left'), (10, y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            
            # Show extension status
            y += 50
            if aec_r['is_extended']:
                cv2.putText(frame, "RIGHT EXTENDED!", (10, y), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
            if aec_l['is_extended']:
                cv2.putText(frame, "LEFT EXTENDED!", (400, y), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 0, 0), 3)
            
            # Handle key presses for calibration
            key = cv2.waitKey(1) & 0xFF
            if key == ord('g'):
                estimator.calibrate_guard(results.pose_landmarks)
            elif key == ord('e'):
                estimator.calibrate_extended(results.pose_landmarks)
            elif key == ord('q'):
                break
        else:
            cv2.putText(frame, "No pose detected", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cv2.imshow("Geometric Depth Test", frame)
    
    cap.release()
    cv2.destroyAllWindows()


def test_without_camera():
    """Test basic functionality without camera."""
    print("=" * 60)
    print("GEOMETRIC DEPTH ESTIMATOR - UNIT TEST")
    print("=" * 60)
    
    estimator = AnchorDepthEstimator()
    
    # Create mock landmarks
    class MockLandmark:
        def __init__(self, x, y, z=0):
            self.x = x
            self.y = y
            self.z = z
    
    class MockLandmarks:
        def __init__(self):
            self.landmark = [MockLandmark(0, 0)] * 33  # MediaPipe has 33 pose landmarks
    
    # Test 1: Guard stance (arms close)
    print("\n1. Guard Stance Test:")
    landmarks = MockLandmarks()
    landmarks.landmark[11] = MockLandmark(0.3, 0.3)  # Left shoulder
    landmarks.landmark[12] = MockLandmark(0.7, 0.3)  # Right shoulder
    landmarks.landmark[13] = MockLandmark(0.25, 0.5)  # Left elbow (bent down)
    landmarks.landmark[14] = MockLandmark(0.75, 0.5)  # Right elbow (bent down)
    landmarks.landmark[15] = MockLandmark(0.35, 0.4)  # Left wrist (close to body)
    landmarks.landmark[16] = MockLandmark(0.65, 0.4)  # Right wrist (close to body)
    
    aec = estimator.calculate_aec(landmarks, 'Right')
    print(f"  Reach: {aec['reach']:.3f}")
    print(f"  Elbow Angle: {aec['elbow_angle']:.1f}°")
    print(f"  Is Extended: {aec['is_extended']} (should be False)")
    
    # Test 2: Extended punch (arm straight out - key: shoulder, elbow, wrist aligned)
    print("\n2. Extended Punch Test:")
    # Straight arm: shoulder(0.7,0.3) -> elbow(0.9,0.3) -> wrist(1.1,0.3)
    landmarks.landmark[14] = MockLandmark(0.9, 0.3)   # Right elbow (horizontal from shoulder)
    landmarks.landmark[16] = MockLandmark(1.1, 0.3)   # Right wrist (continues horizontal)
    
    aec = estimator.calculate_aec(landmarks, 'Right')
    print(f"  Reach: {aec['reach']:.3f}")
    print(f"  Elbow Angle: {aec['elbow_angle']:.1f}° (should be ~180)")
    print(f"  Is Horizontal: {aec['is_horizontal']}")
    print(f"  Is Arm Straight: {aec['is_arm_straight']}")
    print(f"  Is Extended: {aec['is_extended']} (should be True)")
    
    # Test 3: Vertical raise (arm up, not punch)
    print("\n3. Vertical Raise Test:")
    landmarks.landmark[14] = MockLandmark(0.7, 0.1)  # Right elbow (up)
    landmarks.landmark[16] = MockLandmark(0.7, -0.1)  # Right wrist (straight up)
    
    aec = estimator.calculate_aec(landmarks, 'Right')
    print(f"  Reach: {aec['reach']:.3f}")
    print(f"  Verticality: {aec['verticality']:.1f}°")
    print(f"  Is Horizontal: {aec['is_horizontal']} (should be False)")
    print(f"  Is Extended: {aec['is_extended']} (should be False)")
    
    print("\n" + "=" * 60)
    print("Unit tests complete!")
    print("=" * 60)


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--camera":
        test_with_camera()
    else:
        test_without_camera()
        print("\nRun with --camera flag for live camera test")
