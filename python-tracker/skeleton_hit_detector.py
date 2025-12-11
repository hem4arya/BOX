"""
Skeleton-Based Hit Detector with Depth Awareness
=================================================
SIMPLE approach: If hand moves fast enough, it's a HIT.
DEPTH: Detect if movement is FORWARD (toward camera), SIDEWAYS, or IDLE.

Physics-based detection:
- Track wrist position from skeleton (reliable from MediaPipe Pose)
- Calculate velocity (pixels per second)
- Analyze movement DIRECTION using Z-coordinate and position changes
- If velocity > threshold AND moving forward = PUNCH!
"""

import numpy as np
from collections import deque
import time


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
            dict with hit events, velocity, direction, depth percent, and elbow angle
        """
        self.frame_count += 1
        results = {
            'Left': {'hit': False, 'velocity': 0, 'direction': 'IDLE', 'depth_change': 0, 'depth_percent': 0, 'elbow_angle': 90},
            'Right': {'hit': False, 'velocity': 0, 'direction': 'IDLE', 'depth_change': 0, 'depth_percent': 0, 'elbow_angle': 90}
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
            
            # Calculate velocity (change from previous frame)
            prev_pos = self.position_history[hand][-2]['pos']
            velocity = np.linalg.norm(current_pos - prev_pos)
            
            # Detect direction
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
            # Problem: Bent guard arm has short 2D distance (like forward punch)
            # Solution: Use elbow angle to disambiguate!
            # Bent arm (< 130°) = GUARD = effective_depth 0%
            # Straight arm (> 130°) = COULD BE PUNCH = use calculated depth
            
            ELBOW_THRESHOLD = 130  # Degrees - below this is considered "bent"
            
            if elbow_angle < ELBOW_THRESHOLD:
                # Arm is BENT = guard position = no forward extension
                effective_depth = 0.0
            else:
                # Arm is STRAIGHT = scale depth by how straight (130° -> 0%, 180° -> 100%)
                straightness = (elbow_angle - 90) / 90  # 90° -> 0, 180° -> 1
                straightness = max(0, min(1, straightness))
                effective_depth = anthro_depth_percent * straightness
            
            # Store the gated depth
            self.last_depth_percent[hand] = effective_depth
            
            results[hand]['velocity'] = velocity
            results[hand]['direction'] = direction
            results[hand]['depth_change'] = depth_change
            results[hand]['depth_percent'] = effective_depth  # GATED by elbow angle!
            results[hand]['elbow_angle'] = elbow_angle
            results[hand]['is_straight'] = elbow_angle >= ELBOW_THRESHOLD
            
            # ════════════════════════════════════════════════════════════
            # HIT DETECTION: Velocity + Elbow Angle Gated Depth
            # ════════════════════════════════════════════════════════════
            # PUNCH = fast movement + arm STRAIGHT + depth > 40%
            in_cooldown = (self.frame_count - self.last_hit_frame[hand]) < self.cooldown_frames
            
            is_fast = velocity > self.velocity_threshold
            is_straight = elbow_angle >= ELBOW_THRESHOLD  # NEW: require straight arm
            is_forward = effective_depth > 40  # Using gated depth
            
            if is_fast and is_straight and is_forward and not in_cooldown:
                # FORWARD PUNCH DETECTED!
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
                
                # Labels
                cv2.putText(frame, f"{hand}", (bar_x, 440), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(frame, f"D={depth_pct:.0f}%", (bar_x, 460), cv2.FONT_HERSHEY_SIMPLEX, 0.5, depth_color, 2)
                cv2.putText(frame, f"E={elbow:.0f}° {elbow_label}", (bar_x, 480), cv2.FONT_HERSHEY_SIMPLEX, 0.4, elbow_color, 1)
                
                # HIT indicator
                if hit:
                    hit_x = 200 if hand == 'Left' else w - 300
                    cv2.putText(frame, ">>> PUNCH! <<<", (hit_x, 100), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 3)
                    print(f">>> {hand} PUNCH! Depth={depth_pct:.0f}% (Total: {detector.hit_count[hand]})")
            
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
