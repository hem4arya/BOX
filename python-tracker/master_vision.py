"""
Master Vision System for Boxing Game (v2)
==========================================
Unified interface that combines all tracking and detection components.

CHANGES IN V2:
- Uses POSE wrist landmarks (15, 16) instead of HAND landmarks for more stable tracking
- Adds arm velocity calculation using full kinematic chain (shoulder → elbow → wrist)
- Hands detector only used for gesture detection (fist/open)

Architecture:
  Camera → Pose Detection → Arm Velocity → DepthEstimator → FlowValidator → PunchFSM → Events

Usage:
    vision = MasterVisionSystem()
    vision.start()
    
    while running:
        result = vision.process_frame(frame)
        if result['punch_event']:
            game.register_hit(result['hand'])
"""

import cv2
import numpy as np
import time
from collections import deque

try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False
    print("Warning: MediaPipe not available")

# Import our modules
from signal_processor import OneEuroFilter
from depth_estimator import AnchorDepthEstimator
from flow_gated_validator import FlowGatedValidator, TRACKING_COLORS
from punch_fsm import PunchDetector, PunchState, STATE_COLORS


class MasterVisionSystem:
    """
    Unified boxing vision system combining all components.
    
    V2 Changes:
    - Position tracking uses POSE wrist (more stable during occlusion)
    - Arm velocity calculated from full kinematic chain
    - Hand landmarks only for gesture detection
    """
    
    # Pose landmark indices
    POSE_LANDMARKS = {
        'left_shoulder': 11,
        'right_shoulder': 12,
        'left_elbow': 13,
        'right_elbow': 14,
        'left_wrist': 15,
        'right_wrist': 16,
    }
    
    def __init__(self, 
                 camera_id=0,
                 resolution=(640, 480),
                 model_complexity=1,
                 use_hands_for_gesture=True):
        """
        Initialize all components.
        
        Args:
            camera_id: Webcam device ID
            resolution: (width, height)
            model_complexity: MediaPipe model complexity (0=lite, 1=full)
            use_hands_for_gesture: Enable hand detection for gesture recognition
        """
        if not MEDIAPIPE_AVAILABLE:
            raise ImportError("MediaPipe is required for MasterVisionSystem")
        
        # Camera
        self.cap = None
        self.camera_id = camera_id
        self.resolution = resolution
        self.frame_width = resolution[0]
        self.frame_height = resolution[1]
        
        # MediaPipe setup
        self.mp_pose = mp.solutions.pose
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        
        # Primary: Pose for wrist tracking (more stable)
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            model_complexity=model_complexity
        )
        
        # Secondary: Hands for gesture detection only
        self.use_hands_for_gesture = use_hands_for_gesture
        self.hands = None
        if use_hands_for_gesture:
            self.hands = self.mp_hands.Hands(
                max_num_hands=2,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
                model_complexity=0  # Lite model for speed
            )
        
        # ========== COMPONENTS ==========
        
        # 1. Depth Estimator (AEC)
        self.depth_estimator = AnchorDepthEstimator()
        
        # 2. Flow-Gated Validator (Sensor Fusion)
        self.flow_validator = FlowGatedValidator()
        
        # 3. Punch FSM (State Machine)
        self.punch_detector = PunchDetector()
        
        # 4. Position smoothers for final output
        self.position_smoother = {
            'Left': {
                'x': OneEuroFilter(min_cutoff=1.5, beta=0.5),
                'y': OneEuroFilter(min_cutoff=1.5, beta=0.5)
            },
            'Right': {
                'x': OneEuroFilter(min_cutoff=1.5, beta=0.5),
                'y': OneEuroFilter(min_cutoff=1.5, beta=0.5)
            }
        }
        
        # 5. Arm length history for velocity calculation
        self.arm_length_history = {
            'Left': deque(maxlen=5),
            'Right': deque(maxlen=5)
        }
        
        # ========== STATE ==========
        self.prev_gray = None
        self.frame_count = 0
        self.start_time = None
        self.fps_history = deque(maxlen=30)
        
        # Gesture state (from hand detection)
        self.current_gesture = {'Left': 'UNKNOWN', 'Right': 'UNKNOWN'}
        
        # Last results for continuity
        self.last_result = {
            'Left': {'position': None, 'source': 'none', 'aec': None, 'state': 'idle'},
            'Right': {'position': None, 'source': 'none', 'aec': None, 'state': 'idle'}
        }
        
        # Statistics
        self.total_punches = {'Left': 0, 'Right': 0}
        
        print("[VISION] Master Vision System v2 initialized (POSE wrist mode)")
    
    def start(self):
        """Initialize camera and start processing."""
        self.cap = cv2.VideoCapture(self.camera_id, cv2.CAP_DSHOW)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.frame_width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.frame_height)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        
        actual_w = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        actual_h = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        
        self.frame_width = int(actual_w)
        self.frame_height = int(actual_h)
        
        self.start_time = time.time()
        print(f"[VISION] Started with resolution {self.frame_width}x{self.frame_height}")
    
    def stop(self):
        """Release resources."""
        if self.cap:
            self.cap.release()
        if self.pose:
            self.pose.close()
        if self.hands:
            self.hands.close()
        print("[VISION] Stopped")
    
    def calculate_arm_length(self, pose_landmarks, hand='Right'):
        """
        Calculate arm length (shoulder to wrist distance).
        Used for arm extension velocity calculation.
        """
        if hand == 'Right':
            shoulder = pose_landmarks.landmark[self.POSE_LANDMARKS['right_shoulder']]
            wrist = pose_landmarks.landmark[self.POSE_LANDMARKS['right_wrist']]
        else:
            shoulder = pose_landmarks.landmark[self.POSE_LANDMARKS['left_shoulder']]
            wrist = pose_landmarks.landmark[self.POSE_LANDMARKS['left_wrist']]
        
        dx = wrist.x - shoulder.x
        dy = wrist.y - shoulder.y
        
        return np.sqrt(dx**2 + dy**2)
    
    def calculate_arm_velocity(self, hand, timestamp):
        """
        Calculate arm extension velocity from history.
        Positive = extending, Negative = retracting.
        """
        history = self.arm_length_history[hand]
        if len(history) < 2:
            return 0.0
        
        h = list(history)
        dt = h[-1]['time'] - h[-2]['time']
        if dt <= 0:
            return 0.0
        
        return (h[-1]['length'] - h[-2]['length']) / dt
    
    def detect_gesture(self, hand_landmarks):
        """
        Detect hand gesture (FIST or OPEN) from hand landmarks.
        """
        if hand_landmarks is None:
            return 'UNKNOWN'
        
        # Get finger tip and MCP (knuckle) positions
        # If tips are below MCPs, fingers are curled (fist)
        tips = [8, 12, 16, 20]  # Index, Middle, Ring, Pinky tips
        mcps = [5, 9, 13, 17]   # Corresponding MCPs
        
        curled_count = 0
        for tip_idx, mcp_idx in zip(tips, mcps):
            tip = hand_landmarks.landmark[tip_idx]
            mcp = hand_landmarks.landmark[mcp_idx]
            
            # If tip is below MCP (higher Y value = lower on screen), finger is curled
            if tip.y > mcp.y:
                curled_count += 1
        
        if curled_count >= 3:
            return 'FIST'
        else:
            return 'OPEN'
    
    def process_frame(self, frame=None):
        """
        Process a single frame through the entire pipeline.
        
        Uses POSE wrist landmarks for tracking (more stable).
        Uses HAND landmarks only for gesture detection.
        """
        timestamp = time.time()
        self.frame_count += 1
        
        # Capture if needed
        if frame is None:
            if self.cap is None:
                return None
            ret, frame = self.cap.read()
            if not ret:
                return None
        
        # Flip for mirror effect
        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]
        self.frame_width = w
        self.frame_height = h
        
        # Convert
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # ========== PRIMARY: POSE DETECTION ==========
        pose_results = self.pose.process(rgb_frame)
        
        # ========== SECONDARY: HAND DETECTION (gesture only) ==========
        hands_results = None
        if self.hands:
            hands_results = self.hands.process(rgb_frame)
        
        # Update gestures from hand detection
        if hands_results and hands_results.multi_hand_landmarks:
            for hand_landmarks, handedness in zip(
                hands_results.multi_hand_landmarks,
                hands_results.multi_handedness
            ):
                hand = handedness.classification[0].label
                self.current_gesture[hand] = self.detect_gesture(hand_landmarks)
        
        # Initialize results
        results = {
            'frame': frame,
            'timestamp': timestamp,
            'fps': 0,
            'Left': None,
            'Right': None,
            'pose_landmarks': pose_results.pose_landmarks if pose_results else None
        }
        
        # ========== PROCESS HANDS FROM POSE ==========
        if pose_results and pose_results.pose_landmarks:
            pose_landmarks = pose_results.pose_landmarks
            
            for hand in ['Left', 'Right']:
                # NOTE: Camera is mirrored in display, but MediaPipe processes ORIGINAL frame
                # So when user sees their right hand on LEFT of screen, MediaPipe still sees it as RIGHT
                # NO SWAP NEEDED - just use the hand label directly
                
                # Get wrist from POSE landmarks (much more stable!)
                wrist_idx = self.POSE_LANDMARKS['right_wrist' if hand == 'Right' else 'left_wrist']
                wrist = pose_landmarks.landmark[wrist_idx]
                wrist_pos = (wrist.x, wrist.y)
                
                # Check visibility
                if wrist.visibility < 0.5:
                    continue
                
                # Calculate arm length and velocity
                arm_length = self.calculate_arm_length(pose_landmarks, hand)
                self.arm_length_history[hand].append({
                    'length': arm_length,
                    'time': timestamp
                })
                arm_velocity = self.calculate_arm_velocity(hand, timestamp)
                
                # Flow-gated validation
                validated_pos, source, val_confidence = self.flow_validator.validate(
                    hand=hand,
                    mp_position=wrist_pos,
                    mp_confidence=wrist.visibility,
                    gray_frame=gray_frame,
                    frame_width=w,
                    frame_height=h,
                    timestamp=timestamp
                )
                
                # Depth estimation (AEC)
                aec_data = self.depth_estimator.calculate_aec(pose_landmarks, hand)
                
                # Debug: Print AEC values ALWAYS to see guard position too
                # Show simplified output: hand, reach%, extended status
                print(f"[{hand[0]}] R={aec_data['reach_percent']:.0f}% Ext={aec_data['is_extended']}", end="  ")
                
                # Calculate flow magnitude (combine position change and arm velocity)
                flow_magnitude = 0.0
                prev_pos = self.last_result[hand].get('position')
                if prev_pos and validated_pos:
                    dx = validated_pos[0] - prev_pos[0]
                    dy = validated_pos[1] - prev_pos[1]
                    pos_change = np.sqrt(dx**2 + dy**2) * 30
                    # Combine position velocity and arm extension velocity
                    flow_magnitude = pos_change + abs(arm_velocity) * 20
                
                # Punch FSM (with arm velocity)
                punch_event, state = self.punch_detector.update(
                    hand=hand,
                    wrist_position=validated_pos,
                    flow_magnitude=flow_magnitude,
                    aec_data=aec_data,
                    timestamp=timestamp,
                    arm_velocity=arm_velocity
                )
                
                if punch_event == 'PUNCH':
                    self.total_punches[hand] += 1
                
                # Final position smoothing
                if validated_pos:
                    final_x = self.position_smoother[hand]['x'](timestamp, validated_pos[0])
                    final_y = self.position_smoother[hand]['y'](timestamp, validated_pos[1])
                    final_pos = (final_x, final_y)
                else:
                    final_pos = validated_pos
                
                # Store result
                results[hand] = {
                    'position': final_pos,
                    'position_raw': validated_pos,
                    'wrist_z': wrist.z,
                    'arm_length': arm_length,
                    'arm_velocity': arm_velocity,
                    'gesture': self.current_gesture[hand],
                    'source': source,
                    'confidence': val_confidence,
                    'aec': aec_data,
                    'state': state.value if state else 'idle',
                    'state_enum': state,
                    'punch_event': punch_event,
                    'flow_magnitude': flow_magnitude,
                    'color': self.punch_detector.get_color(hand)
                }
                
                self.last_result[hand] = results[hand]
        
        # ========== HANDLE LOST HANDS ==========
        for hand in ['Left', 'Right']:
            if results[hand] is None:
                validated_pos, source, val_confidence = self.flow_validator.validate(
                    hand=hand,
                    mp_position=None,
                    mp_confidence=0.0,
                    gray_frame=gray_frame,
                    frame_width=w,
                    frame_height=h,
                    timestamp=timestamp
                )
                
                if validated_pos and source != 'lost':
                    results[hand] = {
                        'position': validated_pos,
                        'source': source,
                        'confidence': val_confidence,
                        'aec': self.last_result[hand].get('aec'),
                        'state': self.punch_detector.get_state(hand).value,
                        'punch_event': None,
                        'color': TRACKING_COLORS.get(source, (128, 128, 128)),
                        'gesture': self.current_gesture[hand],
                        'predicted': True
                    }
        
        # Update previous frame
        self.prev_gray = gray_frame
        
        # Calculate FPS
        elapsed = timestamp - self.start_time
        if elapsed > 0:
            fps = self.frame_count / elapsed
            self.fps_history.append(fps)
            results['fps'] = np.mean(self.fps_history)
        
        return results
    
    def draw_debug(self, frame, results, draw_skeleton=True, draw_hands=True):
        """Draw debug visualization on frame."""
        if results is None:
            return frame
        
        h, w = frame.shape[:2]
        
        # Draw pose skeleton
        if draw_skeleton and results.get('pose_landmarks'):
            self.mp_drawing.draw_landmarks(
                frame, results['pose_landmarks'], self.mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=self.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=2),
                connection_drawing_spec=self.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1)
            )
        
        # Draw hands
        if draw_hands:
            for hand in ['Left', 'Right']:
                data = results.get(hand)
                if data and data.get('position'):
                    pos = data['position']
                    px = int(pos[0] * w)
                    py = int(pos[1] * h)
                    
                    color = data.get('color', (255, 255, 255))
                    
                    # Draw hand circle
                    cv2.circle(frame, (px, py), 25, color, -1)
                    cv2.circle(frame, (px, py), 27, (255, 255, 255), 2)
                    
                    # Predicted indicator
                    if data.get('predicted'):
                        cv2.circle(frame, (px, py), 30, (0, 165, 255), 2)
                    
                    # State label
                    state = data.get('state', 'idle')
                    gesture = data.get('gesture', 'UNK')
                    label = f"{hand[0]}: {state.upper()}"
                    cv2.putText(frame, label, (px - 50, py - 35),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                    
                    # Gesture indicator
                    cv2.putText(frame, f"[{gesture[:4]}]", (px - 25, py - 55),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 200, 0), 1)
                    
                    # PUNCH indicator
                    if data.get('punch_event'):
                        cv2.putText(frame, "PUNCH!", (px - 50, py + 50),
                                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
                    
                    # Arm velocity indicator
                    arm_vel = data.get('arm_velocity', 0)
                    vel_color = (0, 255, 0) if arm_vel > 0 else (0, 0, 255)
                    cv2.putText(frame, f"V:{arm_vel:.2f}", (px - 30, py + 75),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, vel_color, 1)
                    
                    # AEC bar
                    if data.get('aec'):
                        aec = data['aec']
                        reach = aec.get('reach_percent', 0)
                        bar_w = 60
                        bar_h = 8
                        bar_x = px - bar_w // 2
                        bar_y = py + 35
                        
                        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (50, 50, 50), -1)
                        fill_w = int(bar_w * reach / 100)
                        bar_color = (0, 255, 0) if reach > 70 else (0, 255, 255)
                        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill_w, bar_y + bar_h), bar_color, -1)
                        cv2.putText(frame, f"{reach:.0f}%", (bar_x + bar_w + 5, bar_y + bar_h),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        # HUD
        fps = results.get('fps', 0)
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        cv2.putText(frame, f"Punches - L:{self.total_punches['Left']} R:{self.total_punches['Right']}",
                   (w // 2 - 100, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        
        cv2.putText(frame, "POSE WRIST MODE", (w - 200, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        
        return frame
    
    def get_hand_data(self, hand):
        """Get last known data for a hand."""
        return self.last_result.get(hand)
    
    def get_stats(self):
        """Get system statistics."""
        return {
            'punches': self.total_punches.copy(),
            'frames': self.frame_count,
            'flow_stats': self.flow_validator.get_stats()
        }
    
    def reset(self):
        """Reset all state."""
        self.punch_detector.reset()
        self.flow_validator.reset()
        self.total_punches = {'Left': 0, 'Right': 0}
        self.frame_count = 0
        self.start_time = time.time()
        for hand in ['Left', 'Right']:
            self.arm_length_history[hand].clear()


def main():
    """Standalone test."""
    print("=" * 60)
    print("MASTER VISION SYSTEM v2 (POSE WRIST MODE)")
    print("=" * 60)
    print("Controls:")
    print("  Q - Quit")
    print("  R - Reset counters")
    print("  G - Calibrate GUARD position (press with arms bent)")
    print("=" * 60)
    
    vision = MasterVisionSystem(resolution=(1280, 720))
    vision.start()
    
    last_results = None
    
    try:
        while True:
            results = vision.process_frame()
            
            if results:
                last_results = results
                frame = vision.draw_debug(results['frame'], results)
                cv2.imshow("Master Vision v2", frame)
                
                for hand in ['Left', 'Right']:
                    if results.get(hand) and results[hand].get('punch_event'):
                        print(f">>> {hand} PUNCH! (Total: {vision.total_punches[hand]})")
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                vision.reset()
                print("[RESET] Counters cleared")
            elif key == ord('g'):
                # Calibrate guard position
                if last_results and last_results.get('pose_landmarks'):
                    vision.depth_estimator.calibrate_guard(last_results['pose_landmarks'])
                    print(">>> GUARD CALIBRATED! Reach% will now start from 0 in this position.")
    
    finally:
        vision.stop()
        cv2.destroyAllWindows()
        
        print("\n" + "=" * 60)
        print("FINAL STATS")
        print("=" * 60)
        stats = vision.get_stats()
        print(f"Total punches: L={stats['punches']['Left']} R={stats['punches']['Right']}")
        print(f"Total frames: {stats['frames']}")


if __name__ == "__main__":
    main()
