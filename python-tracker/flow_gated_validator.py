"""
Flow-Gated Validator for High-Speed Hand Tracking
==================================================
Fuses MediaPipe pose estimation with Optical Flow to maintain
tracking during motion blur (fast punches at 30 FPS).

Algorithm:
- CASE A: MediaPipe confidence HIGH → Trust MediaPipe
- CASE B: MediaPipe LOW + Optical Flow HIGH → Trust Flow
- CASE C: Both LOW → Use Ballistic Extrapolation

Reference: "Punch Detection Sensor Fusion Architecture" Section 5
"""

import numpy as np
import cv2
from collections import deque
import time


# Color coding for tracking sources
TRACKING_COLORS = {
    'mediapipe': (0, 255, 0),     # GREEN - High confidence
    'optical_flow': (0, 165, 255), # ORANGE - Bridging blur
    'ballistic': (0, 0, 255),      # RED - Prediction only
    'lost': (128, 128, 128),       # GRAY - Lost tracking
    'none': (64, 64, 64),          # DARK GRAY - Uninitialized
}


class OpticalFlowTracker:
    """
    Lucas-Kanade Optical Flow for tracking hand ROI.
    
    When MediaPipe loses the hand due to blur, optical flow
    can still track the pixel movement of the blurry blob.
    """
    
    def __init__(self, roi_size=80, max_level=3):
        self.roi_size = roi_size
        self.lk_params = dict(
            winSize=(40, 40),
            maxLevel=max_level,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
        )
        self.prev_gray = None
        self.prev_points = None
    
    def initialize(self, gray_frame, wrist_position):
        """
        Initialize tracking from a known wrist position.
        
        Args:
            gray_frame: Grayscale frame
            wrist_position: (x, y) in pixel coordinates
        """
        self.prev_gray = gray_frame.copy()
        self.prev_points = np.array([[wrist_position]], dtype=np.float32)
    
    def track(self, gray_frame):
        """
        Track the wrist using optical flow.
        
        Returns:
            new_position: (x, y) or None if tracking failed
            flow_vector: (dx, dy) motion vector
            quality: 0-1 quality score
        """
        if self.prev_gray is None or self.prev_points is None:
            return None, (0, 0), 0.0
        
        try:
            # Calculate optical flow
            new_points, status, error = cv2.calcOpticalFlowPyrLK(
                self.prev_gray, 
                gray_frame, 
                self.prev_points, 
                None, 
                **self.lk_params
            )
            
            if new_points is None or status[0][0] == 0:
                return None, (0, 0), 0.0
            
            # Calculate flow vector
            flow_x = new_points[0][0][0] - self.prev_points[0][0][0]
            flow_y = new_points[0][0][1] - self.prev_points[0][0][1]
            
            # Quality based on error (lower error = higher quality)
            quality = 1.0 / (1.0 + error[0][0])
            
            # Update state
            self.prev_gray = gray_frame.copy()
            self.prev_points = new_points
            
            return (new_points[0][0][0], new_points[0][0][1]), (flow_x, flow_y), quality
            
        except Exception as e:
            return None, (0, 0), 0.0
    
    def reset(self):
        """Reset tracker state."""
        self.prev_gray = None
        self.prev_points = None


class BallisticExtrapolator:
    """
    Physics-based position prediction when all tracking fails.
    
    Assumes punch follows ballistic trajectory:
    P(t) = P(t-1) + V(t-1)*dt + 0.5*A(t-1)*dt^2
    """
    
    def __init__(self, history_size=5):
        self.position_history = deque(maxlen=history_size)
        self.last_time = None
    
    def update(self, position, timestamp):
        """Add a confirmed position to history."""
        if position is not None:
            self.position_history.append({
                'pos': np.array(position),
                'time': timestamp
            })
        self.last_time = timestamp
    
    def predict(self, current_time):
        """
        Predict position using kinematic equations.
        
        Returns:
            predicted_position: (x, y) or None
            confidence: 0-1 prediction confidence
        """
        if len(self.position_history) < 2:
            return None, 0.0
        
        # Get last two positions
        p1 = self.position_history[-2]
        p2 = self.position_history[-1]
        
        # Calculate velocity
        dt = p2['time'] - p1['time']
        if dt <= 0:
            return p2['pos'], 0.5
        
        velocity = (p2['pos'] - p1['pos']) / dt
        
        # Calculate acceleration if we have 3+ points
        if len(self.position_history) >= 3:
            p0 = self.position_history[-3]
            dt0 = p1['time'] - p0['time']
            if dt0 > 0:
                velocity_prev = (p1['pos'] - p0['pos']) / dt0
                acceleration = (velocity - velocity_prev) / dt
            else:
                acceleration = np.zeros(2)
        else:
            acceleration = np.zeros(2)
        
        # Predict future position
        dt_pred = current_time - p2['time']
        predicted = p2['pos'] + velocity * dt_pred + 0.5 * acceleration * dt_pred**2
        
        # Confidence decreases with prediction time
        confidence = max(0.0, 1.0 - (dt_pred / 0.3))  # 0.3 seconds = 0 confidence
        
        return predicted, confidence
    
    def get_velocity(self):
        """Get current estimated velocity."""
        if len(self.position_history) < 2:
            return np.zeros(2)
        
        p1 = self.position_history[-2]
        p2 = self.position_history[-1]
        
        dt = p2['time'] - p1['time']
        if dt <= 0:
            return np.zeros(2)
        
        return (p2['pos'] - p1['pos']) / dt
    
    def reset(self):
        """Reset history."""
        self.position_history.clear()


class FlowGatedValidator:
    """
    Main sensor fusion class that combines:
    1. MediaPipe pose estimation
    2. Optical Flow tracking
    3. Ballistic extrapolation
    
    Uses a gating mechanism to select the most reliable source.
    """
    
    def __init__(self, confidence_threshold=0.5, flow_threshold=5.0):
        """
        Args:
            confidence_threshold: Minimum MediaPipe confidence to trust
            flow_threshold: Minimum flow magnitude to trust optical flow
        """
        self.confidence_threshold = confidence_threshold
        self.flow_threshold = flow_threshold
        
        # Trackers for each hand
        self.flow_tracker = {
            'Left': OpticalFlowTracker(),
            'Right': OpticalFlowTracker()
        }
        self.ballistic = {
            'Left': BallisticExtrapolator(),
            'Right': BallisticExtrapolator()
        }
        
        # Last known good state
        self.last_position = {'Left': None, 'Right': None}
        self.last_confidence = {'Left': 0.0, 'Right': 0.0}
        self.tracking_source = {'Left': 'none', 'Right': 'none'}
        
        # Statistics
        self.source_counts = {'mediapipe': 0, 'optical_flow': 0, 'ballistic': 0, 'lost': 0}
    
    def validate(self, hand, mp_position, mp_confidence, gray_frame, frame_width, frame_height, timestamp=None):
        """
        Validate/fuse hand position from multiple sources.
        
        Args:
            hand: 'Left' or 'Right'
            mp_position: (x, y) normalized from MediaPipe (0-1)
            mp_confidence: MediaPipe confidence score
            gray_frame: Current grayscale frame
            frame_width, frame_height: Frame dimensions for denormalization
            timestamp: Current time in seconds (defaults to time.time())
            
        Returns:
            position: Best (x, y) in normalized coordinates
            source: 'mediapipe', 'optical_flow', 'ballistic', or 'lost'
            confidence: Combined confidence score
        """
        if timestamp is None:
            timestamp = time.time()
        
        # Convert to pixel coordinates for optical flow
        if mp_position is not None:
            mp_pixel = (mp_position[0] * frame_width, mp_position[1] * frame_height)
        else:
            mp_pixel = None
        
        # CASE A: MediaPipe is confident
        if mp_confidence > self.confidence_threshold and mp_position is not None:
            # Trust MediaPipe, reset optical flow to this position
            self.flow_tracker[hand].initialize(gray_frame, mp_pixel)
            self.ballistic[hand].update(np.array(mp_pixel), timestamp)
            self.last_position[hand] = mp_position
            self.last_confidence[hand] = mp_confidence
            self.tracking_source[hand] = 'mediapipe'
            self.source_counts['mediapipe'] += 1
            
            return mp_position, 'mediapipe', mp_confidence
        
        # CASE B: MediaPipe failed, try Optical Flow
        flow_pos, flow_vector, flow_quality = self.flow_tracker[hand].track(gray_frame)
        flow_magnitude = np.linalg.norm(flow_vector)
        
        if flow_pos is not None and (flow_magnitude > self.flow_threshold or flow_quality > 0.5):
            # Normalize back to 0-1
            normalized = (flow_pos[0] / frame_width, flow_pos[1] / frame_height)
            # Clamp to valid range
            normalized = (max(0, min(1, normalized[0])), max(0, min(1, normalized[1])))
            
            self.ballistic[hand].update(np.array(flow_pos), timestamp)
            self.last_position[hand] = normalized
            self.last_confidence[hand] = flow_quality
            self.tracking_source[hand] = 'optical_flow'
            self.source_counts['optical_flow'] += 1
            
            return normalized, 'optical_flow', flow_quality
        
        # CASE C: Optical Flow failed, try Ballistic
        ballistic_pos, ballistic_conf = self.ballistic[hand].predict(timestamp)
        
        if ballistic_pos is not None and ballistic_conf > 0.3:
            # Normalize back to 0-1
            normalized = (ballistic_pos[0] / frame_width, ballistic_pos[1] / frame_height)
            # Clamp to valid range
            normalized = (max(0, min(1, normalized[0])), max(0, min(1, normalized[1])))
            
            self.last_position[hand] = normalized
            self.last_confidence[hand] = ballistic_conf
            self.tracking_source[hand] = 'ballistic'
            self.source_counts['ballistic'] += 1
            
            return normalized, 'ballistic', ballistic_conf
        
        # CASE D: All failed, return last known position
        self.tracking_source[hand] = 'lost'
        self.source_counts['lost'] += 1
        return self.last_position[hand], 'lost', 0.0
    
    def get_status(self, hand):
        """Get current tracking status for a hand."""
        return {
            'source': self.tracking_source[hand],
            'confidence': self.last_confidence[hand],
            'position': self.last_position[hand],
            'velocity': self.ballistic[hand].get_velocity()
        }
    
    def get_color(self, hand):
        """Get the color for this hand's current tracking source."""
        return TRACKING_COLORS.get(self.tracking_source[hand], (255, 255, 255))
    
    def get_stats(self):
        """Get tracking source statistics."""
        total = sum(self.source_counts.values())
        if total == 0:
            return self.source_counts
        
        return {
            k: f"{v} ({v/total*100:.1f}%)" 
            for k, v in self.source_counts.items()
        }
    
    def reset(self, hand=None):
        """Reset tracker(s)."""
        hands = [hand] if hand else ['Left', 'Right']
        for h in hands:
            self.flow_tracker[h].reset()
            self.ballistic[h].reset()
            self.last_position[h] = None
            self.last_confidence[h] = 0.0
            self.tracking_source[h] = 'none'


class HandTrackingFusion:
    """
    Complete hand tracking fusion system.
    Wraps FlowGatedValidator for easy integration.
    """
    
    def __init__(self):
        self.validator = FlowGatedValidator()
        self.prev_gray = None
    
    def process(self, frame, hand_data, mp_confidence=0.8):
        """
        Process frame with sensor fusion.
        
        Args:
            frame: BGR frame
            hand_data: Dict with 'Left' and 'Right' hand data
            mp_confidence: Default MediaPipe confidence
        
        Returns:
            fused_data: Enhanced hand data with fusion results
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        h, w = frame.shape[:2]
        timestamp = time.time()
        
        fused_data = {}
        
        for hand in ['Left', 'Right']:
            data = hand_data.get(hand, {})
            is_visible = data.get('visible', False)
            
            if is_visible:
                mp_pos = (data['x'], data['y'])
                confidence = data.get('confidence', mp_confidence)
            else:
                mp_pos = None
                confidence = 0.0
            
            # Run fusion
            fused_pos, source, fused_conf = self.validator.validate(
                hand, mp_pos, confidence, gray, w, h, timestamp
            )
            
            # Build fused result
            if fused_pos is not None:
                fused_data[hand] = {
                    'x': fused_pos[0],
                    'y': fused_pos[1],
                    'z': data.get('z', 0),
                    'visible': True,
                    'gesture': data.get('gesture', 'UNKNOWN'),
                    'source': source,
                    'confidence': fused_conf,
                    'color': self.validator.get_color(hand),
                }
            else:
                fused_data[hand] = {
                    'x': 0,
                    'y': 0,
                    'z': 0,
                    'visible': False,
                    'gesture': 'UNKNOWN',
                    'source': 'lost',
                    'confidence': 0.0,
                    'color': TRACKING_COLORS['lost'],
                }
        
        self.prev_gray = gray
        return fused_data
    
    def get_stats(self):
        """Get tracking statistics."""
        return self.validator.get_stats()


def test_flow_gated_validator():
    """Test the flow-gated validator with simulated data."""
    print("=" * 60)
    print("FLOW-GATED VALIDATOR TEST")
    print("=" * 60)
    
    validator = FlowGatedValidator()
    
    # Create a simple test frame
    frame = np.zeros((480, 640), dtype=np.uint8)
    
    print("\n1. Testing MediaPipe confidence...")
    # High confidence - should trust MediaPipe
    pos, source, conf = validator.validate(
        'Right', (0.5, 0.5), 0.9, frame, 640, 480, time.time()
    )
    print(f"   Position: {pos}, Source: {source}, Confidence: {conf:.2f}")
    assert source == 'mediapipe', "Should trust MediaPipe when confident"
    
    print("\n2. Testing low confidence fallback...")
    # Low confidence, no prior tracking - should go to lost
    time.sleep(0.05)
    pos, source, conf = validator.validate(
        'Right', (0.6, 0.5), 0.2, frame, 640, 480, time.time()
    )
    print(f"   Position: {pos}, Source: {source}, Confidence: {conf:.2f}")
    
    print("\n3. Testing ballistic prediction...")
    # Add more history for ballistic
    for i in range(5):
        validator.validate('Right', (0.5 + i*0.1, 0.5), 0.9, frame, 640, 480, time.time())
        time.sleep(0.05)
    
    # Now fail MediaPipe - should use ballistic
    time.sleep(0.05)
    pos, source, conf = validator.validate(
        'Right', None, 0.0, frame, 640, 480, time.time()
    )
    print(f"   Position: {pos}, Source: {source}, Confidence: {conf:.2f}")
    
    print("\n4. Source statistics:")
    stats = validator.get_stats()
    for source, count in stats.items():
        print(f"   {source}: {count}")
    
    print("\n" + "=" * 60)
    print("TEST COMPLETE")
    print("=" * 60)


def test_with_camera():
    """Test with live camera to visualize tracking sources."""
    try:
        import mediapipe as mp
    except ImportError:
        print("MediaPipe not installed!")
        return
    
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    
    validator = FlowGatedValidator()
    
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    print("=" * 60)
    print("FLOW-GATED VALIDATOR CAMERA TEST")
    print("=" * 60)
    print("Color Legend:")
    print("  GREEN  = MediaPipe (high confidence)")
    print("  ORANGE = Optical Flow (bridging blur)")
    print("  RED    = Ballistic (prediction)")
    print("  GRAY   = Lost tracking")
    print("Press Q to quit")
    print("=" * 60)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Detect hands with MediaPipe
        results = hands.process(rgb)
        
        hand_data = {'Left': {'visible': False}, 'Right': {'visible': False}}
        
        if results.multi_hand_landmarks and results.multi_handedness:
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                label = handedness.classification[0].label
                wrist = hand_landmarks.landmark[0]
                hand_data[label] = {
                    'x': wrist.x,
                    'y': wrist.y,
                    'visible': True,
                    'confidence': handedness.classification[0].score
                }
        
        # Run fusion
        timestamp = time.time()
        
        for hand in ['Left', 'Right']:
            data = hand_data[hand]
            if data['visible']:
                mp_pos = (data['x'], data['y'])
                mp_conf = data['confidence']
            else:
                mp_pos = None
                mp_conf = 0.0
            
            fused_pos, source, fused_conf = validator.validate(
                hand, mp_pos, mp_conf, gray, w, h, timestamp
            )
            
            # Draw result
            if fused_pos is not None:
                px = int(fused_pos[0] * w)
                py = int(fused_pos[1] * h)
                color = validator.get_color(hand)
                
                # Draw circle
                cv2.circle(frame, (px, py), 30, color, -1)
                cv2.circle(frame, (px, py), 32, (255, 255, 255), 2)
                
                # Draw label
                cv2.putText(frame, f"{hand[0]}: {source[:4].upper()}", 
                           (px - 30, py - 40),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # Draw legend
        y = 30
        for source_name, color in TRACKING_COLORS.items():
            cv2.rectangle(frame, (10, y-15), (30, y+5), color, -1)
            cv2.putText(frame, source_name.upper(), (40, y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            y += 25
        
        cv2.imshow("Flow-Gated Validator Test", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    
    print("\nFinal Statistics:")
    stats = validator.get_stats()
    for source, count in stats.items():
        print(f"  {source}: {count}")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--camera":
        test_with_camera()
    else:
        test_flow_gated_validator()
        print("\nRun with --camera flag for live camera test")
