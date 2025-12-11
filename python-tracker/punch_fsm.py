"""
Ballistic State Machine for Punch Detection
============================================
Models the temporal morphology of a punch as a lifecycle of states.

States:
  IDLE → ACCELERATION → EXTENSION → RETRACTION → COOLDOWN → IDLE

Transitions are driven by a weighted voting system that polls:
1. Skeleton Voter (velocity from position delta)
2. Flow Voter (optical flow magnitude)
3. Geometry Voter (AEC extension score)

Reference: "Punch Detection Sensor Fusion Architecture" Section 6
"""

import numpy as np
from enum import Enum
from collections import deque
import time


class PunchState(Enum):
    """Punch lifecycle states."""
    IDLE = "idle"              # Guard stance
    CHAMBER = "chamber"        # Preparation (optional)
    ACCELERATION = "accel"     # Launch phase
    EXTENSION = "extend"       # Impact zone
    RETRACTION = "retract"     # Return phase
    COOLDOWN = "cooldown"      # Reset buffer


# State colors for visualization
STATE_COLORS = {
    PunchState.IDLE: (128, 128, 128),      # Gray
    PunchState.CHAMBER: (255, 255, 0),      # Yellow
    PunchState.ACCELERATION: (0, 165, 255), # Orange
    PunchState.EXTENSION: (0, 0, 255),      # Red (PUNCH!)
    PunchState.RETRACTION: (255, 0, 255),   # Magenta
    PunchState.COOLDOWN: (255, 165, 0),     # Blue-orange
}


class SkeletonVoter:
    """
    Votes based on wrist velocity derived from position delta.
    """
    def __init__(self, velocity_threshold=0.5):
        self.velocity_threshold = velocity_threshold
        self.position_history = deque(maxlen=5)
        self.last_time = None
    
    def update(self, wrist_position, timestamp):
        """Update with new position."""
        if wrist_position is not None:
            self.position_history.append({
                'pos': np.array(wrist_position),
                'time': timestamp
            })
        self.last_time = timestamp
    
    def vote(self):
        """
        Return vote score 0-1 based on velocity.
        High velocity = high vote for punch.
        """
        if len(self.position_history) < 2:
            return 0.0
        
        p1 = self.position_history[-2]
        p2 = self.position_history[-1]
        
        dt = p2['time'] - p1['time']
        if dt <= 0:
            return 0.0
        
        velocity = np.linalg.norm(p2['pos'] - p1['pos']) / dt
        
        # Normalize to 0-1
        return min(1.0, velocity / self.velocity_threshold)
    
    def get_velocity(self):
        """Get current velocity magnitude."""
        if len(self.position_history) < 2:
            return 0.0
        
        p1 = self.position_history[-2]
        p2 = self.position_history[-1]
        dt = p2['time'] - p1['time']
        
        if dt <= 0:
            return 0.0
        
        return np.linalg.norm(p2['pos'] - p1['pos']) / dt
    
    def get_velocity_vector(self):
        """Get velocity as (vx, vy) vector."""
        if len(self.position_history) < 2:
            return np.zeros(2)
        
        p1 = self.position_history[-2]
        p2 = self.position_history[-1]
        dt = p2['time'] - p1['time']
        
        if dt <= 0:
            return np.zeros(2)
        
        return (p2['pos'] - p1['pos']) / dt
    
    def reset(self):
        """Reset voter state."""
        self.position_history.clear()


class FlowVoter:
    """
    Votes based on optical flow magnitude.
    """
    def __init__(self, flow_threshold=20.0):
        self.flow_threshold = flow_threshold
        self.current_magnitude = 0.0
    
    def update(self, flow_magnitude):
        """Update with flow magnitude from Flow-Gated Validator."""
        self.current_magnitude = flow_magnitude
    
    def vote(self):
        """
        Return vote score 0-1 based on flow.
        High flow = high vote for punch.
        """
        return min(1.0, self.current_magnitude / self.flow_threshold)
    
    def reset(self):
        """Reset voter state."""
        self.current_magnitude = 0.0


class ArmVelocityVoter:
    """
    Votes based on arm extension velocity (d(arm_length)/dt).
    More robust than wrist position velocity alone.
    """
    def __init__(self, velocity_threshold=0.3):
        self.velocity_threshold = velocity_threshold
        self.current_velocity = 0.0
    
    def update(self, arm_velocity):
        """Update with arm velocity from master_vision."""
        self.current_velocity = arm_velocity if arm_velocity else 0.0
    
    def vote(self):
        """
        Return vote score 0-1 based on arm extension velocity.
        Only vote positive for EXTENDING (positive velocity).
        """
        if self.current_velocity > 0:
            return min(1.0, self.current_velocity / self.velocity_threshold)
        return 0.0
    
    def reset(self):
        """Reset voter state."""
        self.current_velocity = 0.0


class GeometryVoter:
    """
    Votes based on Arm Extension Coefficient (AEC).
    """
    def __init__(self):
        self.aec_data = None
    
    def update(self, aec_data):
        """Update with AEC data from depth_estimator."""
        self.aec_data = aec_data
    
    def vote(self):
        """
        Return vote score 0-1 based on extension.
        Extended arm = high vote for punch.
        """
        if self.aec_data is None:
            return 0.0
        
        # Score based on reach percentage
        reach_score = self.aec_data.get('reach_percent', 0) / 100.0
        
        # Bonus if arm is straight
        if self.aec_data.get('is_arm_straight', False):
            reach_score = min(1.0, reach_score + 0.2)
        
        # Penalty if vertical (not a punch direction)
        if not self.aec_data.get('is_horizontal', True):
            reach_score *= 0.3
        
        return reach_score
    
    def is_extended(self):
        """Check if arm is fully extended."""
        if self.aec_data is None:
            return False
        return self.aec_data.get('is_extended', False)
    
    def is_horizontal(self):
        """Check if movement is horizontal."""
        if self.aec_data is None:
            return True
        return self.aec_data.get('is_horizontal', True)
    
    def reset(self):
        """Reset voter state."""
        self.aec_data = None


class PunchStateMachine:
    """
    Finite State Machine for punch detection.
    
    Transitions are driven by weighted voting from multiple sensors.
    This replaces brittle single-threshold detection.
    """
    
    def __init__(self, 
                 accel_threshold=0.30,     # Vote threshold to enter ACCEL (lowered)
                 extend_threshold=0.45,    # Vote threshold to enter EXTEND (lowered for 9/10)
                 cooldown_frames=6):       # Frames to stay in COOLDOWN (faster reset)
        
        self.state = PunchState.IDLE
        self.accel_threshold = accel_threshold
        self.extend_threshold = extend_threshold
        self.cooldown_frames = cooldown_frames
        
        # Voters (4 voters now)
        self.skeleton_voter = SkeletonVoter()
        self.flow_voter = FlowVoter()
        self.arm_voter = ArmVelocityVoter()
        self.geometry_voter = GeometryVoter()
        
        # Weights for voting (must sum to 1.0)
        self.weights = {
            'skeleton': 0.2,     # Wrist position velocity
            'arm': 0.3,          # Arm extension velocity (NEW)
            'flow': 0.2,         # Optical flow
            'geometry': 0.3      # AEC
        }
        
        # State tracking
        self.cooldown_counter = 0
        self.accel_start_time = None
        self.extension_time = None
        self.punch_events = deque(maxlen=100)
        
        # Debug
        self.last_probability = 0.0
        self.last_votes = {}
    
    def calculate_punch_probability(self):
        """
        Calculate weighted vote from all sensors.
        
        P_punch = w1*V_skeleton + w2*V_arm + w3*V_flow + w4*V_geometry
        """
        skeleton_vote = self.skeleton_voter.vote()
        arm_vote = self.arm_voter.vote()
        flow_vote = self.flow_voter.vote()
        geometry_vote = self.geometry_voter.vote()
        
        probability = (
            self.weights['skeleton'] * skeleton_vote +
            self.weights['arm'] * arm_vote +
            self.weights['flow'] * flow_vote +
            self.weights['geometry'] * geometry_vote
        )
        
        votes = {
            'skeleton': skeleton_vote,
            'arm': arm_vote,
            'flow': flow_vote,
            'geometry': geometry_vote
        }
        
        self.last_probability = probability
        self.last_votes = votes
        
        return probability, votes
    
    def update(self, wrist_position, flow_magnitude=0.0, aec_data=None, timestamp=None, arm_velocity=0.0):
        """
        Update FSM with new sensor data.
        
        Args:
            wrist_position: (x, y) normalized 0-1
            flow_magnitude: Optical flow magnitude
            aec_data: Dict from AnchorDepthEstimator
            timestamp: Current time
            arm_velocity: Arm extension velocity (d(arm_length)/dt)
            
        Returns:
            punch_event: 'PUNCH' if punch detected, None otherwise
            state: Current FSM state
        """
        if timestamp is None:
            timestamp = time.time()
        
        # Update voters
        self.skeleton_voter.update(wrist_position, timestamp)
        self.flow_voter.update(flow_magnitude)
        self.arm_voter.update(arm_velocity)
        self.geometry_voter.update(aec_data)
        
        # Calculate probability
        probability, votes = self.calculate_punch_probability()
        
        punch_event = None
        prev_state = self.state
        
        # QUICK PUNCH: If arm velocity is extremely high + good reach, bypass ACCEL
        arm_vote = votes.get('arm', 0)
        reach_percent = aec_data.get('reach_percent', 0) if aec_data else 0
        is_horizontal = aec_data.get('is_horizontal', True) if aec_data else True
        
        if arm_vote > 0.85 and reach_percent > 55 and is_horizontal:
            if self.state == PunchState.IDLE or self.state == PunchState.ACCELERATION:
                self.state = PunchState.EXTENSION
                self.extension_time = timestamp
                punch_event = 'PUNCH'
                self.punch_events.append({
                    'time': timestamp,
                    'probability': probability,
                    'votes': votes.copy(),
                    'quick_punch': True
                })
                print(f"[QUICK PUNCH] Fast velocity bypass! arm={arm_vote:.2f} reach={reach_percent:.0f}%")
        
        # Normal state transitions (skip if quick punch already triggered)
        if punch_event is None:
            if self.state == PunchState.IDLE:
                # Transition to ACCELERATION if probability high
                if probability > self.accel_threshold:
                    self.state = PunchState.ACCELERATION
                    self.accel_start_time = timestamp
        
            elif self.state == PunchState.ACCELERATION:
                # Check for EXTENSION (arm fully extended)
                is_extended = self.geometry_voter.is_extended()
                is_horizontal = self.geometry_voter.is_horizontal()
                
                if is_extended and is_horizontal:
                    self.state = PunchState.EXTENSION
                    self.extension_time = timestamp
                    punch_event = 'PUNCH'
                    self.punch_events.append({
                        'time': timestamp,
                        'probability': probability,
                        'votes': votes.copy()
                    })
                elif probability < self.accel_threshold * 0.3:  # More lenient abort
                    # Motion stopped without extension - abort
                    self.state = PunchState.IDLE
                elif timestamp - self.accel_start_time > 0.6:  # Slightly longer timeout
                    # Timeout (600ms without reaching extension)
                    self.state = PunchState.IDLE
        
        elif self.state == PunchState.EXTENSION:
            # Transition to RETRACTION when velocity decreases or arm retracts
            velocity = self.skeleton_voter.get_velocity()
            is_extended = self.geometry_voter.is_extended()
            
            if not is_extended or velocity < 0.1:
                self.state = PunchState.RETRACTION
        
        elif self.state == PunchState.RETRACTION:
            # Transition to COOLDOWN when hand returns
            if probability < 0.2:
                self.state = PunchState.COOLDOWN
                self.cooldown_counter = self.cooldown_frames
        
        elif self.state == PunchState.COOLDOWN:
            # Wait before returning to IDLE
            self.cooldown_counter -= 1
            if self.cooldown_counter <= 0:
                self.state = PunchState.IDLE
        
        # Log state transition
        if self.state != prev_state:
            print(f"[FSM] {prev_state.value} → {self.state.value} (P={probability:.2f})")
        
        return punch_event, self.state
    
    def get_debug_info(self):
        """Get debug information for display."""
        return {
            'state': self.state.value,
            'state_enum': self.state,
            'probability': self.last_probability,
            'votes': self.last_votes,
            'punch_count': len(self.punch_events),
            'velocity': self.skeleton_voter.get_velocity(),
        }
    
    def get_color(self):
        """Get color for current state."""
        return STATE_COLORS.get(self.state, (255, 255, 255))
    
    def reset(self):
        """Reset FSM to initial state."""
        self.state = PunchState.IDLE
        self.cooldown_counter = 0
        self.accel_start_time = None
        self.extension_time = None
        self.skeleton_voter.reset()
        self.flow_voter.reset()
        self.arm_voter.reset()
        self.geometry_voter.reset()


class PunchDetector:
    """
    High-level punch detection combining all components.
    
    Usage:
        detector = PunchDetector()
        punch_event, state = detector.update('Right', wrist, flow, aec, time)
        if punch_event == 'PUNCH':
            handle_punch()
    """
    
    def __init__(self):
        self.fsm = {
            'Left': PunchStateMachine(),
            'Right': PunchStateMachine()
        }
        self.total_punches = {'Left': 0, 'Right': 0}
    
    def update(self, hand, wrist_position, flow_magnitude=0.0, aec_data=None, timestamp=None, arm_velocity=0.0):
        """Update detector and check for punch."""
        punch_event, state = self.fsm[hand].update(
            wrist_position, flow_magnitude, aec_data, timestamp, arm_velocity
        )
        
        if punch_event == 'PUNCH':
            self.total_punches[hand] += 1
            print(f"[PUNCH] {hand} hand! Total: {self.total_punches[hand]}")
        
        return punch_event, state
    
    def get_state(self, hand):
        """Get current state for a hand."""
        return self.fsm[hand].state
    
    def get_debug(self, hand):
        """Get debug info for a hand."""
        return self.fsm[hand].get_debug_info()
    
    def get_color(self, hand):
        """Get color for hand's current state."""
        return self.fsm[hand].get_color()
    
    def get_punch_counts(self):
        """Get total punch counts."""
        return self.total_punches.copy()
    
    def reset(self, hand=None):
        """Reset FSM(s)."""
        if hand:
            self.fsm[hand].reset()
        else:
            self.fsm['Left'].reset()
            self.fsm['Right'].reset()


def test_punch_fsm():
    """Test the punch state machine with simulated data."""
    print("=" * 60)
    print("PUNCH STATE MACHINE TEST")
    print("=" * 60)
    
    fsm = PunchStateMachine()
    
    print("\n1. Testing IDLE state...")
    # Low movement should stay IDLE
    for i in range(5):
        _, state = fsm.update((0.5, 0.5), 0.0, None, time.time())
        time.sleep(0.033)
    print(f"   State: {state.value} (should be idle)")
    assert state == PunchState.IDLE, "Should stay IDLE with no movement"
    
    print("\n2. Testing IDLE → ACCELERATION transition...")
    # Rapid position change should trigger ACCEL
    for i in range(5):
        pos = (0.5 + i*0.05, 0.5)
        _, state = fsm.update(pos, 10.0, {'reach_percent': 30}, time.time())
        time.sleep(0.033)
    print(f"   State: {state.value} (should be accel)")
    
    print("\n3. Testing ACCELERATION → EXTENSION transition...")
    # Extended arm should trigger EXTENSION
    aec_data = {
        'reach_percent': 90,
        'is_extended': True,
        'is_arm_straight': True,
        'is_horizontal': True
    }
    punch_event, state = fsm.update((0.9, 0.5), 15.0, aec_data, time.time())
    print(f"   State: {state.value}, Punch Event: {punch_event}")
    assert punch_event == 'PUNCH', "Should fire PUNCH event"
    assert state == PunchState.EXTENSION, "Should be in EXTENSION state"
    
    print("\n4. Testing EXTENSION → RETRACTION transition...")
    time.sleep(0.1)
    for i in range(5):
        pos = (0.9 - i*0.05, 0.5)
        aec_data['is_extended'] = False
        _, state = fsm.update(pos, 5.0, aec_data, time.time())
        time.sleep(0.033)
    print(f"   State: {state.value} (should be retract or cooldown)")
    
    print("\n5. Testing RETRACTION → COOLDOWN → IDLE...")
    for i in range(15):
        _, state = fsm.update((0.5, 0.5), 0.0, None, time.time())
        time.sleep(0.033)
    print(f"   State: {state.value} (should be idle)")
    assert state == PunchState.IDLE, "Should return to IDLE"
    
    print("\n6. Testing vertical slide (should NOT trigger punch)...")
    fsm.reset()
    # Vertical movement with non-horizontal AEC
    for i in range(10):
        pos = (0.5, 0.5 - i*0.05)  # Moving up
        aec_data = {
            'reach_percent': 90,
            'is_extended': True,
            'is_arm_straight': True,
            'is_horizontal': False  # VERTICAL - should veto!
        }
        punch_event, state = fsm.update(pos, 15.0, aec_data, time.time())
        if punch_event == 'PUNCH':
            print("   ERROR: Vertical slide triggered punch!")
            break
        time.sleep(0.033)
    else:
        print(f"   State: {state.value}, No punch (CORRECT - geometry veto)")
    
    print("\n7. Debug info:")
    debug = fsm.get_debug_info()
    print(f"   State: {debug['state']}")
    print(f"   Probability: {debug['probability']:.2f}")
    print(f"   Votes: S={debug['votes'].get('skeleton', 0):.2f} "
          f"F={debug['votes'].get('flow', 0):.2f} "
          f"G={debug['votes'].get('geometry', 0):.2f}")
    print(f"   Punch count: {debug['punch_count']}")
    
    print("\n" + "=" * 60)
    print("TEST COMPLETE")
    print("=" * 60)


def test_with_camera():
    """Test with live camera."""
    try:
        import cv2
        import mediapipe as mp
        from depth_estimator import AnchorDepthEstimator
    except ImportError as e:
        print(f"Missing dependency: {e}")
        return
    
    mp_pose = mp.solutions.pose
    mp_draw = mp.solutions.drawing_utils
    
    pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
    depth_estimator = AnchorDepthEstimator()
    punch_detector = PunchDetector()
    
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    print("=" * 60)
    print("PUNCH FSM CAMERA TEST")
    print("=" * 60)
    print("State Legend:")
    print("  GRAY   = IDLE (guard)")
    print("  ORANGE = ACCELERATION (launching)")
    print("  RED    = EXTENSION (punch!)")
    print("  MAGENTA = RETRACTION (returning)")
    print("Press Q to quit")
    print("=" * 60)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        results = pose.process(rgb)
        
        if results.pose_landmarks:
            mp_draw.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            
            timestamp = time.time()
            
            for hand in ['Right', 'Left']:
                # Get wrist position
                idx = 16 if hand == 'Right' else 15
                wrist = results.pose_landmarks.landmark[idx]
                wrist_pos = (wrist.x, wrist.y)
                
                # Get AEC
                aec = depth_estimator.calculate_aec(results.pose_landmarks, hand)
                
                # Update FSM
                punch_event, state = punch_detector.update(
                    hand, wrist_pos, 0.0, aec, timestamp
                )
                
                # Draw wrist marker with state color
                px, py = int(wrist.x * w), int(wrist.y * h)
                color = punch_detector.get_color(hand)
                cv2.circle(frame, (px, py), 25, color, -1)
                cv2.circle(frame, (px, py), 27, (255, 255, 255), 2)
                
                # Draw state label
                debug = punch_detector.get_debug(hand)
                label = f"{hand[0]}: {state.value.upper()}"
                cv2.putText(frame, label, (px - 40, py - 35),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                
                # Draw probability bar
                prob_w = int(debug['probability'] * 100)
                bar_x = 10 if hand == 'Left' else w - 120
                bar_y = 30
                cv2.rectangle(frame, (bar_x, bar_y), (bar_x + 100, bar_y + 20), (50, 50, 50), -1)
                cv2.rectangle(frame, (bar_x, bar_y), (bar_x + prob_w, bar_y + 20), color, -1)
                cv2.putText(frame, f"P:{debug['probability']:.2f}", (bar_x, bar_y + 40),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Draw punch counts
        counts = punch_detector.get_punch_counts()
        cv2.putText(frame, f"Punches - L:{counts['Left']} R:{counts['Right']}", 
                   (w//2 - 100, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        cv2.imshow("Punch FSM Test", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    
    print(f"\nFinal punch counts: L={counts['Left']} R={counts['Right']}")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--camera":
        test_with_camera()
    else:
        test_punch_fsm()
        print("\nRun with --camera flag for live camera test")
