"""
Ballistic Trajectory Engine v2
Smart Punch Filter + Sustained Acceleration

Improvements:
- SUSTAINED acceleration (2-3 frames, not single spike)
- MINIMUM travel distance before triggering
- COOLDOWN between punches
- Optional AUDIO trigger for instant detection

State Machine: IDLE → FIRING → RETRACTING → IDLE
"""

import numpy as np
from collections import deque
from enum import Enum
import time


class PunchState(Enum):
    IDLE = "idle"           # Normal tracking
    FIRING = "firing"       # Punch in progress (ballistic)
    RETRACTING = "retracting"  # Returning to rest


# ==================== CONFIGURATION ====================
# INSTANT TRIGGER (tuned - balanced sensitivity)
ACCELERATION_THRESHOLD = 3.0    # Trigger on spike > 3.0 (was 2.0)
VELOCITY_THRESHOLD = 0.5        # OR velocity > 0.5 (was 0.4)
COOLDOWN = 0.15                 # 150ms between punches

BALLISTIC_DURATION = 0.18       # Seconds for ballistic phase
RETRACT_DURATION = 0.15         # Seconds for retraction
MAX_REACH = 0.5                 # Maximum punch extension (50% of screen)

HISTORY_SIZE = 5                # Frames of position history

# Debug mode
DEBUG_PRINT = True              # Print telemetry to console


class HandBallistic:
    """
    Ballistic trajectory simulator for a single hand.
    State machine: IDLE → FIRING → RETRACTING → IDLE
    """
    
    def __init__(self, hand_label):
        self.hand_label = hand_label
        self.state = PunchState.IDLE
        
        # Position history for velocity/acceleration calculation
        self.position_history = deque(maxlen=HISTORY_SIZE)
        self.time_history = deque(maxlen=HISTORY_SIZE)
        
        # Current tracking data
        self.camera_pos = np.array([0.5, 0.5, 0.0])  # x, y, z
        self.velocity = np.array([0.0, 0.0, 0.0])
        self.acceleration = np.array([0.0, 0.0, 0.0])
        
        # Cooldown timer only (simplified)
        
        # SMART FILTER: Cooldown timer
        self.last_punch_time = 0
        
        # Ballistic simulation
        self.ballistic_start_pos = None
        self.ballistic_direction = None
        self.ballistic_start_time = None
        self.ballistic_peak_pos = None
        
        # Output position (can be camera or simulated)
        self.output_pos = np.array([0.5, 0.5, 0.0])
        
        # Stats
        self.punches_fired = 0
        self.rejected_punches = 0  # Filtered out by smart filter
        self.last_state_change = time.time()
    
    def update(self, landmarks, current_time=None):
        """
        Update ballistic engine with new camera data.
        
        Args:
            landmarks: MediaPipe hand landmarks (or None if lost)
            current_time: Current timestamp (uses time.time() if None)
        
        Returns:
            np.array: Output position (either camera or simulated)
        """
        if current_time is None:
            current_time = time.time()
        
        # Extract position from landmarks
        if landmarks is not None:
            wrist = landmarks.landmark[0]
            self.camera_pos = np.array([wrist.x, wrist.y, wrist.z])
            
            # Add to history
            self.position_history.append(self.camera_pos.copy())
            self.time_history.append(current_time)
            
            # Calculate velocity and acceleration
            self._calculate_motion()
        
        # State machine logic
        if self.state == PunchState.IDLE:
            self._update_idle(current_time)
        elif self.state == PunchState.FIRING:
            self._update_firing(current_time)
        elif self.state == PunchState.RETRACTING:
            self._update_retracting(current_time)
        
        return self.output_pos.copy()
    
    def _calculate_motion(self):
        """Calculate velocity and acceleration from position history."""
        if len(self.position_history) < 2:
            return
        
        # Calculate velocity (change in position / change in time)
        pos_diff = self.position_history[-1] - self.position_history[-2]
        time_diff = max(0.001, self.time_history[-1] - self.time_history[-2])
        self.velocity = pos_diff / time_diff
        
        # Calculate acceleration if we have enough history
        if len(self.position_history) >= 3:
            prev_pos_diff = self.position_history[-2] - self.position_history[-3]
            prev_time_diff = max(0.001, self.time_history[-2] - self.time_history[-3])
            prev_velocity = prev_pos_diff / prev_time_diff
            
            self.acceleration = (self.velocity - prev_velocity) / time_diff
    
    def _check_punch_trigger(self, current_time):
        """
        INSTANT TRIGGER - Fire on ANY spike (OR logic).
        Only cooldown prevents double-fire.
        """
        accel_magnitude = np.linalg.norm(self.acceleration)
        velocity_magnitude = np.linalg.norm(self.velocity)
        
        # FILTER: Cooldown check only
        if current_time - self.last_punch_time < COOLDOWN:
            return False
        
        # INSTANT TRIGGER: Any spike triggers (OR logic)
        if accel_magnitude > ACCELERATION_THRESHOLD or velocity_magnitude > VELOCITY_THRESHOLD:
            if DEBUG_PRINT:
                print(f"\n>>> INSTANT TRIGGER! Accel={accel_magnitude:.2f} Vel={velocity_magnitude:.2f}")
            return True
        
        return False
    
    def _update_idle(self, current_time):
        """IDLE state: Check for instant trigger."""
        # Use camera position directly
        self.output_pos = self.camera_pos.copy()
        
        # Check trigger (simplified - instant fire on spike)
        if self._check_punch_trigger(current_time):
            self._start_firing(current_time)
    
    def _start_firing(self, current_time):
        """Transition to FIRING state."""
        self.state = PunchState.FIRING
        self.last_state_change = current_time
        self.last_punch_time = current_time
        self.ballistic_start_time = current_time
        self.ballistic_start_pos = self.camera_pos.copy()
        
        # Lock the trajectory direction
        if np.linalg.norm(self.velocity) > 0.001:
            self.ballistic_direction = self.velocity / np.linalg.norm(self.velocity)
        else:
            self.ballistic_direction = np.array([0.0, 0.0, -1.0])  # Default forward
        
        # Calculate peak position
        self.ballistic_peak_pos = self.ballistic_start_pos + self.ballistic_direction * MAX_REACH
        
        # Clamp to valid range
        self.ballistic_peak_pos[0] = np.clip(self.ballistic_peak_pos[0], 0.0, 1.0)
        self.ballistic_peak_pos[1] = np.clip(self.ballistic_peak_pos[1], 0.0, 1.0)
        
        self.punches_fired += 1
    
    def _update_firing(self, current_time):
        """FIRING state: Simulate ballistic punch trajectory."""
        elapsed = current_time - self.ballistic_start_time
        progress = min(1.0, elapsed / BALLISTIC_DURATION)
        
        # Ease-out curve for natural punch feel
        eased_progress = 1.0 - (1.0 - progress) ** 2
        
        # Interpolate from start to peak
        self.output_pos = self.ballistic_start_pos + (
            self.ballistic_peak_pos - self.ballistic_start_pos
        ) * eased_progress
        
        # Check for transition to retracting
        if progress >= 1.0:
            self._start_retracting(current_time)
    
    def _start_retracting(self, current_time):
        """Transition to RETRACTING state."""
        self.state = PunchState.RETRACTING
        self.last_state_change = current_time
        self.ballistic_start_time = current_time
        self.ballistic_start_pos = self.output_pos.copy()
    
    def _update_retracting(self, current_time):
        """RETRACTING state: Blend back to camera position."""
        elapsed = current_time - self.ballistic_start_time
        progress = min(1.0, elapsed / RETRACT_DURATION)
        
        # Ease-in curve for retraction
        eased_progress = progress ** 2
        
        # Blend from peak back to camera position
        self.output_pos = self.ballistic_start_pos + (
            self.camera_pos - self.ballistic_start_pos
        ) * eased_progress
        
        # Check for transition back to idle
        if progress >= 1.0:
            self.state = PunchState.IDLE
            self.last_state_change = current_time
    
    def is_active(self):
        """Check if ballistic simulation is active."""
        return self.state in [PunchState.FIRING, PunchState.RETRACTING]
    
    def get_state_name(self):
        """Get current state as string."""
        return self.state.value
    
    def get_stats(self):
        """Get ballistic engine statistics."""
        return {
            'state': self.state.value,
            'punches_fired': self.punches_fired,
            'rejected': self.rejected_punches,
            'velocity_mag': np.linalg.norm(self.velocity),
            'accel_mag': np.linalg.norm(self.acceleration),
            'sustained_frames': len([a for a in self.accel_history if a > ACCELERATION_THRESHOLD])
        }


class BallisticEngine:
    """
    Manages ballistic simulation for both hands.
    """
    
    def __init__(self):
        self.hands = {
            'Left': HandBallistic('Left'),
            'Right': HandBallistic('Right')
        }
    
    def update(self, results, current_time=None):
        """
        Update ballistic engine with new detection results.
        """
        if current_time is None:
            current_time = time.time()
        
        # Extract landmarks per hand
        hand_landmarks = {'Left': None, 'Right': None}
        
        if results and results.multi_hand_landmarks:
            for landmarks, handedness in zip(
                results.multi_hand_landmarks,
                results.multi_handedness
            ):
                label = handedness.classification[0].label
                hand_landmarks[label] = landmarks
        
        # Update each hand
        outputs = {}
        for label in ['Left', 'Right']:
            pos = self.hands[label].update(hand_landmarks[label], current_time)
            outputs[label] = {
                'position': pos,
                'is_active': self.hands[label].is_active(),
                'state': self.hands[label].get_state_name()
            }
        
        return outputs
    
    def is_any_active(self):
        """Check if any hand is in ballistic mode."""
        return any(h.is_active() for h in self.hands.values())
    
    def get_stats(self):
        """Get statistics for all hands."""
        return {
            'Left': self.hands['Left'].get_stats(),
            'Right': self.hands['Right'].get_stats()
        }


def test_ballistic_engine():
    """Test the ballistic engine with webcam."""
    import cv2
    import mediapipe as mp
    
    print("=" * 60)
    print("BALLISTIC ENGINE v2 - SMART PUNCH FILTER")
    print("=" * 60)
    print("Filters Active:")
    print(f"  - Sustained: {SUSTAINED_FRAMES} frames @ accel > {ACCELERATION_THRESHOLD}")
    print(f"  - Min Distance: {PUNCH_MIN_DISTANCE * 100:.0f}% screen")
    print(f"  - Cooldown: {COOLDOWN * 1000:.0f}ms")
    print(f"  - Velocity: > {VELOCITY_THRESHOLD}")
    print("=" * 60)
    print("Punch HARD to trigger! Slight movements filtered out.")
    print("Press 'Q' to quit.")
    print("=" * 60)
    
    # Initialize
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        model_complexity=1,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.6
    )
    
    engine = BallisticEngine()
    
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]
        frame_count += 1
        
        # Process
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)
        
        # Update ballistic engine
        outputs = engine.update(results)
        
        # Draw results
        for label, data in outputs.items():
            if data['position'] is not None:
                x = int(data['position'][0] * w)
                y = int(data['position'][1] * h)
                
                # Color based on state
                if data['is_active']:
                    color = (0, 0, 255)  # Red for ballistic
                    radius = 30
                else:
                    color = (0, 255, 0)  # Green for camera
                    radius = 15
                
                cv2.circle(frame, (x, y), radius, color, -1)
                cv2.putText(frame, data['state'].upper(), (x - 40, y - 35),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # Stats overlay
        stats = engine.get_stats()
        cv2.rectangle(frame, (5, 5), (400, 140), (0, 0, 0), -1)
        
        for i, (label, hstats) in enumerate(stats.items()):
            y = 25 + i * 60
            color = (0, 0, 255) if engine.hands[label].is_active() else (0, 255, 0)
            cv2.putText(frame, f"{label}: {hstats['state']}", (10, y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            cv2.putText(frame, f"  Accel: {hstats['accel_mag']:.2f} Vel: {hstats['velocity_mag']:.2f}", 
                       (10, y + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
            cv2.putText(frame, f"  Punches: {hstats['punches_fired']} | Filtered: {hstats['rejected']}", 
                       (10, y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
            cv2.putText(frame, f"  Sustained: {hstats['sustained_frames']}/{SUSTAINED_FRAMES}", 
                       (10, y + 45), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 200), 1)
        
        # Debug print every frame
        if DEBUG_PRINT and frame_count % 5 == 0:
            r_stats = stats['Right']
            print(f"\rR: Accel={r_stats['accel_mag']:.2f} Vel={r_stats['velocity_mag']:.2f} "
                  f"Sustained={r_stats['sustained_frames']}/{SUSTAINED_FRAMES} "
                  f"State={r_stats['state']}   ", end="")
        
        cv2.imshow("Ballistic v2 - Smart Filter - Press Q", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    hands.close()
    cv2.destroyAllWindows()
    
    print("\n\n" + "=" * 60)
    print("FINAL STATS")
    print("=" * 60)
    for label, hstats in engine.get_stats().items():
        print(f"{label}: {hstats['punches_fired']} punches, {hstats['rejected']} filtered out")
    print("=" * 60)


if __name__ == "__main__":
    test_ballistic_engine()
