"""
Occlusion Handler - Maintains hand tracking when hands cross/occlude
Uses position prediction and temporal smoothing to recover from temporary tracking loss.
"""

import time
import numpy as np
from collections import deque


class OcclusionHandler:
    """
    Handles hand occlusion using position prediction.
    When a hand is temporarily lost, predicts its position based on recent trajectory.
    """
    
    def __init__(self, timeout=0.5, prediction_frames=10):
        """
        Args:
            timeout: Max time in seconds to predict position (default 0.5s)
            prediction_frames: Max frames to predict before giving up
        """
        self.timeout = timeout
        self.prediction_frames = prediction_frames
        
        # Per-hand state
        self.hand_state = {
            'Left': self._create_hand_state(),
            'Right': self._create_hand_state(),
        }
    
    def _create_hand_state(self):
        """Create initial state for a hand."""
        return {
            'last_seen': 0,
            'last_pos': None,       # (x, y, z)
            'velocity': (0, 0, 0),  # Estimated velocity
            'position_history': deque(maxlen=10),
            'status': 'LOST',       # VISIBLE, OCCLUDED, LOST
            'frames_predicted': 0,
            'gesture': 'UNKNOWN',
        }
    
    def update(self, detected_hands, current_time=None):
        """
        Update hand tracking with occlusion handling.
        
        Args:
            detected_hands: {
                'Left': {'x': 0.5, 'y': 0.5, 'z': 0, 'visible': True, 'gesture': 'FIST'},
                'Right': {...}
            }
            current_time: Current timestamp (uses time.time() if None)
        
        Returns:
            dict: Always returns both hands, with predicted positions if occluded
        """
        if current_time is None:
            current_time = time.time()
        
        result = {'Left': {}, 'Right': {}}
        
        for hand in ['Left', 'Right']:
            state = self.hand_state[hand]
            hand_data = detected_hands.get(hand, {})
            is_visible = hand_data.get('visible', False)
            
            if is_visible:
                # Hand is visible - update state
                x = hand_data.get('x', 0)
                y = hand_data.get('y', 0)
                z = hand_data.get('z', 0)
                
                # Calculate velocity from history
                if state['last_pos'] is not None:
                    dt = current_time - state['last_seen']
                    if dt > 0 and dt < 0.5:  # Reasonable time gap
                        vx = (x - state['last_pos'][0]) / dt
                        vy = (y - state['last_pos'][1]) / dt
                        vz = (z - state['last_pos'][2]) / dt
                        # Smooth velocity
                        alpha = 0.7
                        state['velocity'] = (
                            alpha * vx + (1-alpha) * state['velocity'][0],
                            alpha * vy + (1-alpha) * state['velocity'][1],
                            alpha * vz + (1-alpha) * state['velocity'][2],
                        )
                
                state['last_seen'] = current_time
                state['last_pos'] = (x, y, z)
                state['position_history'].append((x, y, z, current_time))
                state['status'] = 'VISIBLE'
                state['frames_predicted'] = 0
                state['gesture'] = hand_data.get('gesture', 'UNKNOWN')
                
                result[hand] = {
                    'x': x,
                    'y': y,
                    'z': z,
                    'visible': True,
                    'gesture': state['gesture'],
                    'status': 'VISIBLE',
                }
            
            else:
                # Hand not detected - check if we should predict
                time_since_seen = current_time - state['last_seen']
                
                if (time_since_seen < self.timeout and 
                    state['last_pos'] is not None and
                    state['frames_predicted'] < self.prediction_frames):
                    
                    # Predict position based on velocity
                    dt = time_since_seen
                    pred_x = state['last_pos'][0] + state['velocity'][0] * dt
                    pred_y = state['last_pos'][1] + state['velocity'][1] * dt
                    pred_z = state['last_pos'][2] + state['velocity'][2] * dt
                    
                    # Clamp to valid range
                    pred_x = max(0, min(1, pred_x))
                    pred_y = max(0, min(1, pred_y))
                    
                    state['status'] = 'OCCLUDED'
                    state['frames_predicted'] += 1
                    
                    result[hand] = {
                        'x': pred_x,
                        'y': pred_y,
                        'z': pred_z,
                        'visible': True,  # Still "visible" with prediction
                        'gesture': state['gesture'],  # Keep last gesture
                        'status': 'OCCLUDED',
                        'predicted': True,
                    }
                
                else:
                    # Gone too long - truly lost
                    state['status'] = 'LOST'
                    state['frames_predicted'] = 0
                    
                    result[hand] = {
                        'x': 0,
                        'y': 0,
                        'z': 0,
                        'visible': False,
                        'gesture': 'UNKNOWN',
                        'status': 'LOST',
                    }
        
        return result
    
    def get_status(self, hand):
        """Get current status of a hand: VISIBLE, OCCLUDED, or LOST."""
        return self.hand_state[hand]['status']
    
    def get_debug_info(self):
        """Get debug info for display."""
        info = {}
        for hand in ['Left', 'Right']:
            state = self.hand_state[hand]
            info[hand] = {
                'status': state['status'],
                'frames_predicted': state['frames_predicted'],
                'velocity': state['velocity'],
            }
        return info
    
    def reset(self, hand=None):
        """Reset state for a hand or all hands."""
        if hand:
            self.hand_state[hand] = self._create_hand_state()
        else:
            for h in ['Left', 'Right']:
                self.hand_state[h] = self._create_hand_state()


class HandCrossingDetector:
    """
    Detects when hands are crossing each other.
    This is important because crossing often causes detection issues.
    """
    
    def __init__(self, cross_threshold=0.15):
        """
        Args:
            cross_threshold: Distance (normalized) at which hands are considered "crossing"
        """
        self.cross_threshold = cross_threshold
        self.is_crossing = False
        self.crossing_start_time = 0
    
    def update(self, left_pos, right_pos, current_time=None):
        """
        Check if hands are crossing.
        
        Args:
            left_pos: (x, y) of left hand
            right_pos: (x, y) of right hand
        
        Returns:
            bool: True if hands are crossing
        """
        if current_time is None:
            current_time = time.time()
        
        if left_pos is None or right_pos is None:
            self.is_crossing = False
            return False
        
        # Calculate distance between hands
        dx = left_pos[0] - right_pos[0]
        dy = left_pos[1] - right_pos[1]
        distance = np.sqrt(dx*dx + dy*dy)
        
        was_crossing = self.is_crossing
        self.is_crossing = distance < self.cross_threshold
        
        if self.is_crossing and not was_crossing:
            self.crossing_start_time = current_time
        
        return self.is_crossing
    
    def get_crossing_duration(self, current_time=None):
        """Get how long hands have been crossing."""
        if not self.is_crossing:
            return 0
        if current_time is None:
            current_time = time.time()
        return current_time - self.crossing_start_time


def test_occlusion_handler():
    """Test the occlusion handler."""
    print("=" * 50)
    print("OCCLUSION HANDLER TEST")
    print("=" * 50)
    
    handler = OcclusionHandler(timeout=0.5)
    
    # Simulate visible hand
    t = 0
    for i in range(5):
        t += 0.033  # ~30fps
        detected = {
            'Right': {'x': 0.5 + i*0.05, 'y': 0.5, 'z': 0.0, 
                     'visible': True, 'gesture': 'FIST'},
            'Left': {'visible': False},
        }
        result = handler.update(detected, t)
        print(f"Frame {i}: Right status = {result['Right']['status']}, x = {result['Right']['x']:.3f}")
    
    # Simulate occlusion (hand disappears)
    print("\n--- Hand disappears ---")
    for i in range(10):
        t += 0.033
        detected = {
            'Right': {'visible': False},
            'Left': {'visible': False},
        }
        result = handler.update(detected, t)
        print(f"Frame {5+i}: Right status = {result['Right']['status']}, "
              f"x = {result['Right']['x']:.3f}, predicted = {result['Right'].get('predicted', False)}")
    
    # Hand comes back
    print("\n--- Hand returns ---")
    detected = {
        'Right': {'x': 0.9, 'y': 0.5, 'z': 0.0, 'visible': True, 'gesture': 'OPEN'},
        'Left': {'visible': False},
    }
    t += 0.033
    result = handler.update(detected, t)
    print(f"Final: Right status = {result['Right']['status']}, x = {result['Right']['x']:.3f}")
    
    print("\n" + "=" * 50)


if __name__ == "__main__":
    test_occlusion_handler()
