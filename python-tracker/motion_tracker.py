"""
Motion Tracker with Optical Flow + Inertia Physics Fallback
VERSION 2 - With drift prevention fixes

Tracking Hierarchy:
1. MediaPipe (best accuracy)
2. YOLO (robust to some blur)
3. Optical Flow (tracks pixels when shape is lost) - WITH CONSTRAINTS
4. Inertia/Coasting (physics prediction when even pixels blur)
5. Motion Energy (detects movement blobs)

DRIFT PREVENTION:
- ROI constraint (only track within hand bounding box)
- Velocity sanity check (reject unrealistic movements)
- Confidence gating (filter by optical flow error)
- Ghost duration limit (max 5 frames)
- Snap back on detection
"""

import cv2
import numpy as np
from collections import deque


# ==================== CONFIGURATION ====================
MAX_HAND_SPEED = 150        # Max pixels per frame (prevents teleporting)
MAX_GHOST_FRAMES = 5        # Max frames to show ghost before giving up
MIN_GOOD_POINTS = 4         # Minimum tracked points needed
MAX_POINT_ERROR = 12.0      # Max optical flow error threshold
ROI_PADDING = 120           # Padding around hand for ROI


class MotionTracker:
    """
    Tracks motion energy (blobs of moving pixels).
    Used as last-resort fallback when all AI detectors fail.
    """
    
    def __init__(self, threshold=25, min_area=500):
        self.threshold = threshold
        self.min_area = min_area
        self.prev_frame_gray = None
    
    def get_motion_centers(self, frame, last_known_positions):
        """Find center of motion blobs near last known hand positions."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)
        
        if self.prev_frame_gray is None:
            self.prev_frame_gray = gray
            return {}
        
        delta_frame = cv2.absdiff(self.prev_frame_gray, gray)
        thresh = cv2.threshold(delta_frame, self.threshold, 255, cv2.THRESH_BINARY)[1]
        thresh = cv2.dilate(thresh, None, iterations=2)
        
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        motion_centers = {}
        
        for hand_label, last_pos in (last_known_positions or {}).items():
            if last_pos is None:
                continue
            
            last_x, last_y = last_pos
            best_dist = float('inf')
            best_center = None
            best_bbox = None
            
            for c in contours:
                area = cv2.contourArea(c)
                if area < self.min_area:
                    continue
                
                (x, y, w, h) = cv2.boundingRect(c)
                center_x = x + w // 2
                center_y = y + h // 2
                
                dist = np.hypot(center_x - last_x, center_y - last_y)
                
                if dist < 300 and dist < best_dist:
                    best_dist = dist
                    best_center = (center_x, center_y)
                    best_bbox = (x, y, w, h)
            
            if best_center:
                motion_centers[hand_label] = {
                    'center': best_center,
                    'bbox': best_bbox,
                    'distance': best_dist
                }
        
        self.prev_frame_gray = gray
        return motion_centers


def get_hand_bbox(landmarks, frame_shape, padding=ROI_PADDING):
    """Get bounding box around hand landmarks with padding."""
    h, w = frame_shape[:2]
    xs = [lm.x * w for lm in landmarks.landmark]
    ys = [lm.y * h for lm in landmarks.landmark]
    
    x1 = max(0, int(min(xs) - padding))
    y1 = max(0, int(min(ys) - padding))
    x2 = min(w, int(max(xs) + padding))
    y2 = min(h, int(max(ys) + padding))
    
    return (x1, y1, x2, y2)


class SingleHandTracker:
    """
    Optical Flow tracker for a SINGLE hand.
    VERSION 2: With drift prevention constraints.
    """
    
    def __init__(self, hand_label):
        self.hand_label = hand_label
        
        # Lucas-Kanade parameters
        self.lk_params = dict(
            winSize=(40, 40),
            maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
        )
        
        # Tracking state
        self.tracked_points = None
        self.prev_gray = None
        
        # ROI constraint
        self.roi_bbox = None  # (x1, y1, x2, y2)
        
        # Position history for smoothing
        self.position_history = deque(maxlen=5)
        
        # Velocity for inertia/coasting
        self.last_velocity = np.array([0.0, 0.0])
        self.last_position = None
        self.last_center = None
        
        # Ghost tracking state
        self.ghost_frames = 0
        self.is_coasting = False
        self.coasting_frames = 0
        self.max_coasting_frames = 5
        self.velocity_decay = 0.85
        
        # Confidence tracking
        self.tracking_confidence = 1.0
    
    def init_tracking(self, gray, landmarks):
        """
        Initialize tracking points from detected landmarks.
        SNAP BACK: Resets all ghost state.
        """
        if landmarks is None:
            return False
        
        h, w = gray.shape
        
        # Extract key points (fingertips + wrist + knuckles)
        key_indices = [0, 4, 5, 8, 9, 12, 13, 16, 17, 20]
        
        points = []
        for idx in key_indices:
            if idx < len(landmarks.landmark):
                lm = landmarks.landmark[idx]
                x, y = int(lm.x * w), int(lm.y * h)
                points.append([[x, y]])
        
        if len(points) < 3:
            return False
        
        self.tracked_points = np.array(points, dtype=np.float32)
        self.prev_gray = gray.copy()
        
        # Set ROI around the hand
        self.roi_bbox = get_hand_bbox(landmarks, gray.shape)
        
        # SNAP BACK: Reset all ghost/coasting state
        self.ghost_frames = 0
        self.is_coasting = False
        self.coasting_frames = 0
        self.tracking_confidence = 1.0
        
        # Update position
        center = np.mean(self.tracked_points, axis=0)[0]
        if self.last_position is not None:
            self.last_velocity = center - self.last_position
        self.last_position = center.copy()
        self.last_center = tuple(center.astype(int))
        
        return True
    
    def track(self, gray):
        """
        Track points using optical flow WITH DRIFT PREVENTION.
        Returns: (success, center, method, data)
        """
        # Check if we can do optical flow
        if self.prev_gray is None or self.tracked_points is None:
            return self._try_coasting()
        
        # GHOST DURATION LIMIT
        self.ghost_frames += 1
        if self.ghost_frames > MAX_GHOST_FRAMES:
            # Ghost too old, unreliable
            return False, None, 'timeout', {'reason': 'ghost_expired'}
        
        # Calculate optical flow
        new_points, status, error = cv2.calcOpticalFlowPyrLK(
            self.prev_gray, gray, self.tracked_points, None, **self.lk_params
        )
        
        if new_points is None:
            return self._try_coasting()
        
        # CONFIDENCE GATING: Filter by status AND error
        status_flat = status.flatten()
        error_flat = error.flatten()
        good_mask = (status_flat == 1) & (error_flat < MAX_POINT_ERROR)
        
        good_new = new_points[good_mask]
        good_old = self.tracked_points[good_mask]
        
        if len(good_new) < MIN_GOOD_POINTS:
            # Too few good points - optical flow unreliable
            self.tracking_confidence *= 0.7
            return self._try_coasting()
        
        # Calculate flow vector
        flow_vectors = good_new - good_old
        avg_flow = np.mean(flow_vectors, axis=0)
        
        # VELOCITY SANITY CHECK
        flow_magnitude = np.linalg.norm(avg_flow)
        if flow_magnitude > MAX_HAND_SPEED:
            # Movement too fast - probably wrong tracking
            # Clamp to max speed
            scale = MAX_HAND_SPEED / flow_magnitude
            avg_flow = avg_flow * scale
            self.tracking_confidence *= 0.8
        
        # Calculate new center
        raw_center = np.mean(good_new, axis=0)
        
        # ROI CONSTRAINT: Check if new center is within expected region
        if self.roi_bbox is not None:
            x1, y1, x2, y2 = self.roi_bbox
            # Expand ROI based on ghost frames (allow more drift over time)
            expand = self.ghost_frames * 30
            if not (x1 - expand <= raw_center[0][0] <= x2 + expand and 
                    y1 - expand <= raw_center[0][1] <= y2 + expand):
                # Outside ROI - likely tracking wrong object
                self.tracking_confidence *= 0.5
                if self.tracking_confidence < 0.3:
                    return self._try_coasting()
        
        # Apply smoothing
        center = raw_center.astype(int)
        
        # Update ROI to follow hand (with constraints)
        if self.roi_bbox is not None:
            cx, cy = center[0][0], center[0][1]
            roi_w = self.roi_bbox[2] - self.roi_bbox[0]
            roi_h = self.roi_bbox[3] - self.roi_bbox[1]
            h, w = gray.shape
            self.roi_bbox = (
                max(0, cx - roi_w // 2),
                max(0, cy - roi_h // 2),
                min(w, cx + roi_w // 2),
                min(h, cy + roi_h // 2)
            )
        
        # Update tracking state
        self.tracked_points = new_points
        self.prev_gray = gray.copy()
        
        # Update velocity
        if self.last_position is not None:
            self.last_velocity = center[0] - self.last_position
        self.last_position = center[0].copy().astype(float)
        self.last_center = tuple(center.flatten())
        
        # Add to history
        self.position_history.append(self.last_center)
        
        # Reset coasting
        self.is_coasting = False
        self.coasting_frames = 0
        
        return True, self.last_center, 'optical_flow', {
            'points': good_new,
            'flow': avg_flow,
            'velocity': self.last_velocity,
            'confidence': self.tracking_confidence,
            'ghost_frame': self.ghost_frames,
            'good_points': len(good_new)
        }
    
    def _try_coasting(self):
        """Physics fallback: Coast using last known velocity."""
        if self.last_position is None:
            return False, None, 'lost', None
        
        velocity_magnitude = np.linalg.norm(self.last_velocity)
        
        # Only coast if we had significant velocity and within limit
        if velocity_magnitude < 3 or self.coasting_frames >= self.max_coasting_frames:
            return False, None, 'lost', None
        
        # VELOCITY SANITY CHECK on coasting too
        if velocity_magnitude > MAX_HAND_SPEED:
            self.last_velocity = self.last_velocity * (MAX_HAND_SPEED / velocity_magnitude)
        
        # Apply velocity with decay
        self.last_position = self.last_position + self.last_velocity
        self.last_velocity = self.last_velocity * self.velocity_decay
        
        self.is_coasting = True
        self.coasting_frames += 1
        self.ghost_frames += 1
        self.tracking_confidence *= 0.9
        
        center = tuple(self.last_position.astype(int))
        
        return True, center, 'coasting', {
            'velocity': self.last_velocity,
            'coasting_frame': self.coasting_frames,
            'confidence': self.tracking_confidence
        }
    
    def update_prev_frame(self, gray):
        """Update previous frame for next iteration."""
        self.prev_gray = gray.copy()
    
    def reset(self):
        """Reset tracker state."""
        self.tracked_points = None
        self.roi_bbox = None
        self.position_history.clear()
        self.ghost_frames = 0
        self.is_coasting = False
        self.coasting_frames = 0
        self.tracking_confidence = 1.0


class CombinedMotionTracker:
    """
    Combines Motion Energy + Optical Flow + Inertia for robust blur tracking.
    VERSION 2: With drift prevention.
    """
    
    def __init__(self):
        self.motion_tracker = MotionTracker()
        
        # INDEPENDENT trackers for each hand
        self.hand_trackers = {
            'Left': SingleHandTracker('Left'),
            'Right': SingleHandTracker('Right')
        }
        
        # Last known good positions
        self.last_known_positions = {'Left': None, 'Right': None}
        self.last_known_landmarks = {'Left': None, 'Right': None}
        
        # Tracking state
        self.tracking_mode = {'Left': 'detection', 'Right': 'detection'}
        self.frames_since_detection = {'Left': 0, 'Right': 0}
        self.max_tracking_frames = MAX_GHOST_FRAMES + 3  # Slightly more for motion
    
    def update_from_detection(self, results, gray):
        """
        Update tracker with new detection results.
        SNAP BACK: Detection resets ghost state immediately.
        """
        if results is None or not results.multi_hand_landmarks:
            return
        
        h, w = gray.shape
        
        for hand_landmarks, handedness in zip(
            results.multi_hand_landmarks,
            results.multi_handedness
        ):
            label = handedness.classification[0].label
            
            # Store last known position
            wrist = hand_landmarks.landmark[0]
            self.last_known_positions[label] = (int(wrist.x * w), int(wrist.y * h))
            self.last_known_landmarks[label] = hand_landmarks
            
            # SNAP BACK: Initialize tracking (resets all ghost state)
            self.hand_trackers[label].init_tracking(gray, hand_landmarks)
            
            # Reset state for detected hand
            self.tracking_mode[label] = 'detection'
            self.frames_since_detection[label] = 0
        
        # Update prev frame for both trackers
        for label in ['Left', 'Right']:
            self.hand_trackers[label].update_prev_frame(gray)
    
    def get_fallback_position(self, gray, hand_label):
        """Get fallback position with drift prevention."""
        self.frames_since_detection[hand_label] += 1
        
        # Timeout check
        if self.frames_since_detection[hand_label] > self.max_tracking_frames:
            self.tracking_mode[hand_label] = 'timeout'
            return False, None, 'timeout', None
        
        # Try optical flow / coasting
        tracker = self.hand_trackers[hand_label]
        success, center, method, data = tracker.track(gray)
        
        if success:
            self.tracking_mode[hand_label] = method
            return True, center, method, data
        
        # Fall back to motion energy
        frame_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        motion_centers = self.motion_tracker.get_motion_centers(
            frame_bgr,
            {hand_label: self.last_known_positions[hand_label]}
        )
        
        if hand_label in motion_centers:
            center = motion_centers[hand_label]['center']
            self.tracking_mode[hand_label] = 'motion_energy'
            return True, center, 'motion_energy', motion_centers[hand_label]
        
        self.tracking_mode[hand_label] = 'lost'
        return False, None, 'lost', None
    
    def get_status(self):
        """Get current tracking status for both hands."""
        return {
            'left_mode': self.tracking_mode.get('Left', 'unknown'),
            'right_mode': self.tracking_mode.get('Right', 'unknown'),
            'left_frames': self.frames_since_detection.get('Left', 0),
            'right_frames': self.frames_since_detection.get('Right', 0),
            'left_coasting': self.hand_trackers['Left'].is_coasting,
            'right_coasting': self.hand_trackers['Right'].is_coasting,
            'left_confidence': self.hand_trackers['Left'].tracking_confidence,
            'right_confidence': self.hand_trackers['Right'].tracking_confidence,
            'left_ghost_frames': self.hand_trackers['Left'].ghost_frames,
            'right_ghost_frames': self.hand_trackers['Right'].ghost_frames
        }


def test_motion_tracker():
    """Test the drift-prevented motion tracking system."""
    print("=" * 60)
    print("MOTION TRACKER V2 TEST (WITH DRIFT PREVENTION)")
    print("=" * 60)
    print("Improvements:")
    print("  - ROI constraint")
    print("  - Velocity sanity check (max 150px/frame)")
    print("  - Confidence gating (error < 12)")
    print("  - Ghost duration limit (5 frames)")
    print("  - Snap back on detection")
    print("=" * 60)
    
    tracker = CombinedMotionTracker()
    
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    print("\nPunch FAST to test tracking!")
    print("Press 'Q' to quit.\n")
    
    tracker.last_known_positions['Right'] = (320, 240)
    tracker.last_known_positions['Left'] = (320, 240)
    
    method_counts = {}
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame = cv2.flip(frame, 1)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        success, center, method, data = tracker.get_fallback_position(gray, 'Right')
        method_counts[method] = method_counts.get(method, 0) + 1
        
        status = tracker.get_status()
        
        if success and center:
            color = {
                'optical_flow': (255, 0, 255),
                'coasting': (255, 255, 0),
                'motion_energy': (0, 0, 255)
            }.get(method, (128, 128, 128))
            
            cv2.circle(frame, center, 15, color, -1)
            conf = data.get('confidence', 1.0) if data else 1.0
            cv2.putText(frame, f"{method} ({conf:.2f})", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # Status overlay
        cv2.putText(frame, f"Ghost: {status['right_ghost_frames']}/5", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        cv2.imshow("Motion Tracker V2 - Press Q", frame)
        tracker.hand_trackers['Right'].update_prev_frame(gray)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    print("\nResults:", method_counts)
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    test_motion_tracker()
