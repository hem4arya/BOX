"""
Hand Renderer - Visual Persistence for Motion Blur
Draws solid-looking 3D hand skeleton that persists during fast motion

Features:
- Standard skeleton drawing for MediaPipe detection
- Ghost skeleton for optical flow (shifted by flow vector)
- Anti-aliased smooth lines for professional look
"""

import cv2
import numpy as np


# MediaPipe Hand Connections (21 landmarks)
HAND_CONNECTIONS = [
    # Thumb
    (0, 1), (1, 2), (2, 3), (3, 4),
    # Index
    (0, 5), (5, 6), (6, 7), (7, 8),
    # Middle
    (0, 9), (9, 10), (10, 11), (11, 12),
    # Ring
    (0, 13), (13, 14), (14, 15), (15, 16),
    # Pinky
    (0, 17), (17, 18), (18, 19), (19, 20),
    # Palm connections
    (5, 9), (9, 13), (13, 17)
]

# Finger tip indices
FINGERTIPS = [4, 8, 12, 16, 20]

# Joint groups for different sizes
PALM_JOINTS = [0, 1, 5, 9, 13, 17]
KNUCKLES = [2, 6, 10, 14, 18]
TIPS = [3, 4, 7, 8, 11, 12, 15, 16, 19, 20]


class HandRenderer:
    """
    Renders hand skeletons with professional visual quality.
    Supports both detected landmarks and "ghost" persistence during blur.
    """
    
    def __init__(self):
        # Default colors (BGR)
        self.right_color = (0, 255, 0)      # Green
        self.left_color = (255, 100, 0)     # Blue-ish
        self.ghost_color = (0, 215, 255)    # Gold
        self.coasting_color = (255, 255, 0) # Cyan
        
        # Line/joint sizes
        self.bone_thickness = 3
        self.joint_radius = 5
        self.tip_radius = 6
        self.ghost_bone_thickness = 2
        self.ghost_joint_radius = 4
    
    def landmarks_to_points(self, landmarks, frame_shape):
        """
        Convert MediaPipe landmarks to pixel coordinates.
        
        Args:
            landmarks: MediaPipe hand landmarks
            frame_shape: (height, width) of the frame
        
        Returns:
            List of (x, y) tuples for each landmark
        """
        h, w = frame_shape[:2]
        points = []
        
        for lm in landmarks.landmark:
            x = int(lm.x * w)
            y = int(lm.y * h)
            points.append((x, y))
        
        return points
    
    def flat_to_points(self, flat_array, frame_shape):
        """
        Convert flat landmark array [x1,y1,z1,x2,y2,z2,...] to pixel coordinates.
        """
        h, w = frame_shape[:2]
        points = []
        
        for i in range(0, len(flat_array), 3):
            x = int(flat_array[i] * w)
            y = int(flat_array[i + 1] * h)
            points.append((x, y))
        
        return points
    
    def draw_skeleton(self, frame, landmarks, hand_label='Right', alpha=1.0):
        """
        Draw a solid-looking hand skeleton on the frame.
        
        Args:
            frame: BGR image
            landmarks: MediaPipe hand landmarks
            hand_label: 'Left' or 'Right'
            alpha: opacity (1.0 = fully opaque)
        """
        if landmarks is None:
            return frame
        
        points = self.landmarks_to_points(landmarks, frame.shape)
        color = self.right_color if hand_label == 'Right' else self.left_color
        
        return self._draw_hand(frame, points, color, 
                              self.bone_thickness, self.joint_radius, self.tip_radius)
    
    def draw_ghost(self, frame, last_landmarks, flow_vector=(0, 0), hand_label='Right'):
        """
        Draw a "ghost" skeleton shifted by flow vector.
        Used during optical flow tracking when landmarks are estimated.
        
        Args:
            frame: BGR image
            last_landmarks: Last known MediaPipe landmarks
            flow_vector: (dx, dy) to shift all points
            hand_label: 'Left' or 'Right'
        """
        if last_landmarks is None:
            return frame
        
        points = self.landmarks_to_points(last_landmarks, frame.shape)
        
        # Shift all points by flow vector
        dx, dy = int(flow_vector[0]), int(flow_vector[1])
        shifted_points = [(x + dx, y + dy) for (x, y) in points]
        
        return self._draw_hand(frame, shifted_points, self.ghost_color,
                              self.ghost_bone_thickness, self.ghost_joint_radius, 
                              self.ghost_joint_radius)
    
    def draw_ghost_from_flat(self, frame, flat_landmarks, flow_vector=(0, 0)):
        """
        Draw ghost from flat landmark array with flow shift.
        """
        if flat_landmarks is None or len(flat_landmarks) < 63:
            return frame
        
        points = self.flat_to_points(flat_landmarks, frame.shape)
        
        dx, dy = int(flow_vector[0]), int(flow_vector[1])
        shifted_points = [(x + dx, y + dy) for (x, y) in points]
        
        return self._draw_hand(frame, shifted_points, self.ghost_color,
                              self.ghost_bone_thickness, self.ghost_joint_radius,
                              self.ghost_joint_radius)
    
    def draw_coasting_ghost(self, frame, last_landmarks, velocity=(0, 0), frames_coasted=1):
        """
        Draw coasting skeleton with velocity prediction.
        Fades based on how many frames since detection.
        """
        if last_landmarks is None:
            return frame
        
        points = self.landmarks_to_points(last_landmarks, frame.shape)
        
        # Apply velocity * frames for prediction
        dx = int(velocity[0] * frames_coasted)
        dy = int(velocity[1] * frames_coasted)
        shifted_points = [(x + dx, y + dy) for (x, y) in points]
        
        # Fade color based on frames coasted
        fade = max(0.3, 1.0 - frames_coasted * 0.15)
        faded_color = tuple(int(c * fade) for c in self.coasting_color)
        
        return self._draw_hand(frame, shifted_points, faded_color,
                              self.ghost_bone_thickness, self.ghost_joint_radius,
                              self.ghost_joint_radius)
    
    def draw_ballistic_hand(self, frame, last_landmarks, target_pos_norm):
        """
        Draw RED ballistic hand at target position.
        
        Args:
            target_pos_norm: (x, y) normalized position of WRIST
        """
        if last_landmarks is None:
            return frame
        
        points = self.landmarks_to_points(last_landmarks, frame.shape)
        
        # Calculate shift based on wrist position (index 0)
        wrist_curr = points[0]
        h, w = frame.shape[:2]
        wrist_target = (int(target_pos_norm[0] * w), int(target_pos_norm[1] * h))
        
        dx = wrist_target[0] - wrist_curr[0]
        dy = wrist_target[1] - wrist_curr[1]
        
        shifted_points = [(x + dx, y + dy) for (x, y) in points]
        
        # Draw in RED
        return self._draw_hand(frame, shifted_points, (0, 0, 255),
                              self.bone_thickness, self.joint_radius, self.tip_radius)
    
    def _draw_hand(self, frame, points, color, bone_thickness, joint_radius, tip_radius):
        """
        Internal method to draw hand from point list.
        Uses anti-aliased lines for smooth appearance.
        """
        if len(points) < 21:
            return frame
        
        # Draw bones (connections)
        for start_idx, end_idx in HAND_CONNECTIONS:
            if start_idx < len(points) and end_idx < len(points):
                pt1 = points[start_idx]
                pt2 = points[end_idx]
                cv2.line(frame, pt1, pt2, color, bone_thickness, cv2.LINE_AA)
        
        # Draw joints
        for i, pt in enumerate(points):
            if i in FINGERTIPS:
                # Larger circles for fingertips
                cv2.circle(frame, pt, tip_radius, color, -1, cv2.LINE_AA)
            elif i in PALM_JOINTS:
                # Medium circles for palm
                cv2.circle(frame, pt, joint_radius, color, -1, cv2.LINE_AA)
            else:
                # Smaller circles for other joints
                cv2.circle(frame, pt, joint_radius - 1, color, -1, cv2.LINE_AA)
        
        return frame
    
    def draw_tracking_label(self, frame, method, position=None):
        """
        Draw tracking method label at top of frame.
        """
        labels = {
            'mediapipe': ('TRACKING: MediaPipe', (0, 255, 0)),
            'yolo+mediapipe': ('TRACKING: YOLO + MediaPipe', (255, 255, 0)),
            'optical_flow': ('TRACKING: GHOST HAND', (0, 215, 255)),
            'coasting': ('TRACKING: PHYSICS PREDICTION', (255, 255, 0)),
            'motion_energy': ('TRACKING: MOTION BLOB', (0, 0, 255)),
        }
        
        text, color = labels.get(method, (f'TRACKING: {method}', (200, 200, 200)))
        
        # Center at top
        h, w = frame.shape[:2]
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
        x = (w - text_size[0]) // 2
        y = 30
        
        # Background for readability
        cv2.rectangle(frame, (x - 10, y - 25), (x + text_size[0] + 10, y + 10), (0, 0, 0), -1)
        cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)
        
        return frame


def test_renderer():
    """Test the hand renderer with a simple mockup."""
    print("=" * 60)
    print("HAND RENDERER TEST")
    print("=" * 60)
    
    # Create test frame
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    frame[:] = (40, 40, 40)  # Dark gray background
    
    renderer = HandRenderer()
    
    # Mock landmarks (21 points in a hand-like pattern)
    # These are normalized coordinates
    mock_points = [
        # Wrist
        (0.5, 0.8),
        # Thumb
        (0.42, 0.72), (0.38, 0.65), (0.35, 0.58), (0.33, 0.52),
        # Index
        (0.45, 0.58), (0.44, 0.48), (0.43, 0.40), (0.42, 0.33),
        # Middle
        (0.5, 0.55), (0.5, 0.45), (0.5, 0.37), (0.5, 0.30),
        # Ring
        (0.55, 0.58), (0.56, 0.48), (0.57, 0.40), (0.58, 0.33),
        # Pinky
        (0.60, 0.62), (0.62, 0.55), (0.64, 0.48), (0.66, 0.42)
    ]
    
    h, w = frame.shape[:2]
    points = [(int(x * w), int(y * h)) for x, y in mock_points]
    
    # Draw test hand
    frame = renderer._draw_hand(frame, points, (0, 255, 0), 3, 5, 6)
    
    # Draw ghost hand (shifted)
    ghost_points = [(x + 150, y - 50) for x, y in points]
    frame = renderer._draw_hand(frame, ghost_points, (0, 215, 255), 2, 4, 4)
    
    # Labels
    cv2.putText(frame, "Normal Hand", (points[0][0] - 50, points[0][1] + 30),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    cv2.putText(frame, "Ghost Hand", (ghost_points[0][0] - 50, ghost_points[0][1] + 30),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 215, 255), 1)
    
    cv2.imshow("Hand Renderer Test - Press any key", frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    print("Test complete!")


if __name__ == "__main__":
    test_renderer()
