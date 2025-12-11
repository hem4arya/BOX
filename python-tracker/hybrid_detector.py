"""
Hybrid Hand Detector: MediaPipe + YOLO Backup
Combines MediaPipe (accurate) + YOLO (robust to blur/motion)

Strategy:
1. Try MediaPipe first (best accuracy)
2. If MediaPipe fails, use YOLO to detect hands
3. If YOLO finds hands, retry MediaPipe on cropped region
4. Track which method was used
"""

import cv2
import numpy as np
import mediapipe as mp
from ultralytics import YOLO
import os


class HybridHandDetector:
    """
    Combines MediaPipe + YOLO for robust hand detection.
    Falls back to YOLO when MediaPipe loses tracking during fast movements.
    """
    
    def __init__(self, yolo_model='yolov8n.pt'):
        print("Initializing Hybrid Detector...")
        
        # MediaPipe - High Accuracy Mode
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            model_complexity=1,              # UPGRADED: 0 → 1 (high accuracy, stable)
            min_detection_confidence=0.6,
            min_tracking_confidence=0.6
        )
        print("  MediaPipe: Ready (model_complexity=1)")
        
        # YOLO - using nano model for speed
        print(f"  Loading YOLO model: {yolo_model}")
        self.yolo = YOLO(yolo_model)
        self.yolo.to('cuda')  # Use GPU
        print("  YOLO: Ready (GPU)")
        
        # Detection statistics
        self.stats = {
            'mediapipe': 0,
            'yolo+mediapipe': 0,
            'yolo_only': 0,
            'none': 0
        }
        self.total_detections = 0
        self.last_method = 'none'
    
    def detect(self, frame, frame_small=None):
        """
        Detect hands using hybrid approach.
        
        Args:
            frame: Original frame (for display)
            frame_small: Resized frame for processing (optional)
        
        Returns:
            (mediapipe_results, method_used, yolo_bboxes)
        """
        process_frame = frame_small if frame_small is not None else frame
        frame_rgb = cv2.cvtColor(process_frame, cv2.COLOR_BGR2RGB)
        
        self.total_detections += 1
        
        # Step 1: Try MediaPipe first (most accurate)
        results = self.hands.process(frame_rgb)
        
        if results.multi_hand_landmarks:
            self.stats['mediapipe'] += 1
            self.last_method = 'mediapipe'
            return results, 'mediapipe', None
        
        # Step 2: MediaPipe failed - use YOLO
        # Detect persons (class 0) as proxy for hands
        yolo_results = self.yolo(process_frame, classes=[0], verbose=False, conf=0.3)
        
        bboxes = []
        for result in yolo_results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                conf = float(box.conf[0])
                bboxes.append((x1, y1, x2, y2, conf))
        
        if bboxes:
            # Step 3: Try MediaPipe on focused regions
            h, w = process_frame.shape[:2]
            
            for bbox in bboxes:
                x1, y1, x2, y2, conf = bbox
                
                # Expand bbox slightly
                pad = 20
                x1 = max(0, x1 - pad)
                y1 = max(0, y1 - pad)
                x2 = min(w, x2 + pad)
                y2 = min(h, y2 + pad)
                
                # Crop region
                region = process_frame[y1:y2, x1:x2]
                if region.size == 0:
                    continue
                
                region_rgb = cv2.cvtColor(region, cv2.COLOR_BGR2RGB)
                region_results = self.hands.process(region_rgb)
                
                if region_results.multi_hand_landmarks:
                    # Adjust landmarks to full frame coordinates
                    adjusted = self._adjust_landmarks(region_results, (x1, y1, x2, y2), (h, w))
                    self.stats['yolo+mediapipe'] += 1
                    self.last_method = 'yolo+mediapipe'
                    return adjusted, 'yolo+mediapipe', [(x1, y1, x2, y2)]
            
            # Step 4: YOLO detected something but MediaPipe couldn't get landmarks
            self.stats['yolo_only'] += 1
            self.last_method = 'yolo_only'
            return None, 'yolo_only', [(b[0], b[1], b[2], b[3]) for b in bboxes]
        
        # Nothing detected
        self.stats['none'] += 1
        self.last_method = 'none'
        return None, 'none', None
    
    def _adjust_landmarks(self, results, bbox, frame_shape):
        """Adjust cropped region landmarks back to full frame coordinates."""
        x1, y1, x2, y2 = bbox
        region_w = x2 - x1
        region_h = y2 - y1
        frame_h, frame_w = frame_shape
        
        for hand_landmarks in results.multi_hand_landmarks:
            for lm in hand_landmarks.landmark:
                # Convert from region coords to frame coords (normalized)
                lm.x = (lm.x * region_w + x1) / frame_w
                lm.y = (lm.y * region_h + y1) / frame_h
        
        return results
    
    def get_stats(self):
        """Get detection statistics."""
        total = max(1, self.total_detections)
        return {
            'mediapipe': self.stats['mediapipe'],
            'mediapipe_pct': self.stats['mediapipe'] / total * 100,
            'yolo+mediapipe': self.stats['yolo+mediapipe'],
            'yolo+mediapipe_pct': self.stats['yolo+mediapipe'] / total * 100,
            'yolo_only': self.stats['yolo_only'],
            'yolo_only_pct': self.stats['yolo_only'] / total * 100,
            'none': self.stats['none'],
            'none_pct': self.stats['none'] / total * 100,
            'total': self.total_detections,
            'last_method': self.last_method
        }
    
    def close(self):
        """Clean up resources."""
        self.hands.close()


def test_detector():
    """Test the hybrid detector."""
    print("=" * 60)
    print("HYBRID DETECTOR TEST")
    print("=" * 60)
    
    detector = HybridHandDetector()
    
    # Test with webcam
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    print("\nPress 'Q' to quit test.\n")
    
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame = cv2.flip(frame, 1)
        frame_count += 1
        
        results, method, bboxes = detector.detect(frame)
        
        # Draw based on method
        color = {
            'mediapipe': (0, 255, 0),
            'yolo+mediapipe': (255, 255, 0),
            'yolo_only': (0, 165, 255),
            'none': (0, 0, 255)
        }.get(method, (255, 255, 255))
        
        cv2.putText(frame, f"Method: {method}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        # Draw YOLO bboxes if available
        if bboxes:
            for bbox in bboxes:
                x1, y1, x2, y2 = bbox[:4]
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 165, 255), 2)
        
        # Draw landmarks if available
        if results and results.multi_hand_landmarks:
            mp_drawing = mp.solutions.drawing_utils
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks,
                                         mp.solutions.hands.HAND_CONNECTIONS)
        
        cv2.imshow("Hybrid Detector Test", frame)
        
        if frame_count % 60 == 0:
            stats = detector.get_stats()
            print(f"[Frame {frame_count}] "
                  f"MP: {stats['mediapipe_pct']:.1f}% | "
                  f"YOLO+MP: {stats['yolo+mediapipe_pct']:.1f}% | "
                  f"YOLO: {stats['yolo_only_pct']:.1f}% | "
                  f"None: {stats['none_pct']:.1f}%")
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Final stats
    stats = detector.get_stats()
    print("\n" + "=" * 60)
    print("FINAL STATISTICS")
    print("=" * 60)
    print(f"  Total Frames: {stats['total']}")
    print(f"  MediaPipe: {stats['mediapipe']} ({stats['mediapipe_pct']:.1f}%)")
    print(f"  YOLO+MediaPipe: {stats['yolo+mediapipe']} ({stats['yolo+mediapipe_pct']:.1f}%)")
    print(f"  YOLO Only: {stats['yolo_only']} ({stats['yolo_only_pct']:.1f}%)")
    print(f"  None: {stats['none']} ({stats['none_pct']:.1f}%)")
    print("=" * 60)
    
    detector.close()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    test_detector()
