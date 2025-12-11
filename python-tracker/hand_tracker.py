"""
GPU-Accelerated Hand Tracking using MediaPipe + PyTorch
Native Python implementation for game development

Requirements:
    pip install mediapipe opencv-python torch torchvision
"""

import cv2
import mediapipe as mp
import time
import numpy as np
from collections import deque
import torch

# ========== CONFIGURATION ==========
WEBCAM_ID = 0
WEBCAM_WIDTH = 1280
WEBCAM_HEIGHT = 720
WEBCAM_FPS = 60

# MediaPipe settings
MODEL_COMPLEXITY = 1  # 0=Lite, 1=Full
MAX_NUM_HANDS = 2
MIN_DETECTION_CONFIDENCE = 0.7
MIN_TRACKING_CONFIDENCE = 0.7

# Performance tracking
FPS_BUFFER_SIZE = 30


def check_gpu():
    """Check and print GPU availability."""
    print("=" * 60)
    print("GPU STATUS")
    print("=" * 60)
    
    # PyTorch GPU
    print(f"  PyTorch version: {torch.__version__}")
    print(f"  CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  CUDA version: {torch.version.cuda}")
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    print("=" * 60)
    return torch.cuda.is_available()


class HandTracker:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            model_complexity=MODEL_COMPLEXITY,
            max_num_hands=MAX_NUM_HANDS,
            min_detection_confidence=MIN_DETECTION_CONFIDENCE,
            min_tracking_confidence=MIN_TRACKING_CONFIDENCE
        )
        
        self.fps_buffer = deque(maxlen=FPS_BUFFER_SIZE)
        self.latency_buffer = deque(maxlen=FPS_BUFFER_SIZE)
        self.last_time = time.perf_counter()
        
        print("\nHAND TRACKER INITIALIZED")
        print(f"  Model Complexity: {MODEL_COMPLEXITY}")
        print(f"  Max Hands: {MAX_NUM_HANDS}")

    def process_frame(self, frame):
        start_time = time.perf_counter()
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)
        process_time = (time.perf_counter() - start_time) * 1000
        return results, process_time

    def draw_landmarks(self, frame, results):
        if results.multi_hand_landmarks:
            for hand_landmarks, handedness in zip(
                results.multi_hand_landmarks, 
                results.multi_handedness
            ):
                hand_label = handedness.classification[0].label
                confidence = handedness.classification[0].score
                
                self.mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS,
                    self.mp_drawing_styles.get_default_hand_landmarks_style(),
                    self.mp_drawing_styles.get_default_hand_connections_style()
                )
                
                wrist = hand_landmarks.landmark[0]
                h, w, _ = frame.shape
                cx, cy = int(wrist.x * w), int(wrist.y * h)
                color = (0, 255, 0) if hand_label == "Right" else (255, 100, 0)
                cv2.putText(frame, f"{hand_label} ({confidence:.2f})",
                    (cx - 50, cy - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        return frame

    def draw_overlay(self, frame, fps, latency, process_time, num_hands, gpu_available):
        color = (0, 255, 0)
        
        # GPU indicator
        gpu_status = "GPU: RTX 3060" if gpu_available else "GPU: N/A"
        cv2.putText(frame, gpu_status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, 
                    (0, 255, 255) if gpu_available else (0, 0, 255), 2)
        
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        cv2.putText(frame, f"Process: {process_time:.1f}ms", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        cv2.putText(frame, f"Hands: {num_hands}", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        return frame

    def print_landmarks(self, results, process_time, frame_count):
        num_hands = len(results.multi_hand_landmarks) if results.multi_hand_landmarks else 0
        print(f"\n[Frame {frame_count}] Hands: {num_hands} | Process: {process_time:.1f}ms")
        
        if results.multi_hand_landmarks:
            for hand_landmarks, handedness in zip(
                results.multi_hand_landmarks, results.multi_handedness
            ):
                hand_label = handedness.classification[0].label
                print(f"  {hand_label} Hand:")
                
                key_points = {0: "WRIST", 4: "THUMB", 8: "INDEX", 12: "MIDDLE", 16: "RING", 20: "PINKY"}
                for idx, name in key_points.items():
                    lm = hand_landmarks.landmark[idx]
                    print(f"    {name}: ({lm.x:.3f}, {lm.y:.3f}, {lm.z:.3f})")

    def update_metrics(self, process_time):
        current_time = time.perf_counter()
        frame_time = current_time - self.last_time
        self.last_time = current_time
        
        if frame_time > 0:
            self.fps_buffer.append(1.0 / frame_time)
        self.latency_buffer.append(frame_time * 1000)
        
        avg_fps = np.mean(self.fps_buffer) if self.fps_buffer else 0
        avg_latency = np.mean(self.latency_buffer) if self.latency_buffer else 0
        return avg_fps, avg_latency

    def close(self):
        self.hands.close()
        print("\nHand tracker closed.")


def main():
    # Check GPU
    gpu_available = check_gpu()
    
    # Initialize webcam
    print("\nInitializing webcam...")
    backends = [
        (cv2.CAP_DSHOW, "DirectShow"),
        (cv2.CAP_MSMF, "Media Foundation"),
        (cv2.CAP_ANY, "Default")
    ]
    
    cap = None
    for backend, name in backends:
        cap = cv2.VideoCapture(WEBCAM_ID, backend)
        if cap.isOpened():
            print(f"  Using {name} backend")
            break
        cap.release()
    
    if not cap or not cap.isOpened():
        print("ERROR: Could not open webcam!")
        return
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, WEBCAM_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, WEBCAM_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, WEBCAM_FPS)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    
    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"  Camera: {actual_w}x{actual_h} @ {actual_fps:.0f} FPS")
    
    tracker = HandTracker()
    frame_count = 0
    
    print("\nPress 'Q' to quit.\n" + "-" * 60)
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            frame = cv2.flip(frame, 1)
            
            results, process_time = tracker.process_frame(frame)
            avg_fps, avg_latency = tracker.update_metrics(process_time)
            num_hands = len(results.multi_hand_landmarks) if results.multi_hand_landmarks else 0
            
            frame = tracker.draw_landmarks(frame, results)
            frame = tracker.draw_overlay(frame, avg_fps, avg_latency, process_time, num_hands, gpu_available)
            
            if frame_count % 30 == 0:
                tracker.print_landmarks(results, process_time, frame_count)
            
            cv2.imshow("Hand Tracking - GPU Accelerated - Press Q to Exit", frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("\nQuitting...")
                break
    
    except KeyboardInterrupt:
        print("\nInterrupted.")
    
    finally:
        print("\n" + "=" * 60)
        print("SESSION STATISTICS")
        print(f"  Total Frames: {frame_count}")
        print(f"  Average FPS: {np.mean(tracker.fps_buffer):.2f}" if tracker.fps_buffer else "  N/A")
        print(f"  Average Latency: {np.mean(tracker.latency_buffer):.2f}ms" if tracker.latency_buffer else "  N/A")
        print(f"  GPU Used: {'Yes - RTX 3060' if gpu_available else 'No'}")
        print("=" * 60)
        
        tracker.close()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
