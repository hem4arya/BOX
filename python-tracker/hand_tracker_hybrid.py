"""
HIGH-PERFORMANCE Hand Tracking with Hybrid Detection (MediaPipe + YOLO)
Phase 4B: Adding YOLO backup for fast movement tracking

Architecture:
- Thread 1: Camera capture
- Thread 2: Hybrid detection (MediaPipe + YOLO fallback) + LSTM
- Thread 3: Display (main thread)
"""

import cv2
import mediapipe as mp
import time
import numpy as np
from collections import deque
import threading
import queue
import psutil
import os
import torch

from hybrid_detector import HybridHandDetector
from lstm_predictor import PredictiveTracker

# ========== CONFIGURATION ==========
WEBCAM_ID = 0
CAPTURE_WIDTH = 1280
CAPTURE_HEIGHT = 720
PROCESS_WIDTH = 640
PROCESS_HEIGHT = 480

# Queue sizes
FRAME_QUEUE_SIZE = 2
RESULT_QUEUE_SIZE = 2

# LSTM settings
LSTM_ENABLED = True
LSTM_BUFFER_SIZE = 10

# Performance
FPS_BUFFER_SIZE = 60


class PerformanceMonitor:
    def __init__(self):
        self.fps_buffer = deque(maxlen=FPS_BUFFER_SIZE)
        self.latency_buffer = deque(maxlen=FPS_BUFFER_SIZE)
        self.process_buffer = deque(maxlen=FPS_BUFFER_SIZE)
        self.last_time = time.perf_counter()
        self.frame_count = 0
        self.min_fps = float('inf')
        self.max_latency = 0
        self.frame_drops = 0
    
    def update(self, process_time_ms):
        current = time.perf_counter()
        frame_time = current - self.last_time
        self.last_time = current
        self.frame_count += 1
        
        if frame_time > 0:
            fps = 1.0 / frame_time
            self.fps_buffer.append(fps)
            self.min_fps = min(self.min_fps, fps)
        
        latency = frame_time * 1000
        self.latency_buffer.append(latency)
        self.max_latency = max(self.max_latency, latency)
        self.process_buffer.append(process_time_ms)
    
    def record_drop(self):
        self.frame_drops += 1
    
    @property
    def fps(self):
        return np.mean(self.fps_buffer) if self.fps_buffer else 0
    
    @property
    def latency(self):
        return np.mean(self.latency_buffer) if self.latency_buffer else 0
    
    @property
    def process_time(self):
        return np.mean(self.process_buffer) if self.process_buffer else 0


class HybridHandTracker:
    """Multi-threaded hand tracking with MediaPipe + YOLO hybrid detection."""
    
    def __init__(self):
        self.running = False
        self.frame_queue = queue.Queue(maxsize=FRAME_QUEUE_SIZE)
        self.result_queue = queue.Queue(maxsize=RESULT_QUEUE_SIZE)
        
        # Hybrid Detector (MediaPipe + YOLO)
        print("Initializing Hybrid Detection System...")
        self.detector = HybridHandDetector()
        
        # LSTM Predictor
        self.predictor = None
        if LSTM_ENABLED:
            print("Initializing LSTM Predictor...")
            self.predictor = PredictiveTracker(buffer_size=LSTM_BUFFER_SIZE, device='cuda')
        
        self.mp_hands = mp.solutions.hands
        self.cap = None
        self.perf = PerformanceMonitor()
        self.capture_fps = 0
        self.process_fps = 0
        
    def init_camera(self):
        print("Initializing camera...")
        cv2.setUseOptimized(True)
        cv2.setNumThreads(4)
        
        self.cap = cv2.VideoCapture(WEBCAM_ID, cv2.CAP_DSHOW)
        if not self.cap.isOpened():
            self.cap = cv2.VideoCapture(WEBCAM_ID)
        
        if not self.cap.isOpened():
            print("ERROR: Cannot open camera!")
            return False
        
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAPTURE_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAPTURE_HEIGHT)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        
        print(f"  Camera: {CAPTURE_WIDTH}x{CAPTURE_HEIGHT}")
        return True
    
    def _capture_thread_func(self):
        capture_count = 0
        start_time = time.perf_counter()
        
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                continue
            
            frame = cv2.flip(frame, 1)
            
            if self.frame_queue.full():
                try:
                    self.frame_queue.get_nowait()
                    self.perf.record_drop()
                except queue.Empty:
                    pass
            
            self.frame_queue.put((frame, time.perf_counter()))
            capture_count += 1
            
            elapsed = time.perf_counter() - start_time
            if elapsed > 0:
                self.capture_fps = capture_count / elapsed
    
    def _processing_thread_func(self):
        process_count = 0
        start_time = time.perf_counter()
        
        while self.running:
            try:
                frame, timestamp = self.frame_queue.get(timeout=0.1)
            except queue.Empty:
                continue
            
            proc_start = time.perf_counter()
            
            # Resize for processing
            small_frame = cv2.resize(frame, (PROCESS_WIDTH, PROCESS_HEIGHT))
            
            # Hybrid detection (MediaPipe + YOLO)
            results, method, bboxes = self.detector.detect(small_frame)
            
            # LSTM prediction if we have landmarks
            prediction_data = None
            if LSTM_ENABLED and self.predictor and results:
                prediction_data = self.predictor.update(results)
            
            total_proc_time = (time.perf_counter() - proc_start) * 1000
            
            if self.result_queue.full():
                try:
                    self.result_queue.get_nowait()
                except queue.Empty:
                    pass
            
            self.result_queue.put((frame, results, method, bboxes, prediction_data, total_proc_time))
            process_count += 1
            
            elapsed = time.perf_counter() - start_time
            if elapsed > 0:
                self.process_fps = process_count / elapsed
    
    def draw_frame(self, frame, results, method, bboxes, prediction_data):
        """Draw all visualizations on frame."""
        h, w = frame.shape[:2]
        scale_x = w / PROCESS_WIDTH
        scale_y = h / PROCESS_HEIGHT
        
        # Method color
        method_colors = {
            'mediapipe': (0, 255, 0),        # Green
            'yolo+mediapipe': (255, 255, 0), # Cyan
            'yolo_only': (0, 165, 255),      # Orange
            'none': (0, 0, 255)              # Red
        }
        color = method_colors.get(method, (255, 255, 255))
        
        # Draw YOLO bboxes if available
        if bboxes:
            for bbox in bboxes:
                x1, y1, x2, y2 = bbox[:4]
                # Scale to display resolution
                x1, x2 = int(x1 * scale_x), int(x2 * scale_x)
                y1, y2 = int(y1 * scale_y), int(y2 * scale_y)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 165, 255), 2)
                cv2.putText(frame, "YOLO", (x1, y1-5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 1)
        
        # Draw landmarks if available
        if results and results.multi_hand_landmarks:
            for hand_landmarks, handedness in zip(
                results.multi_hand_landmarks,
                results.multi_handedness
            ):
                label = handedness.classification[0].label
                hand_color = (0, 255, 0) if label == "Right" else (255, 100, 0)
                
                # Draw landmarks
                for lm in hand_landmarks.landmark:
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    cv2.circle(frame, (cx, cy), 4, hand_color, -1)
                
                # Draw connections
                for connection in self.mp_hands.HAND_CONNECTIONS:
                    start = hand_landmarks.landmark[connection[0]]
                    end = hand_landmarks.landmark[connection[1]]
                    pt1 = (int(start.x * w), int(start.y * h))
                    pt2 = (int(end.x * w), int(end.y * h))
                    cv2.line(frame, pt1, pt2, hand_color, 2)
                
                # Label
                wrist = hand_landmarks.landmark[0]
                cv2.putText(frame, label, 
                           (int(wrist.x * w) - 30, int(wrist.y * h) - 15),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, hand_color, 2)
        
        return frame
    
    def draw_overlay(self, frame, proc_time, method, detector_stats, pred_stats):
        """Draw performance overlay."""
        cv2.rectangle(frame, (5, 5), (300, 220), (0, 0, 0), -1)
        cv2.rectangle(frame, (5, 5), (300, 220), (0, 255, 0), 1)
        
        y = 22
        cv2.putText(frame, "HYBRID DETECTION (MP + YOLO)", (10, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
        
        # Method indicator
        y += 20
        method_colors = {
            'mediapipe': (0, 255, 0),
            'yolo+mediapipe': (255, 255, 0),
            'yolo_only': (0, 165, 255),
            'none': (0, 0, 255)
        }
        cv2.putText(frame, f"Method: {method}", (10, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, method_colors.get(method, (255,255,255)), 1)
        
        y += 18
        cv2.putText(frame, f"Display FPS: {self.perf.fps:.1f}", (10, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
        
        y += 16
        cv2.putText(frame, f"Process: {proc_time:.1f}ms | Latency: {self.perf.latency:.1f}ms", (10, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        
        # Detection stats
        y += 18
        cv2.putText(frame, "--- Detection Stats ---", (10, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        
        y += 16
        cv2.putText(frame, f"MediaPipe: {detector_stats['mediapipe_pct']:.1f}%", (10, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        
        y += 14
        cv2.putText(frame, f"YOLO+MP: {detector_stats['yolo+mediapipe_pct']:.1f}%", (10, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
        
        y += 14
        cv2.putText(frame, f"YOLO Only: {detector_stats['yolo_only_pct']:.1f}%", (10, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 165, 255), 1)
        
        y += 14
        cv2.putText(frame, f"None: {detector_stats['none_pct']:.1f}%", (10, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
        
        # LSTM stats
        if pred_stats:
            y += 18
            cv2.putText(frame, f"LSTM Predictions: {pred_stats.get('predictions', 0)}", (10, y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 255), 1)
        
        return frame
    
    def run(self):
        if not self.init_camera():
            return
        
        self.running = True
        
        capture_t = threading.Thread(target=self._capture_thread_func, daemon=True)
        process_t = threading.Thread(target=self._processing_thread_func, daemon=True)
        
        capture_t.start()
        process_t.start()
        
        print("\n" + "=" * 60)
        print("HYBRID HAND TRACKER (MediaPipe + YOLO)")
        print("=" * 60)
        print(f"  Detection: MediaPipe primary, YOLO backup")
        print(f"  LSTM: {'Enabled' if LSTM_ENABLED else 'Disabled'}")
        print(f"  GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'}")
        print("\nPress 'Q' to quit.")
        print("-" * 60)
        
        last_frame = None
        last_results = None
        last_method = 'none'
        last_bboxes = None
        last_pred_data = None
        last_proc_time = 0
        
        try:
            while self.running:
                try:
                    frame, results, method, bboxes, pred_data, proc_time = self.result_queue.get(timeout=0.033)
                    last_frame = frame
                    last_results = results
                    last_method = method
                    last_bboxes = bboxes
                    last_pred_data = pred_data
                    last_proc_time = proc_time
                except queue.Empty:
                    if last_frame is None:
                        continue
                    frame = last_frame
                    results = last_results
                    method = last_method
                    bboxes = last_bboxes
                    pred_data = last_pred_data
                    proc_time = last_proc_time
                
                self.perf.update(proc_time)
                
                detector_stats = self.detector.get_stats()
                pred_stats = self.predictor.get_stats() if self.predictor else None
                
                frame = self.draw_frame(frame, results, method, bboxes, pred_data)
                frame = self.draw_overlay(frame, proc_time, method, detector_stats, pred_stats)
                
                cv2.imshow("Hybrid Hand Tracking - Press Q", frame)
                
                if self.perf.frame_count % 60 == 0 and self.perf.frame_count > 0:
                    print(f"[Frame {self.perf.frame_count}] "
                          f"FPS: {self.perf.fps:.1f} | "
                          f"Process: {self.perf.process_time:.1f}ms | "
                          f"Method: {method} | "
                          f"MP: {detector_stats['mediapipe_pct']:.0f}% "
                          f"YOLO: {detector_stats['yolo_only_pct']:.0f}%")
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        
        except KeyboardInterrupt:
            pass
        
        finally:
            self.running = False
            capture_t.join(timeout=1)
            process_t.join(timeout=1)
            
            detector_stats = self.detector.get_stats()
            pred_stats = self.predictor.get_stats() if self.predictor else {}
            
            print("\n" + "=" * 60)
            print("SESSION STATISTICS")
            print("=" * 60)
            print(f"  Total Frames: {self.perf.frame_count}")
            print(f"  Average FPS: {self.perf.fps:.2f}")
            print(f"  Average Latency: {self.perf.latency:.2f}ms")
            print(f"  Average Process Time: {self.perf.process_time:.2f}ms")
            print("-" * 60)
            print("DETECTION METHOD USAGE")
            print(f"  MediaPipe: {detector_stats['mediapipe']} ({detector_stats['mediapipe_pct']:.1f}%)")
            print(f"  YOLO+MediaPipe: {detector_stats['yolo+mediapipe']} ({detector_stats['yolo+mediapipe_pct']:.1f}%)")
            print(f"  YOLO Only: {detector_stats['yolo_only']} ({detector_stats['yolo_only_pct']:.1f}%)")
            print(f"  None: {detector_stats['none']} ({detector_stats['none_pct']:.1f}%)")
            print("-" * 60)
            print("LSTM STATISTICS")
            print(f"  Predictions: {pred_stats.get('predictions', 0)}")
            print(f"  Training Iterations: {pred_stats.get('training_iterations', 0)}")
            print("=" * 60)
            
            self.detector.close()
            self.cap.release()
            cv2.destroyAllWindows()


def main():
    print("=" * 60)
    print("PYTORCH & GPU STATUS")
    print("=" * 60)
    print(f"  PyTorch: {torch.__version__}")
    print(f"  CUDA: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print("=" * 60 + "\n")
    
    tracker = HybridHandTracker()
    tracker.run()


if __name__ == "__main__":
    main()
