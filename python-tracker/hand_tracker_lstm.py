"""
HIGH-PERFORMANCE Multi-Threaded Hand Tracking with LSTM Prediction
Phase 2: Adding intelligence for smoother tracking

Architecture:
- Thread 1: Camera capture
- Thread 2: MediaPipe processing + LSTM prediction
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

from lstm_predictor import PredictiveTracker

# ========== CONFIGURATION ==========
WEBCAM_ID = 0
CAPTURE_WIDTH = 1280
CAPTURE_HEIGHT = 720
PROCESS_WIDTH = 640
PROCESS_HEIGHT = 480

# MediaPipe settings
MODEL_COMPLEXITY = 0
MAX_NUM_HANDS = 2
MIN_DETECTION_CONFIDENCE = 0.5
MIN_TRACKING_CONFIDENCE = 0.5

# Queue sizes
FRAME_QUEUE_SIZE = 2
RESULT_QUEUE_SIZE = 2

# LSTM settings
LSTM_BUFFER_SIZE = 10
LSTM_ENABLED = True

# Performance
FPS_BUFFER_SIZE = 60


class PerformanceMonitor:
    """Track performance metrics."""
    def __init__(self):
        self.fps_buffer = deque(maxlen=FPS_BUFFER_SIZE)
        self.latency_buffer = deque(maxlen=FPS_BUFFER_SIZE)
        self.process_buffer = deque(maxlen=FPS_BUFFER_SIZE)
        self.lstm_buffer = deque(maxlen=FPS_BUFFER_SIZE)
        self.last_time = time.perf_counter()
        self.frame_count = 0
        self.min_fps = float('inf')
        self.max_latency = 0
        self.frame_drops = 0
    
    def update(self, process_time_ms, lstm_time_ms=0):
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
        self.lstm_buffer.append(lstm_time_ms)
    
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
    
    @property
    def lstm_time(self):
        return np.mean(self.lstm_buffer) if self.lstm_buffer else 0


class ThreadedHandTrackerWithLSTM:
    """Multi-threaded hand tracking with LSTM prediction."""
    
    def __init__(self):
        self.running = False
        self.frame_queue = queue.Queue(maxsize=FRAME_QUEUE_SIZE)
        self.result_queue = queue.Queue(maxsize=RESULT_QUEUE_SIZE)
        
        # MediaPipe
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            model_complexity=MODEL_COMPLEXITY,
            max_num_hands=MAX_NUM_HANDS,
            min_detection_confidence=MIN_DETECTION_CONFIDENCE,
            min_tracking_confidence=MIN_TRACKING_CONFIDENCE
        )
        
        # LSTM Predictor
        self.predictor = None
        if LSTM_ENABLED:
            print("Initializing LSTM Predictor...")
            self.predictor = PredictiveTracker(buffer_size=LSTM_BUFFER_SIZE, device='cuda')
        
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
            
            # Resize and process
            small_frame = cv2.resize(frame, (PROCESS_WIDTH, PROCESS_HEIGHT))
            rgb_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(rgb_frame)
            
            mediapipe_time = (time.perf_counter() - proc_start) * 1000
            
            # LSTM prediction
            lstm_start = time.perf_counter()
            prediction_data = None
            if LSTM_ENABLED and self.predictor:
                prediction_data = self.predictor.update(results)
            lstm_time = (time.perf_counter() - lstm_start) * 1000
            
            total_proc_time = mediapipe_time + lstm_time
            
            if self.result_queue.full():
                try:
                    self.result_queue.get_nowait()
                except queue.Empty:
                    pass
            
            self.result_queue.put((frame, results, prediction_data, total_proc_time, lstm_time))
            process_count += 1
            
            elapsed = time.perf_counter() - start_time
            if elapsed > 0:
                self.process_fps = process_count / elapsed
    
    def draw_landmarks(self, frame, results, prediction_data):
        if not results or not results.multi_hand_landmarks:
            return frame
        
        h, w, _ = frame.shape
        
        for hand_landmarks, handedness in zip(
            results.multi_hand_landmarks,
            results.multi_handedness
        ):
            label = handedness.classification[0].label
            color = (0, 255, 0) if label == "Right" else (255, 100, 0)
            
            # Draw detected landmarks
            for lm in hand_landmarks.landmark:
                cx, cy = int(lm.x * w), int(lm.y * h)
                cv2.circle(frame, (cx, cy), 4, color, -1)
            
            # Draw connections
            for connection in self.mp_hands.HAND_CONNECTIONS:
                start = hand_landmarks.landmark[connection[0]]
                end = hand_landmarks.landmark[connection[1]]
                pt1 = (int(start.x * w), int(start.y * h))
                pt2 = (int(end.x * w), int(end.y * h))
                cv2.line(frame, pt1, pt2, color, 2)
            
            # Draw predicted position (ghost hand)
            if prediction_data and label in prediction_data.get('predicted', {}):
                pred = prediction_data['predicted'][label]
                for i in range(0, len(pred), 3):
                    px, py = int(pred[i] * w), int(pred[i + 1] * h)
                    # Draw prediction as smaller, transparent circles
                    ghost_color = (100, 255, 100) if label == "Right" else (255, 180, 100)
                    cv2.circle(frame, (px, py), 3, ghost_color, 1)
            
            # Label
            wrist = hand_landmarks.landmark[0]
            cv2.putText(frame, label, 
                       (int(wrist.x * w) - 30, int(wrist.y * h) - 15),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        return frame
    
    def draw_overlay(self, frame, proc_time, lstm_time, num_hands, pred_stats):
        cv2.rectangle(frame, (5, 5), (280, 200), (0, 0, 0), -1)
        cv2.rectangle(frame, (5, 5), (280, 200), (0, 255, 0), 1)
        
        y = 22
        cv2.putText(frame, "LSTM-ENHANCED TRACKING", (10, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        
        y += 20
        device = pred_stats.get('device', 'N/A') if pred_stats else 'N/A'
        cv2.putText(frame, f"Device: {device.upper()}", (10, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 1)
        
        y += 18
        cv2.putText(frame, f"Display FPS: {self.perf.fps:.1f}", (10, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        y += 18
        cv2.putText(frame, f"Capture FPS: {self.capture_fps:.1f}", (10, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        y += 18
        cv2.putText(frame, f"Process Time: {proc_time:.1f}ms", (10, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        y += 18
        cv2.putText(frame, f"LSTM Time: {lstm_time:.2f}ms", (10, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        y += 18
        cv2.putText(frame, f"Latency: {self.perf.latency:.1f}ms", (10, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        y += 18
        cv2.putText(frame, f"Hands: {num_hands} | Drops: {self.perf.frame_drops}", (10, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        if pred_stats:
            y += 18
            cv2.putText(frame, f"Predictions: {pred_stats.get('predictions', 0)}", (10, y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 255), 1)
            y += 16
            cv2.putText(frame, f"Training: {pred_stats.get('training_iterations', 0)} (loss: {pred_stats.get('last_loss', 0):.4f})", (10, y),
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
        print("LSTM-ENHANCED HAND TRACKER")
        print("=" * 60)
        print(f"  Threads: 3 (Capture, Process+LSTM, Display)")
        print(f"  LSTM enabled: {LSTM_ENABLED}")
        print(f"  LSTM buffer: {LSTM_BUFFER_SIZE} frames")
        print(f"  Device: {self.predictor.device if self.predictor else 'N/A'}")
        print("\nPress 'Q' to quit.")
        print("-" * 60)
        
        last_frame = None
        last_results = None
        last_pred_data = None
        last_proc_time = 0
        last_lstm_time = 0
        
        try:
            while self.running:
                try:
                    frame, results, pred_data, proc_time, lstm_time = self.result_queue.get(timeout=0.033)
                    last_frame = frame
                    last_results = results
                    last_pred_data = pred_data
                    last_proc_time = proc_time
                    last_lstm_time = lstm_time
                except queue.Empty:
                    if last_frame is None:
                        continue
                    frame = last_frame
                    results = last_results
                    pred_data = last_pred_data
                    proc_time = last_proc_time
                    lstm_time = last_lstm_time
                
                self.perf.update(proc_time, lstm_time)
                
                num_hands = len(results.multi_hand_landmarks) if results and results.multi_hand_landmarks else 0
                pred_stats = self.predictor.get_stats() if self.predictor else None
                
                frame = self.draw_landmarks(frame, results, pred_data)
                frame = self.draw_overlay(frame, proc_time, lstm_time, num_hands, pred_stats)
                
                cv2.imshow("LSTM Hand Tracking - Press Q", frame)
                
                if self.perf.frame_count % 60 == 0 and self.perf.frame_count > 0:
                    stats = self.predictor.get_stats() if self.predictor else {}
                    print(f"[Frame {self.perf.frame_count}] "
                          f"FPS: {self.perf.fps:.1f} | "
                          f"Process: {self.perf.process_time:.1f}ms | "
                          f"LSTM: {self.perf.lstm_time:.2f}ms | "
                          f"Latency: {self.perf.latency:.1f}ms | "
                          f"Predictions: {stats.get('predictions', 0)} | "
                          f"Hands: {num_hands}")
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        
        except KeyboardInterrupt:
            pass
        
        finally:
            self.running = False
            capture_t.join(timeout=1)
            process_t.join(timeout=1)
            
            pred_stats = self.predictor.get_stats() if self.predictor else {}
            
            print("\n" + "=" * 60)
            print("SESSION STATISTICS")
            print("=" * 60)
            print(f"  Total Frames: {self.perf.frame_count}")
            print(f"  Average FPS: {self.perf.fps:.2f}")
            print(f"  Min FPS: {self.perf.min_fps:.2f}")
            print(f"  Average Latency: {self.perf.latency:.2f}ms")
            print(f"  Max Latency: {self.perf.max_latency:.2f}ms")
            print(f"  Average Process Time: {self.perf.process_time:.2f}ms")
            print(f"  Average LSTM Time: {self.perf.lstm_time:.2f}ms")
            print(f"  Frame Drops: {self.perf.frame_drops}")
            print("-" * 60)
            print("LSTM STATISTICS")
            print(f"  Device: {pred_stats.get('device', 'N/A')}")
            print(f"  Total Predictions: {pred_stats.get('predictions', 0)}")
            print(f"  Training Iterations: {pred_stats.get('training_iterations', 0)}")
            print(f"  Final Loss: {pred_stats.get('last_loss', 0):.6f}")
            print("=" * 60)
            
            self.hands.close()
            self.cap.release()
            cv2.destroyAllWindows()


def main():
    print("=" * 60)
    print("PYTORCH & GPU STATUS")
    print("=" * 60)
    print(f"  PyTorch: {torch.__version__}")
    print(f"  CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  CUDA version: {torch.version.cuda}")
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print("=" * 60 + "\n")
    
    tracker = ThreadedHandTrackerWithLSTM()
    tracker.run()


if __name__ == "__main__":
    main()
