"""
HIGH-PERFORMANCE Multi-Threaded Hand Tracking (No PyTorch)
Optimized for maximum FPS and minimum latency

Architecture:
- Thread 1: Camera capture only (puts frames in queue)
- Thread 2: MediaPipe processing only (reads from queue, processes, puts results)
- Thread 3: Display/render only (main thread)
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

# ========== CONFIGURATION ==========
WEBCAM_ID = 0
CAPTURE_WIDTH = 1280
CAPTURE_HEIGHT = 720
PROCESS_WIDTH = 640   # Resize BEFORE MediaPipe
PROCESS_HEIGHT = 480

# MediaPipe OPTIMIZED settings (for speed)
MODEL_COMPLEXITY = 0  # 0=Fastest
MAX_NUM_HANDS = 2
MIN_DETECTION_CONFIDENCE = 0.5
MIN_TRACKING_CONFIDENCE = 0.5

# Queue sizes
FRAME_QUEUE_SIZE = 2
RESULT_QUEUE_SIZE = 2

# Performance tracking
FPS_BUFFER_SIZE = 60


class PerformanceMonitor:
    """Track performance metrics."""
    def __init__(self):
        self.fps_buffer = deque(maxlen=FPS_BUFFER_SIZE)
        self.latency_buffer = deque(maxlen=FPS_BUFFER_SIZE)
        self.process_buffer = deque(maxlen=FPS_BUFFER_SIZE)
        self.last_time = time.perf_counter()
        self.frame_count = 0
        self.start_time = time.perf_counter()
        
        # Track min/max
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
            if fps < self.min_fps:
                self.min_fps = fps
        
        latency = frame_time * 1000
        self.latency_buffer.append(latency)
        if latency > self.max_latency:
            self.max_latency = latency
        
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
    
    def get_cpu_usage(self):
        return psutil.Process(os.getpid()).cpu_percent()


class ThreadedHandTracker:
    """Multi-threaded hand tracking pipeline (3 threads)."""
    
    def __init__(self):
        # Threading control
        self.running = False
        
        # Queues with specified sizes
        self.frame_queue = queue.Queue(maxsize=FRAME_QUEUE_SIZE)
        self.result_queue = queue.Queue(maxsize=RESULT_QUEUE_SIZE)
        
        # MediaPipe setup
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            model_complexity=MODEL_COMPLEXITY,
            max_num_hands=MAX_NUM_HANDS,
            min_detection_confidence=MIN_DETECTION_CONFIDENCE,
            min_tracking_confidence=MIN_TRACKING_CONFIDENCE
        )
        
        # Camera
        self.cap = None
        
        # Performance
        self.perf = PerformanceMonitor()
        self.capture_fps = 0
        self.process_fps = 0
        
        # Thread references
        self.capture_thread = None
        self.process_thread = None
        
    def init_camera(self):
        """Initialize camera with all optimized settings."""
        print("Initializing camera...")
        
        # OpenCV optimizations
        cv2.setUseOptimized(True)
        cv2.setNumThreads(4)
        print(f"  OpenCV optimized: {cv2.useOptimized()}")
        
        # Open camera with DirectShow
        self.cap = cv2.VideoCapture(WEBCAM_ID, cv2.CAP_DSHOW)
        
        if not self.cap.isOpened():
            self.cap = cv2.VideoCapture(WEBCAM_ID)
        
        if not self.cap.isOpened():
            print("ERROR: Cannot open camera!")
            return False
        
        # Apply all optimized settings
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAPTURE_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAPTURE_HEIGHT)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # CRITICAL
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        
        # Verify settings
        actual_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = self.cap.get(cv2.CAP_PROP_FPS)
        buffer_size = self.cap.get(cv2.CAP_PROP_BUFFERSIZE)
        
        print(f"  Capture: {actual_w}x{actual_h} @ {actual_fps:.0f} FPS")
        print(f"  Buffer size: {buffer_size}")
        print(f"  Process resolution: {PROCESS_WIDTH}x{PROCESS_HEIGHT}")
        return True
    
    def _capture_thread_func(self):
        """THREAD 1: Camera capture only."""
        capture_count = 0
        start_time = time.perf_counter()
        
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                continue
            
            # Flip immediately
            frame = cv2.flip(frame, 1)
            
            # Drop old frames if queue full
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
        """THREAD 2: MediaPipe processing only."""
        process_count = 0
        start_time = time.perf_counter()
        
        while self.running:
            try:
                frame, timestamp = self.frame_queue.get(timeout=0.1)
            except queue.Empty:
                continue
            
            proc_start = time.perf_counter()
            
            # RESIZE before MediaPipe (critical optimization)
            small_frame = cv2.resize(frame, (PROCESS_WIDTH, PROCESS_HEIGHT))
            
            # Convert to RGB
            rgb_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
            
            # Process with MediaPipe
            results = self.hands.process(rgb_frame)
            
            proc_time = (time.perf_counter() - proc_start) * 1000
            
            # Drop old results if queue full
            if self.result_queue.full():
                try:
                    self.result_queue.get_nowait()
                except queue.Empty:
                    pass
            
            self.result_queue.put((frame, results, proc_time, timestamp))
            process_count += 1
            
            elapsed = time.perf_counter() - start_time
            if elapsed > 0:
                self.process_fps = process_count / elapsed
    
    def draw_landmarks(self, frame, results):
        """Draw hand landmarks on frame."""
        if not results or not results.multi_hand_landmarks:
            return frame
        
        h, w, _ = frame.shape
        
        for hand_landmarks, handedness in zip(
            results.multi_hand_landmarks,
            results.multi_handedness
        ):
            hand_label = handedness.classification[0].label
            color = (0, 255, 0) if hand_label == "Right" else (255, 100, 0)
            
            # Draw landmarks
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
            
            # Label
            wrist = hand_landmarks.landmark[0]
            cv2.putText(frame, hand_label, 
                       (int(wrist.x * w) - 30, int(wrist.y * h) - 15),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        return frame
    
    def draw_overlay(self, frame, proc_time, num_hands):
        """Draw performance overlay."""
        # Background
        cv2.rectangle(frame, (5, 5), (250, 160), (0, 0, 0), -1)
        cv2.rectangle(frame, (5, 5), (250, 160), (0, 255, 0), 1)
        
        y = 25
        cv2.putText(frame, f"NO PYTORCH - Pure MediaPipe", (10, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
        
        y += 20
        cv2.putText(frame, f"Display FPS: {self.perf.fps:.1f}", (10, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        y += 18
        cv2.putText(frame, f"Capture FPS: {self.capture_fps:.1f}", (10, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        y += 18
        cv2.putText(frame, f"Process FPS: {self.process_fps:.1f}", (10, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        y += 18
        cv2.putText(frame, f"Process Time: {proc_time:.1f}ms", (10, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        y += 18
        cv2.putText(frame, f"Latency: {self.perf.latency:.1f}ms", (10, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        y += 18
        cv2.putText(frame, f"Hands: {num_hands} | Drops: {self.perf.frame_drops}", (10, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        return frame
    
    def run(self):
        """THREAD 3 (Main): Display/render loop."""
        if not self.init_camera():
            return
        
        self.running = True
        
        # Start worker threads
        self.capture_thread = threading.Thread(target=self._capture_thread_func, daemon=True, name="CaptureThread")
        self.process_thread = threading.Thread(target=self._processing_thread_func, daemon=True, name="ProcessThread")
        
        self.capture_thread.start()
        self.process_thread.start()
        
        print("\n" + "=" * 60)
        print("MULTI-THREADED HAND TRACKER (NO PYTORCH)")
        print("=" * 60)
        print(f"  Threads: 3 (Capture, Process, Display)")
        print(f"  Queue sizes: frame={FRAME_QUEUE_SIZE}, result={RESULT_QUEUE_SIZE}")
        print(f"  Model complexity: {MODEL_COMPLEXITY}")
        print(f"  Detection confidence: {MIN_DETECTION_CONFIDENCE}")
        print(f"  Tracking confidence: {MIN_TRACKING_CONFIDENCE}")
        print("\nPress 'Q' to quit.")
        print("-" * 60)
        
        last_frame = None
        last_results = None
        last_proc_time = 0
        
        try:
            while self.running:
                try:
                    frame, results, proc_time, timestamp = self.result_queue.get(timeout=0.033)
                    last_frame = frame
                    last_results = results
                    last_proc_time = proc_time
                except queue.Empty:
                    if last_frame is None:
                        continue
                    frame = last_frame
                    results = last_results
                    proc_time = last_proc_time
                
                self.perf.update(proc_time)
                
                num_hands = len(results.multi_hand_landmarks) if results and results.multi_hand_landmarks else 0
                
                frame = self.draw_landmarks(frame, results)
                frame = self.draw_overlay(frame, proc_time, num_hands)
                
                cv2.imshow("Multi-Threaded Hand Tracking (No PyTorch) - Press Q", frame)
                
                # Print stats every 60 frames
                if self.perf.frame_count % 60 == 0 and self.perf.frame_count > 0:
                    cpu = self.perf.get_cpu_usage()
                    print(f"[Frame {self.perf.frame_count}] "
                          f"FPS: {self.perf.fps:.1f} | "
                          f"Process: {self.perf.process_time:.1f}ms | "
                          f"Latency: {self.perf.latency:.1f}ms | "
                          f"Hands: {num_hands} | "
                          f"Drops: {self.perf.frame_drops} | "
                          f"CPU: {cpu:.0f}%")
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        
        except KeyboardInterrupt:
            pass
        
        finally:
            self.running = False
            
            self.capture_thread.join(timeout=1)
            self.process_thread.join(timeout=1)
            
            # Final stats
            print("\n" + "=" * 60)
            print("SESSION STATISTICS")
            print("=" * 60)
            print(f"  Total Frames: {self.perf.frame_count}")
            print(f"  Average FPS: {self.perf.fps:.2f}")
            print(f"  Min FPS: {self.perf.min_fps:.2f}")
            print(f"  Average Latency: {self.perf.latency:.2f}ms")
            print(f"  Max Latency: {self.perf.max_latency:.2f}ms")
            print(f"  Average Process Time: {self.perf.process_time:.2f}ms")
            print(f"  Frame Drops: {self.perf.frame_drops}")
            print(f"  CPU Usage: {self.perf.get_cpu_usage():.0f}%")
            print("=" * 60)
            
            self.hands.close()
            self.cap.release()
            cv2.destroyAllWindows()


def main():
    print("=" * 60)
    print("CONFIGURATION CHECK")
    print("=" * 60)
    print(f"  PyTorch: NOT LOADED (removed)")
    print(f"  MediaPipe model_complexity: {MODEL_COMPLEXITY}")
    print(f"  min_detection_confidence: {MIN_DETECTION_CONFIDENCE}")
    print(f"  min_tracking_confidence: {MIN_TRACKING_CONFIDENCE}")
    print(f"  Capture resolution: {CAPTURE_WIDTH}x{CAPTURE_HEIGHT}")
    print(f"  Process resolution: {PROCESS_WIDTH}x{PROCESS_HEIGHT}")
    print(f"  Frame queue size: {FRAME_QUEUE_SIZE}")
    print(f"  Result queue size: {RESULT_QUEUE_SIZE}")
    print("=" * 60 + "\n")
    
    tracker = ThreadedHandTracker()
    tracker.run()


if __name__ == "__main__":
    main()
