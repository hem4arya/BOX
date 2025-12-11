"""
INVINCIBLE Hand Tracking - Full Pipeline
Hierarchy: MediaPipe → YOLO → Optical Flow → Motion Energy

This tracker NEVER loses the hand position because:
1. MediaPipe: Best accuracy when hand is visible
2. YOLO: Catches hands even with some motion blur
3. Optical Flow: Tracks pixels when shape is lost
4. Motion Energy: Detects movement blobs as last resort
"""

import cv2
import mediapipe as mp
import time
import numpy as np
from collections import deque
import threading
import queue
import torch

from hybrid_detector import HybridHandDetector
from lstm_predictor import PredictiveTracker
from motion_tracker import CombinedMotionTracker
from hand_renderer import HandRenderer
from ik_solver import SimpleIKSolver
from ballistic_engine import BallisticEngine
from telemetry import PunchRecorder, TelemetryDisplay

# ========== CONFIGURATION ==========
WEBCAM_ID = 0
CAPTURE_WIDTH = 1280
CAPTURE_HEIGHT = 720
PROCESS_WIDTH = 640
PROCESS_HEIGHT = 480

FRAME_QUEUE_SIZE = 2
RESULT_QUEUE_SIZE = 2

LSTM_ENABLED = True
LSTM_BUFFER_SIZE = 10

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
    
    @property
    def fps(self):
        return np.mean(self.fps_buffer) if self.fps_buffer else 0
    
    @property
    def latency(self):
        return np.mean(self.latency_buffer) if self.latency_buffer else 0
    
    @property
    def process_time(self):
        return np.mean(self.process_buffer) if self.process_buffer else 0


class InvincibleTracker:
    """
    Multi-threaded hand tracking that NEVER loses track.
    Uses 4-layer fallback system.
    """
    
    def __init__(self):
        self.running = False
        self.frame_queue = queue.Queue(maxsize=FRAME_QUEUE_SIZE)
        self.result_queue = queue.Queue(maxsize=RESULT_QUEUE_SIZE)
        
        # Detection layers
        print("Initializing INVINCIBLE Tracker...")
        print("  [1/4] Hybrid Detector (MediaPipe + YOLO)...")
        self.detector = HybridHandDetector()
        
        print("  [2/4] LSTM Predictor...")
        self.predictor = PredictiveTracker(buffer_size=LSTM_BUFFER_SIZE, device='cuda') if LSTM_ENABLED else None
        
        print("  [3/4] Motion Tracker (Optical Flow + Motion Energy)...")
        self.motion_tracker = CombinedMotionTracker()
        
        print("  [4/4] MediaPipe reference...")
        self.mp_hands = mp.solutions.hands
        
        # Hand renderer for skeleton visualization
        print("  [5/5] Hand Renderer...")
        self.renderer = HandRenderer()
        
        # IK Solver for gesture detection
        print("  [6/6] IK Solver (Gesture Detection)...")
        self.ik_solver = SimpleIKSolver()
        
        # Ballistic Engine and Telemetry
        print("  [7/7] Ballistic Engine & Telemetry...")
        self.ballistic_engine = BallisticEngine()
        self.recorder = PunchRecorder()
        self.telemetry_display = TelemetryDisplay()
        
        # Store last good landmarks for ghost drawing
        self.last_good_landmarks = {'Left': None, 'Right': None}
        self.last_positions = {'Left': None, 'Right': None}
        
        # Gesture detection state
        self.current_gestures = {'Left': 'UNKNOWN', 'Right': 'UNKNOWN'}
        self.finger_curls = {'Left': None, 'Right': None}
        
        # Pose locking (prevent flickering during fast movement)
        self.last_stable_gesture = {'Left': 'UNKNOWN', 'Right': 'UNKNOWN'}
        self.pose_locked = {'Left': False, 'Right': False}
        self.POSE_LOCK_VELOCITY = 0.15  # Lock pose if velocity > this
        
        self.cap = None
        self.perf = PerformanceMonitor()
        self.capture_fps = 0
        
        # Method usage stats
        self.method_stats = {
            'mediapipe': 0,
            'yolo+mediapipe': 0,
            'yolo_only': 0,
            'optical_flow': 0,
            'coasting': 0,
            'motion_energy': 0,
            'none': 0
        }
        
        print("  ✓ All systems ready!")
    
    def init_camera(self):
        print("\nInitializing camera...")
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
    
    def _capture_thread(self):
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
                except queue.Empty:
                    pass
            
            self.frame_queue.put((frame, time.perf_counter()))
            capture_count += 1
            
            elapsed = time.perf_counter() - start_time
            if elapsed > 0:
                self.capture_fps = capture_count / elapsed
    
    def _process_thread(self):
        while self.running:
            try:
                frame, timestamp = self.frame_queue.get(timeout=0.1)
            except queue.Empty:
                continue
            
            proc_start = time.perf_counter()
            
            # Resize for processing
            small_frame = cv2.resize(frame, (PROCESS_WIDTH, PROCESS_HEIGHT))
            gray = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)
            
            # === LAYER 1 & 2: MediaPipe + YOLO ===
            results, method, bboxes = self.detector.detect(small_frame)
            
            fallback_centers = {}
            final_method = method
            
            if results and results.multi_hand_landmarks:
                # Detection successful - update motion tracker
                self.motion_tracker.update_from_detection(results, gray)
                
                # LSTM prediction
                if self.predictor:
                    self.predictor.update(results)
            
            elif method in ['yolo_only', 'none']:
                # === LAYER 3 & 4: Optical Flow / Motion Energy ===
                for hand_label in ['Left', 'Right']:
                    success, center, fallback_method, data = self.motion_tracker.get_fallback_position(gray, hand_label)
                    
                    if success and center:
                        fallback_centers[hand_label] = {
                            'center': center,
                            'method': fallback_method,
                            'data': data
                        }
                        final_method = fallback_method
            
            # === [NEW] BALLISTIC ENGINE UPDATE ===
            ballistic_outputs = self.ballistic_engine.update(results)
            
            # Telemetry Logging
            for label in ['Left', 'Right']:
                hand = self.ballistic_engine.hands[label]
                self.recorder.log_frame(
                    label, 
                    hand.camera_pos, 
                    hand.velocity, 
                    hand.acceleration, 
                    hand.get_state_name()
                )
                self.telemetry_display.update(
                    label,
                    hand.velocity,
                    hand.acceleration,
                    hand.get_state_name()
                )
            
            total_time = (time.perf_counter() - proc_start) * 1000
            
            # Update optical flow prev frame for both hand trackers
            for label in ['Left', 'Right']:
                self.motion_tracker.hand_trackers[label].update_prev_frame(gray)
            
            if self.result_queue.full():
                try:
                    self.result_queue.get_nowait()
                except queue.Empty:
                    pass
            
            self.result_queue.put((frame, small_frame, results, final_method, bboxes, fallback_centers, ballistic_outputs, total_time))
    
    def draw_frame(self, frame, small_frame, results, method, bboxes, fallback_centers, ballistic_outputs=None):
        h, w = frame.shape[:2]
        scale_x = w / PROCESS_WIDTH
        scale_y = h / PROCESS_HEIGHT
        
        # Draw YOLO bboxes
        if bboxes:
            for bbox in bboxes:
                x1, y1, x2, y2 = bbox[:4]
                x1, x2 = int(x1 * scale_x), int(x2 * scale_x)
                y1, y2 = int(y1 * scale_y), int(y2 * scale_y)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 165, 255), 2)
        
        # Draw MediaPipe landmarks and detect gestures
        if results and results.multi_hand_landmarks:
            # Get world landmarks for 3D gesture detection
            world_landmarks_list = results.multi_hand_world_landmarks if hasattr(results, 'multi_hand_world_landmarks') and results.multi_hand_world_landmarks else [None] * len(results.multi_hand_landmarks)
            
            for idx, (hand_landmarks, handedness) in enumerate(zip(
                results.multi_hand_landmarks,
                results.multi_handedness
            )):
                label = handedness.classification[0].label
                
                # Draw skeleton using renderer
                frame = self.renderer.draw_skeleton(frame, hand_landmarks, label)
                
                # Store last good landmarks for ghost drawing
                self.last_good_landmarks[label] = hand_landmarks
                
                # Store position (wrist)
                wrist = hand_landmarks.landmark[0]
                self.last_positions[label] = (int(wrist.x * w), int(wrist.y * h))
                
                # Process world landmarks for gesture detection
                world_lm = world_landmarks_list[idx] if idx < len(world_landmarks_list) else None
                if world_lm is not None:
                    # Calculate finger angles
                    angles = self.ik_solver.calculate_angles(world_lm)
                    
                    # Get velocity from ballistic engine
                    hand_velocity = np.linalg.norm(self.ballistic_engine.hands[label].velocity)
                    
                    # POSE LOCKING: Trust past gesture if moving fast
                    if hand_velocity > self.POSE_LOCK_VELOCITY:
                        # Motion blur zone - camera unreliable
                        gesture = self.last_stable_gesture[label]
                        self.pose_locked[label] = True
                    else:
                        # Stable zone - camera reliable
                        gesture = self.ik_solver.detect_gesture(angles, world_lm)
                        self.last_stable_gesture[label] = gesture
                        self.pose_locked[label] = False
                    
                    self.current_gestures[label] = gesture
                    
                    # Store finger curls
                    self.finger_curls[label] = self.ik_solver.get_finger_info(angles)
                    
                    # Display gesture near hand
                    gesture_x = int(wrist.x * w)
                    gesture_y = int(wrist.y * h) + 40
                    
                    # Gesture color
                    gesture_colors = {
                        'FIST': (0, 0, 255),      # Red
                        'OPEN': (0, 255, 0),      # Green
                        'POINTING': (255, 255, 0), # Cyan
                        'PEACE': (255, 0, 255),   # Magenta
                        'THUMBS_UP': (0, 215, 255), # Gold
                        'UNKNOWN': (128, 128, 128)
                    }
                    color = gesture_colors.get(gesture, (255, 255, 255))
                    
                    # Draw gesture label (add LOCKED indicator)
                    display_text = f"{gesture} [LOCKED]" if self.pose_locked[label] else gesture
                    cv2.putText(frame, display_text, (gesture_x - 50, gesture_y),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)
        
        # Draw fallback ghost skeletons (Optical Flow / Coasting)
        for hand_label, data in fallback_centers.items():
            center = data['center']
            fallback_method = data['method']
            
            # Scale center to display resolution (handle numpy arrays)
            cx = int(float(center[0]) * scale_x)
            cy = int(float(center[1]) * scale_y)
            
            # Get last known landmarks for this hand
            last_lm = self.last_good_landmarks.get(hand_label)
            last_pos = self.last_positions.get(hand_label)
            
            if fallback_method == 'optical_flow' and last_lm is not None:
                # Calculate flow vector to shift the ghost skeleton
                if last_pos:
                    flow_vector = (cx - last_pos[0], cy - last_pos[1])
                else:
                    flow_vector = (0, 0)
                
                # Draw ghost skeleton shifted by flow
                frame = self.renderer.draw_ghost(frame, last_lm, flow_vector, hand_label)
                
                # Update last position
                self.last_positions[hand_label] = (cx, cy)
            
            elif fallback_method == 'coasting' and last_lm is not None:
                # Draw coasting ghost with velocity prediction
                coasting_frame = data.get('data', {}).get('coasting_frame', 1)
                velocity = data.get('data', {}).get('velocity', (0, 0))
                
                frame = self.renderer.draw_coasting_ghost(frame, last_lm, velocity, coasting_frame)
            
            elif fallback_method == 'motion_energy':
                # Red circle for motion blob (no skeleton available)
                cv2.circle(frame, (cx, cy), 25, (0, 0, 255), -1)
                cv2.putText(frame, "FAST MOTION", (cx - 50, cy - 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
        # Draw tracking label at top
        frame = self.renderer.draw_tracking_label(frame, method)
        
        # POSE LOCK VISUAL: Blue border when any pose is locked
        if any(self.pose_locked.values()):
            cv2.rectangle(frame, (0, 0), (w-1, h-1), (255, 100, 0), 4)  # Blue border
            cv2.putText(frame, "POSE LOCKED", (w//2 - 80, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 100, 0), 2, cv2.LINE_AA)
        
        return frame
    
    def draw_overlay(self, frame, proc_time, method, ballistic_outputs=None):
        cv2.rectangle(frame, (5, 5), (320, 180), (0, 0, 0), -1)
        cv2.rectangle(frame, (5, 5), (320, 180), (0, 255, 0), 1)
        
        y = 22
        cv2.putText(frame, "INVINCIBLE TRACKER", (10, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        
        # Method with color
        method_colors = {
            'mediapipe': (0, 255, 0),
            'yolo+mediapipe': (255, 255, 0),
            'yolo_only': (0, 165, 255),
            'optical_flow': (255, 0, 255),
            'coasting': (255, 255, 0),
            'motion_energy': (0, 0, 255),
            'none': (128, 128, 128)
        }
        y += 20
        cv2.putText(frame, f"Method: {method}", (10, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, method_colors.get(method, (255, 255, 255)), 1)
        
        y += 18
        cv2.putText(frame, f"FPS: {self.perf.fps:.1f} | Latency: {self.perf.latency:.1f}ms", (10, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        
        y += 16
        cv2.putText(frame, f"Process: {proc_time:.1f}ms", (10, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        
        # Detection breakdown
        y += 18
        cv2.putText(frame, "--- Method Usage ---", (10, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        
        total = sum(self.method_stats.values()) or 1
        methods_display = [
            ('MP', 'mediapipe', (0, 255, 0)),
            ('YOLO+MP', 'yolo+mediapipe', (255, 255, 0)),
            ('YOLO', 'yolo_only', (0, 165, 255)),
            ('OptFlow', 'optical_flow', (255, 0, 255)),
            ('Coast', 'coasting', (255, 255, 0)),
            ('Motion', 'motion_energy', (0, 0, 255)),
        ]
        
        y += 16
        for name, key, color in methods_display[:3]:
            pct = self.method_stats[key] / total * 100
            cv2.putText(frame, f"{name}: {pct:.0f}%", (10 + methods_display.index((name, key, color)) * 80, y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)
        
        y += 14
        for name, key, color in methods_display[3:]:
            pct = self.method_stats[key] / total * 100
            cv2.putText(frame, f"{name}: {pct:.0f}%", (10 + (methods_display.index((name, key, color)) - 3) * 100, y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)
        
        # Motion tracker status
        motion_status = self.motion_tracker.get_status()
        y += 18
        cv2.putText(frame, f"L: {motion_status['left_mode']} | R: {motion_status['right_mode']}", (10, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 255), 1)
        
        # Check Ballistic Overrides
        if ballistic_outputs:
            h, w = frame.shape[:2]
            for label, data in ballistic_outputs.items():
                if data['is_active']:
                    # Draw Red Ballistic Hand
                    pos = data['position']
                    if self.last_good_landmarks[label]:
                        frame = self.renderer.draw_ballistic_hand(frame, self.last_good_landmarks[label], pos)
                    
                    # Draw Trajectory Line
                    if self.last_positions[label]:
                        start_pt = self.last_positions[label]
                        end_pt = (int(pos[0] * w), int(pos[1] * h))
                        cv2.line(frame, start_pt, end_pt, (0, 0, 255), 2, cv2.LINE_AA)
                        cv2.circle(frame, end_pt, 5, (0, 0, 255), -1)
                        cv2.putText(frame, "BALLISTIC", (end_pt[0] + 10, end_pt[1]), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                
                # Telemetry Text
                text = self.telemetry_display.get_display_text(label)
                peak = self.telemetry_display.get_peak_text(label)
                
                ty = 180 if label == 'Right' else 240
                color = (0, 0, 255) if data['is_active'] else (0, 255, 0)
                cv2.putText(frame, text, (10, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
                cv2.putText(frame, peak, (10, ty + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

        return frame
    
    def run(self):
        if not self.init_camera():
            return
        
        self.running = True
        
        capture_t = threading.Thread(target=self._capture_thread, daemon=True)
        process_t = threading.Thread(target=self._process_thread, daemon=True)
        
        capture_t.start()
        process_t.start()
        
        print("\n" + "=" * 60)
        print("INVINCIBLE HAND TRACKER")
        print("=" * 60)
        print("  Fallback Chain: MediaPipe → YOLO → OpticalFlow → MotionEnergy")
        print(f"  GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'}")
        print("\nPunch fast to test blur tracking!")
        print("Press 'Q' to quit.")
        print("-" * 60)
        
        last_data = None
        
        try:
            while self.running:
                try:
                    data = self.result_queue.get(timeout=0.033)
                    last_data = data
                except queue.Empty:
                    if last_data is None:
                        continue
                    data = last_data
                
                frame, small_frame, results, method, bboxes, fallback_centers, ballistic_outputs, proc_time = data
                
                self.perf.update(proc_time)
                self.method_stats[method] = self.method_stats.get(method, 0) + 1
                
                frame = self.draw_frame(frame, small_frame, results, method, bboxes, fallback_centers, ballistic_outputs)
                frame = self.draw_overlay(frame, proc_time, method, ballistic_outputs)
                
                cv2.imshow("INVINCIBLE Tracker - Press Q", frame)
                
                if self.perf.frame_count % 60 == 0 and self.perf.frame_count > 0:
                    print(f"[Frame {self.perf.frame_count}] "
                          f"FPS: {self.perf.fps:.1f} | "
                          f"Method: {method} | "
                          f"Process: {self.perf.process_time:.1f}ms")
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        
        except KeyboardInterrupt:
            pass
        
        finally:
            self.running = False
            capture_t.join(timeout=1)
            process_t.join(timeout=1)
            
            total = sum(self.method_stats.values()) or 1
            
            print("\n" + "=" * 60)
            print("SESSION STATISTICS")
            print("=" * 60)
            print(f"  Total Frames: {self.perf.frame_count}")
            print(f"  Average FPS: {self.perf.fps:.2f}")
            print(f"  Average Latency: {self.perf.latency:.2f}ms")
            print("-" * 60)
            print("METHOD USAGE")
            for method, count in sorted(self.method_stats.items(), key=lambda x: -x[1]):
                pct = count / total * 100
                print(f"  {method}: {count} ({pct:.1f}%)")
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
    
    tracker = InvincibleTracker()
    tracker.run()


if __name__ == "__main__":
    main()
