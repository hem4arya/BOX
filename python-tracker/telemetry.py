"""
Telemetry Recorder - Punch Data Logger
Records hand movement data for future AI training

Logs:
- Time
- Wrist position (x, y, z)
- Velocity (x, y, z)
- Acceleration (x, y, z)
- State (idle/firing/retracting)
"""

import os
import csv
import time
from collections import deque
from datetime import datetime
import numpy as np


class PunchRecorder:
    """
    Records punch telemetry data for AI training.
    Auto-saves when a punch is detected.
    """
    
    def __init__(self, output_dir='punches'):
        self.output_dir = output_dir
        self.buffer = deque(maxlen=300)  # ~10 seconds at 30fps
        self.is_recording = False
        self.punch_detected = False
        self.punch_count = 0
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"PunchRecorder initialized. Output: {output_dir}/")
    
    def log_frame(self, hand_label, position, velocity, acceleration, state='idle'):
        """
        Log a single frame of data.
        
        Args:
            hand_label: 'Left' or 'Right'
            position: (x, y, z) wrist position
            velocity: (x, y, z) velocity
            acceleration: (x, y, z) acceleration
            state: 'idle', 'firing', or 'retracting'
        """
        timestamp = time.time()
        
        frame_data = {
            'time': timestamp,
            'hand': hand_label,
            'pos_x': position[0] if position is not None else 0,
            'pos_y': position[1] if position is not None else 0,
            'pos_z': position[2] if position is not None else 0,
            'vel_x': velocity[0] if velocity is not None else 0,
            'vel_y': velocity[1] if velocity is not None else 0,
            'vel_z': velocity[2] if velocity is not None else 0,
            'acc_x': acceleration[0] if acceleration is not None else 0,
            'acc_y': acceleration[1] if acceleration is not None else 0,
            'acc_z': acceleration[2] if acceleration is not None else 0,
            'state': state
        }
        
        self.buffer.append(frame_data)
        
        # Detect punch start
        if state == 'firing' and not self.punch_detected:
            self.punch_detected = True
            self.is_recording = True
        
        # Save if punch just completed
        if state == 'idle' and self.punch_detected:
            self.save_recording()
            self.punch_detected = False
    
    def save_recording(self):
        """Save buffered data to CSV file."""
        if len(self.buffer) == 0:
            return
        
        self.punch_count += 1
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{self.output_dir}/punch_{timestamp}_{self.punch_count:03d}.csv"
        
        fieldnames = ['time', 'hand', 'pos_x', 'pos_y', 'pos_z', 
                      'vel_x', 'vel_y', 'vel_z', 
                      'acc_x', 'acc_y', 'acc_z', 'state']
        
        try:
            with open(filename, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(self.buffer)
            
            print(f"Saved punch recording: {filename} ({len(self.buffer)} frames)")
        except Exception as e:
            print(f"Error saving recording: {e}")
        
        # Clear buffer after save
        self.buffer.clear()
        self.is_recording = False
    
    def get_stats(self):
        """Get recorder statistics."""
        return {
            'punches_recorded': self.punch_count,
            'buffer_size': len(self.buffer),
            'is_recording': self.is_recording
        }


class TelemetryDisplay:
    """
    Real-time telemetry display helper.
    Formats motion data for HUD display.
    """
    
    def __init__(self):
        self.last_accel = {'Left': 0, 'Right': 0}
        self.last_vel = {'Left': 0, 'Right': 0}
        self.last_state = {'Left': 'idle', 'Right': 'idle'}
        self.peak_accel = {'Left': 0, 'Right': 0}
    
    def update(self, hand_label, velocity, acceleration, state):
        """Update telemetry values."""
        if velocity is not None:
            self.last_vel[hand_label] = np.linalg.norm(velocity)
        
        if acceleration is not None:
            accel_mag = np.linalg.norm(acceleration)
            self.last_accel[hand_label] = accel_mag
            self.peak_accel[hand_label] = max(self.peak_accel[hand_label], accel_mag)
        
        self.last_state[hand_label] = state
    
    def get_display_text(self, hand_label):
        """Get formatted display text for hand."""
        return (
            f"{hand_label}: Accel={self.last_accel[hand_label]:.4f} "
            f"Vel={self.last_vel[hand_label]:.4f} "
            f"State={self.last_state[hand_label]}"
        )
    
    def get_peak_text(self, hand_label):
        """Get peak acceleration text."""
        return f"Peak: {self.peak_accel[hand_label]:.4f}"
    
    def reset_peaks(self):
        """Reset peak values."""
        self.peak_accel = {'Left': 0, 'Right': 0}


def test_recorder():
    """Test the punch recorder."""
    print("=" * 60)
    print("TELEMETRY RECORDER TEST")
    print("=" * 60)
    
    recorder = PunchRecorder()
    display = TelemetryDisplay()
    
    # Simulate some data
    print("\nSimulating punch sequence...")
    
    for i in range(50):
        t = i / 30.0
        
        # Simulate position
        pos = np.array([0.5 + np.sin(t) * 0.1, 0.5, -0.1 * t])
        vel = np.array([np.cos(t) * 0.3, 0, -0.1])
        acc = np.array([-np.sin(t) * 0.5, 0, 0])
        
        # Simulate state transition
        if 15 <= i <= 25:
            state = 'firing'
        elif 26 <= i <= 35:
            state = 'retracting'
        else:
            state = 'idle'
        
        recorder.log_frame('Right', pos, vel, acc, state)
        display.update('Right', vel, acc, state)
        
        if i % 10 == 0:
            print(display.get_display_text('Right'))
    
    print("\n" + "=" * 60)
    print("Stats:", recorder.get_stats())
    print("=" * 60)


if __name__ == "__main__":
    test_recorder()
