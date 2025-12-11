"""
Depth Manager - 3D Arm Extension Detection (Per-Hand Calibration)
Handles Z-axis calibration for true 3D interaction.

MediaPipe Z convention:
- Typically: Lower Z = closer to camera (extended arm)
- But this varies! Use debug mode to check.

Usage:
1. Calibrate each hand separately:
   - Right: C=Chin, V=Reach
   - Left:  X=Chin, B=Reach
2. get_reach_percent(hand, z) returns 0.0-1.0
"""

import numpy as np
from collections import deque


class DepthCalibrator:
    """
    Per-hand calibration for Z-axis reach detection.
    """
    
    def __init__(self):
        # Per-hand calibration (None = uncalibrated)
        self.calibration = {
            'Left':  {'chin': None, 'reach': None},
            'Right': {'chin': None, 'reach': None},
        }
        
        # Smoothing buffers per hand
        self.z_buffers = {
            'Left': deque(maxlen=5),
            'Right': deque(maxlen=5),
        }
        
        # Last raw Z values (for debug display)
        self.last_raw_z = {'Left': 0.0, 'Right': 0.0}
        self.last_reach_percent = {'Left': 0.0, 'Right': 0.0}
        
        # Debug mode
        self.debug_print = True
        self.frame_count = 0
        
        print("=" * 50)
        print("DEPTH CALIBRATOR - PER-HAND")
        print("=" * 50)
        print("RIGHT HAND: C = Chin | V = Reach")
        print("LEFT HAND:  X = Chin | B = Reach")
        print("=" * 50)
    
    def update_z(self, hand, raw_z):
        """Update with new Z value from hand tracking."""
        self.last_raw_z[hand] = raw_z
        self.z_buffers[hand].append(raw_z)
        
        # Debug print every 30 frames
        self.frame_count += 1
        if self.debug_print and self.frame_count % 30 == 0:
            left_z = self.last_raw_z['Left']
            right_z = self.last_raw_z['Right']
            print(f"Z DEBUG: R={right_z:+.4f} | L={left_z:+.4f}")
    
    def get_smoothed_z(self, hand):
        """Get smoothed Z value for a hand."""
        if len(self.z_buffers[hand]) == 0:
            return self.last_raw_z[hand]
        return np.mean(self.z_buffers[hand])
    
    def calibrate_chin(self, hand):
        """Calibrate chin position (arm retracted) for specific hand."""
        z = self.get_smoothed_z(hand)
        self.calibration[hand]['chin'] = z
        print(f"[CALIBRATE] {hand} CHIN: Z = {z:.4f}")
        self._check_calibration(hand)
        return z
    
    def calibrate_reach(self, hand):
        """Calibrate reach position (arm extended) for specific hand."""
        z = self.get_smoothed_z(hand)
        self.calibration[hand]['reach'] = z
        print(f"[CALIBRATE] {hand} REACH: Z = {z:.4f}")
        self._check_calibration(hand)
        return z
    
    def _check_calibration(self, hand):
        """Check if hand is fully calibrated (both chin and reach set)."""
        cal = self.calibration[hand]
        if cal['chin'] is not None and cal['reach'] is not None:
            z_range = abs(cal['chin'] - cal['reach'])
            direction = "UP" if cal['reach'] > cal['chin'] else "DOWN"
            print(f">>> {hand} FULLY CALIBRATED!")
            print(f"    Chin={cal['chin']:.4f}, Reach={cal['reach']:.4f}")
            print(f"    Range={z_range:.4f}, Z goes {direction} when extending")
    
    def is_hand_calibrated(self, hand):
        """Check if specific hand is fully calibrated."""
        cal = self.calibration[hand]
        return cal['chin'] is not None and cal['reach'] is not None
    
    @property
    def is_calibrated(self):
        """Check if at least one hand is fully calibrated."""
        return self.is_hand_calibrated('Left') or self.is_hand_calibrated('Right')
    
    def get_reach_percent(self, hand, current_z=None):
        """
        Get normalized reach percentage for a specific hand.
        Auto-detects if Z goes up or down when extending.
        
        Returns:
            float: 0.0 (chin) to 1.0 (full reach)
        """
        if current_z is None:
            current_z = self.get_smoothed_z(hand)
        
        cal = self.calibration[hand]
        z_chin = cal['chin']
        z_reach = cal['reach']
        
        # Not calibrated - return 50%
        if z_chin is None or z_reach is None:
            return 0.5
        
        z_range = z_reach - z_chin  # Can be positive or negative
        
        if abs(z_range) < 0.01:
            # Invalid range (too small)
            return 0.5
        
        # Normalize: 0 at chin, 1 at reach
        # This works regardless of whether Z goes up or down
        reach = (current_z - z_chin) / z_range
        
        # Clamp to valid range
        reach = max(0.0, min(1.0, reach))
        
        self.last_reach_percent[hand] = reach
        return reach
    
    def is_in_range(self, hand, threshold=0.5):
        """Check if arm is extended enough to hit."""
        return self.last_reach_percent[hand] >= threshold
    
    def get_debug_text(self, hand):
        """Get debug text for HUD display."""
        cal = self.calibration[hand]
        z = self.last_raw_z[hand]
        reach = self.last_reach_percent[hand]
        
        if cal['calibrated']:
            return f"Z:{z:+.3f} R:{reach:.0%}"
        else:
            return f"Z:{z:+.3f} [NOT CAL]"


class ReachBar:
    """
    Visual reach indicator for Pygame HUD.
    """
    
    def __init__(self, x, y, width, height):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.threshold = 0.85  # 85% hit threshold
    
    def draw(self, surface, reach_percent, pygame):
        """Draw the reach bar on pygame surface."""
        # Background
        pygame.draw.rect(surface, (40, 40, 40), 
                        (self.x, self.y, self.width, self.height))
        
        # Border
        pygame.draw.rect(surface, (100, 100, 100), 
                        (self.x, self.y, self.width, self.height), 2)
        
        # Fill (from bottom up)
        fill_height = int(self.height * min(1.0, reach_percent))
        fill_y = self.y + self.height - fill_height
        
        # Color based on threshold
        if reach_percent >= self.threshold:
            color = (0, 255, 0)  # Green - in range
        elif reach_percent >= 0.3:
            color = (255, 255, 0)  # Yellow - getting close
        else:
            color = (255, 100, 100)  # Red - too far
        
        pygame.draw.rect(surface, color,
                        (self.x + 2, fill_y, self.width - 4, fill_height))
        
        # Threshold line
        thresh_y = self.y + self.height - int(self.height * self.threshold)
        pygame.draw.line(surface, (255, 255, 255),
                        (self.x, thresh_y), (self.x + self.width, thresh_y), 2)
        
        # Label
        font = pygame.font.Font(None, 24)
        text = font.render(f"{int(reach_percent * 100)}%", True, (255, 255, 255))
        surface.blit(text, (self.x, self.y + self.height + 5))
        
        # "REACH" label
        label = font.render("REACH", True, (200, 200, 200))
        surface.blit(label, (self.x - 5, self.y - 20))


def test_depth_manager():
    """Test the depth calibrator."""
    print("=" * 60)
    print("DEPTH MANAGER TEST")
    print("=" * 60)
    
    calibrator = DepthCalibrator()
    
    # Simulate Right hand chin position
    for _ in range(5):
        calibrator.update_z('Right', 0.05)
    calibrator.calibrate_chin('Right')
    
    # Simulate Right hand reach position
    for _ in range(5):
        calibrator.update_z('Right', -0.10)
    calibrator.calibrate_reach('Right')
    
    # Test reach percentages
    test_values = [0.05, 0.0, -0.05, -0.10, -0.15]
    print("\nReach Test (Right hand):")
    for z in test_values:
        reach = calibrator.get_reach_percent('Right', z)
        in_range = "IN RANGE" if calibrator.is_in_range('Right') else "out of range"
        print(f"  Z={z:+.2f} -> Reach={reach:.0%} ({in_range})")
    
    print("\n" + "=" * 60)


if __name__ == "__main__":
    test_depth_manager()
