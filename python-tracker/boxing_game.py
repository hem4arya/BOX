"""
Boxing Game Prototype - Pygame + Hand Tracking
Uses the invincible tracker for real-time hand detection

Features:
- Fullscreen display
- Virtual gloves follow real hands
- Glove state changes with gesture (FIST/OPEN)
- Punching bag target with collision detection
- Depth-based glove scaling
"""

import pygame
import cv2
import numpy as np
import mediapipe as mp
import threading
import queue
import time
from collections import deque
from depth_manager import DepthCalibrator, ReachBar
from fusion_detector import PunchValidator, is_moving_toward_target
from occlusion_handler import OcclusionHandler

# Initialize Pygame
pygame.init()
pygame.mixer.init()

# Get screen info for fullscreen
info = pygame.display.Info()
SCREEN_WIDTH = info.current_w
SCREEN_HEIGHT = info.current_h

# Colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (200, 50, 50)
BLUE = (50, 100, 200)
GREEN = (50, 200, 50)
YELLOW = (255, 220, 50)
ORANGE = (255, 150, 50)
GOLD = (255, 215, 0)     # Gold for in-range punches
DARK_RED = (150, 30, 30)
DARK_BLUE = (30, 70, 150)

# Game settings
FPS = 60
GLOVE_BASE_SIZE = 80
TARGET_SIZE = 120
HIT_COOLDOWN = 0.5  # seconds


class HandTracker:
    """Simplified hand tracker for the boxing game."""
    
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            model_complexity=1,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.6
        )
        
        self.cap = None
        self.running = False
        self.data_queue = queue.Queue(maxsize=2)
        
        # Tracking state
        self.hand_data = {
            'Left': {'x': 0, 'y': 0, 'z': 0, 'gesture': 'UNKNOWN', 'visible': False},
            'Right': {'x': 0, 'y': 0, 'z': 0, 'gesture': 'UNKNOWN', 'visible': False}
        }
        
        self.fps = 0
    
    def start(self):
        """Start the hand tracking thread."""
        self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        self.running = True
        self.thread = threading.Thread(target=self._tracking_loop, daemon=True)
        self.thread.start()
        print("Hand tracker started")
    
    def _tracking_loop(self):
        """Main tracking loop."""
        fps_buffer = deque(maxlen=30)
        last_time = time.perf_counter()
        
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                continue
            
            frame = cv2.flip(frame, 1)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            results = self.hands.process(frame_rgb)
            
            # Reset visibility
            for label in ['Left', 'Right']:
                self.hand_data[label]['visible'] = False
            
            if results.multi_hand_landmarks:
                world_landmarks_list = results.multi_hand_world_landmarks if hasattr(results, 'multi_hand_world_landmarks') and results.multi_hand_world_landmarks else None
                
                for idx, (hand_landmarks, handedness) in enumerate(zip(
                    results.multi_hand_landmarks,
                    results.multi_handedness
                )):
                    label = handedness.classification[0].label
                    
                    # Get wrist position (normalized 0-1) for X/Y
                    wrist = hand_landmarks.landmark[0]
                    self.hand_data[label]['x'] = wrist.x
                    self.hand_data[label]['y'] = wrist.y
                    
                    # Use WORLD landmarks for Z (much better depth!)
                    if world_landmarks_list and idx < len(world_landmarks_list):
                        world_wrist = world_landmarks_list[idx].landmark[0]
                        self.hand_data[label]['z'] = world_wrist.z
                    else:
                        self.hand_data[label]['z'] = wrist.z  # fallback
                    
                    self.hand_data[label]['visible'] = True
                    
                    # Detect gesture from finger positions
                    gesture = self._detect_gesture(hand_landmarks)
                    self.hand_data[label]['gesture'] = gesture
            
            # Calculate FPS
            current = time.perf_counter()
            fps_buffer.append(1.0 / max(0.001, current - last_time))
            last_time = current
            self.fps = np.mean(fps_buffer)
            
            # Quick sleep to prevent CPU hogging
            time.sleep(0.01)
    
    def _detect_gesture(self, landmarks):
        """Simple gesture detection based on fingertip positions."""
        fingertips = [8, 12, 16, 20]  # Index, middle, ring, pinky tips
        mcp_joints = [5, 9, 13, 17]   # Base knuckles
        
        fingers_curled = 0
        
        for tip, mcp in zip(fingertips, mcp_joints):
            tip_y = landmarks.landmark[tip].y
            mcp_y = landmarks.landmark[mcp].y
            
            # If tip is below MCP, finger is curled
            if tip_y > mcp_y:
                fingers_curled += 1
        
        if fingers_curled >= 4:
            return 'FIST'
        elif fingers_curled <= 1:
            return 'OPEN'
        else:
            return 'UNKNOWN'
    
    def get_hand_data(self):
        """Get current hand data."""
        return self.hand_data.copy(), self.fps
    
    def stop(self):
        """Stop the tracker."""
        self.running = False
        if self.cap:
            self.cap.release()
        self.hands.close()


class PunchingBag:
    """Floating punching bag target."""
    
    def __init__(self, x, y, size):
        self.x = x
        self.y = y
        self.size = size
        self.base_size = size
        
        self.hit = False
        self.hit_time = 0
        self.hit_count = 0
        
        # Movement
        self.vx = 0
        self.vy = 0
        self.swing_time = 0
    
    def update(self, dt):
        """Update bag position and state."""
        self.swing_time += dt
        
        # Natural swing
        swing_x = np.sin(self.swing_time * 0.5) * 30
        swing_y = np.sin(self.swing_time * 0.7) * 15
        
        # Apply hit momentum
        self.x += self.vx * dt
        self.y += self.vy * dt
        
        # Dampen velocity
        self.vx *= 0.95
        self.vy *= 0.95
        
        # Return toward center
        center_x = SCREEN_WIDTH // 2
        center_y = SCREEN_HEIGHT // 2 - 50
        
        self.x += (center_x + swing_x - self.x) * 0.02
        self.y += (center_y + swing_y - self.y) * 0.02
        
        # Hit flash decay
        if self.hit and time.time() - self.hit_time > 0.15:
            self.hit = False
    
    def register_hit(self, direction_x, direction_y):
        """Register a hit on the bag."""
        self.hit = True
        self.hit_time = time.time()
        self.hit_count += 1
        
        # Add momentum from punch direction
        self.vx += direction_x * 500
        self.vy += direction_y * 300
    
    def draw(self, screen):
        """Draw the punching bag."""
        # Rope
        pygame.draw.line(screen, (100, 100, 100), 
                        (int(self.x), 0), 
                        (int(self.x), int(self.y - self.size)), 4)
        
        # Bag body
        color = YELLOW if self.hit else RED
        pygame.draw.circle(screen, color, 
                          (int(self.x), int(self.y)), 
                          self.size)
        pygame.draw.circle(screen, DARK_RED, 
                          (int(self.x), int(self.y)), 
                          self.size, 3)
        
        # Target circle
        pygame.draw.circle(screen, WHITE, 
                          (int(self.x), int(self.y)), 
                          self.size // 3, 2)
    
    def check_collision(self, glove_x, glove_y, glove_size):
        """Check if glove collides with bag."""
        dx = glove_x - self.x
        dy = glove_y - self.y
        dist = np.sqrt(dx * dx + dy * dy)
        
        return dist < (self.size + glove_size * 0.7)


class Glove:
    """Virtual boxing glove."""
    
    def __init__(self, side):
        self.side = side  # 'Left' or 'Right'
        self.x = 0
        self.y = 0
        self.z = 0
        self.prev_x = 0
        self.prev_y = 0
        self.vx = 0  # Velocity X
        self.vy = 0  # Velocity Y
        self.size = GLOVE_BASE_SIZE
        self.visible = False
        self.gesture = 'UNKNOWN'
        
        # Hit detection
        self.last_hit_time = 0
        
        # Depth/Reach (for 3D hit detection)
        self.reach_percent = 0.0
        self.in_range = False  # True when reach > 85%
        
        # Trail effect
        self.trail = deque(maxlen=5)
    
    def update(self, hand_data):
        """Update glove from hand data."""
        data = hand_data.get(self.side, {})
        
        self.visible = data.get('visible', False)
        self.gesture = data.get('gesture', 'UNKNOWN')
        
        if self.visible:
            # Store previous position for velocity
            self.prev_x = self.x
            self.prev_y = self.y
            
            # Map normalized coords to screen
            self.x = data['x'] * SCREEN_WIDTH
            self.y = data['y'] * SCREEN_HEIGHT
            self.z = data.get('z', 0)
            
            # Calculate velocity (pixels per frame)
            self.vx = self.x - self.prev_x
            self.vy = self.y - self.prev_y
            
            # Depth-based size scaling
            depth_scale = 1.0 + (-self.z * 3)
            depth_scale = max(0.6, min(1.8, depth_scale))
            self.size = int(GLOVE_BASE_SIZE * depth_scale)
            
            # Add to trail
            self.trail.append((self.x, self.y, self.size))
    
    def can_hit(self):
        """Check if glove can register a hit."""
        if time.time() - self.last_hit_time < HIT_COOLDOWN:
            return False
        return self.gesture == 'FIST'
    
    def register_hit(self):
        """Register a hit."""
        self.last_hit_time = time.time()
    
    def draw(self, screen):
        """Draw the glove."""
        if not self.visible:
            return
        
        # Colors based on side (GOLD override when in range + FIST)
        if self.in_range and self.gesture == 'FIST':
            main_color = GOLD
            dark_color = (180, 150, 0)
        elif self.side == 'Right':
            main_color = RED
            dark_color = DARK_RED
        else:
            main_color = BLUE
            dark_color = DARK_BLUE
        
        x, y = int(self.x), int(self.y)
        
        # Draw trail
        for i, (tx, ty, ts) in enumerate(self.trail):
            alpha = (i + 1) / len(self.trail) * 0.3
            trail_size = int(ts * 0.8)
            trail_surf = pygame.Surface((trail_size * 2, trail_size * 2), pygame.SRCALPHA)
            pygame.draw.circle(trail_surf, (*main_color[:3], int(50 * alpha)), 
                              (trail_size, trail_size), trail_size)
            screen.blit(trail_surf, (int(tx - trail_size), int(ty - trail_size)))
        
        if self.gesture == 'FIST':
            # Filled punch glove
            pygame.draw.circle(screen, main_color, (x, y), self.size)
            pygame.draw.circle(screen, dark_color, (x, y), self.size, 4)
            
            # Punch indicator (extra for in-range)
            if self.in_range:
                pygame.draw.circle(screen, WHITE, (x, y), self.size // 2, 3)
            pygame.draw.circle(screen, WHITE, (x, y), self.size // 3)
        else:
            # Open/guard glove (outline)
            pygame.draw.circle(screen, main_color, (x, y), self.size, 6)
            pygame.draw.circle(screen, dark_color, (x, y), self.size - 10, 3)
        
        # Gesture label with reach info
        if self.in_range and self.gesture == 'FIST':
            gesture_text = "PUNCH READY!"
            text_color = (255, 100, 100)  # Red when ready
        elif self.gesture == 'FIST':
            gesture_text = f"EXTEND! ({int(self.reach_percent*100)}%)"
            text_color = YELLOW
        else:
            gesture_text = f"OPEN ({int(self.reach_percent*100)}%)"
            text_color = (150, 150, 150)
        font = pygame.font.Font(None, 28)
        text = font.render(gesture_text, True, text_color)
        screen.blit(text, (x - text.get_width() // 2, y + self.size + 10))
        
        # Z Debug display
        z_text = font.render(f"Z:{self.z:+.3f}", True, (150, 150, 255))
        screen.blit(z_text, (x - z_text.get_width() // 2, y + self.size + 30))


class BoxingGame:
    """Main boxing game class."""
    
    def __init__(self):
        # Setup display (windowed for stability, can change to FULLSCREEN)
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT), 
                                               pygame.FULLSCREEN | pygame.DOUBLEBUF)
        pygame.display.set_caption("Boxing Prototype")
        
        self.clock = pygame.time.Clock()
        self.running = True
        
        # Game objects
        self.tracker = HandTracker()
        self.bag = PunchingBag(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2 - 50, TARGET_SIZE)
        self.left_glove = Glove('Left')
        self.right_glove = Glove('Right')
        
        # 3D Depth System
        self.depth_calibrator = DepthCalibrator()
        self.reach_bar = ReachBar(SCREEN_WIDTH - 60, 150, 40, 300)
        self.hit_threshold = 0.85  # 85% reach required for hit
        
        # Punch Validation (direction + energy filter)
        self.punch_validator = PunchValidator()
        
        # Occlusion Recovery
        self.occlusion_handler = OcclusionHandler(timeout=0.5)
        # Stats
        self.game_fps = 0
        self.tracking_fps = 0
        self.total_hits = 0
        self.calibrated = False
        
        # Fonts
        self.font_large = pygame.font.Font(None, 72)
        self.font_medium = pygame.font.Font(None, 48)
        self.font_small = pygame.font.Font(None, 32)
        
        # Hit flash
        self.flash_alpha = 0
    
    def start(self):
        """Start the game."""
        self.tracker.start()
        self.run()
    
    def run(self):
        """Main game loop."""
        while self.running:
            dt = self.clock.tick(FPS) / 1000.0
            
            self.handle_events()
            self.update(dt)
            self.draw()
            
            pygame.display.flip()
        
        self.cleanup()
    
    def handle_events(self):
        """Handle pygame events."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.running = False
                elif event.key == pygame.K_r:
                    # Reset score
                    self.total_hits = 0
                    self.bag.hit_count = 0
                # RIGHT hand calibration
                elif event.key == pygame.K_c:
                    self.depth_calibrator.calibrate_chin('Right')
                elif event.key == pygame.K_v:
                    self.depth_calibrator.calibrate_reach('Right')
                    self.calibrated = self.depth_calibrator.is_calibrated
                # LEFT hand calibration
                elif event.key == pygame.K_x:
                    self.depth_calibrator.calibrate_chin('Left')
                elif event.key == pygame.K_b:
                    self.depth_calibrator.calibrate_reach('Left')
                    self.calibrated = self.depth_calibrator.is_calibrated
    
    def update(self, dt):
        """Update game state."""
        # Get hand data
        hand_data, tracking_fps = self.tracker.get_hand_data()
        self.tracking_fps = tracking_fps
        self.game_fps = self.clock.get_fps()
        
        # Apply occlusion recovery (predicts position if hand temporarily lost)
        hand_data = self.occlusion_handler.update(hand_data)
        
        # Update gloves
        self.left_glove.update(hand_data)
        self.right_glove.update(hand_data)
        
        # Update depth calibrator per hand
        if self.right_glove.visible:
            self.depth_calibrator.update_z('Right', self.right_glove.z)
        if self.left_glove.visible:
            self.depth_calibrator.update_z('Left', self.left_glove.z)
        
        # Update reach percent for each glove (per-hand calibration)
        for glove in [self.left_glove, self.right_glove]:
            if glove.visible:
                glove.reach_percent = self.depth_calibrator.get_reach_percent(glove.side, glove.z)
                glove.in_range = glove.reach_percent >= self.hit_threshold
            else:
                glove.reach_percent = 0.0
                glove.in_range = False
        
        # Update bag
        self.bag.update(dt)
        
        # Check collisions with enhanced validation
        for glove in [self.left_glove, self.right_glove]:
            # Update energy filter with velocity magnitude
            vel_mag = np.sqrt(glove.vx**2 + glove.vy**2) / SCREEN_WIDTH  # Normalize
            self.punch_validator.energy_filter.update(vel_mag)
            
            if glove.visible and glove.can_hit():
                # Check 2D collision
                collision = self.bag.check_collision(glove.x, glove.y, glove.size)
                
                # Direction check - moving toward target?
                direction_ok = is_moving_toward_target(
                    (glove.vx, glove.vy),
                    (glove.x, glove.y),
                    (self.bag.x, self.bag.y)
                )
                
                # Full validation: collision + reach + fist + direction
                if collision and glove.in_range and direction_ok:
                    # Calculate punch direction for bag physics
                    dx = self.bag.x - glove.x
                    dy = self.bag.y - glove.y
                    dist = max(1, np.sqrt(dx * dx + dy * dy))
                    
                    self.bag.register_hit(dx / dist, dy / dist)
                    glove.register_hit()
                    self.total_hits += 1
                    self.flash_alpha = 100
        
        # Decay flash
        if self.flash_alpha > 0:
            self.flash_alpha = max(0, self.flash_alpha - 300 * dt)
    
    def draw(self):
        """Draw the game."""
        # Background
        self.screen.fill((30, 30, 40))
        
        # Draw gym floor perspective lines
        for i in range(0, SCREEN_WIDTH, 100):
            pygame.draw.line(self.screen, (40, 40, 50), 
                            (i, SCREEN_HEIGHT), 
                            (SCREEN_WIDTH // 2, SCREEN_HEIGHT // 3), 1)
        
        # Draw punching bag
        self.bag.draw(self.screen)
        
        # Draw gloves
        self.left_glove.draw(self.screen)
        self.right_glove.draw(self.screen)
        
        # Draw reach bar
        reach = max(self.left_glove.reach_percent, self.right_glove.reach_percent)
        self.reach_bar.draw(self.screen, reach, pygame)
        
        # Hit flash overlay
        if self.flash_alpha > 0:
            flash_surf = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.SRCALPHA)
            flash_surf.fill((255, 255, 100, int(self.flash_alpha)))
            self.screen.blit(flash_surf, (0, 0))
        
        # UI
        self.draw_ui()
    
    def draw_ui(self):
        """Draw UI elements."""
        # Title
        title = self.font_large.render("BOXING PROTOTYPE", True, WHITE)
        self.screen.blit(title, (SCREEN_WIDTH // 2 - title.get_width() // 2, 20))
        
        # Score
        score_text = self.font_medium.render(f"HITS: {self.total_hits}", True, YELLOW)
        self.screen.blit(score_text, (SCREEN_WIDTH // 2 - score_text.get_width() // 2, 90))
        
        # Calibration status
        if self.calibrated:
            cal_text = self.font_small.render("CALIBRATED ✓ (FIST + 85% Reach)", True, GREEN)
        else:
            cal_text = self.font_small.render("RIGHT: C=Chin V=Reach | LEFT: X=Chin B=Reach", True, ORANGE)
        self.screen.blit(cal_text, (SCREEN_WIDTH // 2 - cal_text.get_width() // 2, 130))
        
        # FPS display
        fps_text = self.font_small.render(
            f"Game: {self.game_fps:.0f} FPS | Tracking: {self.tracking_fps:.0f} FPS", 
            True, (150, 150, 150))
        self.screen.blit(fps_text, (10, SCREEN_HEIGHT - 60))
        
        # Instructions
        instr = self.font_small.render("FIST+EXTEND to hit | R=Reset | ESC=Exit", 
                                       True, (100, 100, 100))
        self.screen.blit(instr, (SCREEN_WIDTH // 2 - instr.get_width() // 2, 
                                 SCREEN_HEIGHT - 35))
        
        # Hand status
        left_status = "LEFT: " + ("DETECTED" if self.left_glove.visible else "NOT FOUND")
        right_status = "RIGHT: " + ("DETECTED" if self.right_glove.visible else "NOT FOUND")
        
        left_color = GREEN if self.left_glove.visible else RED
        right_color = GREEN if self.right_glove.visible else RED
        
        left_text = self.font_small.render(left_status, True, left_color)
        right_text = self.font_small.render(right_status, True, right_color)
        
        self.screen.blit(left_text, (10, 10))
        self.screen.blit(right_text, (SCREEN_WIDTH - right_text.get_width() - 10, 10))
    
    def cleanup(self):
        """Cleanup resources."""
        self.tracker.stop()
        pygame.quit()


def main():
    print("=" * 60)
    print("BOXING PROTOTYPE - 3D DEPTH")
    print("=" * 60)
    print("PER-HAND CALIBRATION:")
    print("  RIGHT HAND: C = Chin | V = Reach")
    print("  LEFT HAND:  X = Chin | B = Reach")
    print("")
    print("HIT REQUIREMENTS:")
    print("  1. FIST gesture (fingers curled)")
    print("  2. Reach >= 85% (arm extended)")
    print("  3. Collision with target")
    print("")
    print("Watch glove for 'PUNCH READY!' when all conditions met")
    print("=" * 60)
    
    game = BoxingGame()
    game.start()


if __name__ == "__main__":
    main()
