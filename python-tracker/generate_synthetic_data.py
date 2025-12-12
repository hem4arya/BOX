"""
Synthetic Punch Data Generator
==============================
Generates 12,500 biomechanically accurate punch trajectories for classifier training.

Output: synthetic_punches.npy containing:
- 2,500 jabs     (lead hand straight)
- 2,500 crosses  (rear hand straight)  
- 2,500 hooks    (curved lateral)
- 2,500 uppercuts (upward motion)
- 2,500 idle     (no punch)

Each sample: 30 frames × 10 features = 300 values
Features per frame: wrist_x, wrist_y, elbow_x, elbow_y, shoulder_x, shoulder_y,
                    elbow_angle, velocity, direction_x, direction_y
"""

import numpy as np
from pathlib import Path

# ════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ════════════════════════════════════════════════════════════════════════════
FRAMES_PER_SAMPLE = 30
SAMPLES_PER_CLASS = 2500
CLASSES = ['jab', 'cross', 'hook', 'uppercut', 'idle']

# Starting positions (normalized 0-1 coordinates)
# These represent typical guard position
GUARD_POSITION = {
    'left': {
        'shoulder': (0.35, 0.35),
        'elbow': (0.30, 0.50),
        'wrist': (0.32, 0.45)
    },
    'right': {
        'shoulder': (0.65, 0.35),
        'elbow': (0.70, 0.50),
        'wrist': (0.68, 0.45)
    }
}

# Target positions for extended punch (center of body)
CENTER_TARGET = (0.50, 0.40)


# ════════════════════════════════════════════════════════════════════════════
# UTILITY FUNCTIONS
# ════════════════════════════════════════════════════════════════════════════
def ease_out(t):
    """
    Ease-out curve: Quick start, gradual stop.
    Mimics real punch deceleration on impact.
    """
    return 1 - (1 - t) ** 3

def ease_in_out(t):
    """
    Ease-in-out curve: Slow start, fast middle, slow end.
    Good for hooks and uppercuts.
    """
    if t < 0.5:
        return 4 * t ** 3
    else:
        return 1 - (-2 * t + 2) ** 3 / 2

def noise(scale=0.02):
    """Add realistic jitter to coordinates."""
    return np.random.normal(0, scale)

def calculate_velocity(positions):
    """Calculate velocity from position history."""
    if len(positions) < 2:
        return 0.0
    dx = positions[-1][0] - positions[-2][0]
    dy = positions[-1][1] - positions[-2][1]
    return np.sqrt(dx**2 + dy**2)

def calculate_elbow_angle(shoulder, elbow, wrist):
    """
    Calculate elbow angle using dot product.
    90° = bent, 180° = straight
    """
    # Vectors: upper arm and forearm
    upper = (elbow[0] - shoulder[0], elbow[1] - shoulder[1])
    fore = (wrist[0] - elbow[0], wrist[1] - elbow[1])
    
    # Dot product
    dot = upper[0] * fore[0] + upper[1] * fore[1]
    mag_upper = np.sqrt(upper[0]**2 + upper[1]**2)
    mag_fore = np.sqrt(fore[0]**2 + fore[1]**2)
    
    if mag_upper < 0.001 or mag_fore < 0.001:
        return 90.0
    
    cos_angle = np.clip(dot / (mag_upper * mag_fore), -1, 1)
    angle_rad = np.arccos(cos_angle)
    return 180 - np.degrees(angle_rad)  # Convert to elbow angle


# ════════════════════════════════════════════════════════════════════════════
# PUNCH GENERATORS
# ════════════════════════════════════════════════════════════════════════════

def generate_jab(hand='left'):
    """
    Generate JAB trajectory.
    
    Lead hand straight punch:
    - Quick forward extension
    - Wrist moves toward center
    - Elbow angle: 90° → 170°
    - Duration: 8-15 frames (fast)
    """
    frames = FRAMES_PER_SAMPLE
    duration = np.random.randint(8, 15)
    guard = GUARD_POSITION[hand]
    data = []
    positions = []
    
    # Speed variation
    speed_factor = np.random.uniform(0.8, 1.2)
    
    for i in range(frames):
        t = min(1.0, i / (duration * speed_factor))
        progress = ease_out(t)
        
        # Shoulder stays relatively fixed
        shoulder = (
            guard['shoulder'][0] + noise(0.01),
            guard['shoulder'][1] + noise(0.01)
        )
        
        # Wrist extends toward target
        start_wrist = guard['wrist']
        wrist = (
            start_wrist[0] + (CENTER_TARGET[0] - start_wrist[0]) * progress + noise(),
            start_wrist[1] + (CENTER_TARGET[1] - start_wrist[1]) * progress * 0.3 + noise()
        )
        
        # Elbow follows, creating straight line
        elbow_progress = progress * 0.8  # Elbow lags slightly
        elbow = (
            (shoulder[0] + wrist[0]) / 2 + noise(),
            (shoulder[1] + wrist[1]) / 2 + 0.05 + noise()  # Slight offset
        )
        
        positions.append(wrist)
        
        # Calculate derived features
        elbow_angle = 90 + 80 * progress  # 90° → 170°
        velocity = calculate_velocity(positions)
        
        if len(positions) >= 2:
            dx = positions[-1][0] - positions[-2][0]
            dy = positions[-1][1] - positions[-2][1]
        else:
            dx, dy = 0, 0
        
        data.append([
            wrist[0], wrist[1],
            elbow[0], elbow[1],
            shoulder[0], shoulder[1],
            elbow_angle, velocity, dx, dy
        ])
    
    return np.array(data, dtype=np.float32)


def generate_cross(hand='right'):
    """
    Generate CROSS trajectory.
    
    Rear hand power punch:
    - Longer wind-up, more rotation
    - Wrist moves toward center with hip rotation
    - Duration: 10-18 frames
    """
    frames = FRAMES_PER_SAMPLE
    duration = np.random.randint(10, 18)
    guard = GUARD_POSITION[hand]
    data = []
    positions = []
    
    speed_factor = np.random.uniform(0.8, 1.2)
    
    for i in range(frames):
        t = min(1.0, i / (duration * speed_factor))
        progress = ease_out(t)
        
        # Shoulder rotates forward with cross
        rotation = progress * 0.08  # Hip/shoulder rotation factor
        shoulder = (
            guard['shoulder'][0] - rotation + noise(0.01),
            guard['shoulder'][1] + noise(0.01)
        )
        
        # Wrist extends toward target with more power
        start_wrist = guard['wrist']
        wrist = (
            start_wrist[0] + (CENTER_TARGET[0] - start_wrist[0]) * progress + noise(),
            start_wrist[1] + (CENTER_TARGET[1] - start_wrist[1]) * progress * 0.4 + noise()
        )
        
        # Elbow follows
        elbow = (
            (shoulder[0] + wrist[0]) / 2 + noise(),
            (shoulder[1] + wrist[1]) / 2 + 0.05 + noise()
        )
        
        positions.append(wrist)
        
        elbow_angle = 90 + 90 * progress  # Full extension
        velocity = calculate_velocity(positions)
        
        if len(positions) >= 2:
            dx = positions[-1][0] - positions[-2][0]
            dy = positions[-1][1] - positions[-2][1]
        else:
            dx, dy = 0, 0
        
        data.append([
            wrist[0], wrist[1],
            elbow[0], elbow[1],
            shoulder[0], shoulder[1],
            elbow_angle, velocity, dx, dy
        ])
    
    return np.array(data, dtype=np.float32)


def generate_hook(hand='left'):
    """
    Generate HOOK trajectory.
    
    Curved lateral punch:
    - Arcing motion from side to center
    - Elbow stays bent (~90°)
    - Shoulder rotates significantly
    - Duration: 12-20 frames
    """
    frames = FRAMES_PER_SAMPLE
    duration = np.random.randint(12, 20)
    guard = GUARD_POSITION[hand]
    data = []
    positions = []
    
    speed_factor = np.random.uniform(0.8, 1.2)
    
    # Hook swings in an arc
    hook_radius = 0.15  # Arc radius
    
    for i in range(frames):
        t = min(1.0, i / (duration * speed_factor))
        progress = ease_in_out(t)
        
        # Arc angle (0 to 90 degrees)
        arc_angle = progress * np.pi / 2
        
        # Shoulder rotates with hook
        rotation = progress * 0.10
        if hand == 'left':
            shoulder = (
                guard['shoulder'][0] + rotation + noise(0.01),
                guard['shoulder'][1] + noise(0.01)
            )
            # Wrist follows arc from left side
            wrist = (
                guard['wrist'][0] + hook_radius * np.sin(arc_angle) + noise(),
                guard['wrist'][1] - 0.05 * progress + noise()  # Slight rise
            )
        else:
            shoulder = (
                guard['shoulder'][0] - rotation + noise(0.01),
                guard['shoulder'][1] + noise(0.01)
            )
            wrist = (
                guard['wrist'][0] - hook_radius * np.sin(arc_angle) + noise(),
                guard['wrist'][1] - 0.05 * progress + noise()
            )
        
        # Elbow stays bent
        elbow = (
            (shoulder[0] + wrist[0]) / 2 + noise(),
            (shoulder[1] + wrist[1]) / 2 - 0.03 + noise()
        )
        
        positions.append(wrist)
        
        elbow_angle = 90 + 30 * progress  # Stays relatively bent
        velocity = calculate_velocity(positions)
        
        if len(positions) >= 2:
            dx = positions[-1][0] - positions[-2][0]
            dy = positions[-1][1] - positions[-2][1]
        else:
            dx, dy = 0, 0
        
        data.append([
            wrist[0], wrist[1],
            elbow[0], elbow[1],
            shoulder[0], shoulder[1],
            elbow_angle, velocity, dx, dy
        ])
    
    return np.array(data, dtype=np.float32)


def generate_uppercut(hand='right'):
    """
    Generate UPPERCUT trajectory.
    
    Upward punch:
    - Significant Y movement (upward)
    - Minimal X change
    - Elbow opens during ascent
    - Duration: 10-16 frames
    """
    frames = FRAMES_PER_SAMPLE
    duration = np.random.randint(10, 16)
    guard = GUARD_POSITION[hand]
    data = []
    positions = []
    
    speed_factor = np.random.uniform(0.8, 1.2)
    
    for i in range(frames):
        t = min(1.0, i / (duration * speed_factor))
        progress = ease_in_out(t)
        
        # Shoulder dips slightly then rises
        dip = np.sin(t * np.pi) * 0.03  # Slight dip and rise
        shoulder = (
            guard['shoulder'][0] + noise(0.01),
            guard['shoulder'][1] + dip + noise(0.01)
        )
        
        # Wrist rises significantly (Y decreases in screen coords)
        wrist = (
            guard['wrist'][0] + progress * 0.08 + noise(),  # Slight forward
            guard['wrist'][1] - progress * 0.20 + noise()   # Significant upward
        )
        
        # Elbow follows upward motion
        elbow = (
            guard['elbow'][0] + progress * 0.04 + noise(),
            guard['elbow'][1] - progress * 0.10 + noise()
        )
        
        positions.append(wrist)
        
        # Elbow extends during uppercut
        elbow_angle = 80 + 70 * progress
        velocity = calculate_velocity(positions)
        
        if len(positions) >= 2:
            dx = positions[-1][0] - positions[-2][0]
            dy = positions[-1][1] - positions[-2][1]
        else:
            dx, dy = 0, 0
        
        data.append([
            wrist[0], wrist[1],
            elbow[0], elbow[1],
            shoulder[0], shoulder[1],
            elbow_angle, velocity, dx, dy
        ])
    
    return np.array(data, dtype=np.float32)


def generate_idle(hand='left'):
    """
    Generate IDLE trajectory.
    
    No punch - small random movements:
    - Random fluctuations around guard
    - No consistent direction
    - Slow velocity
    - Elbow stays bent
    """
    frames = FRAMES_PER_SAMPLE
    guard = GUARD_POSITION[hand]
    data = []
    positions = []
    
    # Random walk parameters
    drift_scale = 0.01
    
    current = {
        'shoulder': list(guard['shoulder']),
        'elbow': list(guard['elbow']),
        'wrist': list(guard['wrist'])
    }
    
    for i in range(frames):
        # Small random drift
        current['shoulder'][0] += np.random.uniform(-drift_scale, drift_scale)
        current['shoulder'][1] += np.random.uniform(-drift_scale, drift_scale)
        current['elbow'][0] += np.random.uniform(-drift_scale * 1.5, drift_scale * 1.5)
        current['elbow'][1] += np.random.uniform(-drift_scale * 1.5, drift_scale * 1.5)
        current['wrist'][0] += np.random.uniform(-drift_scale * 2, drift_scale * 2)
        current['wrist'][1] += np.random.uniform(-drift_scale * 2, drift_scale * 2)
        
        wrist = tuple(current['wrist'])
        positions.append(wrist)
        
        # Elbow stays bent
        elbow_angle = 85 + np.random.uniform(-10, 10)
        velocity = calculate_velocity(positions)
        
        if len(positions) >= 2:
            dx = positions[-1][0] - positions[-2][0]
            dy = positions[-1][1] - positions[-2][1]
        else:
            dx, dy = 0, 0
        
        data.append([
            current['wrist'][0], current['wrist'][1],
            current['elbow'][0], current['elbow'][1],
            current['shoulder'][0], current['shoulder'][1],
            elbow_angle, velocity, dx, dy
        ])
    
    return np.array(data, dtype=np.float32)


# ════════════════════════════════════════════════════════════════════════════
# DATASET GENERATION
# ════════════════════════════════════════════════════════════════════════════

def generate_dataset(samples_per_class=SAMPLES_PER_CLASS, output_path='synthetic_punches.npy'):
    """
    Generate complete synthetic punch dataset.
    
    Returns:
        X: (N, 30, 10) array of trajectories
        y: (N,) array of class labels
    """
    print("=" * 60)
    print("SYNTHETIC PUNCH DATA GENERATOR")
    print("=" * 60)
    print(f"Samples per class: {samples_per_class}")
    print(f"Total samples: {samples_per_class * 5}")
    print()
    
    all_data = []
    all_labels = []
    
    class_map = {
        'jab': 0,
        'cross': 1,
        'hook': 2,
        'uppercut': 3,
        'idle': 4
    }
    
    generators = {
        'jab': lambda: generate_jab(np.random.choice(['left', 'right'])),
        'cross': lambda: generate_cross(np.random.choice(['left', 'right'])),
        'hook': lambda: generate_hook(np.random.choice(['left', 'right'])),
        'uppercut': lambda: generate_uppercut(np.random.choice(['left', 'right'])),
        'idle': lambda: generate_idle(np.random.choice(['left', 'right']))
    }
    
    for punch_type in CLASSES:
        print(f"Generating {samples_per_class} {punch_type.upper()} samples...")
        
        for i in range(samples_per_class):
            sample = generators[punch_type]()
            all_data.append(sample)
            all_labels.append(class_map[punch_type])
            
            if (i + 1) % 500 == 0:
                print(f"  {i + 1}/{samples_per_class} complete")
    
    # Convert to numpy arrays
    X = np.array(all_data, dtype=np.float32)
    y = np.array(all_labels, dtype=np.int32)
    
    # Shuffle
    indices = np.random.permutation(len(X))
    X = X[indices]
    y = y[indices]
    
    print()
    print(f"Dataset shape: X={X.shape}, y={y.shape}")
    print(f"Class distribution:")
    for name, idx in class_map.items():
        count = np.sum(y == idx)
        print(f"  {name.upper()}: {count}")
    
    # Save
    output_path = Path(output_path)
    np.savez(output_path.with_suffix('.npz'), X=X, y=y, 
             class_names=np.array(['jab', 'cross', 'hook', 'uppercut', 'idle']))
    print()
    print(f"✓ Saved to {output_path.with_suffix('.npz')}")
    
    return X, y


# ════════════════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate synthetic punch training data')
    parser.add_argument('--samples', type=int, default=2500, help='Samples per class')
    parser.add_argument('--output', type=str, default='synthetic_punches', help='Output filename')
    args = parser.parse_args()
    
    X, y = generate_dataset(
        samples_per_class=args.samples,
        output_path=args.output
    )
    
    print()
    print("=" * 60)
    print("GENERATION COMPLETE!")
    print("=" * 60)
    print(f"Total samples: {len(y)}")
    print(f"Feature shape per sample: {X.shape[1:]} (frames x features)")
    print()
    print("Usage:")
    print("  data = np.load('synthetic_punches.npz')")
    print("  X, y = data['X'], data['y']")
    print("  class_names = data['class_names']")
