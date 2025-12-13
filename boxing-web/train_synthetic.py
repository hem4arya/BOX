"""
Train punch classifier with synthetic data.

Generates 10000 synthetic punch sequences and trains a 1D-CNN classifier.
Output: punch_classifier.pth (state_dict only) + punch_classifier.onnx
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import os

# ============================================================================
# MODEL
# ============================================================================

class PunchClassifierCNN(nn.Module):
    """1D-CNN for punch classification"""
    
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(10, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(32)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(64)
        self.conv3 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(128)
        self.pool = nn.MaxPool1d(2)
        self.adaptive_pool = nn.AdaptiveAvgPool1d(1)
        self.dropout1 = nn.Dropout(0.3)
        self.dropout2 = nn.Dropout(0.2)
        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, 5)  # 5 classes
    
    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.pool(torch.relu(self.bn1(self.conv1(x))))
        x = self.pool(torch.relu(self.bn2(self.conv2(x))))
        x = self.adaptive_pool(torch.relu(self.bn3(self.conv3(x))))
        x = x.view(x.size(0), -1)
        x = self.dropout1(x)
        x = torch.relu(self.fc1(x))
        x = self.dropout2(x)
        return self.fc2(x)

# ============================================================================
# SYNTHETIC DATA GENERATOR
# ============================================================================

def generate_synthetic_punch(punch_type, frames=30):
    """
    Generate synthetic punch sequence.
    
    Features (10 per frame):
    [left_wrist_x, left_wrist_y, right_wrist_x, right_wrist_y,
     left_velocity, right_velocity, left_elbow_angle, right_elbow_angle,
     left_depth, right_depth]
    
    Punch types: 0=JAB, 1=CROSS, 2=HOOK, 3=UPPERCUT, 4=IDLE
    """
    data = np.zeros((frames, 10))
    noise = 0.05
    
    # Base positions (normalized 0-1)
    left_base = [0.3, 0.5]  # Left wrist rest position
    right_base = [0.7, 0.5]  # Right wrist rest position
    
    if punch_type == 0:  # JAB (left forward punch)
        for i in range(frames):
            t = i / frames
            # Ramp up, then back
            if t < 0.5:
                ext = t * 2
            else:
                ext = (1 - t) * 2
            
            data[i, 0] = left_base[0] + ext * 0.3 + np.random.randn() * noise  # left_wrist_x
            data[i, 1] = left_base[1] + np.random.randn() * noise  # left_wrist_y
            data[i, 2] = right_base[0] + np.random.randn() * noise  # right_wrist_x
            data[i, 3] = right_base[1] + np.random.randn() * noise  # right_wrist_y
            data[i, 4] = ext * 2.0  # left_velocity (high during punch)
            data[i, 5] = 0.1  # right_velocity (low)
            data[i, 6] = 180 - ext * 30  # left_elbow_angle (straighter during punch)
            data[i, 7] = 160  # right_elbow_angle (bent, at rest)
            data[i, 8] = ext * 80  # left_depth (high during punch)
            data[i, 9] = 10  # right_depth (low)
            
    elif punch_type == 1:  # CROSS (right forward punch)
        for i in range(frames):
            t = i / frames
            if t < 0.5:
                ext = t * 2
            else:
                ext = (1 - t) * 2
            
            data[i, 0] = left_base[0] + np.random.randn() * noise
            data[i, 1] = left_base[1] + np.random.randn() * noise
            data[i, 2] = right_base[0] - ext * 0.3 + np.random.randn() * noise  # right moves left (forward)
            data[i, 3] = right_base[1] + np.random.randn() * noise
            data[i, 4] = 0.1
            data[i, 5] = ext * 2.5  # right_velocity (high, cross is powerful)
            data[i, 6] = 160
            data[i, 7] = 180 - ext * 35
            data[i, 8] = 10
            data[i, 9] = ext * 90
            
    elif punch_type == 2:  # HOOK (right lateral punch)
        for i in range(frames):
            t = i / frames
            if t < 0.5:
                ext = t * 2
            else:
                ext = (1 - t) * 2
            
            data[i, 0] = left_base[0] + np.random.randn() * noise
            data[i, 1] = left_base[1] + np.random.randn() * noise
            data[i, 2] = right_base[0] - ext * 0.4 + np.random.randn() * noise  # right swings left
            data[i, 3] = right_base[1] + np.random.randn() * noise
            data[i, 4] = 0.1
            data[i, 5] = ext * 1.8
            data[i, 6] = 160
            data[i, 7] = 90 + ext * 20  # Elbow stays bent for hook
            data[i, 8] = 10
            data[i, 9] = ext * 50
            
    elif punch_type == 3:  # UPPERCUT (right upward punch)
        for i in range(frames):
            t = i / frames
            if t < 0.5:
                ext = t * 2
            else:
                ext = (1 - t) * 2
            
            data[i, 0] = left_base[0] + np.random.randn() * noise
            data[i, 1] = left_base[1] + np.random.randn() * noise
            data[i, 2] = right_base[0] + np.random.randn() * noise
            data[i, 3] = right_base[1] - ext * 0.3 + np.random.randn() * noise  # right moves UP
            data[i, 4] = 0.1
            data[i, 5] = ext * 1.5
            data[i, 6] = 160
            data[i, 7] = 100 + ext * 40  # Elbow straightens upward
            data[i, 8] = 10
            data[i, 9] = ext * 40
            
    else:  # IDLE (subtle movement)
        for i in range(frames):
            data[i, 0] = left_base[0] + np.random.randn() * 0.02
            data[i, 1] = left_base[1] + np.random.randn() * 0.02
            data[i, 2] = right_base[0] + np.random.randn() * 0.02
            data[i, 3] = right_base[1] + np.random.randn() * 0.02
            data[i, 4] = 0.05 + np.random.rand() * 0.05
            data[i, 5] = 0.05 + np.random.rand() * 0.05
            data[i, 6] = 150 + np.random.randn() * 5
            data[i, 7] = 150 + np.random.randn() * 5
            data[i, 8] = 5 + np.random.rand() * 5
            data[i, 9] = 5 + np.random.rand() * 5
    
    return data.astype(np.float32)


def generate_dataset(n_samples=10000):
    """Generate balanced dataset with n_samples total."""
    samples_per_class = n_samples // 5
    
    X = []
    y = []
    
    for punch_type in range(5):
        for _ in range(samples_per_class):
            X.append(generate_synthetic_punch(punch_type))
            y.append(punch_type)
    
    X = np.array(X)
    y = np.array(y)
    
    # Shuffle
    idx = np.random.permutation(len(X))
    X = X[idx]
    y = y[idx]
    
    return X, y

# ============================================================================
# TRAINING
# ============================================================================

def train():
    print('🥊 Generating 10000 synthetic punch samples...')
    X, y = generate_dataset(10000)
    
    # Split 80/20
    split = int(len(X) * 0.8)
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]
    
    print(f'   Training: {len(X_train)}, Validation: {len(X_val)}')
    
    # Create data loaders
    train_dataset = TensorDataset(torch.tensor(X_train), torch.tensor(y_train))
    val_dataset = TensorDataset(torch.tensor(X_val), torch.tensor(y_val))
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64)
    
    # Model, loss, optimizer
    model = PunchClassifierCNN()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5)
    
    best_acc = 0
    
    print('🏋️ Training for 50 epochs...')
    for epoch in range(50):
        # Train
        model.train()
        train_loss = 0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        # Validate
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                outputs = model(X_batch)
                _, predicted = outputs.max(1)
                total += y_batch.size(0)
                correct += (predicted == y_batch).sum().item()
        
        acc = correct / total
        scheduler.step(1 - acc)
        
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), 'punch_classifier.pth')
        
        if (epoch + 1) % 10 == 0:
            print(f'   Epoch {epoch+1}/50 - Acc: {acc*100:.1f}% (Best: {best_acc*100:.1f}%)')
    
    print(f'✅ Training complete! Best accuracy: {best_acc*100:.1f}%')
    
    # Load best model and export to ONNX
    model.load_state_dict(torch.load('punch_classifier.pth', weights_only=True))
    model.eval()
    
    dummy_input = torch.randn(1, 30, 10)
    torch.onnx.export(
        model,
        dummy_input,
        'punch_classifier.onnx',
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch'}, 'output': {0: 'batch'}},
        opset_version=11,
        do_constant_folding=True,
    )
    
    size_kb = os.path.getsize('punch_classifier.onnx') / 1024
    print(f'✅ Exported to punch_classifier.onnx ({size_kb:.1f} KB)')
    print('')
    print('Next: Move punch_classifier.onnx to assets/')


if __name__ == '__main__':
    train()
