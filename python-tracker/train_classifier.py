"""
Punch Classifier Training Script
=================================
Trains a 1D Convolutional Neural Network on synthetic punch data.

Input: synthetic_punches.npz
Output: punch_classifier.pth

Classes: JAB, CROSS, HOOK, UPPERCUT, IDLE
Architecture: 1D-CNN with 3 conv layers

Expected: ~90% accuracy after 30 epochs
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path
import time


# ════════════════════════════════════════════════════════════════════════════
# 1D-CNN MODEL
# ════════════════════════════════════════════════════════════════════════════
class PunchClassifierCNN(nn.Module):
    """
    1D Convolutional Neural Network for punch classification.
    
    Input: (batch, 30, 10) - 30 frames, 10 features per frame
    Output: (batch, 5) - 5 punch classes
    """
    
    def __init__(self, num_features=10, num_classes=5):
        super(PunchClassifierCNN, self).__init__()
        
        # Reshape input from (batch, frames, features) to (batch, features, frames)
        # CNN expects (batch, channels, sequence)
        
        self.conv1 = nn.Sequential(
            nn.Conv1d(num_features, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2)  # 30 -> 15
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2)  # 15 -> 7
        )
        
        self.conv3 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)  # 7 -> 1
        )
        
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, num_classes)
        )
    
    def forward(self, x):
        # x: (batch, 30, 10)
        # Transpose to (batch, 10, 30) for Conv1d
        x = x.transpose(1, 2)
        
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        
        x = x.squeeze(-1)  # Remove last dim: (batch, 128)
        x = self.classifier(x)
        
        return x


# ════════════════════════════════════════════════════════════════════════════
# TRAINING FUNCTIONS
# ════════════════════════════════════════════════════════════════════════════

def load_data(data_path='synthetic_punches.npz', train_ratio=0.8):
    """Load and split synthetic data into train/test sets."""
    print(f"Loading data from {data_path}...")
    
    data = np.load(data_path)
    X = data['X']  # (N, 30, 10)
    y = data['y']  # (N,)
    class_names = data['class_names']
    
    print(f"  Loaded {len(y)} samples")
    print(f"  Features shape: {X.shape}")
    print(f"  Classes: {list(class_names)}")
    
    # Shuffle
    indices = np.random.permutation(len(X))
    X, y = X[indices], y[indices]
    
    # Split
    split_idx = int(len(X) * train_ratio)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    print(f"  Train: {len(X_train)}, Test: {len(X_test)}")
    
    return X_train, y_train, X_test, y_test, class_names


def train_model(model, train_loader, test_loader, epochs=30, lr=0.001, device='cpu'):
    """Train the model."""
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    
    model.to(device)
    
    best_acc = 0.0
    history = {'train_loss': [], 'test_acc': []}
    
    print("\n" + "=" * 60)
    print("TRAINING")
    print("=" * 60)
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        
        for batch_X, batch_y in train_loader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        history['train_loss'].append(avg_loss)
        
        # Evaluate
        model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch_X, batch_y in test_loader:
                batch_X = batch_X.to(device)
                batch_y = batch_y.to(device)
                
                outputs = model(batch_X)
                _, predicted = torch.max(outputs, 1)
                total += batch_y.size(0)
                correct += (predicted == batch_y).sum().item()
        
        acc = 100 * correct / total
        history['test_acc'].append(acc)
        
        if acc > best_acc:
            best_acc = acc
            best_state = model.state_dict().copy()
        
        scheduler.step()
        
        # Progress bar style output
        bar = "█" * int(acc / 5) + "░" * (20 - int(acc / 5))
        print(f"Epoch {epoch+1:2d}/{epochs} | Loss: {avg_loss:.4f} | Acc: {acc:5.1f}% |{bar}|")
    
    # Restore best weights
    model.load_state_dict(best_state)
    
    return model, history, best_acc


def evaluate_model(model, test_loader, class_names, device='cpu'):
    """Detailed evaluation with confusion matrix."""
    
    model.eval()
    model.to(device)
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            batch_X = batch_X.to(device)
            outputs = model(batch_X)
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(batch_y.numpy())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    print("\n" + "=" * 60)
    print("EVALUATION")
    print("=" * 60)
    
    # Per-class accuracy
    print("\nPer-Class Performance:")
    print("-" * 40)
    for i, name in enumerate(class_names):
        mask = all_labels == i
        if mask.sum() > 0:
            class_acc = (all_preds[mask] == all_labels[mask]).mean() * 100
            print(f"  {name.upper():10s}: {class_acc:5.1f}%")
    
    # Overall accuracy
    overall_acc = (all_preds == all_labels).mean() * 100
    print("-" * 40)
    print(f"  {'OVERALL':10s}: {overall_acc:5.1f}%")
    
    return overall_acc


# ════════════════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("PUNCH CLASSIFIER TRAINING")
    print("=" * 60)
    
    # Check for GPU
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    
    # Load data
    X_train, y_train, X_test, y_test, class_names = load_data()
    
    # Create data loaders
    train_dataset = TensorDataset(
        torch.FloatTensor(X_train),
        torch.LongTensor(y_train)
    )
    test_dataset = TensorDataset(
        torch.FloatTensor(X_test),
        torch.LongTensor(y_test)
    )
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    
    # Create model
    model = PunchClassifierCNN(num_features=10, num_classes=5)
    print(f"\nModel Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Train
    start_time = time.time()
    model, history, best_acc = train_model(
        model, train_loader, test_loader,
        epochs=30, lr=0.001, device=device
    )
    train_time = time.time() - start_time
    
    print(f"\nTraining Time: {train_time:.1f} seconds")
    print(f"Best Accuracy: {best_acc:.1f}%")
    
    # Detailed evaluation
    evaluate_model(model, test_loader, class_names, device)
    
    # Save model
    output_path = Path('punch_classifier.pth')
    torch.save({
        'model_state_dict': model.state_dict(),
        'class_names': list(class_names),
        'input_shape': (30, 10),
        'best_accuracy': best_acc
    }, output_path)
    
    print("\n" + "=" * 60)
    print("COMPLETE!")
    print("=" * 60)
    print(f"✓ Model saved to: {output_path}")
    print(f"✓ Best accuracy: {best_acc:.1f}%")
    print()
    print("Usage:")
    print("  checkpoint = torch.load('punch_classifier.pth')")
    print("  model = PunchClassifierCNN()")
    print("  model.load_state_dict(checkpoint['model_state_dict'])")


if __name__ == '__main__':
    main()
