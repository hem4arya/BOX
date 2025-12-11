"""
LSTM Predictor for Hand Landmark Prediction
Predicts hand positions 2-3 frames ahead to:
1. Compensate for tracking loss during fast movements
2. Create smoother hand motion
3. Reduce perceived latency
"""

import torch
import torch.nn as nn
import numpy as np
from collections import deque


class HandPredictor(nn.Module):
    """
    LSTM model to predict hand landmark positions.
    Input: Last N frames of landmarks (21 landmarks × 3 coords = 63 features)
    Output: Predicted next frame landmarks
    """
    def __init__(self, input_size=63, hidden_size=128, num_layers=2, output_size=63):
        super(HandPredictor, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.1)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, output_size)
        )
    
    def forward(self, x):
        # x shape: (batch, sequence_length, input_size)
        batch_size = x.size(0)
        
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
        
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])  # Take last timestep
        return out


class LandmarkBuffer:
    """
    Stores history of landmarks for prediction.
    """
    def __init__(self, buffer_size=10):
        self.buffer_size = buffer_size
        self.left_hand_history = deque(maxlen=buffer_size)
        self.right_hand_history = deque(maxlen=buffer_size)
    
    def add_landmarks(self, landmarks, hand_label):
        """Convert MediaPipe landmarks to flat array and store."""
        if landmarks is None:
            return
        
        flat = []
        for lm in landmarks.landmark:
            flat.extend([lm.x, lm.y, lm.z])
        
        if hand_label == "Left":
            self.left_hand_history.append(flat)
        else:
            self.right_hand_history.append(flat)
    
    def get_sequence(self, hand_label):
        """Get landmark history as tensor for prediction."""
        history = self.left_hand_history if hand_label == "Left" else self.right_hand_history
        
        if len(history) < self.buffer_size:
            return None
        
        return torch.tensor([list(history)], dtype=torch.float32)
    
    def is_ready(self, hand_label):
        """Check if we have enough history for prediction."""
        history = self.left_hand_history if hand_label == "Left" else self.right_hand_history
        return len(history) >= self.buffer_size
    
    def clear(self, hand_label=None):
        """Clear history."""
        if hand_label is None or hand_label == "Left":
            self.left_hand_history.clear()
        if hand_label is None or hand_label == "Right":
            self.right_hand_history.clear()


class SimpleInterpolator:
    """Interpolate between frames for smoother display."""
    
    def __init__(self):
        self.previous_landmarks = {}
    
    def interpolate(self, current_flat, label, alpha=0.5):
        """
        Blend previous and current landmarks.
        alpha: 0 = all previous, 1 = all current
        """
        if label not in self.previous_landmarks:
            self.previous_landmarks[label] = current_flat
            return current_flat
        
        prev = np.array(self.previous_landmarks[label])
        curr = np.array(current_flat)
        
        # Linear interpolation
        blended = prev * (1 - alpha) + curr * alpha
        
        self.previous_landmarks[label] = current_flat
        return blended.tolist()
    
    def lerp_landmarks(self, lm1, lm2, alpha):
        """Linear interpolation between two landmark arrays."""
        return [l1 * (1 - alpha) + l2 * alpha for l1, l2 in zip(lm1, lm2)]


class PredictiveTracker:
    """
    Main class combining MediaPipe results with LSTM predictions.
    """
    
    def __init__(self, buffer_size=10, device='cuda'):
        self.device = device if torch.cuda.is_available() else 'cpu'
        self.buffer_size = buffer_size
        
        print(f"  PredictiveTracker device: {self.device}")
        if self.device == 'cuda':
            print(f"  GPU: {torch.cuda.get_device_name(0)}")
        
        self.buffer = LandmarkBuffer(buffer_size)
        self.interpolator = SimpleInterpolator()
        
        # Create model
        self.model = HandPredictor().to(self.device)
        self.model.eval()
        
        # Online learning
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        self.criterion = nn.MSELoss()
        
        # Training schedule
        self.train_every_n_frames = 30
        self.frame_count = 0
        self.training_enabled = True
        
        # Statistics
        self.predictions_made = 0
        self.training_iterations = 0
        self.last_loss = 0.0
        
        # Store last prediction for training comparison
        self.last_predictions = {}
    
    def update(self, mediapipe_results):
        """
        Process MediaPipe results, store history, make predictions.
        Returns: dict with 'detected' and 'predicted' landmarks
        """
        output = {
            'detected': {},
            'predicted': {},
            'smoothed': {}
        }
        
        if mediapipe_results is None or not mediapipe_results.multi_hand_landmarks:
            return output
        
        for hand_landmarks, handedness in zip(
            mediapipe_results.multi_hand_landmarks,
            mediapipe_results.multi_handedness
        ):
            label = handedness.classification[0].label
            
            # Store in buffer
            self.buffer.add_landmarks(hand_landmarks, label)
            output['detected'][label] = hand_landmarks
            
            # Convert to flat array
            current_flat = []
            for lm in hand_landmarks.landmark:
                current_flat.extend([lm.x, lm.y, lm.z])
            
            # Make prediction if buffer ready
            if self.buffer.is_ready(label):
                sequence = self.buffer.get_sequence(label).to(self.device)
                
                with torch.no_grad():
                    predicted = self.model(sequence)
                
                pred_np = predicted.cpu().numpy()[0]
                output['predicted'][label] = pred_np
                self.predictions_made += 1
                
                # Store for training comparison
                self.last_predictions[label] = pred_np
                
                # Create smoothed output (blend detected + predicted)
                smoothed = self.interpolator.interpolate(current_flat, label, alpha=0.7)
                output['smoothed'][label] = smoothed
                
                # Online learning
                self.frame_count += 1
                if self.training_enabled and self.frame_count % self.train_every_n_frames == 0:
                    self._online_train(label, current_flat)
        
        return output
    
    def _online_train(self, label, actual_flat):
        """Train model on actual vs predicted (online learning)."""
        if not self.buffer.is_ready(label):
            return
        
        actual_tensor = torch.tensor([actual_flat], dtype=torch.float32).to(self.device)
        sequence = self.buffer.get_sequence(label).to(self.device)
        
        self.model.train()
        self.optimizer.zero_grad()
        
        predicted = self.model(sequence)
        loss = self.criterion(predicted, actual_tensor)
        loss.backward()
        self.optimizer.step()
        
        self.model.eval()
        self.training_iterations += 1
        self.last_loss = loss.item()
    
    def get_stats(self):
        """Return performance statistics."""
        return {
            'predictions': self.predictions_made,
            'training_iterations': self.training_iterations,
            'last_loss': self.last_loss,
            'device': self.device
        }
    
    def flat_to_landmarks(self, flat_array):
        """Convert flat array back to landmark-like structure for drawing."""
        landmarks = []
        for i in range(0, len(flat_array), 3):
            landmarks.append({
                'x': flat_array[i],
                'y': flat_array[i + 1],
                'z': flat_array[i + 2]
            })
        return landmarks


# Test function
def test_predictor():
    print("=" * 60)
    print("LSTM PREDICTOR TEST")
    print("=" * 60)
    
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # Test model creation
    predictor = PredictiveTracker(buffer_size=10)
    
    # Test forward pass with dummy data
    dummy_sequence = torch.randn(1, 10, 63).to(predictor.device)
    output = predictor.model(dummy_sequence)
    
    print(f"\nModel test:")
    print(f"  Input shape: {dummy_sequence.shape}")
    print(f"  Output shape: {output.shape}")
    print(f"  Device: {predictor.device}")
    
    print("\n✅ LSTM Predictor initialized successfully!")
    print("=" * 60)


if __name__ == "__main__":
    test_predictor()
