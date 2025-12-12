"""
Convert PyTorch punch classifier to ONNX format for Rust WASM inference.

Usage:
    python convert_to_onnx.py

Requirements:
    pip install torch onnx

Input:  punch_classifier.pth (trained weights)
Output: punch_classifier.onnx (~164KB)
"""

import torch
import torch.nn as nn

class PunchClassifierCNN(nn.Module):
    """1D-CNN for punch classification (matches training architecture exactly)"""
    
    def __init__(self):
        super().__init__()
        # Convolutional layers
        self.conv1 = nn.Conv1d(10, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(32)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(64)
        self.conv3 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(128)
        
        # Pooling layers
        self.pool = nn.MaxPool1d(2)
        self.adaptive_pool = nn.AdaptiveAvgPool1d(1)
        
        # Fully connected layers
        self.dropout1 = nn.Dropout(0.3)
        self.dropout2 = nn.Dropout(0.2)
        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, 5)  # 5 classes: jab, cross, hook, uppercut, idle
    
    def forward(self, x):
        # Input: (batch, 30, 10) → transpose to (batch, 10, 30) for Conv1d
        x = x.transpose(1, 2)
        
        # Conv block 1
        x = self.pool(torch.relu(self.bn1(self.conv1(x))))  # (B, 32, 15)
        
        # Conv block 2
        x = self.pool(torch.relu(self.bn2(self.conv2(x))))  # (B, 64, 7)
        
        # Conv block 3
        x = self.adaptive_pool(torch.relu(self.bn3(self.conv3(x))))  # (B, 128, 1)
        
        # Flatten
        x = x.view(x.size(0), -1)  # (B, 128)
        
        # Classifier
        x = self.dropout1(x)
        x = torch.relu(self.fc1(x))
        x = self.dropout2(x)
        return self.fc2(x)  # (B, 5) logits


def main():
    # Load trained weights
    model = PunchClassifierCNN()
    
    try:
        state_dict = torch.load('punch_classifier.pth', map_location='cpu')
        model.load_state_dict(state_dict)
        print('✅ Loaded punch_classifier.pth')
    except FileNotFoundError:
        print('❌ punch_classifier.pth not found!')
        print('   Make sure the trained model file is in the current directory.')
        return
    
    # Set to eval mode (important for BatchNorm and Dropout)
    model.eval()
    
    # Dummy input: (batch=1, frames=30, features=10)
    dummy_input = torch.randn(1, 30, 10)
    
    # Export to ONNX
    torch.onnx.export(
        model,
        dummy_input,
        'punch_classifier.onnx',
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch'},
            'output': {0: 'batch'}
        },
        opset_version=11,
        do_constant_folding=True,
    )
    
    print('✅ Exported to punch_classifier.onnx')
    
    # Verify file size
    import os
    size_kb = os.path.getsize('punch_classifier.onnx') / 1024
    print(f'   File size: {size_kb:.1f} KB')
    print('')
    print('Next steps:')
    print('1. Copy punch_classifier.onnx to boxing-web/assets/')
    print('2. Run: wasm-pack build --target web --out-dir pkg')


if __name__ == '__main__':
    main()
