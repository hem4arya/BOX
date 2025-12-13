"""
Convert PyTorch punch classifier to ONNX format for Rust WASM inference.
Handles both raw state_dict and checkpoint format.
"""

import torch
import torch.nn as nn
import os

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
        self.fc2 = nn.Linear(64, 5)
    
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


def main():
    model = PunchClassifierCNN()
    
    try:
        checkpoint = torch.load('punch_classifier.pth', map_location='cpu', weights_only=False)
        
        # Handle checkpoint format (model_state key) or raw state_dict
        if isinstance(checkpoint, dict) and 'model_state' in checkpoint:
            state_dict = checkpoint['model_state']
            print('✅ Loaded checkpoint with model_state key')
        else:
            state_dict = checkpoint
            print('✅ Loaded raw state_dict')
        
        model.load_state_dict(state_dict)
    except FileNotFoundError:
        print('❌ punch_classifier.pth not found!')
        return
    except Exception as e:
        print(f'❌ Error loading model: {e}')
        return
    
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
    
    print('✅ Exported to punch_classifier.onnx')
    size_kb = os.path.getsize('punch_classifier.onnx') / 1024
    print(f'   File size: {size_kb:.1f} KB')


if __name__ == '__main__':
    main()
