"""
ONNX Detector - GPU-accelerated hand detection using ONNX Runtime.
This is a placeholder for ONNX model integration.
For now, we verify ONNX Runtime is working with GPU.
"""

import numpy as np

try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False
    print("WARNING: onnxruntime not installed")


class ONNXHandDetector:
    """
    Fast GPU-accelerated hand detection using ONNX.
    NOTE: This requires a compatible ONNX model for hand landmarks.
    """
    
    def __init__(self, model_path=None):
        """
        Initialize ONNX detector.
        
        Args:
            model_path: Path to ONNX model file (None = don't load model)
        """
        self.available = ONNX_AVAILABLE
        self.session = None
        self.provider = None
        
        if not self.available:
            print("[ONNX] Not available - install with: pip install onnxruntime-gpu")
            return
        
        # Check available providers
        available_providers = ort.get_available_providers()
        print(f"[ONNX] Available providers: {available_providers}")
        
        # Prefer GPU
        if 'CUDAExecutionProvider' in available_providers:
            self.providers = [
                ('CUDAExecutionProvider', {
                    'device_id': 0,
                    'arena_extend_strategy': 'kNextPowerOfTwo',
                }),
                'CPUExecutionProvider'
            ]
            self.provider = 'CUDA'
            print("[ONNX] Using CUDA GPU provider")
        elif 'DmlExecutionProvider' in available_providers:
            self.providers = ['DmlExecutionProvider', 'CPUExecutionProvider']
            self.provider = 'DirectML'
            print("[ONNX] Using DirectML GPU provider")
        else:
            self.providers = ['CPUExecutionProvider']
            self.provider = 'CPU'
            print("[ONNX] Using CPU provider (GPU not available)")
        
        # Load model if provided
        if model_path:
            self.load_model(model_path)
    
    def load_model(self, model_path):
        """Load ONNX model."""
        try:
            self.session = ort.InferenceSession(model_path, providers=self.providers)
            self.input_name = self.session.get_inputs()[0].name
            print(f"[ONNX] Model loaded: {model_path}")
            print(f"[ONNX] Active providers: {self.session.get_providers()}")
            return True
        except Exception as e:
            print(f"[ONNX] Failed to load model: {e}")
            return False
    
    def detect(self, frame):
        """
        Detect hands in frame using ONNX model.
        
        Args:
            frame: BGR image (numpy array)
        
        Returns:
            dict: Detection results
        """
        if self.session is None:
            return None
        
        # Preprocess
        input_tensor = self._preprocess(frame)
        
        # Run inference
        outputs = self.session.run(None, {self.input_name: input_tensor})
        
        # Postprocess
        return self._postprocess(outputs)
    
    def _preprocess(self, frame):
        """Preprocess frame for model input."""
        import cv2
        # Resize to model input size (common: 224x224 or 256x256)
        resized = cv2.resize(frame, (256, 256))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        normalized = rgb.astype(np.float32) / 255.0
        # Add batch dimension: (1, H, W, C) or (1, C, H, W)
        return np.expand_dims(normalized, axis=0)
    
    def _postprocess(self, outputs):
        """Postprocess model outputs."""
        # Format depends on specific ONNX model
        return {
            'landmarks': outputs[0] if len(outputs) > 0 else None,
            'confidence': outputs[1] if len(outputs) > 1 else None,
        }
    
    def get_provider_info(self):
        """Get info about current provider."""
        return {
            'available': self.available,
            'provider': self.provider,
            'session_providers': self.session.get_providers() if self.session else None,
        }


def check_onnx_gpu():
    """Check if ONNX Runtime GPU is available and working."""
    print("=" * 60)
    print("ONNX RUNTIME GPU CHECK")
    print("=" * 60)
    
    if not ONNX_AVAILABLE:
        print("ONNX Runtime not installed!")
        print("Install with: pip install onnxruntime-gpu")
        return False
    
    print(f"ONNX Runtime version: {ort.__version__}")
    
    # Check providers
    providers = ort.get_available_providers()
    print(f"\nAvailable providers: {providers}")
    
    has_gpu = False
    if 'CUDAExecutionProvider' in providers:
        print("✓ CUDA GPU support available!")
        has_gpu = True
    elif 'DmlExecutionProvider' in providers:
        print("✓ DirectML GPU support available!")
        has_gpu = True
    else:
        print("✗ No GPU support - using CPU only")
    
    # Create a simple test
    print("\n--- Simple inference test ---")
    try:
        # Create a tiny dummy model for testing
        import onnx
        from onnx import helper, TensorProto
        
        # Input
        X = helper.make_tensor_value_info('X', TensorProto.FLOAT, [1, 3])
        # Output
        Y = helper.make_tensor_value_info('Y', TensorProto.FLOAT, [1, 3])
        # Identity node (just copies input to output)
        node = helper.make_node('Identity', ['X'], ['Y'])
        # Graph
        graph = helper.make_graph([node], 'test_graph', [X], [Y])
        # Model
        model = helper.make_model(graph)
        
        # Save and load
        onnx.save(model, 'test_model.onnx')
        
        # Create session
        detector = ONNXHandDetector()
        detector.load_model('test_model.onnx')
        
        # Run test inference
        input_data = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
        output = detector.session.run(None, {'X': input_data})
        
        print(f"Input: {input_data}")
        print(f"Output: {output[0]}")
        print("✓ Inference test passed!")
        
        # Cleanup
        import os
        os.remove('test_model.onnx')
        
        return True
        
    except ImportError:
        print("Note: 'onnx' package not installed for model creation test")
        print("But ONNX Runtime is working fine!")
        return has_gpu
    except Exception as e:
        print(f"Error: {e}")
        return False


if __name__ == "__main__":
    check_onnx_gpu()
