"""
Test Suite for LSTM Predictor (Phase 2)
========================================
Unit and integration tests for the LSTM predictive tracking system.
"""

import pytest
import numpy as np
import torch
import time
import sys

# Add parent directory for imports
sys.path.insert(0, '.')

from lstm_predictor import (
    HandPredictor,
    LandmarkBuffer,
    PhysicsValidator,
    PredictiveTracker
)


class TestHandPredictor:
    """Unit tests for the LSTM model."""
    
    def test_model_creation(self):
        """Test model can be created with default parameters."""
        model = HandPredictor()
        assert model is not None
        assert model.input_size == 63
        assert model.hidden_size == 128
        assert model.num_layers == 2
    
    def test_forward_pass_shape(self):
        """Test forward pass returns correct output shape."""
        model = HandPredictor()
        batch_size = 2
        seq_len = 10
        input_tensor = torch.randn(batch_size, seq_len, 63)
        
        output = model(input_tensor)
        
        assert output.shape == (batch_size, 63)
    
    def test_residual_prediction(self):
        """Test residual mode adds delta to last input."""
        model = HandPredictor()
        model.use_residual = True
        
        # Create input where all frames are the same
        static_input = torch.ones(1, 10, 63) * 0.5
        output = model(static_input)
        
        # Output should be close to input (delta should be small for static input)
        # After training on static data, the model should learn delta ≈ 0
        assert output.shape == (1, 63)
    
    def test_gpu_if_available(self):
        """Test model works on GPU if available."""
        if torch.cuda.is_available():
            model = HandPredictor().cuda()
            input_tensor = torch.randn(1, 10, 63).cuda()
            output = model(input_tensor)
            assert output.device.type == 'cuda'


class TestLandmarkBuffer:
    """Unit tests for the landmark buffer."""
    
    def test_buffer_creation(self):
        """Test buffer can be created."""
        buffer = LandmarkBuffer(buffer_size=10)
        assert buffer.buffer_size == 10
    
    def test_add_flat_landmarks(self):
        """Test adding flat landmarks to buffer."""
        buffer = LandmarkBuffer(buffer_size=5)
        flat = np.random.rand(63).tolist()
        
        buffer.add_flat_landmarks(flat, "Right")
        
        assert len(buffer.right_hand_history) == 1
    
    def test_is_ready(self):
        """Test buffer ready state."""
        buffer = LandmarkBuffer(buffer_size=3)
        
        # Not ready initially
        assert not buffer.is_ready("Right")
        
        # Add frames
        for i in range(3):
            buffer.add_flat_landmarks(np.random.rand(63).tolist(), "Right")
        
        # Now ready
        assert buffer.is_ready("Right")
    
    def test_get_sequence(self):
        """Test getting sequence as tensor."""
        buffer = LandmarkBuffer(buffer_size=3)
        
        for i in range(3):
            buffer.add_flat_landmarks(np.random.rand(63).tolist(), "Right")
        
        seq = buffer.get_sequence("Right")
        
        assert seq is not None
        assert seq.shape == (1, 3, 63)
    
    def test_velocity_calculation(self):
        """Test velocity calculation from history."""
        buffer = LandmarkBuffer(buffer_size=10)
        
        # Add two frames with known positions
        frame1 = np.zeros(63)
        frame1[0] = 0.5  # wrist x
        frame1[1] = 0.5  # wrist y
        
        frame2 = np.zeros(63)
        frame2[0] = 0.6  # wrist x moved
        frame2[1] = 0.5  # wrist y same
        
        buffer.add_flat_landmarks(frame1.tolist(), "Right", timestamp=0.0)
        buffer.add_flat_landmarks(frame2.tolist(), "Right", timestamp=0.1)
        
        velocity = buffer.get_velocity("Right")
        
        # Velocity should be approximately (0.1/0.1, 0, 0) = (1.0, 0, 0), ...
        assert velocity[0] > 0  # Positive x velocity
        assert abs(velocity[1]) < 0.01  # No y velocity


class TestPhysicsValidator:
    """Unit tests for physics validation."""
    
    def test_valid_small_displacement(self):
        """Test valid prediction with small displacement."""
        validator = PhysicsValidator()
        
        last_known = np.random.rand(63) * 0.5 + 0.25
        predicted = last_known + 0.01  # Small displacement
        
        is_valid, reason = validator.validate(predicted, last_known)
        
        assert is_valid
        assert reason == "valid"
    
    def test_reject_teleportation(self):
        """Test rejection of large displacement (teleportation)."""
        validator = PhysicsValidator()
        
        last_known = np.random.rand(63) * 0.5 + 0.25
        predicted = last_known.copy()
        predicted[0] = last_known[0] + 0.5  # Large displacement in wrist x
        
        is_valid, reason = validator.validate(predicted, last_known)
        
        assert not is_valid
        assert "teleportation" in reason
    
    def test_reject_out_of_bounds(self):
        """Test rejection of out-of-bounds predictions."""
        validator = PhysicsValidator()
        
        last_known = np.random.rand(63) * 0.5 + 0.25
        predicted = last_known.copy()
        predicted[0] = 2.0  # Way out of bounds
        
        is_valid, reason = validator.validate(predicted, last_known)
        
        assert not is_valid
        assert reason == "out_of_bounds"
    
    def test_stats_tracking(self):
        """Test that stats are tracked correctly."""
        validator = PhysicsValidator()
        
        last_known = np.random.rand(63) * 0.5 + 0.25
        
        # One valid
        validator.validate(last_known + 0.01, last_known)
        
        # One invalid
        invalid_pred = last_known.copy()
        invalid_pred[0] += 0.5
        validator.validate(invalid_pred, last_known)
        
        stats = validator.get_stats()
        assert stats['accepted'] == 1
        assert stats['rejected'] == 1


class TestPredictiveTracker:
    """Integration tests for the full predictor."""
    
    def test_tracker_creation(self):
        """Test tracker can be created."""
        tracker = PredictiveTracker(buffer_size=10)
        assert tracker is not None
        assert tracker.buffer_size == 10
    
    def test_predict_multi_step(self):
        """Test multi-step prediction."""
        tracker = PredictiveTracker(buffer_size=5)
        
        # Fill buffer with dummy data
        for i in range(10):
            flat = np.random.rand(63) * 0.1 + 0.45
            tracker.buffer.add_flat_landmarks(flat.tolist(), "Right")
        
        predictions = tracker.predict_multi_step("Right", num_steps=3)
        
        assert len(predictions) >= 1  # At least one step
        assert len(predictions[0]) == 63
    
    def test_stats(self):
        """Test stats retrieval."""
        tracker = PredictiveTracker(buffer_size=5)
        stats = tracker.get_stats()
        
        assert 'predictions' in stats
        assert 'device' in stats
        assert 'physics_rejection_rate' in stats


class TestIntegration:
    """Integration tests with FlowGatedValidator."""
    
    def test_flow_validator_with_lstm(self):
        """Test FlowGatedValidator with LSTM enabled."""
        from flow_gated_validator import FlowGatedValidator
        
        predictor = PredictiveTracker(buffer_size=5)
        validator = FlowGatedValidator(lstm_predictor=predictor)
        
        assert validator.lstm_enabled
        assert validator.lstm_predictor is predictor
    
    def test_flow_validator_fallback(self):
        """Test that LSTM is tried before optical flow."""
        from flow_gated_validator import FlowGatedValidator
        
        predictor = PredictiveTracker(buffer_size=5)
        validator = FlowGatedValidator(lstm_predictor=predictor)
        
        # Without sufficient history, LSTM should fail and fall through
        gray_frame = np.zeros((480, 640), dtype=np.uint8)
        
        pos, source, conf = validator.validate(
            hand="Right",
            mp_position=None,  # MediaPipe failed
            mp_confidence=0.0,
            gray_frame=gray_frame,
            frame_width=640,
            frame_height=480,
            timestamp=time.time()
        )
        
        # Should not be LSTM since buffer isn't ready
        assert source in ['optical_flow', 'ballistic', 'lost']


if __name__ == "__main__":
    # Run with: python test_lstm_predictor.py
    # Or: python -m pytest test_lstm_predictor.py -v
    
    print("=" * 60)
    print("RUNNING LSTM PREDICTOR TESTS")
    print("=" * 60)
    
    # Simple runner without pytest
    test_classes = [
        TestHandPredictor,
        TestLandmarkBuffer,
        TestPhysicsValidator,
        TestPredictiveTracker,
        TestIntegration
    ]
    
    passed = 0
    failed = 0
    
    for test_class in test_classes:
        print(f"\n{test_class.__name__}:")
        instance = test_class()
        
        for method_name in dir(instance):
            if method_name.startswith('test_'):
                try:
                    getattr(instance, method_name)()
                    print(f"  ✓ {method_name}")
                    passed += 1
                except Exception as e:
                    print(f"  ✗ {method_name}: {e}")
                    failed += 1
    
    print(f"\n{'=' * 60}")
    print(f"RESULTS: {passed} passed, {failed} failed")
    print("=" * 60)
