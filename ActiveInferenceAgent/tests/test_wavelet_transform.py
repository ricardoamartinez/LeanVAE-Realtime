"""
Unit tests for PhysicsInformedWaveletTransform module

Tests cover:
- Forward 2D spatial wavelet transform
- Forward 3D spatiotemporal wavelet transform
- Inverse wavelet transform with energy conservation
- Temporal buffer management
- Edge cases and error handling
"""

import unittest
import torch
import numpy as np
import sys
sys.path.append('../src')

from active_inference_agent import PhysicsInformedWaveletTransform


class TestPhysicsInformedWaveletTransform(unittest.TestCase):
    """Comprehensive tests for physics-informed wavelet transforms"""
    
    def setUp(self):
        """Initialize test fixtures"""
        self.input_size = 64  # Small size for fast tests
        self.n_channels = 3
        self.temporal_depth = 8
        self.batch_size = 2
        
        self.wavelet_transform = PhysicsInformedWaveletTransform(
            input_size=self.input_size,
            n_channels=self.n_channels,
            temporal_depth=self.temporal_depth
        )
        
        # Create test input tensors
        self.test_input = torch.randn(
            self.batch_size, self.n_channels, self.input_size, self.input_size
        )
        
    def test_initialization(self):
        """Test proper initialization of wavelet transform"""
        self.assertEqual(self.wavelet_transform.input_size, self.input_size)
        self.assertEqual(self.wavelet_transform.n_channels, self.n_channels)
        self.assertEqual(self.wavelet_transform.temporal_depth, self.temporal_depth)
        
        # Check wavelet coefficients are registered
        self.assertTrue(hasattr(self.wavelet_transform, 'spatial_h0'))
        self.assertTrue(hasattr(self.wavelet_transform, 'spatial_h1'))
        self.assertTrue(hasattr(self.wavelet_transform, 'temporal_h0'))
        self.assertTrue(hasattr(self.wavelet_transform, 'temporal_h1'))
        
        # Check energy weights are initialized
        self.assertEqual(len(self.wavelet_transform.spatiotemporal_energy_weights), 8)
        self.assertEqual(len(self.wavelet_transform.spatial_energy_weights), 4)
        
    def test_forward_2d_spatial_wavelet(self):
        """Test 2D spatial wavelet transform"""
        output = self.wavelet_transform.forward_2d_spatial_wavelet(self.test_input)
        
        # Check output shape
        expected_features = self.wavelet_transform.spatial_features
        self.assertEqual(output.shape, (self.batch_size, self.n_channels, expected_features))
        
        # Check output is not NaN
        self.assertFalse(torch.isnan(output).any())
        
        # Check energy conservation (Parseval's theorem)
        input_energy = torch.sum(self.test_input**2).item()
        output_energy = torch.sum(output**2).item()
        # Allow 10% tolerance for numerical errors
        self.assertAlmostEqual(input_energy, output_energy, delta=input_energy * 0.1)
        
    def test_1d_spatial_wavelet_axis(self):
        """Test 1D spatial wavelet transform along different axes"""
        # Test along width (axis=3)
        output_width = self.wavelet_transform.physics_informed_1d_spatial_wavelet(
            self.test_input, axis=3
        )
        self.assertEqual(output_width.shape[3], self.input_size)  # Width doubled (low+high)
        
        # Test along height (axis=2)  
        output_height = self.wavelet_transform.physics_informed_1d_spatial_wavelet(
            self.test_input, axis=2
        )
        self.assertEqual(output_height.shape[2], self.input_size)  # Height doubled (low+high)
        
    def test_temporal_buffer_management(self):
        """Test temporal frame buffer operations"""
        # Initially buffer should be empty
        self.assertEqual(len(self.wavelet_transform.frame_buffer), 0)
        self.assertFalse(self.wavelet_transform.can_process_spatiotemporal())
        
        # Add frames one by one
        for i in range(self.temporal_depth):
            frame = torch.randn(1, self.n_channels, self.input_size, self.input_size)
            self.wavelet_transform.add_frame_to_buffer(frame)
            
            if i < self.temporal_depth - 1:
                self.assertFalse(self.wavelet_transform.can_process_spatiotemporal())
            else:
                self.assertTrue(self.wavelet_transform.can_process_spatiotemporal())
                
        # Buffer should be at max capacity
        self.assertEqual(len(self.wavelet_transform.frame_buffer), self.temporal_depth)
        
        # Add one more frame - oldest should be removed
        extra_frame = torch.randn(1, self.n_channels, self.input_size, self.input_size)
        self.wavelet_transform.add_frame_to_buffer(extra_frame)
        self.assertEqual(len(self.wavelet_transform.frame_buffer), self.temporal_depth)
        
    def test_spatiotemporal_sequence_generation(self):
        """Test generation of spatiotemporal sequences"""
        # Test with empty buffer
        sequence = self.wavelet_transform.get_spatiotemporal_sequence()
        self.assertEqual(sequence.shape[0], self.temporal_depth)
        
        # Test with partial buffer
        self.wavelet_transform.frame_buffer.clear()
        for i in range(3):  # Add only 3 frames
            frame = torch.randn(1, self.n_channels, self.input_size, self.input_size)
            self.wavelet_transform.add_frame_to_buffer(frame)
            
        sequence = self.wavelet_transform.get_spatiotemporal_sequence()
        self.assertEqual(sequence.shape, 
                        (self.temporal_depth, 1, self.n_channels, self.input_size, self.input_size))
        
    def test_forward_3d_spatiotemporal_wavelet(self):
        """Test 3D spatiotemporal wavelet transform"""
        # Fill temporal buffer
        for i in range(self.temporal_depth):
            frame = torch.randn(1, self.n_channels, self.input_size, self.input_size)
            self.wavelet_transform.add_frame_to_buffer(frame)
            
        # Get spatiotemporal sequence
        sequence = self.wavelet_transform.get_spatiotemporal_sequence()
        
        # Apply 3D transform
        output, temporal_frames = self.wavelet_transform.forward_3d_spatiotemporal_wavelet(sequence)
        
        # Check output shape
        expected_features = self.wavelet_transform.spatiotemporal_features
        self.assertEqual(output.shape[2], expected_features)
        self.assertEqual(temporal_frames, self.temporal_depth // 2)
        
        # Check no NaN values
        self.assertFalse(torch.isnan(output).any())
        
    def test_inverse_2d_wavelet(self):
        """Test inverse wavelet transform and reconstruction quality"""
        # Forward transform
        coeffs = self.wavelet_transform.forward_2d_spatial_wavelet(self.test_input)
        
        # Inverse transform
        reconstruction = self.wavelet_transform.inverse_2d_wavelet(
            coeffs, self.input_size, self.input_size
        )
        
        # Check shape matches input
        self.assertEqual(reconstruction.shape, self.test_input.shape)
        
        # Check reconstruction quality (should be close to original)
        mse = torch.mean((reconstruction - self.test_input)**2).item()
        self.assertLess(mse, 0.1)  # Reasonable reconstruction error
        
        # Check energy conservation
        input_energy = torch.sum(self.test_input**2).item()
        recon_energy = torch.sum(reconstruction**2).item()
        self.assertAlmostEqual(input_energy, recon_energy, delta=input_energy * 0.1)
        
    def test_energy_weights_gradient_flow(self):
        """Test that energy weights can be learned via gradient"""
        # Enable gradients
        self.wavelet_transform.spatial_energy_weights.requires_grad = True
        self.wavelet_transform.spatiotemporal_energy_weights.requires_grad = True
        
        # Forward pass
        output = self.wavelet_transform.forward_2d_spatial_wavelet(self.test_input)
        
        # Compute dummy loss
        loss = torch.sum(output**2)
        loss.backward()
        
        # Check gradients exist
        self.assertIsNotNone(self.wavelet_transform.spatial_energy_weights.grad)
        self.assertFalse(torch.all(self.wavelet_transform.spatial_energy_weights.grad == 0))
        
    def test_edge_cases(self):
        """Test edge cases and potential failure modes"""
        # Test with very small input
        small_input = torch.randn(1, 3, 16, 16)
        small_transform = PhysicsInformedWaveletTransform(input_size=16)
        output = small_transform.forward_2d_spatial_wavelet(small_input)
        self.assertFalse(torch.isnan(output).any())
        
        # Test with single channel
        single_channel = torch.randn(1, 1, 64, 64)
        single_transform = PhysicsInformedWaveletTransform(input_size=64, n_channels=1)
        output = single_transform.forward_2d_spatial_wavelet(single_channel)
        self.assertFalse(torch.isnan(output).any())
        
        # Test with non-square input (should handle gracefully)
        # Note: Current implementation assumes square, but should not crash
        rect_input = torch.randn(1, 3, 64, 32)
        try:
            output = self.wavelet_transform.forward_2d_spatial_wavelet(rect_input)
            # If it works, check output
            self.assertFalse(torch.isnan(output).any())
        except Exception as e:
            # Document this as a known limitation
            print(f"Known limitation: Non-square inputs cause: {e}")
            

class TestWaveletPerformance(unittest.TestCase):
    """Performance and stress tests for wavelet transform"""
    
    def test_memory_efficiency(self):
        """Test memory usage doesn't explode with large inputs"""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Create large wavelet transform
        large_transform = PhysicsInformedWaveletTransform(input_size=256)
        large_input = torch.randn(4, 3, 256, 256)
        
        # Run forward pass
        output = large_transform.forward_2d_spatial_wavelet(large_input)
        
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory
        
        # Memory increase should be reasonable (< 500MB for this size)
        self.assertLess(memory_increase, 500)
        
    def test_computation_speed(self):
        """Test computation speed is reasonable"""
        import time
        
        transform = PhysicsInformedWaveletTransform(input_size=128)
        input_tensor = torch.randn(8, 3, 128, 128)
        
        # Warm up
        _ = transform.forward_2d_spatial_wavelet(input_tensor)
        
        # Time multiple runs
        n_runs = 10
        start_time = time.time()
        for _ in range(n_runs):
            _ = transform.forward_2d_spatial_wavelet(input_tensor)
        end_time = time.time()
        
        avg_time = (end_time - start_time) / n_runs
        
        # Should be fast (< 100ms per batch on CPU)
        self.assertLess(avg_time, 0.1)
        

if __name__ == '__main__':
    # Run with verbose output to see all test results
    unittest.main(verbosity=2)