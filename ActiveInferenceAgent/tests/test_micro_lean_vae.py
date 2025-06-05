"""
Unit tests for MicroLeanVAE module

Tests cover:
- VAE encoding/decoding
- Reparameterization trick
- Temporal dynamics integration
- Collapse detection and recovery
- End-to-end forward pass
"""

import unittest
import torch
import torch.nn as nn
import numpy as np
import sys
sys.path.append('../src')

from active_inference_agent import MicroLeanVAE, CollapseDetector


class TestMicroLeanVAE(unittest.TestCase):
    """Comprehensive tests for MicroLeanVAE"""
    
    def setUp(self):
        """Initialize test fixtures"""
        self.input_size = 64
        self.latent_dim = 16
        self.batch_size = 4
        
        self.vae = MicroLeanVAE(
            input_size=self.input_size,
            latent_dim=self.latent_dim
        )
        
        # Create test input
        self.test_input = torch.rand(
            self.batch_size, 3, self.input_size, self.input_size
        )
        
    def test_initialization(self):
        """Test VAE initialization"""
        self.assertEqual(self.vae.input_size, self.input_size)
        self.assertEqual(self.vae.latent_dim, self.latent_dim)
        
        # Check components are initialized
        self.assertIsNotNone(self.vae.wavelet_transform)
        self.assertIsNotNone(self.vae.encoder)
        self.assertIsNotNone(self.vae.decoder)
        self.assertIsNotNone(self.vae.koopman_operator)
        self.assertIsNotNone(self.vae.motion_detector)
        
        # Check wavelet features were calculated
        self.assertGreater(self.vae.wavelet_features, 0)
        
    def test_encode(self):
        """Test encoding to latent space"""
        mean, logvar = self.vae.encode(self.test_input)
        
        # Check shapes
        self.assertEqual(mean.shape, (self.batch_size, self.latent_dim))
        self.assertEqual(logvar.shape, (self.batch_size, self.latent_dim))
        
        # Check no NaN values
        self.assertFalse(torch.isnan(mean).any())
        self.assertFalse(torch.isnan(logvar).any())
        
        # Check logvar is reasonable (not too large or small)
        self.assertTrue(torch.all(logvar > -10))
        self.assertTrue(torch.all(logvar < 10))
        
    def test_reparameterize(self):
        """Test reparameterization trick"""
        mean = torch.randn(self.batch_size, self.latent_dim)
        logvar = torch.randn(self.batch_size, self.latent_dim) * 0.1
        
        # Test in training mode
        self.vae.train()
        z_train = self.vae.reparameterize(mean, logvar)
        self.assertEqual(z_train.shape, mean.shape)
        
        # Should be different from mean due to noise
        self.assertFalse(torch.allclose(z_train, mean))
        
        # Test in eval mode
        self.vae.eval()
        z_eval = self.vae.reparameterize(mean, logvar)
        
        # Should be equal to mean (no noise)
        self.assertTrue(torch.allclose(z_eval, mean))
        
    def test_decode(self):
        """Test decoding from latent space"""
        z = torch.randn(self.batch_size, self.latent_dim)
        reconstruction = self.vae.decode(z)
        
        # Check shape
        self.assertEqual(reconstruction.shape, self.test_input.shape)
        
        # Check values are in [0, 1] range (after sigmoid)
        self.assertTrue(torch.all(reconstruction >= 0))
        self.assertTrue(torch.all(reconstruction <= 1))
        
        # Check no NaN values
        self.assertFalse(torch.isnan(reconstruction).any())
        
    def test_forward_pass(self):
        """Test complete forward pass"""
        reconstruction, mean, logvar = self.vae(self.test_input)
        
        # Check all outputs
        self.assertEqual(reconstruction.shape, self.test_input.shape)
        self.assertEqual(mean.shape, (self.batch_size, self.latent_dim))
        self.assertEqual(logvar.shape, (self.batch_size, self.latent_dim))
        
        # Check no NaN values
        self.assertFalse(torch.isnan(reconstruction).any())
        self.assertFalse(torch.isnan(mean).any())
        self.assertFalse(torch.isnan(logvar).any())
        
    def test_temporal_dynamics(self):
        """Test temporal dynamics integration"""
        # First pass - no previous latent
        recon1, mean1, logvar1 = self.vae(self.test_input)
        self.assertIsNotNone(self.vae.prev_latent)
        
        # Second pass - should use temporal dynamics
        self.vae.train()  # Temporal dynamics only in training
        recon2, mean2, logvar2 = self.vae(self.test_input)
        
        # Outputs should be different due to temporal evolution
        self.assertFalse(torch.allclose(recon1, recon2, atol=1e-3))
        
    def test_gradient_flow(self):
        """Test gradient flow through VAE"""
        self.test_input.requires_grad = True
        
        # Forward pass
        reconstruction, mean, logvar = self.vae(self.test_input)
        
        # Compute VAE loss
        recon_loss = nn.functional.mse_loss(reconstruction, self.test_input)
        kl_loss = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp())
        total_loss = recon_loss + 0.01 * kl_loss
        
        # Backward pass
        total_loss.backward()
        
        # Check gradients exist
        self.assertIsNotNone(self.test_input.grad)
        self.assertFalse(torch.all(self.test_input.grad == 0))
        
    def test_spatiotemporal_encoding(self):
        """Test spatiotemporal encoding mode"""
        # Fill temporal buffer first
        for _ in range(8):
            self.vae.wavelet_transform.add_frame_to_buffer(self.test_input[0:1])
            
        # Enable spatiotemporal processing
        mean, logvar = self.vae.encode(self.test_input[0:1], use_spatiotemporal=True)
        
        # Should still produce valid output
        self.assertEqual(mean.shape, (1, self.latent_dim))
        self.assertEqual(logvar.shape, (1, self.latent_dim))
        self.assertFalse(torch.isnan(mean).any())
        self.assertFalse(torch.isnan(logvar).any())
        
    def test_reconstruction_quality(self):
        """Test reconstruction quality metrics"""
        # Use a simple pattern that should be easy to reconstruct
        test_pattern = torch.zeros_like(self.test_input)
        test_pattern[:, :, 20:40, 20:40] = 1.0  # White square
        
        # Forward pass
        reconstruction, _, _ = self.vae(test_pattern)
        
        # Calculate MSE
        mse = torch.mean((reconstruction - test_pattern)**2).item()
        
        # Should achieve reasonable reconstruction
        self.assertLess(mse, 0.1)  # Threshold may need tuning
        
    def test_latent_space_properties(self):
        """Test properties of learned latent space"""
        # Encode multiple samples
        latents = []
        for _ in range(10):
            x = torch.rand(self.batch_size, 3, self.input_size, self.input_size)
            mean, _ = self.vae.encode(x)
            latents.append(mean)
            
        latents = torch.cat(latents, dim=0)
        
        # Check latent space has reasonable properties
        # Mean should be near zero (due to KL regularization)
        latent_mean = torch.mean(latents, dim=0)
        self.assertTrue(torch.all(torch.abs(latent_mean) < 2.0))
        
        # Should have reasonable variance (not collapsed)
        latent_std = torch.std(latents, dim=0)
        self.assertTrue(torch.all(latent_std > 0.01))
        self.assertTrue(torch.all(latent_std < 10.0))


class TestCollapseDetector(unittest.TestCase):
    """Tests for VAE collapse detection"""
    
    def setUp(self):
        """Initialize test fixtures"""
        self.detector = CollapseDetector()
        
    def test_initialization(self):
        """Test detector initialization"""
        self.assertEqual(len(self.detector.mean_history), 0)
        self.assertEqual(len(self.detector.std_history), 0)
        self.assertEqual(self.detector.collapse_count, 0)
        
    def test_collapse_detection_normal(self):
        """Test detection with normal VAE output"""
        # Normal reconstruction
        reconstruction = torch.rand(4, 3, 64, 64) * 0.8 + 0.1  # Range [0.1, 0.9]
        mean = torch.randn(4, 32) * 0.5
        logvar = torch.randn(4, 32) * 0.1 - 2.0  # Reasonable variance
        
        is_collapsed, score, indicators = self.detector.detect_collapse(
            reconstruction, mean, logvar
        )
        
        self.assertFalse(is_collapsed)
        self.assertLess(score, 0.5)
        
    def test_collapse_detection_collapsed(self):
        """Test detection with collapsed VAE"""
        # Collapsed reconstruction (low variance)
        reconstruction = torch.ones(4, 3, 64, 64) * 0.5  # Constant output
        mean = torch.zeros(4, 32)  # Zero mean
        logvar = torch.ones(4, 32) * -10  # Very low variance
        
        is_collapsed, score, indicators = self.detector.detect_collapse(
            reconstruction, mean, logvar
        )
        
        self.assertTrue(is_collapsed)
        self.assertGreater(score, 0.4)
        self.assertTrue(indicators['low_reconstruction_std'])
        self.assertTrue(indicators['low_latent_diversity'])
        
    def test_collapse_recovery_strategies(self):
        """Test recovery strategy selection"""
        # Test different collapse types
        indicators1 = {
            'low_reconstruction_std': True,
            'extreme_reconstruction_mean': False,
            'low_latent_diversity': False,
            'decreasing_std_trend': False
        }
        strategy1 = self.detector.get_recovery_strategy(indicators1)
        self.assertEqual(strategy1, "diversity_injection")
        
        indicators2 = {
            'low_reconstruction_std': False,
            'extreme_reconstruction_mean': True,
            'low_latent_diversity': False,
            'decreasing_std_trend': False
        }
        strategy2 = self.detector.get_recovery_strategy(indicators2)
        self.assertEqual(strategy2, "range_normalization")
        
    def test_history_tracking(self):
        """Test history tracking for trend detection"""
        # Add declining std values
        for i in range(10):
            reconstruction = torch.randn(4, 3, 64, 64) * (0.5 - i * 0.04)
            mean = torch.randn(4, 32)
            logvar = torch.randn(4, 32) * 0.1
            
            _, _, indicators = self.detector.detect_collapse(
                reconstruction, mean, logvar
            )
            
        # Should detect decreasing trend after enough samples
        self.assertTrue(len(self.detector.std_history) >= 5)


class TestVAEWeakPoints(unittest.TestCase):
    """Tests to identify weak points in VAE implementation"""
    
    def test_extreme_latent_dimensions(self):
        """Test with very small and large latent dimensions"""
        # Very small latent dim
        small_vae = MicroLeanVAE(input_size=32, latent_dim=2)
        x = torch.rand(2, 3, 32, 32)
        reconstruction, mean, logvar = small_vae(x)
        self.assertFalse(torch.isnan(reconstruction).any())
        
        # Large latent dim
        large_vae = MicroLeanVAE(input_size=32, latent_dim=128)
        reconstruction, mean, logvar = large_vae(x)
        self.assertFalse(torch.isnan(reconstruction).any())
        
    def test_single_channel_input(self):
        """Test with grayscale input"""
        # This should fail as current implementation expects 3 channels
        vae = MicroLeanVAE(input_size=64, latent_dim=16)
        x_gray = torch.rand(4, 1, 64, 64)
        
        with self.assertRaises(Exception):
            # Should raise error due to channel mismatch
            reconstruction, mean, logvar = vae(x_gray)
            
    def test_batch_size_edge_cases(self):
        """Test with various batch sizes"""
        vae = MicroLeanVAE(input_size=64, latent_dim=16)
        
        # Single sample
        x_single = torch.rand(1, 3, 64, 64)
        recon, _, _ = vae(x_single)
        self.assertEqual(recon.shape[0], 1)
        
        # Large batch
        x_large = torch.rand(32, 3, 64, 64)
        recon, _, _ = vae(x_large)
        self.assertEqual(recon.shape[0], 32)
        
    def test_numerical_stability(self):
        """Test numerical stability with extreme inputs"""
        vae = MicroLeanVAE(input_size=64, latent_dim=16)
        
        # Very small values
        x_small = torch.rand(4, 3, 64, 64) * 1e-6
        recon, mean, logvar = vae(x_small)
        self.assertFalse(torch.isnan(recon).any())
        self.assertFalse(torch.isinf(logvar).any())
        
        # Values near 1
        x_high = torch.ones(4, 3, 64, 64) * 0.999
        recon, mean, logvar = vae(x_high)
        self.assertFalse(torch.isnan(recon).any())
        
    def test_memory_efficiency(self):
        """Test memory usage with large inputs"""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Create VAE with larger input
        vae = MicroLeanVAE(input_size=128, latent_dim=32)
        x = torch.rand(8, 3, 128, 128)
        
        # Multiple forward passes
        for _ in range(5):
            recon, mean, logvar = vae(x)
            
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory
        
        # Should not leak memory excessively
        self.assertLess(memory_increase, 1000)  # Less than 1GB increase


if __name__ == '__main__':
    unittest.main(verbosity=2)