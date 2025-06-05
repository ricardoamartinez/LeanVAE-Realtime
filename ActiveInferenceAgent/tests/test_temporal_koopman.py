"""
Unit tests for TemporalKoopmanOperator module

Tests cover:
- Temporal state prediction
- Stability monitoring
- EMA state tracking
- Temporal buffer management
- Edge cases and numerical stability
"""

import unittest
import torch
import numpy as np
import sys
sys.path.append('../src')

from active_inference_agent import TemporalKoopmanOperator, TemporalStabilityMonitor


class TestTemporalStabilityMonitor(unittest.TestCase):
    """Tests for temporal stability monitoring"""
    
    def setUp(self):
        """Initialize test fixtures"""
        self.monitor = TemporalStabilityMonitor(tolerance=0.05, smoothing_factor=0.9)
        
    def test_initialization(self):
        """Test monitor initialization"""
        self.assertEqual(self.monitor.tolerance, 0.05)
        self.assertEqual(self.monitor.smoothing_factor, 0.9)
        self.assertIsNone(self.monitor.exponential_average)
        self.assertEqual(len(self.monitor.stability_history), 0)
        
    def test_stability_monitoring(self):
        """Test stability measurement and exponential smoothing"""
        batch_size = 4
        latent_dim = 32
        
        # Create stable evolution (small change)
        z_before = torch.randn(batch_size, latent_dim)
        z_after = z_before + torch.randn_like(z_before) * 0.01  # Small perturbation
        
        is_stable, measure = self.monitor.monitor_latent_stability(z_before, z_after)
        self.assertTrue(is_stable)
        self.assertLess(measure, self.monitor.tolerance)
        
        # Create unstable evolution (large change)
        z_unstable = z_before + torch.randn_like(z_before) * 1.0  # Large perturbation
        is_stable, measure = self.monitor.monitor_latent_stability(z_before, z_unstable)
        self.assertFalse(is_stable)
        self.assertGreater(measure, self.monitor.tolerance)
        
    def test_exponential_smoothing(self):
        """Test exponential average calculation"""
        # First measurement initializes the average
        z1 = torch.randn(2, 16)
        z2 = z1 + 0.1
        _, measure1 = self.monitor.monitor_latent_stability(z1, z2)
        self.assertEqual(self.monitor.exponential_average, measure1)
        
        # Subsequent measurements use smoothing
        z3 = z1 + 0.2
        _, measure2 = self.monitor.monitor_latent_stability(z1, z3)
        
        # Check exponential average formula
        expected_avg = (self.monitor.smoothing_factor * measure1 + 
                       (1 - self.monitor.smoothing_factor) * torch.norm(z3 - z1).mean().item())
        self.assertAlmostEqual(self.monitor.exponential_average, expected_avg, places=5)
        
    def test_statistics(self):
        """Test stability statistics calculation"""
        # Generate some measurements
        for i in range(10):
            z1 = torch.randn(2, 16)
            z2 = z1 + torch.randn_like(z1) * (0.01 if i < 8 else 0.1)
            self.monitor.monitor_latent_stability(z1, z2)
            
        stats = self.monitor.get_stability_statistics()
        
        self.assertIn('avg_stability', stats)
        self.assertIn('max_instability', stats)
        self.assertIn('instability_rate', stats)
        self.assertIn('total_instabilities', stats)
        
        # Check values are reasonable
        self.assertGreater(stats['avg_stability'], 0)
        self.assertGreater(stats['max_instability'], stats['avg_stability'])
        self.assertGreaterEqual(stats['instability_rate'], 0)
        self.assertLessEqual(stats['instability_rate'], 1)


class TestTemporalKoopmanOperator(unittest.TestCase):
    """Comprehensive tests for temporal Koopman operator"""
    
    def setUp(self):
        """Initialize test fixtures"""
        self.latent_dim = 32
        self.temporal_window = 5
        self.batch_size = 4
        
        self.koopman = TemporalKoopmanOperator(
            latent_dim=self.latent_dim,
            temporal_window=self.temporal_window
        )
        
        # Create test tensors
        self.z_current = torch.randn(self.batch_size, self.latent_dim)
        
    def test_initialization(self):
        """Test proper initialization"""
        self.assertEqual(self.koopman.latent_dim, self.latent_dim)
        self.assertEqual(self.koopman.temporal_window, self.temporal_window)
        self.assertIsNone(self.koopman.ema_state)
        self.assertIsNone(self.koopman.z_previous)
        
        # Check networks are initialized
        self.assertIsNotNone(self.koopman.temporal_predictor)
        self.assertIsNotNone(self.koopman.consistency_network)
        self.assertIsNotNone(self.koopman.stability_monitor)
        
        # Check temporal weights
        self.assertEqual(len(self.koopman.temporal_weights), self.temporal_window)
        
    def test_temporal_prediction(self):
        """Test temporal state prediction"""
        # First prediction (no history)
        z_predicted = self.koopman.predict_next_latent_state(self.z_current)
        
        # Should return valid output
        self.assertEqual(z_predicted.shape, self.z_current.shape)
        self.assertFalse(torch.isnan(z_predicted).any())
        
        # Prediction should be close to input initially (due to residual connection)
        diff = torch.norm(z_predicted - self.z_current).item()
        self.assertLess(diff, self.latent_dim)  # Reasonable bound
        
    def test_ema_state_tracking(self):
        """Test exponential moving average state tracking"""
        # First call initializes EMA
        _ = self.koopman.predict_next_latent_state(self.z_current)
        self.assertIsNotNone(self.koopman.ema_state)
        self.assertTrue(torch.allclose(self.koopman.ema_state, self.z_current, atol=1e-5))
        
        # Second call updates EMA
        z_new = torch.randn_like(self.z_current)
        _ = self.koopman.predict_next_latent_state(z_new)
        
        # EMA should be between old and new states
        expected_ema = self.koopman.ema_alpha * self.z_current + (1 - self.koopman.ema_alpha) * z_new
        self.assertTrue(torch.allclose(self.koopman.ema_state, expected_ema, atol=1e-5))
        
    def test_temporal_buffer_update(self):
        """Test temporal buffer management"""
        # Buffer starts empty
        self.assertEqual(len(self.koopman.temporal_buffer), 0)
        
        # Add states
        for i in range(self.temporal_window + 2):
            z = torch.randn(self.batch_size, self.latent_dim)
            self.koopman.update_temporal_buffer(z)
            
        # Buffer should be at max capacity
        self.assertEqual(len(self.koopman.temporal_buffer), self.temporal_window)
        
    def test_temporal_context_retrieval(self):
        """Test relevant temporal context retrieval"""
        # Fill buffer with diverse states
        for i in range(self.temporal_window):
            z = torch.randn(self.batch_size, self.latent_dim) * (i + 1)  # Different scales
            self.koopman.update_temporal_buffer(z)
            
        # Query with a state similar to one in buffer
        z_query = self.koopman.temporal_buffer[2] + torch.randn_like(self.koopman.temporal_buffer[2]) * 0.1
        z_prev, z_prev_prev = self.koopman.get_relevant_temporal_context(z_query)
        
        # Should return non-None states
        self.assertIsNotNone(z_prev)
        # z_prev_prev might be None if only one relevant state found
        
    def test_forward_pass(self):
        """Test complete forward pass"""
        # Run forward pass
        z_predicted = self.koopman(self.z_current)
        
        # Check output
        self.assertEqual(z_predicted.shape, self.z_current.shape)
        self.assertFalse(torch.isnan(z_predicted).any())
        
        # Buffer should be updated
        self.assertEqual(len(self.koopman.temporal_buffer), 1)
        
    def test_gradient_flow(self):
        """Test gradient flow through the operator"""
        self.z_current.requires_grad = True
        
        # Forward pass
        z_predicted = self.koopman(self.z_current)
        
        # Compute dummy loss
        loss = torch.sum(z_predicted**2)
        loss.backward()
        
        # Check gradients exist
        self.assertIsNotNone(self.z_current.grad)
        self.assertFalse(torch.all(self.z_current.grad == 0))
        
    def test_numerical_stability(self):
        """Test numerical stability with extreme inputs"""
        # Test with very large inputs
        z_large = torch.randn(self.batch_size, self.latent_dim) * 1000
        z_predicted = self.koopman(z_large)
        self.assertFalse(torch.isnan(z_predicted).any())
        self.assertFalse(torch.isinf(z_predicted).any())
        
        # Test with very small inputs
        z_small = torch.randn(self.batch_size, self.latent_dim) * 1e-6
        z_predicted = self.koopman(z_small)
        self.assertFalse(torch.isnan(z_predicted).any())
        
        # Test with zero input
        z_zero = torch.zeros(self.batch_size, self.latent_dim)
        z_predicted = self.koopman(z_zero)
        self.assertFalse(torch.isnan(z_predicted).any())
        
    def test_temporal_consistency(self):
        """Test temporal consistency across multiple steps"""
        states = []
        
        # Generate trajectory
        z = self.z_current
        for _ in range(10):
            z = self.koopman(z)
            states.append(z.clone())
            
        # Check states don't diverge too much
        for i in range(1, len(states)):
            diff = torch.norm(states[i] - states[i-1]).item()
            # Difference should be bounded (not exploding)
            self.assertLess(diff, 10.0)
            
        # Check states don't collapse
        final_norm = torch.norm(states[-1]).item()
        self.assertGreater(final_norm, 0.1)  # Not collapsed to zero
        
    def test_potential_energy_computation(self):
        """Test potential energy calculation"""
        q = torch.randn(self.batch_size, self.latent_dim // 2)
        V = self.koopman.potential_energy(q)
        
        # Check shape
        self.assertEqual(V.shape, (self.batch_size, 1))
        
        # Check value is correct
        expected_V = 0.5 * torch.sum(q**2, dim=-1, keepdim=True)
        self.assertTrue(torch.allclose(V, expected_V))
        
    def test_smooth_latent_evolution(self):
        """Test smooth latent evolution method"""
        q = torch.randn(self.batch_size, self.latent_dim // 2)
        p = torch.randn(self.batch_size, self.latent_dim // 2)
        
        q_new, p_new = self.koopman.smooth_latent_evolution(q, p)
        
        # Check outputs have correct shape
        self.assertEqual(q_new.shape, q.shape)
        self.assertEqual(p_new.shape, p.shape)
        
        # Check momentum decay
        expected_p = p * 0.95  # momentum_decay
        self.assertTrue(torch.allclose(p_new, expected_p))
        
        # Check position update
        expected_q = q + 0.1 * p_new
        self.assertTrue(torch.allclose(q_new, expected_q))


class TestKoopmanWeakPoints(unittest.TestCase):
    """Tests specifically designed to find weak points and edge cases"""
    
    def test_temporal_window_edge_cases(self):
        """Test behavior with different temporal window sizes"""
        # Very small window
        small_koopman = TemporalKoopmanOperator(latent_dim=16, temporal_window=1)
        z = torch.randn(2, 16)
        output = small_koopman(z)
        self.assertFalse(torch.isnan(output).any())
        
        # Large window
        large_koopman = TemporalKoopmanOperator(latent_dim=16, temporal_window=50)
        output = large_koopman(z)
        self.assertFalse(torch.isnan(output).any())
        
    def test_batch_size_sensitivity(self):
        """Test with various batch sizes"""
        koopman = TemporalKoopmanOperator(latent_dim=32)
        
        # Single sample
        z_single = torch.randn(1, 32)
        output = koopman(z_single)
        self.assertEqual(output.shape, (1, 32))
        
        # Large batch
        z_large = torch.randn(128, 32)
        output = koopman(z_large)
        self.assertEqual(output.shape, (128, 32))
        
    def test_repeated_identical_inputs(self):
        """Test behavior with repeated identical inputs"""
        koopman = TemporalKoopmanOperator(latent_dim=32)
        z = torch.randn(4, 32)
        
        outputs = []
        for _ in range(10):
            output = koopman(z.clone())  # Same input repeatedly
            outputs.append(output.clone())
            
        # Outputs should stabilize (not keep changing)
        for i in range(5, 10):
            diff = torch.norm(outputs[i] - outputs[i-1]).item()
            self.assertLess(diff, 0.1)  # Should converge
            

if __name__ == '__main__':
    unittest.main(verbosity=2)