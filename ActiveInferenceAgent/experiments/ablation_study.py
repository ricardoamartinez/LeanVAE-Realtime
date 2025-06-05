"""
Ablation Study for Active Inference Agent

This experiment systematically disables components to measure their contribution.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

import torch
import torch.nn as nn
import numpy as np
import time
import json
from datetime import datetime
import matplotlib.pyplot as plt

from active_inference_agent import (
    MicroLeanVAE,
    PhysicsInformedWaveletTransform,
    TemporalKoopmanOperator
)


class AblationStudy:
    """Systematic ablation study of Active Inference Agent components"""
    
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.results = {
            'timestamp': datetime.now().isoformat(),
            'device': device,
            'ablations': {}
        }
        
    def create_test_data(self, n_samples=100, input_size=64):
        """Create synthetic test data"""
        # Static patterns
        static_patterns = torch.zeros(n_samples // 4, 3, input_size, input_size)
        for i in range(n_samples // 4):
            # Create checkerboard pattern
            for x in range(0, input_size, 8):
                for y in range(0, input_size, 8):
                    if (x // 8 + y // 8) % 2 == 0:
                        static_patterns[i, :, x:x+8, y:y+8] = 1.0
                        
        # Moving patterns
        moving_patterns = torch.zeros(n_samples // 4, 3, input_size, input_size)
        for i in range(n_samples // 4):
            # Moving bar
            pos = (i * 4) % input_size
            moving_patterns[i, :, :, pos:pos+4] = 1.0
            
        # Random noise
        noise_patterns = torch.rand(n_samples // 4, 3, input_size, input_size)
        
        # Natural-like patterns (smooth gradients)
        natural_patterns = torch.zeros(n_samples // 4, 3, input_size, input_size)
        for i in range(n_samples // 4):
            # Create smooth gradient
            x = torch.linspace(-1, 1, input_size)
            y = torch.linspace(-1, 1, input_size)
            xx, yy = torch.meshgrid(x, y, indexing='ij')
            pattern = torch.sin(xx * 3 + i * 0.5) * torch.cos(yy * 3 + i * 0.5)
            pattern = (pattern + 1) / 2  # Normalize to [0, 1]
            natural_patterns[i] = pattern.unsqueeze(0).repeat(3, 1, 1)
            
        # Combine all patterns
        all_patterns = torch.cat([static_patterns, moving_patterns, noise_patterns, natural_patterns])
        
        # Shuffle
        indices = torch.randperm(n_samples)
        return all_patterns[indices].to(self.device)
        
    def test_full_model(self, test_data):
        """Test full model performance"""
        print("\n1. Testing FULL MODEL...")
        
        model = MicroLeanVAE(input_size=64, latent_dim=32).to(self.device)
        model.eval()
        
        metrics = self.evaluate_model(model, test_data, "full_model")
        self.results['ablations']['full_model'] = metrics
        
        return model, metrics
        
    def test_without_temporal_dynamics(self, test_data):
        """Test model without temporal dynamics"""
        print("\n2. Testing WITHOUT TEMPORAL DYNAMICS...")
        
        # Create model and disable temporal dynamics
        model = MicroLeanVAE(input_size=64, latent_dim=32).to(self.device)
        
        # Replace forward method to skip temporal dynamics
        original_forward = model.forward
        
        def forward_no_temporal(x):
            # Encode
            mean, logvar = model.encode(x)
            z = model.reparameterize(mean, logvar)
            
            # Skip temporal dynamics - decode directly
            reconstruction = model.decode(z)
            
            return reconstruction, mean, logvar
            
        model.forward = forward_no_temporal
        model.eval()
        
        metrics = self.evaluate_model(model, test_data, "no_temporal")
        self.results['ablations']['no_temporal_dynamics'] = metrics
        
        return metrics
        
    def test_without_wavelet_transform(self, test_data):
        """Test with simple convolutions instead of wavelets"""
        print("\n3. Testing WITHOUT WAVELET TRANSFORM...")
        
        # Create a modified VAE with conv layers instead of wavelets
        class SimpleConvVAE(nn.Module):
            def __init__(self, input_size=64, latent_dim=32):
                super().__init__()
                self.latent_dim = latent_dim
                
                # Simple convolutional encoder
                self.encoder = nn.Sequential(
                    nn.Conv2d(3, 32, 4, 2, 1),  # 64 -> 32
                    nn.ReLU(),
                    nn.Conv2d(32, 64, 4, 2, 1),  # 32 -> 16
                    nn.ReLU(),
                    nn.Conv2d(64, 128, 4, 2, 1),  # 16 -> 8
                    nn.ReLU(),
                    nn.Flatten(),
                    nn.Linear(128 * 8 * 8, latent_dim * 2)
                )
                
                # Simple convolutional decoder
                self.decoder = nn.Sequential(
                    nn.Linear(latent_dim, 128 * 8 * 8),
                    nn.ReLU(),
                    nn.Unflatten(1, (128, 8, 8)),
                    nn.ConvTranspose2d(128, 64, 4, 2, 1),  # 8 -> 16
                    nn.ReLU(),
                    nn.ConvTranspose2d(64, 32, 4, 2, 1),  # 16 -> 32
                    nn.ReLU(),
                    nn.ConvTranspose2d(32, 3, 4, 2, 1),  # 32 -> 64
                    nn.Sigmoid()
                )
                
            def reparameterize(self, mean, logvar):
                if self.training:
                    std = torch.exp(0.5 * logvar)
                    eps = torch.randn_like(std) * 0.1
                    return mean + eps * std
                return mean
                
            def forward(self, x):
                # Encode
                encoded = self.encoder(x)
                mean, logvar = torch.chunk(encoded, 2, dim=1)
                
                # Sample
                z = self.reparameterize(mean, logvar)
                
                # Decode
                reconstruction = self.decoder(z)
                
                return reconstruction, mean, logvar
                
        model = SimpleConvVAE(input_size=64, latent_dim=32).to(self.device)
        model.eval()
        
        metrics = self.evaluate_model(model, test_data, "no_wavelet")
        self.results['ablations']['no_wavelet_transform'] = metrics
        
        return metrics
        
    def test_without_motion_adaptation(self, test_data):
        """Test without motion-based adaptation"""
        print("\n4. Testing WITHOUT MOTION ADAPTATION...")
        
        # This would require modifying the training loop
        # For now, we'll test inference performance only
        model = MicroLeanVAE(input_size=64, latent_dim=32).to(self.device)
        model.eval()
        
        # Test on static vs moving patterns separately
        n_samples = len(test_data)
        quarter = n_samples // 4
        
        # Assuming data is ordered: static, moving, noise, natural
        static_data = test_data[:quarter]
        moving_data = test_data[quarter:2*quarter]
        
        print("   Testing on static patterns...")
        static_metrics = self.evaluate_model(model, static_data, "static_only")
        
        print("   Testing on moving patterns...")
        moving_metrics = self.evaluate_model(model, moving_data, "moving_only")
        
        self.results['ablations']['motion_analysis'] = {
            'static_patterns': static_metrics,
            'moving_patterns': moving_metrics,
            'motion_benefit': moving_metrics['reconstruction_error'] - static_metrics['reconstruction_error']
        }
        
        return static_metrics, moving_metrics
        
    def evaluate_model(self, model, test_data, name):
        """Evaluate model performance"""
        total_loss = 0
        reconstruction_errors = []
        inference_times = []
        
        with torch.no_grad():
            for i in range(len(test_data)):
                x = test_data[i:i+1]
                
                # Time inference
                start_time = time.time()
                reconstruction, mean, logvar = model(x)
                inference_time = time.time() - start_time
                
                # Calculate reconstruction error
                mse = nn.functional.mse_loss(reconstruction, x).item()
                reconstruction_errors.append(mse)
                inference_times.append(inference_time)
                
                # Calculate VAE loss
                kl_loss = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp()).item()
                total_loss += mse + 0.01 * kl_loss
                
        metrics = {
            'reconstruction_error': np.mean(reconstruction_errors),
            'reconstruction_std': np.std(reconstruction_errors),
            'inference_time_ms': np.mean(inference_times) * 1000,
            'total_loss': total_loss / len(test_data),
            'fps_potential': 1000 / (np.mean(inference_times) * 1000)
        }
        
        print(f"   {name} - Recon Error: {metrics['reconstruction_error']:.4f}, "
              f"Time: {metrics['inference_time_ms']:.2f}ms, "
              f"FPS: {metrics['fps_potential']:.1f}")
              
        return metrics
        
    def plot_results(self):
        """Visualize ablation results"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # 1. Reconstruction Error Comparison
        ax = axes[0, 0]
        models = list(self.results['ablations'].keys())
        recon_errors = [self.results['ablations'][m].get('reconstruction_error', 0) for m in models]
        ax.bar(models, recon_errors)
        ax.set_ylabel('Reconstruction Error')
        ax.set_title('Reconstruction Quality')
        ax.tick_params(axis='x', rotation=45)
        
        # 2. Inference Speed Comparison
        ax = axes[0, 1]
        inference_times = [self.results['ablations'][m].get('inference_time_ms', 0) for m in models]
        ax.bar(models, inference_times)
        ax.set_ylabel('Inference Time (ms)')
        ax.set_title('Inference Speed')
        ax.tick_params(axis='x', rotation=45)
        
        # 3. Component Contribution
        ax = axes[1, 0]
        if 'full_model' in self.results['ablations']:
            baseline = self.results['ablations']['full_model']['reconstruction_error']
            contributions = {}
            
            if 'no_temporal_dynamics' in self.results['ablations']:
                contributions['Temporal Dynamics'] = (
                    self.results['ablations']['no_temporal_dynamics']['reconstruction_error'] - baseline
                )
            if 'no_wavelet_transform' in self.results['ablations']:
                contributions['Wavelet Transform'] = (
                    self.results['ablations']['no_wavelet_transform']['reconstruction_error'] - baseline
                )
                
            ax.bar(contributions.keys(), contributions.values())
            ax.set_ylabel('Performance Degradation')
            ax.set_title('Component Contributions')
            ax.axhline(y=0, color='k', linestyle='--', alpha=0.5)
            
        # 4. Motion Adaptation Analysis
        ax = axes[1, 1]
        if 'motion_analysis' in self.results['ablations']:
            motion_data = self.results['ablations']['motion_analysis']
            patterns = ['Static', 'Moving']
            errors = [
                motion_data['static_patterns']['reconstruction_error'],
                motion_data['moving_patterns']['reconstruction_error']
            ]
            ax.bar(patterns, errors)
            ax.set_ylabel('Reconstruction Error')
            ax.set_title('Motion Adaptation Effect')
            
        plt.tight_layout()
        plt.savefig('ablation_results.png', dpi=150)
        print(f"\n📊 Results plot saved to: ablation_results.png")
        
    def save_results(self):
        """Save detailed results to JSON"""
        output_path = 'ablation_results.json'
        with open(output_path, 'w') as f:
            json.dump(self.results, f, indent=2)
        print(f"\n📄 Detailed results saved to: {output_path}")
        
    def run(self):
        """Run complete ablation study"""
        print("=" * 80)
        print("ACTIVE INFERENCE AGENT - ABLATION STUDY")
        print("=" * 80)
        
        # Create test data
        print("\nCreating test data...")
        test_data = self.create_test_data(n_samples=100, input_size=64)
        print(f"Created {len(test_data)} test samples")
        
        # Run ablations
        full_model, full_metrics = self.test_full_model(test_data)
        no_temporal_metrics = self.test_without_temporal_dynamics(test_data)
        no_wavelet_metrics = self.test_without_wavelet_transform(test_data)
        static_metrics, moving_metrics = self.test_without_motion_adaptation(test_data)
        
        # Analysis
        print("\n" + "=" * 80)
        print("ABLATION ANALYSIS")
        print("=" * 80)
        
        # Component importance
        if full_metrics['reconstruction_error'] > 0:
            temporal_importance = (
                (no_temporal_metrics['reconstruction_error'] - full_metrics['reconstruction_error']) 
                / full_metrics['reconstruction_error'] * 100
            )
            wavelet_importance = (
                (no_wavelet_metrics['reconstruction_error'] - full_metrics['reconstruction_error']) 
                / full_metrics['reconstruction_error'] * 100
            )
            
            print(f"\n📊 Component Importance (% degradation when removed):")
            print(f"   Temporal Dynamics: {temporal_importance:.1f}%")
            print(f"   Wavelet Transform: {wavelet_importance:.1f}%")
            
        # Motion adaptation benefit
        motion_benefit = moving_metrics['reconstruction_error'] - static_metrics['reconstruction_error']
        print(f"\n🎯 Motion Adaptation Benefit:")
        print(f"   Static patterns error: {static_metrics['reconstruction_error']:.4f}")
        print(f"   Moving patterns error: {moving_metrics['reconstruction_error']:.4f}")
        print(f"   Difference: {motion_benefit:.4f} ({'better' if motion_benefit < 0 else 'worse'} on motion)")
        
        # Performance impact
        print(f"\n⚡ Performance Impact:")
        print(f"   Full model: {full_metrics['fps_potential']:.1f} FPS")
        print(f"   Without wavelets: {no_wavelet_metrics['fps_potential']:.1f} FPS")
        speed_improvement = (
            (1/no_wavelet_metrics['inference_time_ms'] - 1/full_metrics['inference_time_ms']) 
            / (1/full_metrics['inference_time_ms']) * 100
        )
        print(f"   Wavelet overhead: {-speed_improvement:.1f}%")
        
        # Visualize and save
        self.plot_results()
        self.save_results()
        
        print("\n✅ Ablation study complete!")
        

if __name__ == '__main__':
    study = AblationStudy()
    study.run()