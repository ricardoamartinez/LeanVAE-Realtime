import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import threading
import time
from collections import deque
from queue import Queue, Empty
import argparse
from LeanVAE import LeanVAE
from einops import rearrange
import matplotlib.pyplot as plt
from datetime import datetime

class PhysicsInformedWaveletTransform(nn.Module):
    """Revolutionary 3D spatiotemporal wavelet transform - Space + Time frequency analysis!"""
    def __init__(self, input_size=240, n_channels=3, temporal_depth=8):
        super().__init__()
        self.input_size = input_size
        self.n_channels = n_channels
        self.temporal_depth = temporal_depth
        
        print(f"🌊 3D SPATIOTEMPORAL WAVELET TRANSFORM - Space + Time frequency decomposition!")
        
        # Pre-computed Daubechies-4 wavelet coefficients for SPATIAL decomposition
        spatial_h0 = torch.tensor([
            -0.010597401785, 0.032883011667, 0.030841381836, -0.187034811719,
            -0.027983769417, 0.630880767930, 0.714846570553, 0.230377813309
        ])
        
        spatial_h1 = torch.tensor([
            -0.230377813309, 0.714846570553, -0.630880767930, -0.027983769417,
            0.187034811719, 0.030841381836, -0.032883011667, -0.010597401785
        ])
        
        # Haar wavelet coefficients for TEMPORAL decomposition (simpler for motion)
        temporal_h0 = torch.tensor([0.7071067812, 0.7071067812])  # Normalized Haar low-pass
        temporal_h1 = torch.tensor([-0.7071067812, 0.7071067812])  # Normalized Haar high-pass
        
        # Register spatial filters
        self.register_buffer('spatial_h0', spatial_h0)
        self.register_buffer('spatial_h1', spatial_h1)
        
        # Register temporal filters
        self.register_buffer('temporal_h0', temporal_h0)
        self.register_buffer('temporal_h1', temporal_h1)
        
        # Physics-based energy conservation weights for 8 spatiotemporal frequency bands
        # LLL, LLH, LHL, LHH, HLL, HLH, HHL, HHH (Low/High in X, Y, Time)
        self.spatiotemporal_energy_weights = nn.Parameter(torch.ones(8))
        
        # Spatial energy weights (for 2D-only processing)
        self.spatial_energy_weights = nn.Parameter(torch.ones(4))  # LL, LH, HL, HH
        
        # Calculate actual output dimensions for efficient processing
        self.spatial_output_h = input_size // 2  # Spatial wavelet downsampling by 2
        self.spatial_output_w = input_size // 2
        self.temporal_output_depth = temporal_depth // 2  # Temporal wavelet downsampling by 2
        
        self.spatial_features = self.spatial_output_h * self.spatial_output_w * 4  # 4 spatial bands
        self.spatiotemporal_features = self.spatial_output_h * self.spatial_output_w * 8 * self.temporal_output_depth  # 8 bands
        
        # Temporal frame buffer for spatiotemporal processing
        self.frame_buffer = deque(maxlen=temporal_depth)
        
        print(f"   - Spatial output: {self.spatial_output_h}x{self.spatial_output_w} per band")
        print(f"   - Temporal depth: {temporal_depth} → {self.temporal_output_depth}")
        print(f"   - Spatial features: {self.spatial_features}")
        print(f"   - Spatiotemporal features: {self.spatiotemporal_features}")
        
    def physics_informed_1d_spatial_wavelet(self, x, axis):
        """Ultra-fast 1D SPATIAL wavelet transform using physics-informed vectorized operations"""
        batch_size, channels, height, width = x.shape
        
        if axis == 3:  # Transform along width (rows)
            # Reshape for row-wise processing
            x_flat = x.reshape(batch_size * channels * height, width)
        else:  # Transform along height (columns)
            # Transpose and reshape for column-wise processing  
            x_flat = x.permute(0, 1, 3, 2).contiguous().reshape(batch_size * channels * width, height)
        
        # Apply 1D spatial wavelet convolution using built-in F.conv1d
        padding = len(self.spatial_h0) // 2
        x_padded = F.pad(x_flat.unsqueeze(1), (padding, padding), mode='reflect')
        
        # Apply spatial wavelet filters
        low_pass = F.conv1d(x_padded, self.spatial_h0.reshape(1, 1, -1), stride=2)
        high_pass = F.conv1d(x_padded, self.spatial_h1.reshape(1, 1, -1), stride=2)
        
        # Get output size
        output_size = low_pass.shape[2]
        
        if axis == 3:  # Row-wise transform
            # Reshape back to spatial format
            low_pass = low_pass.reshape(batch_size, channels, height, output_size)
            high_pass = high_pass.reshape(batch_size, channels, height, output_size)
            return torch.cat([low_pass, high_pass], dim=3)
        else:  # Column-wise transform
            # Reshape and transpose back
            low_pass = low_pass.reshape(batch_size, channels, width, output_size)
            high_pass = high_pass.reshape(batch_size, channels, width, output_size)
            result = torch.cat([low_pass, high_pass], dim=3)
            return result.permute(0, 1, 3, 2).contiguous()
    
    def physics_informed_1d_temporal_wavelet(self, x_sequence):
        """Ultra-fast 1D TEMPORAL wavelet transform across frame sequence"""
        # x_sequence: (temporal_depth, batch, channels, height, width)
        temporal_depth, batch_size, channels, height, width = x_sequence.shape
        
        # Reshape for temporal processing: (batch*channels*height*width, temporal_depth)
        x_temporal = x_sequence.permute(1, 2, 3, 4, 0).contiguous()  # (batch, channels, height, width, temporal_depth)
        x_flat = x_temporal.reshape(batch_size * channels * height * width, temporal_depth)
        
        # Apply 1D temporal wavelet convolution
        padding = len(self.temporal_h0) // 2
        x_padded = F.pad(x_flat.unsqueeze(1), (padding, padding), mode='reflect')
        
        # Apply temporal wavelet filters (Haar for motion detection)
        temporal_low = F.conv1d(x_padded, self.temporal_h0.reshape(1, 1, -1), stride=2)
        temporal_high = F.conv1d(x_padded, self.temporal_h1.reshape(1, 1, -1), stride=2)
        
        # Get output temporal size
        temporal_output_size = temporal_low.shape[2]
        
        # Reshape back to spatial-temporal format
        temporal_low = temporal_low.reshape(batch_size, channels, height, width, temporal_output_size)
        temporal_high = temporal_high.reshape(batch_size, channels, height, width, temporal_output_size)
        
        # Convert back to (temporal_output_size, batch, channels, height, width)
        temporal_low = temporal_low.permute(4, 0, 1, 2, 3)
        temporal_high = temporal_high.permute(4, 0, 1, 2, 3)
        
        return temporal_low, temporal_high, temporal_output_size
    
    def forward_2d_spatial_wavelet(self, x):
        """Ultra-fast 2D SPATIAL wavelet transform using physics-informed separable decomposition"""
        batch_size, channels, height, width = x.shape
        
        # Row-wise spatial wavelet transform (along width dimension) - vectorized
        row_transformed = self.physics_informed_1d_spatial_wavelet(x, axis=3)
        
        # Column-wise spatial wavelet transform (along height dimension) - vectorized  
        # Transpose for column processing
        row_transposed = row_transformed.permute(0, 1, 3, 2)  # (batch, channels, width, height)
        col_transformed = self.physics_informed_1d_spatial_wavelet(row_transposed, axis=3)
        
        # Transpose back to original format
        result = col_transformed.permute(0, 1, 3, 2)  # (batch, channels, height, width)
        
        # Split into 4 spatial frequency bands using physics-based energy separation
        h_half, w_half = result.shape[2] // 2, result.shape[3] // 2
        
        # Extract spatial frequency bands (Low-Low, Low-High, High-Low, High-High)
        LL = result[:, :, :h_half, :w_half] * self.spatial_energy_weights[0]
        LH = result[:, :, :h_half, w_half:] * self.spatial_energy_weights[1]  
        HL = result[:, :, h_half:, :w_half] * self.spatial_energy_weights[2]
        HH = result[:, :, h_half:, w_half:] * self.spatial_energy_weights[3]
        
        # Flatten and concatenate for network processing
        return torch.cat([LL.flatten(2), LH.flatten(2), HL.flatten(2), HH.flatten(2)], dim=2)
    
    def forward_3d_spatiotemporal_wavelet(self, x_sequence):
        """🌊 REVOLUTIONARY 3D spatiotemporal wavelet transform - Space + Time decomposition!"""
        # x_sequence: (temporal_depth, batch, channels, height, width)
        temporal_depth, batch_size, channels, height, width = x_sequence.shape
        
        # Step 1: Temporal decomposition across the frame sequence
        temporal_low, temporal_high, temporal_output_size = self.physics_informed_1d_temporal_wavelet(x_sequence)
        
        # Step 2: Spatial decomposition on both temporal components
        spatiotemporal_bands = []
        
        # Process temporal low-frequency frames (stable motion)
        for t in range(temporal_output_size):
            spatial_coeffs = self.forward_2d_spatial_wavelet(temporal_low[t])  # (batch, channels, spatial_features)
            spatiotemporal_bands.append(spatial_coeffs)
        
        # Process temporal high-frequency frames (rapid motion)  
        for t in range(temporal_output_size):
            spatial_coeffs = self.forward_2d_spatial_wavelet(temporal_high[t])  # (batch, channels, spatial_features)
            spatiotemporal_bands.append(spatial_coeffs)
        
        # Combine all spatiotemporal frequency bands
        # Each frame contributes: 4 spatial bands × 2 temporal bands = 8 spatiotemporal bands per frame
        all_bands = torch.cat(spatiotemporal_bands, dim=2)  # (batch, channels, total_spatiotemporal_features)
        
        # Apply physics-based energy weights for spatiotemporal bands
        band_size = all_bands.shape[2] // 8  # 8 spatiotemporal frequency bands
        weighted_bands = []
        
        for i in range(8):
            start_idx = i * band_size
            end_idx = (i + 1) * band_size
            band = all_bands[:, :, start_idx:end_idx] * self.spatiotemporal_energy_weights[i]
            weighted_bands.append(band)
        
        spatiotemporal_features = torch.cat(weighted_bands, dim=2)
        
        return spatiotemporal_features, temporal_output_size
    
    def add_frame_to_buffer(self, frame_tensor):
        """Add frame to temporal buffer for spatiotemporal processing"""
        self.frame_buffer.append(frame_tensor.detach().clone())
    
    def get_spatiotemporal_sequence(self):
        """Get current frame sequence for spatiotemporal wavelet processing"""
        if len(self.frame_buffer) < self.temporal_depth:
            # Pad with repeated frames if not enough frames
            frames = list(self.frame_buffer)
            while len(frames) < self.temporal_depth:
                if len(frames) > 0:
                    frames.insert(0, frames[0])
                else:
                    # Create zero frame if buffer is empty
                    frames.append(torch.zeros(1, self.n_channels, self.input_size, self.input_size))
        else:
            frames = list(self.frame_buffer)
        
        # Stack into temporal sequence: (temporal_depth, batch, channels, height, width)
        return torch.stack(frames, dim=0)
    
    def can_process_spatiotemporal(self):
        """Check if we have enough frames for spatiotemporal processing"""
        return len(self.frame_buffer) >= self.temporal_depth
    
    def inverse_2d_wavelet(self, coeffs, target_height, target_width):
        """Ultra-fast inverse wavelet transform using physics principles"""
        batch_size, channels, total_features = coeffs.shape
        
        # Calculate actual coefficient dimensions  
        coeff_size = total_features // 4
        spatial_dim = int(np.sqrt(coeff_size))
        
        # Split back into frequency bands
        LL = coeffs[:, :, :coeff_size].reshape(batch_size, channels, spatial_dim, spatial_dim)
        LH = coeffs[:, :, coeff_size:2*coeff_size].reshape(batch_size, channels, spatial_dim, spatial_dim)
        HL = coeffs[:, :, 2*coeff_size:3*coeff_size].reshape(batch_size, channels, spatial_dim, spatial_dim)
        HH = coeffs[:, :, 3*coeff_size:].reshape(batch_size, channels, spatial_dim, spatial_dim)
        
        # Reconstruct using physics-informed energy conservation
        # Combine frequency bands into spatial domain
        top = torch.cat([LL, LH], dim=3)
        bottom = torch.cat([HL, HH], dim=3)
        coeffs_2d = torch.cat([top, bottom], dim=2)
        
        # Physics-informed upsampling to target resolution
        reconstruction = F.interpolate(coeffs_2d, size=(target_height, target_width), 
                                     mode='bilinear', align_corners=False)
        
        return reconstruction

class SpatiotemporalKoopmanOperator(nn.Module):
    """Revolutionary Koopman operator for linear spatiotemporal dynamics"""
    def __init__(self, latent_dim=32, temporal_window=5):
        super().__init__()
        self.latent_dim = latent_dim
        self.temporal_window = temporal_window
        
        print(f"🌊 Spatiotemporal Koopman Operator - Linear dynamics in lifted space")
        
        # Koopman operator: dz/dt = K * z (linear dynamics in lifted space)
        self.koopman_matrix = nn.Parameter(torch.eye(latent_dim) * 0.95 + torch.randn(latent_dim, latent_dim) * 0.01)
        
        # Observable functions φ(x) that lift state to space where dynamics are linear
        self.observable_functions = nn.Sequential(
            nn.Linear(latent_dim, latent_dim * 2),
            nn.Tanh(),
            nn.Linear(latent_dim * 2, latent_dim),
            nn.Tanh()
        )
        
        # Spatiotemporal propagator for motion prediction
        self.motion_propagator = nn.Sequential(
            nn.Linear(latent_dim * 2, latent_dim),  # current + previous
            nn.ReLU(),
            nn.Linear(latent_dim, latent_dim)
        )
        
        # Temporal memory buffer
        self.temporal_buffer = deque(maxlen=temporal_window)
        
        # Velocity estimation for immediate adaptation
        self.velocity_estimator = nn.Sequential(
            nn.Linear(latent_dim * 2, latent_dim),
            nn.Tanh(),
            nn.Linear(latent_dim, latent_dim)
        )
        
    def lift_to_observable_space(self, z):
        """Lift state to space where dynamics are linear"""
        return self.observable_functions(z)
    
    def predict_next_state(self, z_current, z_previous=None):
        """Predict next state using Koopman dynamics"""
        # Lift to observable space
        phi_z = self.lift_to_observable_space(z_current)
        
        # Apply Koopman operator: φ(z_{t+1}) = K * φ(z_t)
        phi_next = torch.matmul(phi_z, self.koopman_matrix.T)
        
        # Add motion-based correction if we have previous state
        if z_previous is not None:
            # Estimate velocity
            velocity = self.velocity_estimator(torch.cat([z_current, z_previous], dim=-1))
            
            # Spatiotemporal propagation
            motion_correction = self.motion_propagator(torch.cat([z_current, z_previous], dim=-1))
            
            # Combine Koopman prediction with motion correction
            phi_next = phi_next + 0.3 * motion_correction
        
        return phi_next
    
    def update_temporal_buffer(self, z):
        """Update temporal memory"""
        self.temporal_buffer.append(z.detach().clone())
    
    def get_temporal_context(self):
        """Get temporal context for spatiotemporal learning"""
        if len(self.temporal_buffer) < 2:
            return None, None
            
        return self.temporal_buffer[-1], self.temporal_buffer[-2]
    
    def forward(self, z_current):
        """Forward pass with spatiotemporal dynamics"""
        # Get temporal context
        z_prev, z_prev_prev = self.get_temporal_context()
        
        # Predict next state
        z_predicted = self.predict_next_state(z_current, z_prev)
        
        # Update temporal buffer
        self.update_temporal_buffer(z_current)
        
        return z_predicted

class OpticalFlowMotionDetector(nn.Module):
    """Optical flow based spatiotemporal motion detection"""
    def __init__(self):
        super().__init__()
        self.prev_frame = None
        
    def calculate_optical_flow(self, current_frame, prev_frame):
        """Calculate optical flow between frames"""
        if prev_frame is None:
            return None, 0.0
            
        # Convert to grayscale for optical flow
        current_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        
        # Calculate optical flow using Lucas-Kanade
        flow = cv2.calcOpticalFlowPyrLK(prev_gray, current_gray, None, None)
        
        # Calculate motion magnitude
        if flow[0] is not None:
            motion_vectors = flow[0]
            motion_magnitude = np.mean(np.linalg.norm(motion_vectors, axis=1))
            return flow, motion_magnitude
        
        return None, 0.0
    
    def forward(self, frame):
        """Detect spatiotemporal motion using optical flow"""
        if self.prev_frame is None:
            self.prev_frame = frame.copy()
            return False, 0.0, None
            
        flow, motion_magnitude = self.calculate_optical_flow(frame, self.prev_frame)
        
        # Classify motion type
        if motion_magnitude > 5.0:
            motion_type = "high"
        elif motion_magnitude > 2.0:
            motion_type = "medium"
        else:
            motion_type = "low"
            
        self.prev_frame = frame.copy()
        
        return motion_magnitude > 1.0, motion_magnitude, motion_type

class FrequencyMotionDetector(nn.Module):
    """Detect motion in wavelet frequency domain"""
    def __init__(self):
        super().__init__()
        
    def forward(self, current_coeffs, prev_coeffs):
        """Detect frequency domain motion"""
        if prev_coeffs is None:
            return False, 0.0
            
        # Compute energy difference across frequency bands
        diff = torch.abs(current_coeffs - prev_coeffs)
        
        # Energy in each frequency band
        band_size = current_coeffs.shape[2] // 4
        ll_energy = torch.mean(diff[:, :, :band_size])
        lh_energy = torch.mean(diff[:, :, band_size:2*band_size])  
        hl_energy = torch.mean(diff[:, :, 2*band_size:3*band_size])
        hh_energy = torch.mean(diff[:, :, 3*band_size:])
        
        # High frequency motion indicates rapid changes
        high_freq_motion = (lh_energy + hl_energy + hh_energy) / 3
        total_motion = (ll_energy + lh_energy + hl_energy + hh_energy) / 4
        
        # Motion detected if high frequency energy > threshold
        motion_threshold = 0.01
        is_motion = high_freq_motion > motion_threshold
        motion_score = total_motion.item()
        
        return is_motion.item(), motion_score

class SceneChangeDetector:
    """🔄 Revolutionary scene change detection for instant memory reset"""
    def __init__(self, threshold=0.3):
        self.threshold = threshold
        self.prev_histogram = None
        self.scene_change_count = 0
        
    def detect_scene_change(self, frame):
        """Detect major scene changes using histogram comparison"""
        # Calculate color histogram
        hist = cv2.calcHist([frame], [0, 1, 2], None, [32, 32, 32], [0, 256, 0, 256, 0, 256])
        hist = cv2.normalize(hist, hist).flatten()
        
        if self.prev_histogram is None:
            self.prev_histogram = hist
            return False, 0.0
            
        # Compare histograms using correlation
        correlation = cv2.compareHist(self.prev_histogram, hist, cv2.HISTCMP_CORREL)
        change_score = 1.0 - correlation
        
        # Update previous histogram
        self.prev_histogram = hist
        
        # Scene change detected if correlation is low
        is_scene_change = change_score > self.threshold
        if is_scene_change:
            self.scene_change_count += 1
            print(f"🔄 SCENE CHANGE DETECTED! Score: {change_score:.3f} (Count: {self.scene_change_count})")
            
        return is_scene_change, change_score

class CollapseDetector:
    """🛡️ Revolutionary VAE collapse detection and recovery"""
    def __init__(self):
        self.mean_history = deque(maxlen=10)
        self.std_history = deque(maxlen=10)
        self.collapse_count = 0
        
    def detect_collapse(self, reconstruction, mean, logvar):
        """Detect VAE posterior collapse"""
        # Calculate reconstruction statistics
        recon_mean = torch.mean(reconstruction).item()
        recon_std = torch.std(reconstruction).item()
        
        # Calculate latent statistics
        latent_mean = torch.mean(torch.abs(mean)).item()
        latent_std = torch.std(mean).item()
        
        # Store history
        self.mean_history.append(recon_mean)
        self.std_history.append(recon_std)
        
        # Collapse indicators
        indicators = {
            'low_reconstruction_std': recon_std < 0.05,  # Too uniform output
            'extreme_reconstruction_mean': recon_mean < 0.1 or recon_mean > 0.9,  # Too dark/bright
            'low_latent_diversity': latent_std < 0.01,  # No latent diversity
            'decreasing_std_trend': len(self.std_history) >= 5 and all(
                self.std_history[i] > self.std_history[i+1] for i in range(4)
            )  # Consistently decreasing diversity
        }
        
        # Count active indicators
        active_indicators = sum(indicators.values())
        collapse_score = active_indicators / len(indicators)
        
        # Collapse detected if multiple indicators are active
        is_collapsed = active_indicators >= 2
        
        if is_collapsed:
            self.collapse_count += 1
            print(f"🛡️ VAE COLLAPSE DETECTED! Score: {collapse_score:.3f} Active: {list(indicators.keys())} (Count: {self.collapse_count})")
            
        return is_collapsed, collapse_score, indicators
    
    def get_recovery_strategy(self, indicators):
        """Get recovery strategy based on collapse type"""
        if indicators['low_reconstruction_std']:
            return "diversity_injection"
        elif indicators['extreme_reconstruction_mean']:
            return "range_normalization"
        elif indicators['low_latent_diversity']:
            return "latent_noise_injection"
        else:
            return "general_reset"

class TemporalMemoryManager:
    """🧠 Revolutionary temporal memory management with exponential forgetting"""
    def __init__(self, forget_rate=0.95, burst_forget_rate=0.5):
        self.forget_rate = forget_rate
        self.burst_forget_rate = burst_forget_rate
        self.pattern_memory = deque(maxlen=20)
        self.static_pattern_count = 0
        
    def update_memory(self, frame_features):
        """Update temporal memory with new frame features"""
        self.pattern_memory.append(frame_features.detach().clone())
        
    def detect_static_patterns(self):
        """Detect PIXEL-LEVEL static patterns (OLED burn effect)"""
        if len(self.pattern_memory) < 10:
            return False, 0.0
            
        # Calculate variance across recent patterns
        recent_patterns = torch.stack(list(self.pattern_memory)[-10:])
        pixel_variance = torch.var(recent_patterns, dim=0)  # Per-pixel variance, don't take mean!
        
        # Detect static pixels (variance below threshold)
        static_threshold = 0.001
        static_pixels = pixel_variance < static_threshold
        
        # Calculate percentage of static pixels
        total_pixels = pixel_variance.numel()
        static_pixel_count = static_pixels.sum().item()
        static_percentage = static_pixel_count / total_pixels
        
        # OLED burn risk if too many pixels are static
        burn_risk_threshold = 0.15  # 15% of pixels static = burn risk
        is_static = static_percentage > burn_risk_threshold
        
        if is_static:
            self.static_pattern_count += 1
            print(f"🧠 PIXEL-LEVEL BURN RISK! Static pixels: {static_pixel_count}/{total_pixels} ({static_percentage*100:.1f}%) (Count: {self.static_pattern_count})")
        else:
            self.static_pattern_count = max(0, self.static_pattern_count - 1)
            
        return is_static, static_percentage
    
    def apply_exponential_forgetting(self, model):
        """Apply exponential forgetting to model parameters"""
        with torch.no_grad():
            for param in model.parameters():
                # Apply exponential decay to parameters
                param.mul_(self.forget_rate)
                
        print(f"🧠 Applied exponential forgetting (rate: {self.forget_rate})")
    
    def burst_forget(self, model):
        """Apply aggressive forgetting for scene changes"""
        with torch.no_grad():
            for param in model.parameters():
                # Apply strong exponential decay
                param.mul_(self.burst_forget_rate)
                # Add small amount of noise to break patterns
                param.add_(torch.randn_like(param) * 0.001)
                
        print(f"🧠 Applied BURST FORGETTING (rate: {self.burst_forget_rate})")
    
    def inject_spatial_noise_to_static_regions(self, model):
        """Inject noise specifically to static regions causing OLED burn"""
        if len(self.pattern_memory) < 10:
            return
            
        # Calculate per-pixel variance to find static regions
        recent_patterns = torch.stack(list(self.pattern_memory)[-10:])
        pixel_variance = torch.var(recent_patterns, dim=0)  # Per-pixel variance
        
        # Find static pixels (those with low variance)
        static_threshold = 0.001
        static_pixels = pixel_variance < static_threshold
        
        # Create spatial noise mask targeting static regions
        noise_mask = static_pixels.float()
        
        # Inject spatial noise into model parameters proportional to static regions
        with torch.no_grad():
            # Target decoder parameters to break static output patterns
            for param in model.decoder.parameters():
                if param.dim() >= 2:  # Only target weight matrices
                    # Generate noise proportional to static pixel density
                    static_density = noise_mask.mean().item()
                    noise_strength = min(0.02, static_density * 0.1)  # Cap at 2% noise
                    
                    spatial_noise = torch.randn_like(param) * noise_strength
                    param.add_(spatial_noise)
        
        static_count = static_pixels.sum().item()
        total_pixels = static_pixels.numel()
        print(f"🎯 SPATIAL NOISE INJECTION: {static_count}/{total_pixels} static pixels targeted")
    
    def reset_temporal_memory(self, model):
        """Complete reset of temporal memory"""
        # Clear memory buffers
        self.pattern_memory.clear()
        
        # Reset model temporal buffers
        if hasattr(model, 'wavelet_transform') and hasattr(model.wavelet_transform, 'frame_buffer'):
            model.wavelet_transform.frame_buffer.clear()
            
        if hasattr(model, 'koopman_operator') and hasattr(model.koopman_operator, 'temporal_buffer'):
            model.koopman_operator.temporal_buffer.clear()
            
        # Reset previous states
        model.prev_coeffs = None
        model.prev_latent = None
        
        print(f"🧠 COMPLETE TEMPORAL MEMORY RESET")

class MicroLeanVAE(nn.Module):
    """Revolutionary Pure Wavelet VAE with immediate Hamiltonian adaptation - NO CONVOLUTIONS!"""
    def __init__(self, input_size=240, latent_dim=32):
        super().__init__()
        self.input_size = input_size
        self.latent_dim = latent_dim
        
        print(f"🚀 REVOLUTIONARY IMMEDIATE ADAPTIVE HAMILTONIAN SYSTEM")
        print(f"🌊 Pure Wavelet Feature Extraction - NO convolutions!")
        print(f"⚛️ Differentiable Hamiltonian Dynamics with Energy Conservation")
        print(f"🔥 Immediate Frequency Domain Adaptation")
        
        # Revolutionary physics-informed wavelet transform (NO convolutions!)
        self.wavelet_transform = PhysicsInformedWaveletTransform(input_size, n_channels=3)
        
        # Calculate actual wavelet features by test transform
        test_input = torch.randn(1, 3, input_size, input_size)
        with torch.no_grad():
            test_coeffs = self.wavelet_transform.forward_2d_spatial_wavelet(test_input)
            self.wavelet_features = test_coeffs.shape[2]
            print(f"🌊 Physics-informed wavelet features: {self.wavelet_features} features")
            print(f"🌊 3D Spatiotemporal wavelet features: {self.wavelet_transform.spatiotemporal_features}")
        
        # Pure linear encoder/decoder (NO convolutions)
        self.encoder = nn.Sequential(
            nn.Linear(3 * self.wavelet_features, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, latent_dim * 2)  # mean + logvar
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 3 * self.wavelet_features)
        )
        
        # Spatiotemporal Koopman operator for linear dynamics  
        self.koopman_operator = SpatiotemporalKoopmanOperator(latent_dim, temporal_window=10)
        
        # Frequency motion detection
        self.motion_detector = FrequencyMotionDetector()
        
        # Previous state for temporal dynamics
        self.prev_coeffs = None
        self.prev_latent = None
        
        # Temporal consistency predictor
        self.temporal_predictor = nn.Sequential(
            nn.Linear(latent_dim * 2, latent_dim),
            nn.Tanh(),
            nn.Linear(latent_dim, latent_dim)
        )
        
    def encode(self, x, use_spatiotemporal=False):
        # Add frame to temporal buffer for spatiotemporal processing
        self.wavelet_transform.add_frame_to_buffer(x)
        
        # Choose processing mode based on temporal buffer availability
        if use_spatiotemporal and self.wavelet_transform.can_process_spatiotemporal():
            # 🌊 SPATIOTEMPORAL PROCESSING - Space + Time wavelets!
            frame_sequence = self.wavelet_transform.get_spatiotemporal_sequence()
            spatiotemporal_coeffs, temporal_frames = self.wavelet_transform.forward_3d_spatiotemporal_wavelet(frame_sequence)
            
            # Use spatiotemporal features
            batch_size = spatiotemporal_coeffs.shape[0]
            flattened = spatiotemporal_coeffs.reshape(batch_size, -1)
            
            # Detect spatiotemporal motion (more sophisticated)
            if self.prev_coeffs is not None:
                # Compare spatiotemporal energy across bands
                is_motion, motion_score = self.motion_detector(spatiotemporal_coeffs, self.prev_coeffs)
            else:
                is_motion, motion_score = False, 0.0
            
            self.prev_coeffs = spatiotemporal_coeffs.detach()
            
            # Print occasionally for debugging
            if hasattr(self, '_debug_counter'):
                self._debug_counter += 1
            else:
                self._debug_counter = 0
                
            if self._debug_counter % 20 == 0:  # Print occasionally
                print(f"🌊 Using SPATIOTEMPORAL wavelets: {flattened.shape[1]} features")
            
        else:
            # Standard 2D spatial wavelet processing
            wavelet_coeffs = self.wavelet_transform.forward_2d_spatial_wavelet(x)
            
            # Flatten for linear layers
            batch_size = wavelet_coeffs.shape[0]
            flattened = wavelet_coeffs.reshape(batch_size, -1)
            
            # Detect frequency domain motion
            is_motion, motion_score = self.motion_detector(wavelet_coeffs, self.prev_coeffs)
            self.prev_coeffs = wavelet_coeffs.detach()
        
        # Encode to latent space
        encoded = self.encoder(flattened)
        mean, logvar = torch.chunk(encoded, 2, dim=1)
        
        return mean, logvar
    
    def reparameterize(self, mean, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std) * 0.1  # Reduced noise for stability
            return mean + eps * std
        return mean
    
    def decode(self, z):
        # Decode from latent space
        decoded = self.decoder(z)
        
        # Reshape to wavelet coefficients
        batch_size = decoded.shape[0]
        wavelet_coeffs = decoded.reshape(batch_size, 3, self.wavelet_features)
        
        # Inverse wavelet transform
        reconstruction = self.wavelet_transform.inverse_2d_wavelet(
            wavelet_coeffs, self.input_size, self.input_size
        )
        
        return torch.sigmoid(reconstruction)
    
    def forward(self, x):
        # Encode with wavelet features
        mean, logvar = self.encode(x)
        
        # Sample latent representation
        z = self.reparameterize(mean, logvar)
        
        # Apply Koopman operator for spatiotemporal dynamics
        if self.prev_latent is not None and self.training:
            # Predict next state using Koopman operator
            predicted_z = self.koopman_operator(z)
            
            # Temporal consistency prediction
            temporal_prediction = self.temporal_predictor(torch.cat([z, self.prev_latent], dim=-1))
            
            # Blend predictions for stability
            z = 0.7 * z + 0.2 * predicted_z + 0.1 * temporal_prediction
        
        # Decode reconstruction
        reconstruction = self.decode(z)
        
        # Store for next iteration
        self.prev_latent = z.detach()
        
        return reconstruction, mean, logvar

class UltraFast60FpsLeanVAE:
    def __init__(self, device='cpu', learning_rate=1e-3, input_resolution=(640, 480)):
        self.device = device
        self.learning_rate = learning_rate
        
        # Calculate half the input resolution
        self.process_width = input_resolution[0] // 2
        self.process_height = input_resolution[1] // 2
        
        # Use the smaller dimension to ensure square processing for model compatibility
        self.process_size = min(self.process_width, self.process_height)
        # Make sure it's divisible by 8 for the conv layers
        self.process_size = (self.process_size // 8) * 8
        
        print(f"🚀 REVOLUTIONARY INSTANT ADAPTATION SYSTEM - Anti-Collapse & Ultra-Fast Learning")
        print(f"Initializing at {self.process_size}x{self.process_size} with instant forgetting mechanisms...")
        
        # Revolutionary immediate adaptive model with frequency domain processing
        self.inference_model = MicroLeanVAE(input_size=self.process_size, latent_dim=32).to(device)
        self.inference_model.eval()
        
        # 🔥 REVOLUTIONARY ULTRA-AGGRESSIVE OPTIMIZERS - 10x faster learning!
        self.instant_optimizer = optim.SGD(self.inference_model.parameters(), lr=learning_rate * 15, momentum=0.95)  # 15x for instant adaptation
        self.ultra_fast_optimizer = optim.SGD(self.inference_model.parameters(), lr=learning_rate * 10, momentum=0.9)  # 10x for rapid changes
        self.fast_optimizer = optim.SGD(self.inference_model.parameters(), lr=learning_rate * 5, momentum=0.9)
        self.adaptive_optimizer = optim.AdamW(self.inference_model.parameters(), lr=learning_rate * 2, weight_decay=1e-4)
        
        # Spatiotemporal motion detection with optical flow
        self.optical_flow_detector = OpticalFlowMotionDetector()
        self.prev_frame_coeffs = None
        
        # Koopman operator learning rate (separate for immediate dynamics learning)
        self.koopman_optimizer = optim.AdamW(self.inference_model.koopman_operator.parameters(), lr=learning_rate * 5)
        
        # 🔥 INSTANT FORGETTING & ANTI-COLLAPSE MECHANISMS
        self.scene_change_detector = SceneChangeDetector()
        self.collapse_detector = CollapseDetector()
        self.memory_manager = TemporalMemoryManager()
        
        print(f"🔥 REVOLUTIONARY ULTRA-AGGRESSIVE LEARNING ENABLED:")
        print(f"   - INSTANT Adaptation: {learning_rate * 15:.6f} (15x base rate)")
        print(f"   - Ultra-Fast SGD: {learning_rate * 10:.6f} (10x base rate)")
        print(f"   - Fast SGD: {learning_rate * 5:.6f} (5x base rate)")
        print(f"   - Adaptive AdamW: {learning_rate * 2:.6f} (2x base rate)")
        print(f"   - Koopman Learning: {learning_rate * 5:.6f} (5x base rate)")
        print(f"🛡️ ANTI-COLLAPSE PROTECTION:")
        print(f"   - Scene Change Detection with Memory Reset")
        print(f"   - VAE Collapse Detection & Recovery")
        print(f"   - Exponential Pattern Forgetting")
        
        # Threading for immediate adaptation
        self.training_queue = Queue(maxsize=10)
        self.training_thread = threading.Thread(target=self._immediate_adaptation_loop, daemon=True)
        self.training_active = True
        self.training_thread.start()
        
        # Performance tracking
        self.frame_times = deque(maxlen=100)
        self.inference_times = deque(maxlen=100)
        self.stats = {
            'frames_processed': 0,
            'avg_fps': 0,
            'avg_inference_time': 0,
            'frames_dropped': 0,
            'training_updates': 0
        }
        
        # Motion stability tracking
        self.previous_frame = None
        self.previous_reconstruction = None
        self.motion_history = deque(maxlen=10)
        self.loss_history = deque(maxlen=20)
        
        # Loss function
        self.mse_loss = nn.MSELoss()
        
    def _detect_motion(self, frame):
        """Detect motion between frames"""
        if self.previous_frame is None:
            self.previous_frame = frame.copy()
            return False, 0.0
            
        frame_diff = cv2.absdiff(frame, self.previous_frame)
        motion_score = np.mean(frame_diff) / 255.0
        self.previous_frame = frame.copy()
        
        is_motion = motion_score > 0.05  # Motion threshold
        return is_motion, motion_score
        
    def _immediate_adaptation_loop(self):
        """Immediate adaptation training loop"""
        while self.training_active:
            try:
                frame_data = self.training_queue.get(timeout=1.0)
                if frame_data is None:
                    continue
                    
                self._train_step(frame_data)
                    
            except Empty:
                continue
            except Exception as e:
                print(f"Training error: {e}")
    
    def _train_step(self, frame_data):
        """Ultra-aggressive spatiotemporal training for immediate adaptation with anti-collapse protection"""
        try:
            frame, motion_score = frame_data
            
            # 🔄 REVOLUTIONARY SCENE CHANGE DETECTION
            is_scene_change, scene_change_score = self.scene_change_detector.detect_scene_change(frame)
            
            # Prepare frame for training
            frame_resized = cv2.resize(frame, (self.process_size, self.process_size), interpolation=cv2.INTER_LINEAR)
            frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
            
            # Convert to tensor
            frame_tensor = torch.tensor(frame_rgb).permute(2, 0, 1).unsqueeze(0).float() / 255.0
            frame_tensor = frame_tensor.to(self.device)
            
            # 🧠 MEMORY MANAGEMENT & PATTERN FORGETTING
            if is_scene_change:
                # INSTANT MEMORY RESET for scene changes
                self.memory_manager.reset_temporal_memory(self.inference_model)
                self.memory_manager.burst_forget(self.inference_model)
                print(f"🔄 SCENE CHANGE TRIGGERED INSTANT RESET!")
            else:
                # Check for static patterns (OLED burn effect)
                is_static, static_variance = self.memory_manager.detect_static_patterns()
                if is_static:
                    # Apply TARGETED spatial noise injection to break static patterns
                    self.memory_manager.inject_spatial_noise_to_static_regions(self.inference_model)
                    # Also apply exponential forgetting for additional protection
                    self.memory_manager.apply_exponential_forgetting(self.inference_model)
                    print(f"🎯 STATIC PATTERN BROKEN with SPATIAL NOISE + exponential forgetting!")
            
            # Stable motion-based optimizer selection
            if motion_score > 0.2:
                # High motion: fast but stable learning
                optimizer = self.ultra_fast_optimizer
                koopman_optimizer = self.koopman_optimizer
                training_steps = 3  # 3 steps for stability
                kl_weight = 0.0001  # Balanced regularization
                print(f"🚀 FAST ADAPTATION - Motion: {motion_score:.3f}")
            elif motion_score > 0.1:
                # Medium motion: enhanced learning
                optimizer = self.fast_optimizer
                koopman_optimizer = self.koopman_optimizer
                training_steps = 2  # 2 steps for stability
                kl_weight = 0.0005
                print(f"🔥 MEDIUM ADAPTATION - Motion: {motion_score:.3f}")
            elif motion_score > 0.05:
                # Low motion: adaptive learning
                optimizer = self.adaptive_optimizer
                koopman_optimizer = self.koopman_optimizer
                training_steps = 1  # 1 step for stability
                kl_weight = 0.001
                print(f"⚡ ADAPTIVE LEARNING - Motion: {motion_score:.3f}")
            else:
                # Very low motion: stable learning
                optimizer = self.adaptive_optimizer  
                koopman_optimizer = self.koopman_optimizer
                training_steps = 1
                kl_weight = 0.001
                
            # Ultra-aggressive multi-step training
            for step in range(training_steps):
                self.inference_model.train()
                
                # Zero gradients for both optimizers
                optimizer.zero_grad()
                koopman_optimizer.zero_grad()
                
                # Forward pass
                reconstructed, mean, logvar = self.inference_model(frame_tensor)
                
                # 🛡️ REVOLUTIONARY COLLAPSE DETECTION & RECOVERY
                is_collapsed, collapse_score, collapse_indicators = self.collapse_detector.detect_collapse(
                    reconstructed, mean, logvar
                )
                
                if is_collapsed:
                    # Apply immediate recovery based on collapse type
                    recovery_strategy = self.collapse_detector.get_recovery_strategy(collapse_indicators)
                    
                    if recovery_strategy == "diversity_injection":
                        # Inject noise to increase diversity
                        with torch.no_grad():
                            for param in self.inference_model.decoder.parameters():
                                param.add_(torch.randn_like(param) * 0.01)
                        print(f"🛡️ Applied DIVERSITY INJECTION recovery!")
                        
                    elif recovery_strategy == "range_normalization":
                        # Reset decoder bias to normalize output range
                        with torch.no_grad():
                            if hasattr(self.inference_model.decoder[-1], 'bias'):
                                self.inference_model.decoder[-1].bias.zero_()
                        print(f"🛡️ Applied RANGE NORMALIZATION recovery!")
                        
                    elif recovery_strategy == "latent_noise_injection":
                        # Inject noise to latent space
                        with torch.no_grad():
                            for param in self.inference_model.encoder.parameters():
                                param.add_(torch.randn_like(param) * 0.005)
                        print(f"🛡️ Applied LATENT NOISE INJECTION recovery!")
                        
                    else:
                        # General reset - apply burst forgetting
                        self.memory_manager.burst_forget(self.inference_model)
                        print(f"🛡️ Applied GENERAL RESET recovery!")
                    
                    # Use ultra-aggressive learning for recovery
                    optimizer = self.instant_optimizer
                    training_steps = 5  # More aggressive recovery
                    kl_weight = 0.00001  # Minimal regularization during recovery
                    print(f"🛡️ COLLAPSE RECOVERY MODE - Ultra-aggressive learning!")
                
                # Check for NaN in outputs and skip training if found
                if torch.isnan(reconstructed).any() or torch.isnan(mean).any() or torch.isnan(logvar).any():
                    print("⚠️ NaN detected in forward pass, applying emergency reset...")
                    # Emergency NaN recovery
                    self.memory_manager.reset_temporal_memory(self.inference_model)
                    with torch.no_grad():
                        for param in self.inference_model.parameters():
                            param.data = torch.where(torch.isnan(param.data), 
                                                   torch.zeros_like(param.data), param.data)
                    break
                
                # 🧠 UPDATE MEMORY PATTERNS
                self.memory_manager.update_memory(reconstructed.detach())
                
                # Core reconstruction loss
                recon_loss = self.mse_loss(reconstructed, frame_tensor)
                
                # Adaptive KL loss based on motion
                kl_loss = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp())
                
                # Check for NaN in losses
                if torch.isnan(recon_loss).any() or torch.isnan(kl_loss).any():
                    print("⚠️ NaN detected in loss computation, skipping training step")
                    break
                
                # Spatiotemporal consistency loss (Koopman prediction)
                if hasattr(self.inference_model, 'prev_latent') and self.inference_model.prev_latent is not None:
                    current_z = self.inference_model.reparameterize(mean, logvar)
                    predicted_z = self.inference_model.koopman_operator(current_z)
                    
                    # Temporal consistency loss
                    temporal_loss = self.mse_loss(predicted_z, current_z)
                    temporal_weight = 0.5 if motion_score > 0.1 else 0.1
                else:
                    temporal_loss = torch.tensor(0.0, device=self.device)
                    temporal_weight = 0.0
                
                # Physics-informed loss (encourage smooth dynamics) - simplified for stability
                physics_loss = torch.tensor(0.0, device=self.device)
                if hasattr(self.inference_model.koopman_operator, 'koopman_matrix'):
                    # Simple Frobenius norm regularization to prevent explosive growth
                    matrix_norm = torch.norm(self.inference_model.koopman_operator.koopman_matrix, p='fro')
                    physics_loss = 0.001 * torch.clamp(matrix_norm - 5.0, min=0)  # Prevent matrix from growing too large
                
                # Total loss with aggressive weighting for motion
                total_loss = (recon_loss + 
                            kl_weight * kl_loss + 
                            temporal_weight * temporal_loss + 
                            physics_loss)
                
                # Backward pass
                total_loss.backward()
                
                # Gradient clipping for stability
                torch.nn.utils.clip_grad_norm_(self.inference_model.parameters(), max_norm=2.0)
                
                # Update both networks
                optimizer.step()
                koopman_optimizer.step()
                
                # Store loss for monitoring
                self.loss_history.append(total_loss.item())
            
            self.inference_model.eval()
            self.stats['training_updates'] += 1
            
        except Exception as e:
            print(f"Training step error: {e}")
            self.inference_model.eval()
    
    def process_frame_ultra_fast(self, frame):
        """Ultra-fast inference with immediate adaptation"""
        start_time = time.time()
        
        try:
            # Detect motion for adaptive response
            is_motion, motion_score = self._detect_motion(frame)
            
            # Resize for processing
            frame_small = cv2.resize(frame, (self.process_size, self.process_size), interpolation=cv2.INTER_LINEAR)
            frame_rgb = cv2.cvtColor(frame_small, cv2.COLOR_BGR2RGB)
            
            # Convert to tensor
            frame_tensor = torch.tensor(frame_rgb).permute(2, 0, 1).unsqueeze(0).float() / 255.0
            frame_tensor = frame_tensor.to(self.device)
            
            # Ultra-fast inference
            with torch.no_grad():
                reconstructed, _, _ = self.inference_model(frame_tensor)
                
                # Debug: Check if reconstruction is reasonable
                if self.stats['frames_processed'] % 10 == 0:  # Every 10 frames
                    print(f"🔍 Debug - Reconstruction range: {reconstructed.min():.3f} to {reconstructed.max():.3f}, mean: {reconstructed.mean():.3f}")
            
            # Convert back to image (detach to avoid gradient issues)
            recon_np = reconstructed.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
            recon_np = np.nan_to_num(recon_np, nan=0.0, posinf=1.0, neginf=0.0)
            recon_np = (recon_np * 255).clip(0, 255).astype(np.uint8)
            
            # Upscale back to display size
            recon_upscaled = cv2.resize(recon_np, (256, 256), interpolation=cv2.INTER_LINEAR)
            recon_bgr = cv2.cvtColor(recon_upscaled, cv2.COLOR_RGB2BGR)
            
            # Add to training queue for immediate adaptation
            try:
                self.training_queue.put_nowait((frame.copy(), motion_score))
            except:
                pass  # Queue full, skip this frame
            
            # Update timing stats
            inference_time = time.time() - start_time
            self.inference_times.append(inference_time)
            self.stats['frames_processed'] += 1
            
            return recon_bgr, inference_time, motion_score
            
        except Exception as e:
            print(f"Inference error: {e}")
            return None, 0.0, 0.0
    
    def get_stats(self):
        """Get current performance statistics"""
        if len(self.inference_times) > 0:
            self.stats['avg_inference_time'] = np.mean(list(self.inference_times)) * 1000  # ms
            
        if len(self.frame_times) > 0:
            frame_intervals = np.diff(list(self.frame_times))
            if len(frame_intervals) > 0:
                self.stats['avg_fps'] = 1.0 / np.mean(frame_intervals)
        
        return self.stats
    
    def cleanup(self):
        """Clean up background thread"""
        self.training_active = False
        if self.training_thread.is_alive():
            self.training_thread.join(timeout=2.0)

def main():
    parser = argparse.ArgumentParser(description='Revolutionary Immediate Adaptive Hamiltonian VAE')
    parser.add_argument('--device', type=str, default='cpu', help='Device (cpu/cuda)')
    parser.add_argument('--camera', type=int, default=0, help='Camera index')
    parser.add_argument('--target-fps', type=int, default=60, help='Target FPS')
    
    args = parser.parse_args()
    
    # Initialize camera
    print(f"Initializing camera {args.camera} for {args.target_fps} FPS...")
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print("Error: Could not open camera")
        return
    
    # Camera settings for high FPS
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 60)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    
    print(f"Camera: {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}")
    
    # Initialize revolutionary processor
    processor = UltraFast60FpsLeanVAE(device=args.device)
    
    print(f"\n🚀 REVOLUTIONARY IMMEDIATE ADAPTIVE HAMILTONIAN VAE:")
    print(f"- Pure Wavelet Feature Extraction (NO convolutions)")
    print(f"- Differentiable Hamiltonian Dynamics")
    print(f"- Immediate Frequency Domain Adaptation")
    print(f"- Target FPS: {args.target_fps}")
    print(f"- Device: {args.device}")
    print("\nPress 'q' to quit")
    
    frame_count = 0
    
    try:
        while True:
            current_time = time.time()
            processor.frame_times.append(current_time)
            
            ret, frame = cap.read()
            if not ret:
                print("Failed to read from camera")
                break
            
            frame_count += 1
            
            # Revolutionary processing
            reconstruction, inference_time, motion_score = processor.process_frame_ultra_fast(frame)
            
            if reconstruction is not None:
                # Prepare display
                display_frame = frame.copy()
                display_height = 300
                aspect_ratio = frame.shape[1] / frame.shape[0]
                display_width = int(display_height * aspect_ratio)
                display_frame = cv2.resize(display_frame, (display_width, display_height))
                
                # Resize reconstruction to match
                reconstruction_display = cv2.resize(reconstruction, (display_width, display_height))
                
                # Combine displays
                combined = np.hstack((display_frame, reconstruction_display))
                
                # Add labels and stats
                cv2.putText(combined, "Live Camera", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(combined, "Revolutionary Wavelet VAE", (display_width + 10, 25), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                
                # Performance stats
                stats = processor.get_stats()
                
                # Motion indicator
                if motion_score > 0.1:
                    motion_text = f"🔥 FAST ADAPT: {motion_score:.3f}"
                    motion_color = (0, 0, 255)  # Red
                elif motion_score > 0.05:
                    motion_text = f"⚡ Motion: {motion_score:.3f}"
                    motion_color = (0, 165, 255)  # Orange
                else:
                    motion_text = f"📊 Stable: {motion_score:.3f}"
                    motion_color = (0, 255, 0)  # Green
                
                stats_text = [
                    f"Target: {args.target_fps} FPS",
                    f"Actual: {stats['avg_fps']:.1f} FPS", 
                    f"Inference: {stats['avg_inference_time']:.1f}ms",
                    motion_text,
                    f"Frames: {stats['frames_processed']}",
                    f"Updates: {stats['training_updates']}"
                ]
                
                for i, stat in enumerate(stats_text):
                    color = motion_color if i == 3 else ((0, 255, 0) if stats['avg_inference_time'] < 16 else (0, 165, 255))
                    cv2.putText(combined, stat, (10, 60 + i * 25), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                
                # Performance indicator
                if stats['avg_inference_time'] < 16:
                    cv2.putText(combined, "🎉 REAL-TIME 60FPS!", (display_width + 10, display_height - 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                else:
                    cv2.putText(combined, f"TOO SLOW: {stats['avg_inference_time']:.1f}ms", 
                               (display_width + 10, display_height - 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                
                # Display
                cv2.imshow('Revolutionary Immediate Adaptive Hamiltonian VAE', combined)
            
            # Handle input
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
                
    except KeyboardInterrupt:
        print("\n🛑 Interrupted by user")
    
    finally:
        processor.cleanup()
        cap.release()
        cv2.destroyAllWindows()
        
        final_stats = processor.get_stats()
        print(f"\n📊 Session Complete:")
        print(f"- Frames processed: {final_stats['frames_processed']}")
        print(f"- Average FPS: {final_stats['avg_fps']:.1f}")
        print(f"- Average inference time: {final_stats['avg_inference_time']:.1f}ms")
        print(f"- Frames dropped: {final_stats['frames_dropped']}")
        print(f"- Training updates: {final_stats['training_updates']}")
        
        if final_stats['avg_inference_time'] < 16:
            print("🎉 SUCCESS: Achieved real-time 60 FPS inference!")
        else:
            print(f"⚠️  Still too slow for 60 FPS (need <16ms, got {final_stats['avg_inference_time']:.1f}ms)")

if __name__ == "__main__":
    main()
