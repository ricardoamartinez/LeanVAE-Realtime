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

class DebugUtils:
    """Utility class for systematic debugging"""
    
    @staticmethod
    def check_tensor_health(tensor, name):
        """Check tensor for NaN/inf values and mode collapse indicators"""
        if tensor is None:
            print(f"❌ {name}: None tensor!")
            return False
            
        if not isinstance(tensor, torch.Tensor):
            print(f"❌ {name}: Not a tensor! Type: {type(tensor)}")
            return False
            
        # Check for NaN/inf
        has_nan = torch.isnan(tensor).any()
        has_inf = torch.isinf(tensor).any()
        
        # Check for mode collapse (all values same)
        std = torch.std(tensor).item()
        mean = torch.mean(tensor).item()
        min_val = torch.min(tensor).item()
        max_val = torch.max(tensor).item()
        
        status = "✅"
        if has_nan:
            status = "🚨 NaN"
        elif has_inf:
            status = "🚨 Inf"
        elif std < 1e-6:
            status = "🚨 MODE COLLAPSE"
            
        print(f"{status} {name}: shape={tensor.shape}, std={std:.6f}, mean={mean:.6f}, range=[{min_val:.6f}, {max_val:.6f}]")
        
        return not (has_nan or has_inf or std < 1e-6)

class DebugSpatiotemporalWaveletTransform(nn.Module):
    """Debug version of wavelet transform with health checks"""
    def __init__(self):
        super().__init__()
        
        # Daubechies-4 wavelet coefficients
        self.register_buffer('db4_h', torch.tensor([
            0.230377813309, 0.714846570553, 0.630880767930, -0.027983769417,
            -0.187034811719, 0.030841381836, 0.032883011667, -0.010597401785
        ], dtype=torch.float32))
        
        self.register_buffer('db4_g', torch.tensor([
            -0.010597401785, -0.032883011667, 0.030841381836, 0.187034811719,
            -0.027983769417, -0.630880767930, 0.714846570553, -0.230377813309
        ], dtype=torch.float32))
    
    def forward_2d_wavelet(self, x):
        """2D wavelet with debug checks"""
        print(f"\n🔍 DEBUG: 2D Wavelet Transform")
        
        if not DebugUtils.check_tensor_health(x, "Input"):
            return None, None, None, None
            
        B, C, H, W = x.shape
        print(f"📊 Input shape: {x.shape}")
        
        try:
            # Row-wise transform
            x_reshaped = x.reshape(B * C, H, W)
            DebugUtils.check_tensor_health(x_reshaped, "X reshaped")
            
            h_filter = self.db4_h.reshape(1, 1, -1)
            g_filter = self.db4_g.reshape(1, 1, -1)
            
            # Apply filters along width dimension
            low_w = F.conv1d(x_reshaped.reshape(B * C * H, 1, W), h_filter, padding=4, stride=2)
            high_w = F.conv1d(x_reshaped.reshape(B * C * H, 1, W), g_filter, padding=4, stride=2)
            
            DebugUtils.check_tensor_health(low_w, "Low W")
            DebugUtils.check_tensor_health(high_w, "High W")
            
            W_new = low_w.shape[-1]
            print(f"📏 Width: {W} -> {W_new}")
            
            # Reshape for column-wise transform
            low_w = low_w.reshape(B * C, H, W_new)
            high_w = high_w.reshape(B * C, H, W_new)
            
            # Column-wise transform
            low_low = F.conv1d(low_w.transpose(1, 2).contiguous().reshape(B * C * W_new, 1, H), 
                              h_filter, padding=4, stride=2)
            low_high = F.conv1d(low_w.transpose(1, 2).contiguous().reshape(B * C * W_new, 1, H), 
                               g_filter, padding=4, stride=2)
            high_low = F.conv1d(high_w.transpose(1, 2).contiguous().reshape(B * C * W_new, 1, H), 
                               h_filter, padding=4, stride=2)
            high_high = F.conv1d(high_w.transpose(1, 2).contiguous().reshape(B * C * W_new, 1, H), 
                                g_filter, padding=4, stride=2)
            
            H_new = low_low.shape[-1]
            print(f"📏 Height: {H} -> {H_new}")
            
            # Final reshape
            LL = low_low.reshape(B * C, W_new, H_new).transpose(1, 2).reshape(B, C, H_new, W_new)
            LH = low_high.reshape(B * C, W_new, H_new).transpose(1, 2).reshape(B, C, H_new, W_new)
            HL = high_low.reshape(B * C, W_new, H_new).transpose(1, 2).reshape(B, C, H_new, W_new)
            HH = high_high.reshape(B * C, W_new, H_new).transpose(1, 2).reshape(B, C, H_new, W_new)
            
            DebugUtils.check_tensor_health(LL, "LL component")
            DebugUtils.check_tensor_health(LH, "LH component")
            DebugUtils.check_tensor_health(HL, "HL component")
            DebugUtils.check_tensor_health(HH, "HH component")
            
            return LL, LH, HL, HH
            
        except Exception as e:
            print(f"❌ 2D Wavelet Transform failed: {e}")
            return None, None, None, None

class DebugWaveletNeuralOperator(nn.Module):
    """Debug version of WNO"""
    def __init__(self, channels, modes_x, modes_y, modes_t):
        super().__init__()
        self.channels = channels
        self.modes_x = modes_x
        self.modes_y = modes_y  
        self.modes_t = modes_t
        
        print(f"🧠 Initializing WNO: channels={channels}, modes=({modes_x}, {modes_y}, {modes_t})")
        
        # Initialize with smaller, more stable values
        self.wavelet_weights_low = nn.Parameter(torch.randn(channels, channels, modes_x, modes_y, modes_t) * 0.01)
        self.wavelet_weights_high = nn.Parameter(torch.randn(channels, channels, modes_x, modes_y, modes_t) * 0.01)
        
        self.skip_connection = nn.Conv3d(channels, channels, 1)
        self.energy_projector = nn.Linear(channels, 1)
        
    def forward(self, low_coeffs, high_coeffs):
        """WNO forward with debugging"""
        print(f"\n🔍 DEBUG: WNO Forward")
        
        if not DebugUtils.check_tensor_health(low_coeffs, "WNO Low Input"):
            return low_coeffs, high_coeffs
            
        if not DebugUtils.check_tensor_health(high_coeffs, "WNO High Input"):
            return low_coeffs, high_coeffs
            
        try:
            B, spatial_modes, C, H, W, T = low_coeffs.shape
            print(f"📊 WNO Input shape: {low_coeffs.shape}")
            
            # Reshape for processing
            low_flat = low_coeffs.reshape(B, spatial_modes * C, H, W, T)
            high_flat = high_coeffs.reshape(B, spatial_modes * C, H, W, T)
            
            DebugUtils.check_tensor_health(low_flat, "WNO Low Flat")
            DebugUtils.check_tensor_health(high_flat, "WNO High Flat")
            
            # Truncate to available modes (safety check)
            actual_modes_x = min(self.modes_x, H)
            actual_modes_y = min(self.modes_y, W)
            actual_modes_t = min(self.modes_t, T)
            
            print(f"📏 Truncating to modes: ({actual_modes_x}, {actual_modes_y}, {actual_modes_t})")
            
            low_truncated = low_flat[:, :, :actual_modes_x, :actual_modes_y, :actual_modes_t]
            high_truncated = high_flat[:, :, :actual_modes_x, :actual_modes_y, :actual_modes_t]
            
            # Use smaller weight matrices to match actual dimensions
            low_weights = self.wavelet_weights_low[:, :, :actual_modes_x, :actual_modes_y, :actual_modes_t]
            high_weights = self.wavelet_weights_high[:, :, :actual_modes_x, :actual_modes_y, :actual_modes_t]
            
            # Neural operator application
            low_evolved = torch.einsum('bcxyt,dcxyt->bdxyt', low_truncated, low_weights)
            high_evolved = torch.einsum('bcxyt,dcxyt->bdxyt', high_truncated, high_weights)
            
            DebugUtils.check_tensor_health(low_evolved, "WNO Low Evolved")
            DebugUtils.check_tensor_health(high_evolved, "WNO High Evolved")
            
            # Pad back to original size
            low_padded = torch.zeros_like(low_flat)
            high_padded = torch.zeros_like(high_flat)
            low_padded[:, :, :actual_modes_x, :actual_modes_y, :actual_modes_t] = low_evolved
            high_padded[:, :, :actual_modes_x, :actual_modes_y, :actual_modes_t] = high_evolved
            
            # Combine and add skip connection
            combined = low_padded + high_padded
            combined = combined + self.skip_connection(low_flat)
            
            DebugUtils.check_tensor_health(combined, "WNO Combined")
            
            # Reshape back
            output_low = combined.reshape(B, spatial_modes, C, H, W, T)
            output_high = torch.zeros_like(high_coeffs)
            
            return output_low, output_high
            
        except Exception as e:
            print(f"❌ WNO Forward failed: {e}")
            return low_coeffs, high_coeffs

class DebugPhysicsInformedPropagator(nn.Module):
    """Debug version of physics propagator"""
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        print(f"⚛️ Initializing Physics Propagator: channels={channels}")
        
        # Initialize with smaller values for stability
        self.kinetic_operator = nn.Linear(channels, channels)
        self.potential_operator = nn.Linear(channels, channels)
        
        # Initialize weights to small values
        nn.init.normal_(self.kinetic_operator.weight, 0, 0.01)
        nn.init.normal_(self.potential_operator.weight, 0, 0.01)
        
        self.dt = nn.Parameter(torch.tensor(0.001))  # Much smaller time step
        
    def hamiltonian(self, q, p):
        """Compute Hamiltonian with debug checks"""
        try:
            kinetic = 0.5 * torch.sum(p * self.kinetic_operator(p), dim=-1)
            potential = torch.sum(q * self.potential_operator(q), dim=-1)
            H = kinetic + potential
            
            DebugUtils.check_tensor_health(kinetic, "Kinetic Energy")
            DebugUtils.check_tensor_health(potential, "Potential Energy")
            DebugUtils.check_tensor_health(H, "Hamiltonian")
            
            return H
        except Exception as e:
            print(f"❌ Hamiltonian computation failed: {e}")
            return torch.zeros_like(q.sum(dim=-1))
    
    def forward(self, state):
        """Physics forward with extensive debugging"""
        print(f"\n🔍 DEBUG: Physics Propagator")
        
        if not DebugUtils.check_tensor_health(state, "Physics Input State"):
            return state, torch.tensor(0.0, device=state.device)
            
        try:
            # Split state into position and momentum
            mid = state.shape[-1] // 2
            q = state[..., :mid]
            p = state[..., mid:]
            
            DebugUtils.check_tensor_health(q, "Position q")
            DebugUtils.check_tensor_health(p, "Momentum p")
            
            # Simplified physics (avoid autograd complications)
            # Just apply linear transformations instead of symplectic integration
            q_evolved = q + self.dt * self.kinetic_operator(p)
            p_evolved = p - self.dt * self.potential_operator(q)
            
            DebugUtils.check_tensor_health(q_evolved, "Evolved Position")
            DebugUtils.check_tensor_health(p_evolved, "Evolved Momentum")
            
            # Energy computation
            H_initial = self.hamiltonian(q, p)
            H_final = self.hamiltonian(q_evolved, p_evolved)
            energy_loss = torch.abs(H_final - H_initial)
            
            DebugUtils.check_tensor_health(energy_loss, "Energy Loss")
            
            evolved_state = torch.cat([q_evolved, p_evolved], dim=-1)
            DebugUtils.check_tensor_health(evolved_state, "Final Evolved State")
            
            return evolved_state, energy_loss
            
        except Exception as e:
            print(f"❌ Physics Propagator failed: {e}")
            return state, torch.tensor(0.0, device=state.device)

class DebugSpatiotemporalWNOVAE(nn.Module):
    """Debug version of the main model"""
    def __init__(self, input_size=240, latent_dim=6, temporal_window=5):
        super().__init__()
        self.input_size = input_size
        self.latent_dim = latent_dim
        self.temporal_window = temporal_window
        
        print(f"🚀 Initializing Debug Spatiotemporal WNO-VAE:")
        print(f"   - Input size: {input_size}x{input_size}")
        print(f"   - Latent dim: {latent_dim}")
        print(f"   - Temporal window: {temporal_window}")
        
        # Components with debug capabilities
        self.wavelet_transform = DebugSpatiotemporalWaveletTransform()
        
        # More conservative WNO parameters
        self.wno_encoder = DebugWaveletNeuralOperator(
            channels=3, 
            modes_x=min(16, input_size//8),  # Much smaller modes for stability
            modes_y=min(16, input_size//8), 
            modes_t=min(2, temporal_window//2)
        )
        
        self.physics_propagator = DebugPhysicsInformedPropagator(latent_dim)
        
        # Simpler encoder/decoder
        self.spatial_encoder = nn.Sequential(
            nn.Conv2d(3*4, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten()
        )
        
        self.latent_encoder = nn.Linear(64, latent_dim * 2)
        self.latent_decoder = nn.Linear(latent_dim, 64)
        
        self.spatial_decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3*4, 3, padding=1),
            nn.Sigmoid()
        )
        
        # Frame buffer
        self.frame_buffer = deque(maxlen=temporal_window)
        
    def encode_frame(self, frame):
        """Encode with debug checks"""
        print(f"\n🔍 DEBUG: Frame Encoding")
        
        if not DebugUtils.check_tensor_health(frame, "Input Frame"):
            return torch.zeros(1, self.latent_dim), torch.zeros(1, self.latent_dim)
            
        try:
            # 2D wavelet transform
            LL, LH, HL, HH = self.wavelet_transform.forward_2d_wavelet(frame)
            
            if LL is None:
                print("❌ Wavelet transform failed, using identity")
                # Fallback to simple processing
                combined = frame.repeat(1, 4, 1, 1)  # Just repeat channels
            else:
                combined = torch.cat([LL, LH, HL, HH], dim=1)
                
            DebugUtils.check_tensor_health(combined, "Wavelet Combined")
            
            # Spatial encoding
            spatial_features = self.spatial_encoder(combined)
            DebugUtils.check_tensor_health(spatial_features, "Spatial Features")
            
            # Latent encoding
            latent_params = self.latent_encoder(spatial_features)
            DebugUtils.check_tensor_health(latent_params, "Latent Params")
            
            mean, logvar = torch.chunk(latent_params, 2, dim=1)
            
            DebugUtils.check_tensor_health(mean, "Latent Mean")
            DebugUtils.check_tensor_health(logvar, "Latent LogVar")
            
            return mean, logvar
            
        except Exception as e:
            print(f"❌ Frame encoding failed: {e}")
            return torch.zeros(1, self.latent_dim), torch.zeros(1, self.latent_dim)
    
    def decode_frame(self, latent):
        """Decode with debug checks"""
        print(f"\n🔍 DEBUG: Frame Decoding")
        
        if not DebugUtils.check_tensor_health(latent, "Latent for Decoding"):
            return torch.zeros(1, 3, self.input_size, self.input_size)
            
        try:
            # Decode to spatial features
            spatial_features = self.latent_decoder(latent)
            DebugUtils.check_tensor_health(spatial_features, "Decoded Spatial Features")
            
            # Reshape for conv operations
            spatial_features = spatial_features.reshape(-1, 64, 1, 1)
            
            # Spatial decoding
            wavelet_coeffs = self.spatial_decoder(spatial_features)
            DebugUtils.check_tensor_health(wavelet_coeffs, "Decoded Wavelet Coeffs")
            
            # Use just the LL component (simplified)
            LL = wavelet_coeffs[:, :3]  # First 3 channels as LL
            
            # Interpolate to target size
            reconstructed = F.interpolate(LL, size=(self.input_size, self.input_size), mode='bilinear')
            DebugUtils.check_tensor_health(reconstructed, "Final Reconstruction")
            
            return reconstructed
            
        except Exception as e:
            print(f"❌ Frame decoding failed: {e}")
            return torch.zeros(1, 3, self.input_size, self.input_size)
    
    def forward(self, frame):
        """Main forward with comprehensive debugging"""
        print(f"\n🎯 DEBUG: Main Forward Pass")
        print(f"Frame buffer size: {len(self.frame_buffer)}")
        
        # Add frame to buffer
        self.frame_buffer.append(frame)
        
        # Standard single-frame processing
        mean, logvar = self.encode_frame(frame)
        
        # Reparameterization with debug
        if self.training:
            std = torch.exp(0.5 * logvar)
            DebugUtils.check_tensor_health(std, "Reparameterization Std")
            eps = torch.randn_like(std)
            latent = mean + eps * std
        else:
            latent = mean
            
        DebugUtils.check_tensor_health(latent, "Final Latent")
        
        # Physics processing (only if enough history)
        energy_loss = torch.tensor(0.0, device=frame.device)
        
        # Decode frame
        reconstructed = self.decode_frame(latent)
        
        return reconstructed, mean, logvar, energy_loss

# Test function to validate components in isolation
def test_components_individually():
    """Test each component separately to identify failures"""
    print("\n" + "="*60)
    print("🧪 TESTING COMPONENTS IN ISOLATION")
    print("="*60)
    
    device = 'cpu'
    
    # Test 1: Simple tensor operations
    print("\n1️⃣ Testing basic tensor operations...")
    test_tensor = torch.randn(1, 3, 64, 64)
    DebugUtils.check_tensor_health(test_tensor, "Test Tensor")
    
    # Test 2: 2D Wavelet Transform
    print("\n2️⃣ Testing 2D Wavelet Transform...")
    wavelet = DebugSpatiotemporalWaveletTransform()
    LL, LH, HL, HH = wavelet.forward_2d_wavelet(test_tensor)
    
    # Test 3: Simple VAE components
    print("\n3️⃣ Testing Simple VAE components...")
    encoder = nn.Sequential(
        nn.Conv2d(3, 16, 3, padding=1),
        nn.ReLU(),
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Linear(16, 12)  # 6*2 for mean + logvar
    )
    
    with torch.no_grad():
        encoded = encoder(test_tensor)
        DebugUtils.check_tensor_health(encoded, "Simple Encoder Output")
    
    # Test 4: Physics propagator
    print("\n4️⃣ Testing Physics Propagator...")
    physics = DebugPhysicsInformedPropagator(6)
    test_state = torch.randn(1, 6) * 0.1  # Small values
    evolved_state, energy_loss = physics(test_state)
    
    print("\n✅ Component testing complete!")

if __name__ == "__main__":
    test_components_individually()
    
    # Test the full debug model
    print("\n" + "="*60)
    print("🎯 TESTING FULL DEBUG MODEL")
    print("="*60)
    
    try:
        model = DebugSpatiotemporalWNOVAE(input_size=64, latent_dim=6, temporal_window=3)
        test_frame = torch.randn(1, 3, 64, 64) * 0.5 + 0.5  # Normalized
        
        with torch.no_grad():
            result = model(test_frame)
            if len(result) == 4:
                recon, mean, logvar, energy_loss = result
                print(f"\n🎉 Full model test successful!")
                DebugUtils.check_tensor_health(recon, "Final Reconstruction")
            else:
                print(f"❌ Model returned {len(result)} values instead of 4")
                
    except Exception as e:
        print(f"❌ Full model test failed: {e}")
        import traceback
        traceback.print_exc()
