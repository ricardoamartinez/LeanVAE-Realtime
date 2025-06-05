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
    def check_tensor_health(tensor, name, verbose=False):
        """Check tensor for NaN/inf values and mode collapse indicators"""
        if tensor is None:
            if verbose: print(f"❌ {name}: None tensor!")
            return False
            
        if not isinstance(tensor, torch.Tensor):
            if verbose: print(f"❌ {name}: Not a tensor! Type: {type(tensor)}")
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
        elif std < 0.05:  # Low variance warning
            status = "⚠️ LOW VARIANCE"
            
        if verbose:
            print(f"{status} {name}: shape={tensor.shape}, std={std:.6f}, mean={mean:.6f}, range=[{min_val:.6f}, {max_val:.6f}]")
        
        return not (has_nan or has_inf or std < 1e-6)

class MicroLeanVAE(nn.Module):
    """Ultra-lightweight VAE - the WORKING version"""
    def __init__(self, input_size=320, latent_dim=6):
        super().__init__()
        self.input_size = input_size
        self.latent_dim = latent_dim
        
        # Calculate dimensions for dynamic input size
        final_size = input_size // 8  # After 3 stride-2 convolutions
        
        # Enhanced encoder for better color representation
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 24, 4, stride=2, padding=1),  # input_size -> input_size/2
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(24, 48, 4, stride=2, padding=1), # input_size/2 -> input_size/4
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(48, 96, 4, stride=2, padding=1), # input_size/4 -> input_size/8
            nn.LeakyReLU(0.2, inplace=True),
            nn.Flatten(),
            nn.Linear(96 * final_size * final_size, latent_dim * 2)  # mean + logvar
        )
        
        # Enhanced decoder for better color reconstruction
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 96 * final_size * final_size),
            nn.ReLU(inplace=True),
            nn.Unflatten(1, (96, final_size, final_size)),
            nn.ConvTranspose2d(96, 48, 4, stride=2, padding=1), # final_size -> final_size*2
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(48, 24, 4, stride=2, padding=1), # final_size*2 -> final_size*4
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(24, 3, 4, stride=2, padding=1),  # final_size*4 -> input_size
            nn.Sigmoid()  # Better for color range [0, 1]
        )
        
    def encode(self, x):
        h = self.encoder(x)
        mean, logvar = torch.chunk(h, 2, dim=1)
        return mean, logvar
    
    def reparameterize(self, mean, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mean + eps * std
        return mean
    
    def decode(self, z):
        return self.decoder(z)
    
    def forward(self, x):
        mean, logvar = self.encode(x)
        z = self.reparameterize(mean, logvar)
        recon = self.decode(z)
        return recon, mean, logvar

class FixedPhysicsInformedPropagator(nn.Module):
    """FIXED Physics propagator with correct dimensions"""
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        print(f"⚛️ Initializing FIXED Physics Propagator: channels={channels}")
        
        # FIXED: Use correct dimensions for position/momentum
        half_channels = channels // 2
        
        # Kinetic energy operator for momentum (p): 3D -> 3D  
        self.kinetic_operator = nn.Linear(half_channels, half_channels)
        # Potential energy operator for position (q): 3D -> 3D
        self.potential_operator = nn.Linear(half_channels, half_channels)
        
        # Initialize with small values for stability
        nn.init.normal_(self.kinetic_operator.weight, 0, 0.01)
        nn.init.normal_(self.potential_operator.weight, 0, 0.01)
        nn.init.zeros_(self.kinetic_operator.bias)
        nn.init.zeros_(self.potential_operator.bias)
        
        self.dt = nn.Parameter(torch.tensor(0.001))  # Small time step
        
    def hamiltonian(self, q, p):
        """Compute Hamiltonian with correct dimensions"""
        try:
            # q and p are both [batch, 3]
            kinetic = 0.5 * torch.sum(p * self.kinetic_operator(p), dim=-1)  # [batch]
            potential = torch.sum(q * self.potential_operator(q), dim=-1)   # [batch]
            H = kinetic + potential  # [batch]
            return H
        except Exception as e:
            print(f"❌ Hamiltonian computation failed: {e}")
            return torch.zeros(q.shape[0], device=q.device)
    
    def forward(self, state):
        """Physics forward with correct dimensions"""
        try:
            # Split state into position and momentum
            batch_size = state.shape[0]
            mid = state.shape[-1] // 2
            q = state[..., :mid]      # [batch, 3] - position
            p = state[..., mid:]      # [batch, 3] - momentum
            
            # Simplified symplectic integration with correct dimensions
            # Update momentum using force from potential
            force = -self.potential_operator(q)  # F = -∇V(q), [batch, 3]
            p_new = p + self.dt * force           # [batch, 3]
            
            # Update position using velocity from kinetic
            velocity = self.kinetic_operator(p_new)  # v = ∇T(p), [batch, 3]  
            q_new = q + self.dt * velocity            # [batch, 3]
            
            # Energy conservation check
            H_initial = self.hamiltonian(q, p)
            H_final = self.hamiltonian(q_new, p_new)
            energy_loss = torch.mean(torch.abs(H_final - H_initial))
            
            # Combine evolved state
            evolved_state = torch.cat([q_new, p_new], dim=-1)  # [batch, 6]
            
            return evolved_state, energy_loss
            
        except Exception as e:
            print(f"❌ Physics Propagator failed: {e}")
            import traceback
            traceback.print_exc()
            return state, torch.tensor(0.0, device=state.device)

class FixedSpatiotemporalWaveletTransform(nn.Module):
    """FIXED wavelet transform with better error handling"""
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
        """2D wavelet with better error handling and fallback"""
        try:
            B, C, H, W = x.shape
            
            # Row-wise transform
            x_reshaped = x.reshape(B * C, H, W)
            h_filter = self.db4_h.reshape(1, 1, -1)
            g_filter = self.db4_g.reshape(1, 1, -1)
            
            # Apply filters along width dimension
            low_w = F.conv1d(x_reshaped.reshape(B * C * H, 1, W), h_filter, padding=4, stride=2)
            high_w = F.conv1d(x_reshaped.reshape(B * C * H, 1, W), g_filter, padding=4, stride=2)
            
            W_new = low_w.shape[-1]
            
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
            
            # Final reshape
            LL = low_low.reshape(B * C, W_new, H_new).transpose(1, 2).reshape(B, C, H_new, W_new)
            LH = low_high.reshape(B * C, W_new, H_new).transpose(1, 2).reshape(B, C, H_new, W_new)
            HL = high_low.reshape(B * C, W_new, H_new).transpose(1, 2).reshape(B, C, H_new, W_new)
            HH = high_high.reshape(B * C, W_new, H_new).transpose(1, 2).reshape(B, C, H_new, W_new)
            
            # Check for mode collapse
            for name, component in [("LL", LL), ("LH", LH), ("HL", HL), ("HH", HH)]:
                if not DebugUtils.check_tensor_health(component, f"Wavelet {name}"):
                    print(f"⚠️ Wavelet component {name} has issues, using identity fallback")
                    return self._identity_fallback(x)
            
            return LL, LH, HL, HH
            
        except Exception as e:
            print(f"❌ 2D Wavelet Transform failed: {e}, using identity fallback")
            return self._identity_fallback(x)
    
    def _identity_fallback(self, x):
        """Fallback when wavelet transform fails"""
        B, C, H, W = x.shape
        H_new, W_new = H // 2, W // 2
        
        # Simple downsampling as fallback
        LL = F.avg_pool2d(x, 2)
        LH = torch.zeros(B, C, H_new, W_new, device=x.device) + 0.1 * torch.randn(B, C, H_new, W_new, device=x.device)
        HL = torch.zeros(B, C, H_new, W_new, device=x.device) + 0.1 * torch.randn(B, C, H_new, W_new, device=x.device)  
        HH = torch.zeros(B, C, H_new, W_new, device=x.device) + 0.1 * torch.randn(B, C, H_new, W_new, device=x.device)
        
        return LL, LH, HL, HH

class HybridWNOVAE(nn.Module):
    """Hybrid model that can switch between simple and complex modes"""
    def __init__(self, input_size=240, latent_dim=6, complexity_level=1):
        super().__init__()
        self.input_size = input_size
        self.latent_dim = latent_dim
        self.complexity_level = complexity_level  # 1=simple, 2=+wavelets, 3=+physics
        
        print(f"🔄 Initializing Hybrid WNO-VAE (complexity level {complexity_level}):")
        print(f"   - Input size: {input_size}x{input_size}")
        print(f"   - Latent dim: {latent_dim}")
        
        # Always have the simple, working model as base
        self.simple_vae = MicroLeanVAE(input_size=input_size, latent_dim=latent_dim)
        
        # Level 2: Add wavelets
        if complexity_level >= 2:
            self.wavelet_transform = FixedSpatiotemporalWaveletTransform()
            
            # Wavelet-aware encoder/decoder
            self.wavelet_encoder = nn.Sequential(
                nn.Conv2d(3*4, 32, 3, padding=1),  # 4 wavelet components
                nn.ReLU(),
                nn.Conv2d(32, 64, 3, stride=2, padding=1),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(64, latent_dim * 2)
            )
            
            self.wavelet_decoder = nn.Sequential(
                nn.Linear(latent_dim, 64),
                nn.ReLU(),
                nn.Unflatten(1, (64, 1, 1)),
                nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),
                nn.ReLU(),
                nn.ConvTranspose2d(32, 3*4, 3, padding=1),
                nn.Sigmoid()
            )
        
        # Level 3: Add physics
        if complexity_level >= 3:
            self.physics_propagator = FixedPhysicsInformedPropagator(latent_dim)
            self.use_physics = True
        else:
            self.use_physics = False
            
        # Frame buffer for temporal processing
        self.frame_buffer = deque(maxlen=5)
        
    def encode_frame_simple(self, frame):
        """Simple encoding using the working MicroLeanVAE"""
        return self.simple_vae.encode(frame)
        
    def encode_frame_wavelet(self, frame):
        """Wavelet-based encoding"""
        try:
            # Apply 2D wavelet transform
            LL, LH, HL, HH = self.wavelet_transform.forward_2d_wavelet(frame)
            
            # Combine wavelet coefficients
            wavelet_combined = torch.cat([LL, LH, HL, HH], dim=1)
            
            # Check for mode collapse in combined wavelets
            if not DebugUtils.check_tensor_health(wavelet_combined, "Wavelet Combined"):
                print("⚠️ Wavelet encoding failed, falling back to simple encoding")
                return self.encode_frame_simple(frame)
            
            # Encode through wavelet encoder
            latent_params = self.wavelet_encoder(wavelet_combined)
            mean, logvar = torch.chunk(latent_params, 2, dim=1)
            
            return mean, logvar
            
        except Exception as e:
            print(f"❌ Wavelet encoding failed: {e}, falling back to simple")
            return self.encode_frame_simple(frame)
    
    def decode_frame_simple(self, latent):
        """Simple decoding using the working MicroLeanVAE"""
        return self.simple_vae.decode(latent)
        
    def decode_frame_wavelet(self, latent):
        """Wavelet-based decoding"""
        try:
            # Decode to wavelet coefficients
            wavelet_coeffs = self.wavelet_decoder(latent)
            
            # Check for mode collapse in decoded coefficients
            if not DebugUtils.check_tensor_health(wavelet_coeffs, "Decoded Wavelet Coeffs"):
                print("⚠️ Wavelet decoding failed, falling back to simple decoding")
                return self.decode_frame_simple(latent)
            
            # For now, just use LL component (low-frequency)
            # TODO: Implement full inverse wavelet transform
            LL = wavelet_coeffs[:, :3]  # First 3 channels as LL
            
            # Interpolate to target size
            reconstructed = F.interpolate(LL, size=(self.input_size, self.input_size), mode='bilinear')
            
            return reconstructed
            
        except Exception as e:
            print(f"❌ Wavelet decoding failed: {e}, falling back to simple")
            return self.decode_frame_simple(latent)
    
    def apply_physics(self, latent):
        """Apply physics-informed evolution to latent"""
        if not self.use_physics:
            return latent, torch.tensor(0.0, device=latent.device)
            
        try:
            evolved_latent, energy_loss = self.physics_propagator(latent)
            
            # Check for mode collapse in physics output
            if not DebugUtils.check_tensor_health(evolved_latent, "Physics Evolved Latent"):
                print("⚠️ Physics evolution failed, using original latent")
                return latent, torch.tensor(0.0, device=latent.device)
                
            return evolved_latent, energy_loss
            
        except Exception as e:
            print(f"❌ Physics evolution failed: {e}, using original latent")
            return latent, torch.tensor(0.0, device=latent.device)
    
    def forward(self, frame):
        """Forward pass with complexity level switching"""
        # Add frame to buffer
        self.frame_buffer.append(frame)
        
        # Encoding based on complexity level
        if self.complexity_level >= 2:
            mean, logvar = self.encode_frame_wavelet(frame)
        else:
            mean, logvar = self.encode_frame_simple(frame)
        
        # Reparameterization
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            latent = mean + eps * std
        else:
            latent = mean
        
        # Physics evolution
        energy_loss = torch.tensor(0.0, device=frame.device)
        if self.complexity_level >= 3:
            latent, energy_loss = self.apply_physics(latent)
        
        # Decoding based on complexity level
        if self.complexity_level >= 2:
            reconstructed = self.decode_frame_wavelet(latent)
        else:
            reconstructed = self.decode_frame_simple(latent)
        
        return reconstructed, mean, logvar, energy_loss

def test_hybrid_progression():
    """Test hybrid model at each complexity level"""
    print("\n" + "="*60)
    print("🧪 TESTING HYBRID MODEL PROGRESSION")
    print("="*60)
    
    device = 'cpu'
    input_size = 64
    test_frame = torch.randn(1, 3, input_size, input_size) * 0.5 + 0.5  # Normalized
    
    for level in [1, 2, 3]:
        print(f"\n🔄 Testing Complexity Level {level}")
        print("-" * 40)
        
        try:
            model = HybridWNOVAE(input_size=input_size, latent_dim=6, complexity_level=level)
            
            with torch.no_grad():
                result = model(test_frame)
                if len(result) == 4:
                    recon, mean, logvar, energy_loss = result
                    
                    print(f"✅ Level {level} successful!")
                    DebugUtils.check_tensor_health(recon, f"Level {level} Reconstruction", verbose=True)
                    DebugUtils.check_tensor_health(mean, f"Level {level} Mean", verbose=True)
                    DebugUtils.check_tensor_health(logvar, f"Level {level} LogVar", verbose=True)
                    
                    if level >= 3:
                        DebugUtils.check_tensor_health(energy_loss, f"Level {level} Energy Loss", verbose=True)
                        
                else:
                    print(f"❌ Level {level} returned {len(result)} values instead of 4")
                    
        except Exception as e:
            print(f"❌ Level {level} failed: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    test_hybrid_progression()
    
    print("\n" + "="*60)
    print("🎯 TESTING FIXED PHYSICS PROPAGATOR")
    print("="*60)
    
    # Test fixed physics propagator specifically
    try:
        physics = FixedPhysicsInformedPropagator(6)
        test_state = torch.randn(2, 6) * 0.1  # Batch of 2, small values
        
        print(f"Input state shape: {test_state.shape}")
        evolved_state, energy_loss = physics(test_state)
        
        print(f"✅ Fixed Physics Propagator works!")
        DebugUtils.check_tensor_health(evolved_state, "Evolved State", verbose=True)
        DebugUtils.check_tensor_health(energy_loss, "Energy Loss", verbose=True)
        
    except Exception as e:
        print(f"❌ Fixed Physics Propagator failed: {e}")
        import traceback
        traceback.print_exc()
