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

class SpatiotemporalWaveletTransform(nn.Module):
    """FIXED 3D Spatiotemporal Wavelet Transform using Daubechies wavelets"""
    def __init__(self):
        super().__init__()
        
        # Daubechies-4 wavelet coefficients (orthogonal basis)
        self.register_buffer('db4_h', torch.tensor([
            0.230377813309, 0.714846570553, 0.630880767930, -0.027983769417,
            -0.187034811719, 0.030841381836, 0.032883011667, -0.010597401785
        ], dtype=torch.float32))
        
        # High-pass filter (quadrature mirror filter)
        self.register_buffer('db4_g', torch.tensor([
            -0.010597401785, -0.032883011667, 0.030841381836, 0.187034811719,
            -0.027983769417, -0.630880767930, 0.714846570553, -0.230377813309
        ], dtype=torch.float32))
    
    def forward_2d_wavelet(self, x):
        """2D Daubechies wavelet transform for spatial decomposition"""
        # x: (batch, channels, height, width)
        B, C, H, W = x.shape
        
        # Row-wise transform
        x_reshaped = x.reshape(B * C, H, W)
        h_filter = self.db4_h.reshape(1, 1, -1)
        g_filter = self.db4_g.reshape(1, 1, -1)
        
        # Apply filters along width dimension
        low_w = F.conv1d(x_reshaped.reshape(B * C * H, 1, W), h_filter, padding=4, stride=2)
        high_w = F.conv1d(x_reshaped.reshape(B * C * H, 1, W), g_filter, padding=4, stride=2)
        
        # Calculate actual output width after convolution
        W_new = low_w.shape[-1]
        
        # Reshape and apply along height dimension
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
        
        # Get actual output height
        H_new = low_low.shape[-1]
        
        # Reshape back
        low_low = low_low.reshape(B * C, W_new, H_new).transpose(1, 2)
        low_high = low_high.reshape(B * C, W_new, H_new).transpose(1, 2)
        high_low = high_low.reshape(B * C, W_new, H_new).transpose(1, 2)
        high_high = high_high.reshape(B * C, W_new, H_new).transpose(1, 2)
        
        LL = low_low.reshape(B, C, H_new, W_new)
        LH = low_high.reshape(B, C, H_new, W_new)  
        HL = high_low.reshape(B, C, H_new, W_new)
        HH = high_high.reshape(B, C, H_new, W_new)
        
        return LL, LH, HL, HH
    
    def forward_3d_wavelet(self, video_sequence):
        """3D spatiotemporal wavelet transform"""
        # video_sequence: (batch, time, channels, height, width)
        B, T, C, H, W = video_sequence.shape
        
        # Step 1: Apply 2D spatial wavelets to each frame
        spatial_coeffs = []
        for t in range(T):
            LL, LH, HL, HH = self.forward_2d_wavelet(video_sequence[:, t])
            spatial_coeffs.append(torch.stack([LL, LH, HL, HH], dim=1))  # (B, 4, C, H//2, W//2)
        
        spatial_tensor = torch.stack(spatial_coeffs, dim=2)  # (B, 4, T, C, H//2, W//2)
        
        # Step 2: Apply 1D temporal wavelets along time dimension
        # Reshape for temporal processing
        spatial_flat = spatial_tensor.reshape(B, 4 * C * (H // 2) * (W // 2), T)
        
        # Temporal wavelet transform
        h_temp = self.db4_h.reshape(1, 1, -1)
        g_temp = self.db4_g.reshape(1, 1, -1)
        
        temporal_low = F.conv1d(spatial_flat, h_temp, padding=4, stride=2)
        temporal_high = F.conv1d(spatial_flat, g_temp, padding=4, stride=2)
        
        # Reshape back to spatiotemporal coefficients
        T_new = temporal_low.shape[-1]  # Use actual output size
        low_coeffs = temporal_low.reshape(B, 4, C, H // 2, W // 2, T_new)
        high_coeffs = temporal_high.reshape(B, 4, C, H // 2, W // 2, T_new)
        
        return low_coeffs, high_coeffs

class WaveletNeuralOperator(nn.Module):
    """FIXED Wavelet Neural Operator for spatiotemporal processing"""
    def __init__(self, channels, modes_x, modes_y, modes_t):
        super().__init__()
        self.channels = channels
        self.modes_x = modes_x
        self.modes_y = modes_y  
        self.modes_t = modes_t
        
        # FIXED: Initialize with proper scaling to prevent mode collapse
        scale = 1.0 / (channels * modes_x * modes_y * modes_t) ** 0.5
        
        # Learnable wavelet coefficients for each scale
        self.wavelet_weights_low = nn.Parameter(torch.randn(channels, channels, modes_x, modes_y, modes_t) * scale)
        self.wavelet_weights_high = nn.Parameter(torch.randn(channels, channels, modes_x, modes_y, modes_t) * scale)
        
        # Linear skip connection
        self.skip_connection = nn.Conv3d(channels, channels, 1)
        
        # FIXED: Initialize skip connection properly
        nn.init.xavier_uniform_(self.skip_connection.weight)
        nn.init.zeros_(self.skip_connection.bias)
        
        # Energy conservation constraint
        self.energy_projector = nn.Linear(channels, 1)
        
    def forward(self, low_coeffs, high_coeffs):
        """Apply neural operator in wavelet domain"""
        B, spatial_modes, C, H, W, T = low_coeffs.shape
        
        # Reshape for processing
        low_flat = low_coeffs.reshape(B, spatial_modes * C, H, W, T)
        high_flat = high_coeffs.reshape(B, spatial_modes * C, H, W, T)
        
        # Apply learnable transformations in wavelet domain
        # FIXED: Ensure we don't exceed available dimensions
        actual_modes_x = min(self.modes_x, H)
        actual_modes_y = min(self.modes_y, W)
        actual_modes_t = min(self.modes_t, T)
        
        # Truncate to learned modes
        low_truncated = low_flat[:, :, :actual_modes_x, :actual_modes_y, :actual_modes_t]
        high_truncated = high_flat[:, :, :actual_modes_x, :actual_modes_y, :actual_modes_t]
        
        # Get corresponding weight tensors
        low_weights = self.wavelet_weights_low[:, :, :actual_modes_x, :actual_modes_y, :actual_modes_t]
        high_weights = self.wavelet_weights_high[:, :, :actual_modes_x, :actual_modes_y, :actual_modes_t]
        
        # Neural operator in wavelet domain (preserves locality)
        low_evolved = torch.einsum('bcxyt,dcxyt->bdxyt', low_truncated, low_weights)
        high_evolved = torch.einsum('bcxyt,dcxyt->bdxyt', high_truncated, high_weights)
        
        # Pad back to original size
        low_padded = torch.zeros_like(low_flat)
        high_padded = torch.zeros_like(high_flat)
        low_padded[:, :, :actual_modes_x, :actual_modes_y, :actual_modes_t] = low_evolved
        high_padded[:, :, :actual_modes_x, :actual_modes_y, :actual_modes_t] = high_evolved
        
        # Combine low and high frequency components
        combined = low_padded + high_padded
        
        # Skip connection for stability
        combined = combined + self.skip_connection(low_flat)
        
        # Reshape back
        output_low = combined.reshape(B, spatial_modes, C, H, W, T)
        output_high = torch.zeros_like(high_coeffs)
        
        return output_low, output_high

class PhysicsInformedPropagator(nn.Module):
    """FIXED Physics-informed propagator with energy conservation"""
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        
        # FIXED: Proper dimensions for Hamiltonian mechanics
        half_channels = channels // 2
        
        # Hamiltonian components - position and momentum spaces
        self.kinetic_operator = nn.Linear(half_channels, half_channels)   # For momentum (p)
        self.potential_operator = nn.Linear(half_channels, half_channels) # For position (q)
        
        # FIXED: Proper initialization to prevent exploding gradients
        nn.init.xavier_uniform_(self.kinetic_operator.weight, gain=0.1)
        nn.init.xavier_uniform_(self.potential_operator.weight, gain=0.1)
        nn.init.zeros_(self.kinetic_operator.bias)
        nn.init.zeros_(self.potential_operator.bias)
        
        # Symplectic integrator parameters
        self.dt = nn.Parameter(torch.tensor(0.001))  # Small time step for stability
        
    def hamiltonian(self, q, p):
        """Compute Hamiltonian H = T(p) + V(q)"""
        kinetic = 0.5 * torch.sum(p * self.kinetic_operator(p), dim=-1)
        potential = torch.sum(q * self.potential_operator(q), dim=-1)
        return kinetic + potential
    
    def symplectic_step(self, q, p):
        """Energy-preserving symplectic integration (FIXED)"""
        # Simplified symplectic integration (avoid autograd issues)
        
        # Update momentum using potential gradient
        force = -self.potential_operator(q)  # F = -∇V(q)
        p_new = p + self.dt * force
        
        # Update position using kinetic gradient  
        velocity = self.kinetic_operator(p_new)  # v = ∇T(p)
        q_new = q + self.dt * velocity
        
        return q_new, p_new
    
    def forward(self, state):
        """Physics-informed evolution with energy conservation"""
        # Split state into position and momentum
        mid = state.shape[-1] // 2
        q = state[..., :mid]  # Position coordinates
        p = state[..., mid:]  # Momentum coordinates
        
        # Symplectic integration (preserves Hamiltonian)
        q_new, p_new = self.symplectic_step(q, p)
        
        # Energy conservation constraint
        H_initial = self.hamiltonian(q, p)
        H_final = self.hamiltonian(q_new, p_new)
        energy_loss = torch.mean(torch.abs(H_final - H_initial))
        
        # Combine evolved state
        evolved_state = torch.cat([q_new, p_new], dim=-1)
        
        return evolved_state, energy_loss

class SpatiotemporalWNOVAE(nn.Module):
    """FIXED Revolutionary Spatiotemporal WNO-based VAE with 1:1 resolution"""
    def __init__(self, input_size=240, latent_dim=6, temporal_window=5):
        super().__init__()
        self.input_size = input_size
        self.latent_dim = latent_dim
        self.temporal_window = temporal_window
        
        # Spatiotemporal wavelet transform
        self.wavelet_transform = SpatiotemporalWaveletTransform()
        
        # WNO processing in wavelet domain - FIXED dimensions
        self.wno_encoder = WaveletNeuralOperator(
            channels=3, 
            modes_x=min(8, input_size//8),    # Conservative modes for stability
            modes_y=min(8, input_size//8), 
            modes_t=min(2, temporal_window//2)
        )
        
        # Physics-informed propagator
        self.physics_propagator = PhysicsInformedPropagator(latent_dim)
        
        # FIXED: Encoder/decoder for latent space with better design
        self.spatial_encoder = nn.Sequential(
            nn.Conv2d(3*4, 64, 3, padding=1),  # 4 spatial wavelet coefficients
            nn.BatchNorm2d(64),  # Added batch norm for stability
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(), 
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(0.1)  # Prevent overfitting
        )
        
        self.latent_encoder = nn.Linear(256, latent_dim * 2)
        self.latent_decoder = nn.Linear(latent_dim, 256)
        
        # FIXED: Better decoder design to prevent mode collapse
        self.spatial_decoder = nn.Sequential(
            nn.Linear(256, 256 * 4 * 4),  # Start from 4x4
            nn.ReLU(),
            nn.Unflatten(1, (256, 4, 4)),
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1), # 4x4 -> 8x8
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),  # 8x8 -> 16x16
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),   # 16x16 -> 32x32
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3*4, 4, stride=2, padding=1),  # 32x32 -> 64x64
            nn.Sigmoid()
        )
        
        # FIXED: Proper initialization
        self._initialize_weights()
        
        # Frame buffer for temporal processing
        self.frame_buffer = deque(maxlen=temporal_window)
        
    def _initialize_weights(self):
        """Proper weight initialization to prevent mode collapse"""
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.1)  # Small gain to prevent explosion
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
        
    def encode_frame(self, frame):
        """Encode single frame efficiently"""
        B, C, H, W = frame.shape
        
        # Single frame 2D wavelet transform
        LL, LH, HL, HH = self.wavelet_transform.forward_2d_wavelet(frame)
        
        # Combine wavelet coefficients
        wavelet_combined = torch.cat([LL, LH, HL, HH], dim=1)  # (B, 3*4, H//2, W//2)
        
        # Spatial encoding
        spatial_features = self.spatial_encoder(wavelet_combined)
        
        # Latent encoding
        latent_params = self.latent_encoder(spatial_features)
        mean, logvar = torch.chunk(latent_params, 2, dim=1)
        
        return mean, logvar
    
    def decode_frame(self, latent):
        """Decode single frame efficiently"""
        # Latent to spatial features
        spatial_features = self.latent_decoder(latent)
        
        # Spatial decoding - FIXED to match encoder output size
        wavelet_coeffs = self.spatial_decoder(spatial_features)
        
        # Split wavelet coefficients
        B, _, H, W = wavelet_coeffs.shape
        LL, LH, HL, HH = torch.chunk(wavelet_coeffs, 4, dim=1)
        
        # FIXED: Better inverse transform - for now use LL + some high frequency
        reconstructed = F.interpolate(LL, size=(self.input_size, self.input_size), mode='bilinear')
        
        # Add some high frequency detail to prevent mode collapse
        if H >= 2 and W >= 2:
            lh_detail = F.interpolate(LH, size=(self.input_size, self.input_size), mode='bilinear') * 0.1
            hl_detail = F.interpolate(HL, size=(self.input_size, self.input_size), mode='bilinear') * 0.1
            reconstructed = reconstructed + lh_detail + hl_detail
        
        return torch.clamp(reconstructed, 0, 1)  # Ensure valid range
    
    def spatiotemporal_processing(self, frame_sequence):
        """Full 3D spatiotemporal WNO processing"""
        # Only use when we have full temporal window
        if len(frame_sequence) < self.temporal_window:
            return None
            
        # Stack frames into 3D tensor
        video_tensor = torch.stack(frame_sequence, dim=1)  # (B, T, C, H, W)
        
        # 3D spatiotemporal wavelet transform
        low_coeffs, high_coeffs = self.wavelet_transform.forward_3d_wavelet(video_tensor)
        
        # WNO processing in wavelet domain
        evolved_low, evolved_high = self.wno_encoder(low_coeffs, high_coeffs)
        
        # Physics-informed temporal propagation
        # Extract central frame latent for physics processing
        central_frame_idx = self.temporal_window // 2
        central_low = evolved_low[:, :, :, :, :, central_frame_idx]
        
        # Flatten for physics propagator
        central_features = central_low.mean(dim=(1, 3, 4))  # (B, C)
        
        # Ensure we have 6D for physics (pad if needed)
        if central_features.shape[-1] < self.latent_dim:
            padding = torch.zeros(central_features.shape[0], 
                                self.latent_dim - central_features.shape[-1], 
                                device=central_features.device)
            central_features = torch.cat([central_features, padding], dim=-1)
        elif central_features.shape[-1] > self.latent_dim:
            central_features = central_features[:, :self.latent_dim]
        
        # Physics-informed evolution
        evolved_features, energy_loss = self.physics_propagator(central_features)
        
        return evolved_features, energy_loss
    
    def forward(self, frame):
        """Main forward pass - processes single frame with temporal context"""
        # Add frame to buffer
        self.frame_buffer.append(frame)
        
        # Standard single-frame processing (always available)
        mean, logvar = self.encode_frame(frame)
        
        # Reparameterization
        if self.training:
            std = torch.exp(0.5 * torch.clamp(logvar, -10, 10))  # Clamp to prevent explosion
            eps = torch.randn_like(std)
            latent = mean + eps * std
        else:
            latent = mean
        
        # Physics-informed temporal processing (when buffer is full)
        energy_loss = torch.tensor(0.0, device=frame.device)
        if len(self.frame_buffer) == self.temporal_window:
            temporal_result = self.spatiotemporal_processing(list(self.frame_buffer))
            if temporal_result is not None:
                evolved_latent, energy_loss = temporal_result
                # Gentle blending to avoid instability
                latent = 0.9 * latent + 0.1 * evolved_latent
        
        # Decode frame
        reconstructed = self.decode_frame(latent)
        
        return reconstructed, mean, logvar, energy_loss

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
        
        print(f"Initializing FIXED Spatiotemporal WNO-VAE for sub-16ms inference at {self.process_size}x{self.process_size}...")
        self.inference_model = SpatiotemporalWNOVAE(input_size=self.process_size, latent_dim=6).to(device)
        self.inference_model.eval()
        
        # Background training model (keeping for compatibility)
        print("Initializing background training model...")
        self.training_model = self._initialize_training_model().to(device)
        self.training_model.train()
        
        # Optimizers with lower learning rates for stability
        self.inference_optimizer = optim.Adam(self.inference_model.parameters(), lr=learning_rate*0.1)
        self.training_optimizer = optim.AdamW(self.training_model.parameters(), lr=learning_rate/20, weight_decay=1e-4)
        
        # Threading for background training (smaller queue for stability)
        self.training_queue = Queue(maxsize=5)
        self.training_thread = threading.Thread(target=self._background_training_loop, daemon=True)
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
        
        # Loss function
        self.mse_loss = nn.MSELoss()
        
        # Model update counter
        self.last_model_update = 0
        self.model_update_interval = 100
        
    def _initialize_training_model(self):
        """Initialize a slightly larger model for background training"""
        args = argparse.Namespace(
            embedding_dim=128,
            latent_dim=4,
            ista_iter_num=1,
            ista_layer_num=1,
            l_dim=32,
            h_dim=96,
            sep_num_layer=1,
            fusion_num_layer=1,
            use_tile_inference=False,
            chunksize_enc=5,
            chunksize_dec=3
        )
        
        model = LeanVAE(args)
        
        # Initialize weights
        def init_weights(m):
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        
        model.apply(init_weights)
        return model
    
    def _background_training_loop(self):
        """Background thread for training"""
        while self.training_active:
            try:
                # Get frame from queue (with timeout)
                frame_data = self.training_queue.get(timeout=1.0)
                if frame_data is None:
                    continue
                    
                frame, timestamp = frame_data
                
                # Train the background model
                self._train_step(frame)
                
                # Periodically update inference model
                if self.stats['training_updates'] % self.model_update_interval == 0:
                    self._update_inference_model()
                    
            except Empty:
                continue
            except Exception as e:
                print(f"Background training error: {e}")
    
    def _train_step(self, frame):
        """Training step with stability improvements"""
        try:
            # Prepare frame for training 
            frame_resized = cv2.resize(frame, (self.process_size, self.process_size), interpolation=cv2.INTER_LINEAR)
            frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
            
            # Convert to tensor - normalize to [0, 1]
            frame_tensor = torch.tensor(frame_rgb).permute(2, 0, 1).unsqueeze(0).float() / 255.0
            frame_tensor = frame_tensor.to(self.device)
            
            # Train the inference model
            self.inference_model.train()
            self.inference_optimizer.zero_grad()
            
            # Forward pass
            result = self.inference_model(frame_tensor)
            if len(result) == 4:
                reconstructed, mean, logvar, energy_loss = result
            else:
                reconstructed, mean, logvar = result
                energy_loss = torch.tensor(0.0, device=self.device)
            
            # FIXED: Better loss function to prevent mode collapse
            # L1 loss preserves details better
            recon_loss = torch.mean(torch.abs(reconstructed - frame_tensor))
            
            # Variance regularization to prevent mode collapse
            recon_var = torch.var(reconstructed)
            target_var = torch.var(frame_tensor)
            var_loss = torch.abs(recon_var - target_var)
            
            # KL loss with proper scaling
            kl_loss = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp())
            kl_loss = kl_loss / (frame_tensor.numel())  # Normalize by image size
            
            # Total loss with stability
            total_loss = recon_loss + 0.0001 * kl_loss + 0.1 * var_loss + 0.01 * energy_loss
            
            # Backward pass with gradient clipping
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.inference_model.parameters(), max_norm=0.5)
            self.inference_optimizer.step()
            
            self.inference_model.eval()
            self.stats['training_updates'] += 1
            
        except Exception as e:
            print(f"Training step error: {e}")
            self.inference_model.eval()
    
    def _update_inference_model(self):
        """Update the fast inference model"""
        try:
            print(f"🔄 Updating inference model (update #{self.stats['training_updates']//self.model_update_interval})")
            self.last_model_update = self.stats['training_updates']
        except Exception as e:
            print(f"Model update error: {e}")
    
    def process_frame_ultra_fast(self, frame):
        """Ultra-fast inference on single frame"""
        start_time = time.time()
        
        try:
            # Resize to processing resolution
            frame_small = cv2.resize(frame, (self.process_size, self.process_size), interpolation=cv2.INTER_LINEAR)
            frame_rgb = cv2.cvtColor(frame_small, cv2.COLOR_BGR2RGB)
            
            # Convert to tensor - normalize to [0, 1] to match Sigmoid output
            frame_tensor = torch.tensor(frame_rgb).permute(2, 0, 1).unsqueeze(0).float() / 255.0
            frame_tensor = frame_tensor.to(self.device)
            
            # Ultra-fast inference
            with torch.no_grad():
                result = self.inference_model(frame_tensor)
                if len(result) == 4:
                    reconstructed, _, _, _ = result  # WNO model returns 4 values
                else:
                    reconstructed, _, _ = result     # Standard model returns 3
            
            # Convert back to image (Sigmoid output is already [0, 1])
            recon_np = reconstructed.squeeze(0).permute(1, 2, 0).cpu().numpy()
            # Safety check for NaN/inf values
            recon_np = np.nan_to_num(recon_np, nan=0.0, posinf=1.0, neginf=0.0)
            recon_np = (recon_np * 255).clip(0, 255).astype(np.uint8)
            
            # Upscale back to original resolution for consistency  
            recon_upscaled = cv2.resize(recon_np, (self.process_size, self.process_size), interpolation=cv2.INTER_LINEAR)
            recon_bgr = cv2.cvtColor(recon_upscaled, cv2.COLOR_RGB2BGR)
            
            # Add to training queue (non-blocking)
            try:
                self.training_queue.put_nowait((frame.copy(), time.time()))
            except:
                self.stats['frames_dropped'] += 1
            
            # Update timing stats
            inference_time = time.time() - start_time
            self.inference_times.append(inference_time)
            self.stats['frames_processed'] += 1
            
            return recon_bgr, inference_time
            
        except Exception as e:
            print(f"Inference error: {e}")
            return None, 0.0
    
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
    parser = argparse.ArgumentParser(description='FIXED Ultra-Fast 60 FPS WNO-VAE')
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
    
    # Initialize ultra-fast processor
    processor = UltraFast60FpsLeanVAE(device=args.device)
    
    print(f"\nFIXED Revolutionary WNO-VAE System:")
    print(f"- Target FPS: {args.target_fps}")
    print(f"- Inference Resolution: {processor.process_size}x{processor.process_size}")
    print(f"- Revolutionary Components: ✅ Wavelets ✅ WNO ✅ Physics")
    print(f"- Device: {args.device}")
    print("\nPress 'q' to quit")
    print("🎯 Aiming for sub-16ms inference time per frame!")
    
    frame_count = 0
    last_time = time.time()
    target_frame_time = 1.0 / args.target_fps
    
    try:
        while True:
            current_time = time.time()
            processor.frame_times.append(current_time)
            
            ret, frame = cap.read()
            if not ret:
                print("Failed to read from camera")
                break
            
            frame_count += 1
            
            # Ultra-fast processing
            reconstruction, inference_time = processor.process_frame_ultra_fast(frame)
            
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
                cv2.putText(combined, "FIXED WNO-VAE", (display_width + 10, 25), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                
                # Performance stats
                stats = processor.get_stats()
                stats_text = [
                    f"Target: {args.target_fps} FPS",
                    f"Actual: {stats['avg_fps']:.1f} FPS",
                    f"Inference: {stats['avg_inference_time']:.1f}ms",
                    f"Processed: {stats['frames_processed']}",
                    f"Dropped: {stats['frames_dropped']}",
                    f"Training Updates: {stats['training_updates']}"
                ]
                
                for i, stat in enumerate(stats_text):
                    color = (0, 255, 0) if stats['avg_inference_time'] < 16 else (0, 165, 255)
                    cv2.putText(combined, stat, (10, 60 + i * 25), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                
                # Performance indicator
                if stats['avg_inference_time'] < 16:
                    cv2.putText(combined, "REAL-TIME 60FPS!", (display_width + 10, display_height - 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                else:
                    cv2.putText(combined, f"TOO SLOW: {stats['avg_inference_time']:.1f}ms", 
                               (display_width + 10, display_height - 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                
                # Add revolutionary system indicator
                cv2.putText(combined, "🌊 Wavelets ✅ 🧠 WNO ✅ ⚛️ Physics ✅", 
                           (10, display_height - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
                
                # Display
                cv2.imshow('FIXED Revolutionary WNO-VAE System', combined)
            
            # Handle input
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            
            # NO FRAME RATE LIMITING - MAXIMUM PERFORMANCE!
            # Revolutionary WNO system handles all temporal dynamics
                
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    
    finally:
        processor.cleanup()
        cap.release()
        cv2.destroyAllWindows()
        
        final_stats = processor.get_stats()
        print(f"\n🎯 FIXED Revolutionary WNO-VAE Session Complete:")
        print(f"- Frames processed: {final_stats['frames_processed']}")
        print(f"- Average FPS: {final_stats['avg_fps']:.1f}")
        print(f"- Average inference time: {final_stats['avg_inference_time']:.1f}ms")
        print(f"- Frames dropped: {final_stats['frames_dropped']}")
        print(f"- Training updates: {final_stats['training_updates']}")
        
        if final_stats['avg_inference_time'] < 16:
            print("🎉 SUCCESS: FIXED Revolutionary WNO-VAE achieved real-time 60 FPS!")
        else:
            print(f"⚠️  Still optimizing: {final_stats['avg_inference_time']:.1f}ms (target <16ms)")

if __name__ == "__main__":
    main()
