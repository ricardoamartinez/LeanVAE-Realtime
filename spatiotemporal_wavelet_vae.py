import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import deque
import cv2
import time
import threading
from queue import Queue, Empty

class SpatiotemporalWaveletTransform(nn.Module):
    """3D Spatiotemporal Wavelet Transform for space-time frequency analysis"""
    
    def __init__(self, temporal_depth=8):
        super().__init__()
        self.temporal_depth = temporal_depth
        
        # Daubechies-4 wavelet coefficients for spatial decomposition
        self.spatial_coeffs = torch.tensor([
            0.230377813309, 0.714846570553, 0.630880767930, -0.027983769417,
            -0.187034811719, 0.030841381836, 0.032883011667, -0.010597401785
        ], dtype=torch.float32)
        
        # Haar wavelet coefficients for temporal decomposition (simpler for motion)
        self.temporal_coeffs = torch.tensor([
            0.7071067812, 0.7071067812  # Normalized Haar
        ], dtype=torch.float32)
        
        # Create spatial filters
        self.register_buffer('spatial_low', self.spatial_coeffs.view(1, 1, -1))
        self.register_buffer('spatial_high', self.spatial_coeffs.flip(0).view(1, 1, -1) * torch.tensor([-1, 1] * 4, dtype=torch.float32))
        
        # Create temporal filters  
        self.register_buffer('temporal_low', self.temporal_coeffs.view(1, 1, -1))
        self.register_buffer('temporal_high', self.temporal_coeffs.flip(0).view(1, 1, -1) * torch.tensor([-1, 1], dtype=torch.float32))
        
        # Learnable spatiotemporal energy weights
        self.energy_weights = nn.Parameter(torch.ones(8))  # 8 spatiotemporal frequency bands
        
    def spatial_wavelet_2d(self, x):
        """2D spatial wavelet decomposition"""
        B, C, H, W = x.shape
        
        # Reshape for row-wise convolution: (B*C, 1, H, W) -> (B*C*H, 1, W)
        x_rows = x.view(B*C*H, 1, W)
        
        # Row-wise decomposition
        rows_low = F.conv1d(x_rows, self.spatial_low, padding=4, stride=2)
        rows_high = F.conv1d(x_rows, self.spatial_high, padding=4, stride=2)
        
        W_new = rows_low.shape[-1]
        
        # Reshape back: (B*C*H, 1, W_new) -> (B, C, H, W_new)
        rows_low = rows_low.view(B, C, H, W_new)
        rows_high = rows_high.view(B, C, H, W_new)
        
        # Column-wise decomposition on both low and high frequency rows
        # Reshape for column-wise: (B, C, H, W_new) -> (B*C*W_new, 1, H)
        low_cols = rows_low.permute(0, 1, 3, 2).contiguous().view(B*C*W_new, 1, H)
        high_cols = rows_high.permute(0, 1, 3, 2).contiguous().view(B*C*W_new, 1, H)
        
        # Apply column filters
        LL = F.conv1d(low_cols, self.spatial_low, padding=4, stride=2)
        LH = F.conv1d(low_cols, self.spatial_high, padding=4, stride=2)
        HL = F.conv1d(high_cols, self.spatial_low, padding=4, stride=2)
        HH = F.conv1d(high_cols, self.spatial_high, padding=4, stride=2)
        
        H_new = LL.shape[-1]
        
        # Reshape back to spatial format
        LL = LL.view(B, C, W_new, H_new).permute(0, 1, 3, 2)  # (B, C, H_new, W_new)
        LH = LH.view(B, C, W_new, H_new).permute(0, 1, 3, 2)
        HL = HL.view(B, C, W_new, H_new).permute(0, 1, 3, 2)
        HH = HH.view(B, C, W_new, H_new).permute(0, 1, 3, 2)
        
        return LL, LH, HL, HH, (H_new, W_new)
    
    def temporal_wavelet_1d(self, sequence):
        """1D temporal wavelet decomposition across frame sequence"""
        # sequence: (T, B, C, H, W)
        T, B, C, H, W = sequence.shape
        
        # Reshape for temporal convolution: (B*C*H*W, 1, T)
        temporal_data = sequence.permute(1, 2, 3, 4, 0).contiguous().view(B*C*H*W, 1, T)
        
        # Apply temporal filters
        temporal_low = F.conv1d(temporal_data, self.temporal_low, padding=1, stride=2)
        temporal_high = F.conv1d(temporal_data, self.temporal_high, padding=1, stride=2)
        
        T_new = temporal_low.shape[-1]
        
        # Reshape back: (B*C*H*W, 1, T_new) -> (T_new, B, C, H, W)
        temporal_low = temporal_low.view(B, C, H, W, T_new).permute(4, 0, 1, 2, 3)
        temporal_high = temporal_high.view(B, C, H, W, T_new).permute(4, 0, 1, 2, 3)
        
        return temporal_low, temporal_high, T_new
    
    def forward_3d_spatiotemporal(self, frame_sequence):
        """3D spatiotemporal wavelet decomposition"""
        # frame_sequence: (T, B, C, H, W)
        T, B, C, H, W = frame_sequence.shape
        
        # Step 1: Temporal decomposition first
        temporal_low, temporal_high, T_new = self.temporal_wavelet_1d(frame_sequence)
        
        # Step 2: Spatial decomposition on both temporal components
        # Process temporal_low frames
        low_spatial_bands = []
        for t in range(T_new):
            LL, LH, HL, HH, spatial_dims = self.spatial_wavelet_2d(temporal_low[t])
            low_spatial_bands.append([LL, LH, HL, HH])
        
        # Process temporal_high frames
        high_spatial_bands = []
        for t in range(T_new):
            LL, LH, HL, HH, spatial_dims = self.spatial_wavelet_2d(temporal_high[t])
            high_spatial_bands.append([LL, LH, HL, HH])
        
        # Combine into 8 spatiotemporal frequency bands
        # Low temporal frequency bands: LLL, LLH, LHL, LHH
        # High temporal frequency bands: HLL, HLH, HHL, HHH
        
        bands_3d = []
        for t in range(T_new):
            # Low temporal frequency spatial bands
            LLL = low_spatial_bands[t][0]  # Low-Low-Low
            LLH = low_spatial_bands[t][1]  # Low-Low-High  
            LHL = low_spatial_bands[t][2]  # Low-High-Low
            LHH = low_spatial_bands[t][3]  # Low-High-High
            
            # High temporal frequency spatial bands
            HLL = high_spatial_bands[t][0]  # High-Low-Low
            HLH = high_spatial_bands[t][1]  # High-Low-High
            HHL = high_spatial_bands[t][2]  # High-High-Low
            HHH = high_spatial_bands[t][3]  # High-High-High
            
            bands_3d.append([LLL, LLH, LHL, LHH, HLL, HLH, HHL, HHH])
        
        # Apply learnable energy weights
        weighted_bands = []
        for t in range(T_new):
            weighted_frame_bands = []
            for i, band in enumerate(bands_3d[t]):
                weighted_band = band * self.energy_weights[i]
                weighted_frame_bands.append(weighted_band)
            weighted_bands.append(weighted_frame_bands)
        
        # Flatten all bands for feature extraction
        all_coefficients = []
        for t in range(T_new):
            for band in weighted_bands[t]:
                all_coefficients.append(band.flatten(1))  # Flatten spatial dimensions
        
        # Concatenate all coefficients: (8 bands * T_new frames, B, spatial_features)
        spatiotemporal_features = torch.cat(all_coefficients, dim=1)  # (B, total_features)
        
        return spatiotemporal_features, spatial_dims, T_new
    
    def get_spatiotemporal_motion_energy(self, frame_sequence):
        """Analyze motion energy across spatiotemporal frequency bands"""
        features, _, _ = self.forward_3d_spatiotemporal(frame_sequence)
        
        # Calculate energy in each spatiotemporal band
        temporal_motion_energy = torch.norm(features[:, :features.shape[1]//2])  # Low temporal frequencies
        spatial_motion_energy = torch.norm(features[:, features.shape[1]//2:])   # High temporal frequencies
        
        return temporal_motion_energy.item(), spatial_motion_energy.item()

class PhysicsInformedSpatiotemporalPropagator(nn.Module):
    """Physics-informed propagator for spatiotemporal dynamics with energy conservation"""
    
    def __init__(self, base_feature_dim, hidden_dim=256):
        super().__init__()
        self.base_feature_dim = base_feature_dim
        self.hidden_dim = hidden_dim
        
        # Dynamic linear layers that will be created on first forward pass
        self.position_net = None
        self.velocity_net = None
        
        # Energy conservation matrix (symmetric for energy preservation)
        self.energy_matrix = nn.Parameter(torch.eye(hidden_dim) * 0.1)
        
        # Temporal propagation with spatial coupling
        self.temporal_dynamics = nn.Linear(hidden_dim, hidden_dim)
        self.spatial_coupling = nn.Linear(hidden_dim, hidden_dim)
        
    def _create_dynamic_layers(self, feature_dim):
        """Create linear layers based on actual feature dimension"""
        if self.position_net is None:
            self.position_net = nn.Linear(feature_dim, self.hidden_dim)
            self.velocity_net = nn.Linear(feature_dim, self.hidden_dim)
            # Move to same device as input
            if hasattr(self.energy_matrix, 'device'):
                self.position_net = self.position_net.to(self.energy_matrix.device)
                self.velocity_net = self.velocity_net.to(self.energy_matrix.device)
            print(f"🔧 Created dynamic propagator layers for {feature_dim} features")
        
    def forward(self, spatiotemporal_features, dt=0.016):  # 60 FPS timestep
        """Forward propagation with spatiotemporal physics"""
        
        # Create dynamic layers if not created yet
        feature_dim = spatiotemporal_features.shape[1]
        self._create_dynamic_layers(feature_dim)
        
        # Extract position and velocity in phase space
        q = torch.tanh(self.position_net(spatiotemporal_features))  # Position
        p = torch.tanh(self.velocity_net(spatiotemporal_features))  # Momentum/velocity
        
        # Hamiltonian dynamics: dq/dt = ∂H/∂p, dp/dt = -∂H/∂q
        # Simplified Hamiltonian: H = 0.5 * p^T * p + 0.5 * q^T * M * q
        
        # Kinetic energy gradient: ∂H/∂p = p
        dq_dt = p
        
        # Potential energy gradient: ∂H/∂q = M * q (with spatial coupling)
        spatial_potential = self.spatial_coupling(q)
        temporal_potential = self.temporal_dynamics(q)
        dp_dt = -(self.energy_matrix @ spatial_potential.T + temporal_potential.T).T
        
        # Symplectic Euler integration (energy preserving)
        p_new = p + dt * dp_dt
        q_new = q + dt * dq_dt
        
        # Combine position and momentum for final representation
        evolved_features = torch.cat([q_new, p_new], dim=1)
        
        # Calculate energy for conservation loss
        kinetic_energy = 0.5 * torch.sum(p_new**2, dim=1)
        potential_energy = 0.5 * torch.sum(q_new * (self.energy_matrix @ q_new.T).T, dim=1)
        total_energy = kinetic_energy + potential_energy
        
        return evolved_features, total_energy.mean()

class SpatiotemporalWaveletVAE(nn.Module):
    """Spatiotemporal Wavelet VAE with 3D space-time frequency analysis"""
    
    def __init__(self, input_shape=(3, 240, 240), latent_dim=512, temporal_depth=8):
        super().__init__()
        self.input_shape = input_shape
        self.latent_dim = latent_dim
        self.temporal_depth = temporal_depth
        
        # Initialize spatiotemporal wavelet transform
        self.wavelet_transform = SpatiotemporalWaveletTransform(temporal_depth)
        
        # Calculate feature dimensions with test input
        with torch.no_grad():
            test_sequence = torch.randn(temporal_depth, 1, *input_shape)
            test_features, self.spatial_dims, self.temporal_dims = self.wavelet_transform.forward_3d_spatiotemporal(test_sequence)
            self.feature_dim = test_features.shape[1]
            print(f"🌊 Spatiotemporal Wavelet Features: {self.feature_dim} (spatial: {self.spatial_dims}, temporal: {self.temporal_dims})")
        
        # Physics-informed propagator (will create dynamic layers based on actual feature dim)
        self.physics_propagator = PhysicsInformedSpatiotemporalPropagator(self.feature_dim)
        
        # VAE components (will be created dynamically)
        self.encoder = None
        self.decoder = None
        self.base_feature_dim = self.feature_dim
        
        # Frame buffer for temporal processing
        self.frame_buffer = deque(maxlen=temporal_depth)
        
    def _create_dynamic_vae(self, feature_dim):
        """Create VAE layers based on actual feature dimension"""
        if self.encoder is None:
            self.encoder = nn.Sequential(
                nn.Linear(feature_dim, self.latent_dim * 2),
                nn.ReLU(),
                nn.Linear(self.latent_dim * 2, self.latent_dim * 2)
            )
            
            self.decoder = nn.Sequential(
                nn.Linear(self.latent_dim, self.latent_dim * 2),
                nn.ReLU(),
                nn.Linear(self.latent_dim * 2, feature_dim)
            )
            print(f"🧠 Created dynamic VAE layers for {feature_dim} features")
        
    def encode(self, spatiotemporal_features):
        """Encode spatiotemporal features to latent space"""
        # Create dynamic VAE if not created yet
        feature_dim = spatiotemporal_features.shape[1]
        self._create_dynamic_vae(feature_dim)
        
        encoded = self.encoder(spatiotemporal_features)
        mu, logvar = encoded.chunk(2, dim=1)
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        """Reparameterization trick"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        """Decode latent to spatiotemporal features"""
        if self.decoder is None:
            raise RuntimeError("Decoder not initialized. Call encode() first.")
        return torch.sigmoid(self.decoder(z))
    
    def inverse_spatiotemporal_wavelet(self, features):
        """Reconstruct frame from spatiotemporal wavelet features"""
        # Simplified reconstruction: create a basic frame from the decoded features
        # In practice, this would involve full 3D inverse wavelet transform
        
        B, feature_dim = features.shape
        H, W = self.spatial_dims
        
        # Calculate how to distribute features across spatial dimensions
        target_spatial = self.input_shape[0] * H * W  # C * H * W
        
        if feature_dim < target_spatial:
            # If decoded features are smaller (like latent size), use a simple mapping
            # Map latent features to spatial domain through linear interpolation
            feature_map = features.view(B, -1, 1, 1)  # (B, feature_dim, 1, 1)
            
            # Interpolate to spatial dimensions
            spatial_features = F.interpolate(
                feature_map, 
                size=(H, W), 
                mode='bilinear', 
                align_corners=False
            )  # (B, feature_dim, H, W)
            
            # Take only the first 3 channels for RGB
            if spatial_features.shape[1] >= self.input_shape[0]:
                reconstruction = spatial_features[:, :self.input_shape[0]]  # (B, 3, H, W)
            else:
                # Repeat channels if not enough
                reconstruction = spatial_features.repeat(1, self.input_shape[0] // spatial_features.shape[1] + 1, 1, 1)
                reconstruction = reconstruction[:, :self.input_shape[0]]
        else:
            # If features are large enough, extract spatial bands properly
            feature_per_band = H * W * self.input_shape[0]  # H * W * C
            
            # Extract spatial bands from the decoded features
            if feature_dim >= 4 * feature_per_band:
                # Extract first 4 bands (simplified from 8 spatiotemporal bands)
                frame_features = features[:, :4 * feature_per_band]
                
                # Reshape to spatial bands
                LL = frame_features[:, :feature_per_band].view(B, self.input_shape[0], H, W)
                LH = frame_features[:, feature_per_band:2*feature_per_band].view(B, self.input_shape[0], H, W)
                HL = frame_features[:, 2*feature_per_band:3*feature_per_band].view(B, self.input_shape[0], H, W)
                HH = frame_features[:, 3*feature_per_band:4*feature_per_band].view(B, self.input_shape[0], H, W)
                
                # Combine spatial bands
                reconstruction = LL + 0.3 * (LH + HL + HH)
            else:
                # Fallback: distribute available features across spatial dimensions
                available_per_channel = feature_dim // self.input_shape[0]
                spatial_dim = int(np.sqrt(available_per_channel))
                
                if spatial_dim * spatial_dim * self.input_shape[0] <= feature_dim:
                    partial_features = features[:, :spatial_dim*spatial_dim*self.input_shape[0]]
                    reconstruction = partial_features.view(B, self.input_shape[0], spatial_dim, spatial_dim)
                else:
                    # Final fallback: use mean pooling
                    mean_features = torch.mean(features, dim=1, keepdim=True)  # (B, 1)
                    reconstruction = mean_features.view(B, 1, 1, 1).expand(B, self.input_shape[0], H, W)
        
        # Interpolate to final output size
        final_reconstruction = F.interpolate(
            reconstruction, 
            size=self.input_shape[1:], 
            mode='bilinear', 
            align_corners=False
        )
        
        return final_reconstruction
    
    def forward(self, frame_sequence):
        """Forward pass with spatiotemporal processing"""
        # frame_sequence: (T, B, C, H, W)
        
        # Extract spatiotemporal wavelet features
        spatiotemporal_features, _, _ = self.wavelet_transform.forward_3d_spatiotemporal(frame_sequence)
        
        # Apply physics-informed propagation
        evolved_features, energy_loss = self.physics_propagator(spatiotemporal_features)
        
        # VAE encoding
        mu, logvar = self.encode(evolved_features)
        z = self.reparameterize(mu, logvar)
        
        # VAE decoding
        decoded_features = self.decode(z)
        
        # Reconstruct frame
        reconstruction = self.inverse_spatiotemporal_wavelet(decoded_features)
        
        return reconstruction, mu, logvar, energy_loss
    
    def add_frame(self, frame):
        """Add frame to temporal buffer"""
        self.frame_buffer.append(frame)
        
    def get_frame_sequence(self):
        """Get current frame sequence tensor"""
        if len(self.frame_buffer) < self.temporal_depth:
            # Pad with repeated first frame if not enough frames
            frames = list(self.frame_buffer)
            while len(frames) < self.temporal_depth:
                frames.insert(0, frames[0] if frames else torch.zeros(1, *self.input_shape))
        else:
            frames = list(self.frame_buffer)
        
        return torch.stack(frames, dim=0)  # (T, B, C, H, W)

class SpatiotemporalMotionDetector:
    """Motion detection in spatiotemporal frequency domain"""
    
    def __init__(self, threshold_low=0.01, threshold_high=0.05):
        self.threshold_low = threshold_low
        self.threshold_high = threshold_high
        self.prev_temporal_energy = 0.0
        self.prev_spatial_energy = 0.0
        
    def detect_spatiotemporal_motion(self, model, frame_sequence):
        """Detect motion in spatiotemporal frequency domain"""
        if len(frame_sequence) < 2:
            return False, 0.0
            
        # Get spatiotemporal motion energy
        temporal_energy, spatial_energy = model.wavelet_transform.get_spatiotemporal_motion_energy(frame_sequence)
        
        # Calculate energy changes
        temporal_change = abs(temporal_energy - self.prev_temporal_energy)
        spatial_change = abs(spatial_energy - self.prev_spatial_energy)
        
        # Combined spatiotemporal motion score
        motion_score = 0.6 * temporal_change + 0.4 * spatial_change
        
        # Update previous energies
        self.prev_temporal_energy = temporal_energy
        self.prev_spatial_energy = spatial_energy
        
        # Determine motion level
        if motion_score > self.threshold_high:
            return "HIGH", motion_score
        elif motion_score > self.threshold_low:
            return "MEDIUM", motion_score
        else:
            return "LOW", motion_score

def main():
    """Test spatiotemporal wavelet VAE system"""
    print("🚀 REVOLUTIONARY SPATIOTEMPORAL WAVELET VAE")
    print("🌊 3D Space-Time Frequency Analysis with Physics-Informed Dynamics")
    
    # Initialize model
    model = SpatiotemporalWaveletVAE(
        input_shape=(3, 240, 240),
        latent_dim=512,
        temporal_depth=8
    )
    
    # Initialize motion detector
    motion_detector = SpatiotemporalMotionDetector()
    
    # Initialize camera
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 240)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
    cap.set(cv2.CAP_PROP_FPS, 60)
    
    # Initialize optimizers for different motion levels
    optimizer_stable = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    optimizer_medium = torch.optim.SGD(model.parameters(), lr=3e-4, momentum=0.9)
    optimizer_fast = torch.optim.SGD(model.parameters(), lr=8e-4, momentum=0.95)
    
    print("🎥 Starting spatiotemporal camera processing...")
    frame_count = 0
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            start_time = time.time()
            
            # Preprocess frame - resize to square 240x240
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_resized = cv2.resize(frame_rgb, (240, 240))  # Ensure square dimensions
            frame_tensor = torch.from_numpy(frame_resized).float().permute(2, 0, 1).unsqueeze(0) / 255.0
            
            # Add to temporal buffer
            model.add_frame(frame_tensor)
            
            # Get frame sequence for spatiotemporal processing
            if len(model.frame_buffer) >= model.temporal_depth:
                frame_sequence = model.get_frame_sequence()
                
                # Detect spatiotemporal motion
                motion_level, motion_score = motion_detector.detect_spatiotemporal_motion(model, frame_sequence)
                
                # Forward pass for inference (no gradients)
                with torch.no_grad():
                    reconstruction_display, mu_display, logvar_display, energy_loss_display = model(frame_sequence)
                
                # Display reconstruction
                recon_frame = reconstruction_display[0].permute(1, 2, 0).clamp(0, 1).numpy()
                recon_frame_bgr = cv2.cvtColor((recon_frame * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
                
                # Add spatiotemporal motion info to display
                cv2.putText(recon_frame_bgr, f"Motion: {motion_level} ({motion_score:.3f})", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(recon_frame_bgr, f"Energy: {energy_loss_display:.4f}", 
                           (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                cv2.putText(recon_frame_bgr, f"Latent std: {torch.std(mu_display).item():.4f}", 
                           (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
                # Show spatiotemporal reconstruction
                cv2.imshow('Spatiotemporal Wavelet VAE', recon_frame_bgr)
                
                # Background training based on motion level (separate forward pass with gradients)
                if motion_level == "HIGH":
                    # High motion: aggressive training
                    for _ in range(3):
                        reconstruction, mu, logvar, energy_loss = model(frame_sequence)
                        loss = torch.nn.functional.mse_loss(reconstruction, frame_sequence[-1]) + \
                               0.1 * energy_loss + \
                               0.001 * (-0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()))
                        optimizer_fast.zero_grad()
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                        optimizer_fast.step()
                elif motion_level == "MEDIUM":
                    # Medium motion: moderate training
                    reconstruction, mu, logvar, energy_loss = model(frame_sequence)
                    loss = torch.nn.functional.mse_loss(reconstruction, frame_sequence[-1]) + \
                           0.05 * energy_loss + \
                           0.005 * (-0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()))
                    optimizer_medium.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer_medium.step()
                else:
                    # Low motion: stable training
                    if frame_count % 3 == 0:  # Train every 3rd frame
                        reconstruction, mu, logvar, energy_loss = model(frame_sequence)
                        loss = torch.nn.functional.mse_loss(reconstruction, frame_sequence[-1]) + \
                               0.02 * energy_loss + \
                               0.01 * (-0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()))
                        optimizer_stable.zero_grad()
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                        optimizer_stable.step()
            
            # Calculate and display performance
            inference_time = (time.time() - start_time) * 1000
            fps = 1.0 / (time.time() - start_time) if time.time() - start_time > 0 else 0
            
            frame_count += 1
            print(f"Frame {frame_count}: {inference_time:.1f}ms ({fps:.1f} FPS) | "
                  f"Motion: {motion_level if 'motion_level' in locals() else 'N/A'}")
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    except KeyboardInterrupt:
        print("\n🔄 Gracefully shutting down...")
    
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("🎉 Spatiotemporal Wavelet VAE session complete!")

if __name__ == "__main__":
    main()
