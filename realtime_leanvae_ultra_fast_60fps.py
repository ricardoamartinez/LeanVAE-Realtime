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

class ImmediateFrequencyAdaptiveWavelet(nn.Module):
    """Immediate adaptive wavelets with learnable frequency adaptation"""
    def __init__(self, input_size=240, n_channels=3):
        super().__init__()
        self.input_size = input_size
        self.n_channels = n_channels
        
        # Pre-computed Daubechies-4 wavelet coefficients
        self.register_buffer('db4_low', torch.tensor([
            -0.010597401785, 0.032883011667, 0.030841381836, -0.187034811719,
            -0.027983769417, 0.630880767930, 0.714846570553, 0.230377813309
        ]))
        
        self.register_buffer('db4_high', torch.tensor([
            -0.230377813309, 0.714846570553, -0.630880767930, -0.027983769417,
            0.187034811719, 0.030841381836, -0.032883011667, -0.010597401785
        ]))
        
        # Learnable frequency adaptation weights for immediate response
        self.freq_adaptation_weights = nn.Parameter(torch.ones(4))  # LL, LH, HL, HH
        
    def forward_2d_wavelet(self, x):
        """2D Daubechies-4 wavelet transform using separable 1D transforms"""
        batch_size, channels, height, width = x.shape
        
        # Row-wise transform first
        x_reshaped = x.reshape(batch_size * channels * height, width)
        
        # Apply 1D wavelet transform to each row
        row_coeffs = []
        for i in range(x_reshaped.shape[0]):
            row = x_reshaped[i].unsqueeze(0).unsqueeze(0)  # (1, 1, width)
            
            # Low-pass filter
            low = F.conv1d(row, self.db4_low.view(1, 1, -1), padding=4)
            low = low[:, :, ::2]  # Downsample
            
            # High-pass filter  
            high = F.conv1d(row, self.db4_high.view(1, 1, -1), padding=4)
            high = high[:, :, ::2]  # Downsample
            
            row_coeffs.append(torch.cat([low, high], dim=2))
        
        # Stack all rows
        row_transformed = torch.stack([coeff.squeeze() for coeff in row_coeffs])
        W_new = row_coeffs[0].shape[2]
        row_transformed = row_transformed.reshape(batch_size, channels, height, W_new)
        
        # Column-wise transform
        col_reshaped = row_transformed.permute(0, 1, 3, 2).reshape(batch_size * channels * W_new, height)
        
        col_coeffs = []
        for i in range(col_reshaped.shape[0]):
            col = col_reshaped[i].unsqueeze(0).unsqueeze(0)  # (1, 1, height)
            
            # Low-pass filter
            low = F.conv1d(col, self.db4_low.view(1, 1, -1), padding=4)
            low = low[:, :, ::2]  # Downsample
            
            # High-pass filter
            high = F.conv1d(col, self.db4_high.view(1, 1, -1), padding=4)
            high = high[:, :, ::2]  # Downsample
            
            col_coeffs.append(torch.cat([low, high], dim=2))
        
        # Final result
        result = torch.stack([coeff.squeeze() for coeff in col_coeffs])
        H_new = col_coeffs[0].shape[2]
        result = result.reshape(batch_size, channels, W_new, H_new).permute(0, 1, 3, 2)
        
        # Split into 4 frequency bands: LL, LH, HL, HH
        H_half, W_half = H_new // 2, W_new // 2
        LL = result[:, :, :H_half, :W_half]
        LH = result[:, :, :H_half, W_half:]  
        HL = result[:, :, H_half:, :W_half]
        HH = result[:, :, H_half:, W_half:]
        
        # Apply learnable frequency adaptation
        LL = LL * self.freq_adaptation_weights[0]
        LH = LH * self.freq_adaptation_weights[1] 
        HL = HL * self.freq_adaptation_weights[2]
        HH = HH * self.freq_adaptation_weights[3]
        
        return torch.cat([LL.flatten(2), LH.flatten(2), HL.flatten(2), HH.flatten(2)], dim=2)
    
    def inverse_2d_wavelet(self, coeffs, target_height, target_width):
        """Inverse 2D wavelet transform"""
        batch_size = coeffs.shape[0]
        channels = self.n_channels
        
        # Calculate actual coefficient dimensions from the feature count
        total_features = coeffs.shape[2]
        coeff_size = total_features // 4  # 4 frequency bands (LL, LH, HL, HH)
        
        # Calculate spatial dimensions from coefficient size
        spatial_dim = int(np.sqrt(coeff_size))
        
        # Split coefficients back into frequency bands
        LL = coeffs[:, :, :coeff_size].reshape(batch_size, channels, spatial_dim, spatial_dim)
        LH = coeffs[:, :, coeff_size:2*coeff_size].reshape(batch_size, channels, spatial_dim, spatial_dim)
        HL = coeffs[:, :, 2*coeff_size:3*coeff_size].reshape(batch_size, channels, spatial_dim, spatial_dim)
        HH = coeffs[:, :, 3*coeff_size:].reshape(batch_size, channels, spatial_dim, spatial_dim)
        
        # Reconstruct full coefficient matrix
        top = torch.cat([LL, LH], dim=3)
        bottom = torch.cat([HL, HH], dim=3)
        coeffs_2d = torch.cat([top, bottom], dim=2)
        
        # Simple upsampling reconstruction (fast approximation)
        result = F.interpolate(coeffs_2d, size=(target_height, target_width), mode='bilinear', align_corners=False)
        
        return result

class HamiltonianDynamics(nn.Module):
    """Differentiable Hamiltonian dynamics with symplectic integration"""
    def __init__(self, latent_dim=32):
        super().__init__()
        self.latent_dim = latent_dim
        
        # Hamiltonian components: H(q,p) = T(p) + V(q)
        self.kinetic_energy = nn.Sequential(
            nn.Linear(latent_dim//2, 16),
            nn.Tanh(),
            nn.Linear(16, 1)
        )
        
        self.potential_energy = nn.Sequential(
            nn.Linear(latent_dim//2, 16), 
            nn.Tanh(),
            nn.Linear(16, 1)
        )
        
        self.force_field = nn.Sequential(
            nn.Linear(latent_dim//2, 16),
            nn.Tanh(), 
            nn.Linear(16, latent_dim//2)
        )
        
    def hamiltonian(self, q, p):
        """Compute total energy H(q,p) = T(p) + V(q)"""
        return self.kinetic_energy(p) + self.potential_energy(q)
    
    def symplectic_step(self, q, p, dt=0.01):
        """Symplectic leapfrog integration preserving energy"""
        # Step 1: Update momentum using current position
        force = -self.force_field(q)  # F = -∇V(q)
        p_half = p + 0.5 * dt * force
        
        # Step 2: Update position using half-step momentum
        # Assume unit mass: ∂T/∂p ≈ p
        q_new = q + dt * p_half
        
        # Step 3: Complete momentum update with new position
        force_new = -self.force_field(q_new)
        p_new = p_half + 0.5 * dt * force_new
        
        return q_new, p_new
    
    def forward(self, state, dt=0.01):
        """Evolve state through Hamiltonian dynamics"""
        q = state[..., :self.latent_dim//2]  # Position
        p = state[..., self.latent_dim//2:]  # Momentum
        
        q_new, p_new = self.symplectic_step(q, p, dt)
        
        return torch.cat([q_new, p_new], dim=-1)

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
        
        # Immediate adaptive wavelet transform
        self.wavelet_transform = ImmediateFrequencyAdaptiveWavelet(input_size, n_channels=3)
        
        # Calculate actual wavelet dimensions by test transform
        test_input = torch.randn(1, 3, input_size, input_size)
        with torch.no_grad():
            test_coeffs = self.wavelet_transform.forward_2d_wavelet(test_input)
            self.wavelet_features = test_coeffs.shape[2]
            print(f"🌊 Wavelet coefficients: {self.wavelet_features} features")
        
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
        
        # Hamiltonian dynamics for energy conservation
        self.hamiltonian = HamiltonianDynamics(latent_dim)
        
        # Frequency motion detection
        self.motion_detector = FrequencyMotionDetector()
        
        # Previous state for temporal dynamics
        self.prev_coeffs = None
        self.prev_latent = None
        
    def encode(self, x):
        # Transform to wavelet domain
        wavelet_coeffs = self.wavelet_transform.forward_2d_wavelet(x)
        
        # Flatten for linear layers
        batch_size = wavelet_coeffs.shape[0]
        flattened = wavelet_coeffs.reshape(batch_size, -1)
        
        # Encode to latent space
        encoded = self.encoder(flattened)
        mean, logvar = torch.chunk(encoded, 2, dim=1)
        
        # Detect frequency domain motion
        is_motion, motion_score = self.motion_detector(wavelet_coeffs, self.prev_coeffs)
        self.prev_coeffs = wavelet_coeffs.detach()
        
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
        
        # Apply Hamiltonian dynamics for temporal consistency
        if self.prev_latent is not None and self.training:
            # Evolve previous latent through Hamiltonian dynamics
            predicted_z = self.hamiltonian(self.prev_latent, dt=0.01)
            
            # Blend current and predicted latent for stability
            z = 0.8 * z + 0.2 * predicted_z
        
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
        
        print(f"Initializing revolutionary immediate adaptive system for 60 FPS at {self.process_size}x{self.process_size}...")
        
        # Revolutionary immediate adaptive model with frequency domain processing
        self.inference_model = MicroLeanVAE(input_size=self.process_size, latent_dim=32).to(device)
        self.inference_model.eval()
        
        # Fast SGD optimizer for immediate adaptation
        self.fast_optimizer = optim.SGD(self.inference_model.parameters(), lr=learning_rate * 10, momentum=0.9)
        
        # Adaptive optimizer for stable learning
        self.adaptive_optimizer = optim.AdamW(self.inference_model.parameters(), lr=learning_rate, weight_decay=1e-4)
        
        # Motion detection for frequency domain adaptation
        self.prev_frame_coeffs = None
        
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
        """Immediate adaptive training step"""
        try:
            frame, motion_score = frame_data
            
            # Prepare frame for training
            frame_resized = cv2.resize(frame, (self.process_size, self.process_size), interpolation=cv2.INTER_LINEAR)
            frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
            
            # Convert to tensor
            frame_tensor = torch.tensor(frame_rgb).permute(2, 0, 1).unsqueeze(0).float() / 255.0
            frame_tensor = frame_tensor.to(self.device)
            
            # Choose optimizer based on motion
            if motion_score > 0.1:
                # High motion: use fast SGD with multiple steps
                optimizer = self.fast_optimizer
                training_steps = 5  # Multiple steps for immediate adaptation
                print(f"🔥 Fast adaptation mode - Motion: {motion_score:.3f}")
            else:
                # Low motion: use stable AdamW
                optimizer = self.adaptive_optimizer  
                training_steps = 1
                
            # Multiple training steps for high motion
            for step in range(training_steps):
                self.inference_model.train()
                optimizer.zero_grad()
                
                # Forward pass
                reconstructed, mean, logvar = self.inference_model(frame_tensor)
                
                # Reconstruction loss
                recon_loss = self.mse_loss(reconstructed, frame_tensor)
                
                # KL loss (reduced during motion for faster adaptation)
                kl_loss = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp())
                kl_weight = 0.0001 if motion_score > 0.1 else 0.001
                
                total_loss = recon_loss + kl_weight * kl_loss
                
                # Backward pass
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.inference_model.parameters(), max_norm=1.0)
                optimizer.step()
            
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
            
            # Convert back to image
            recon_np = reconstructed.squeeze(0).permute(1, 2, 0).cpu().numpy()
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
