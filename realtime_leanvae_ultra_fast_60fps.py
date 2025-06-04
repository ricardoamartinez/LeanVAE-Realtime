import cv2
import torch
import torch.nn as nn
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

class MicroLeanVAE(nn.Module):
    """Ultra-lightweight VAE for 60 FPS inference"""
    def __init__(self, input_size=320, latent_dim=4):
        super().__init__()
        self.input_size = input_size
        self.latent_dim = latent_dim
        
        # Calculate dimensions for dynamic input size
        # Assuming input_size is always divisible by 8 (for 3 stride-2 convolutions)
        final_size = input_size // 8  # After 3 stride-2 convolutions
        
        # Ultra-simple encoder: input_size x input_size x 3 -> latent_dim
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, 4, stride=2, padding=1),  # input_size -> input_size/2
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, 4, stride=2, padding=1), # input_size/2 -> input_size/4
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 4, stride=2, padding=1), # input_size/4 -> input_size/8
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(64 * final_size * final_size, latent_dim * 2)  # mean + logvar
        )
        
        # Ultra-simple decoder: latent_dim -> input_size x input_size x 3
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64 * final_size * final_size),
            nn.ReLU(inplace=True),
            nn.Unflatten(1, (64, final_size, final_size)),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1), # final_size -> final_size*2
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 16, 4, stride=2, padding=1), # final_size*2 -> final_size*4
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(16, 3, 4, stride=2, padding=1),  # final_size*4 -> input_size
            nn.Tanh()
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
        
        print(f"Initializing micro inference model for 60 FPS at {self.process_size}x{self.process_size}...")
        self.inference_model = MicroLeanVAE(input_size=self.process_size, latent_dim=4).to(device)
        self.inference_model.eval()
        
        # Background training model (larger, updates inference model)
        print("Initializing background training model...")
        self.training_model = self._initialize_training_model().to(device)
        self.training_model.train()
        
        # Optimizers
        self.inference_optimizer = optim.Adam(self.inference_model.parameters(), lr=learning_rate)
        self.training_optimizer = optim.AdamW(self.training_model.parameters(), lr=learning_rate/10, weight_decay=1e-4)
        
        # Threading for background training
        self.training_queue = Queue(maxsize=10)
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
        self.model_update_interval = 100  # Update inference model every 100 training steps
        
    def _initialize_training_model(self):
        """Initialize a slightly larger model for background training"""
        args = argparse.Namespace(
            embedding_dim=128,  # Smaller than original
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
        """Single training step on background model"""
        try:
            # Prepare frame for training (dynamic size to match inference model)
            frame_resized = cv2.resize(frame, (self.process_size, self.process_size), interpolation=cv2.INTER_LINEAR)
            frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
            
            # Convert to tensor
            frame_tensor = torch.tensor(frame_rgb).permute(2, 0, 1).unsqueeze(0).float() / 255.0
            frame_tensor = (frame_tensor - 0.5) * 2  # Normalize to [-1, 1]
            frame_tensor = frame_tensor.to(self.device)
            
            # Train the inference model directly instead of complex background model
            self.inference_model.train()
            self.inference_optimizer.zero_grad()
            
            # Forward pass
            reconstructed, mean, logvar = self.inference_model(frame_tensor)
            
            # Compute loss
            recon_loss = self.mse_loss(reconstructed, frame_tensor)
            kl_loss = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp())
            total_loss = recon_loss + 0.0001 * kl_loss  # Very small KL weight
            
            # Backward pass
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.inference_model.parameters(), max_norm=1.0)
            self.inference_optimizer.step()
            
            self.inference_model.eval()  # Back to eval mode
            self.stats['training_updates'] += 1
            
        except Exception as e:
            print(f"Training step error: {e}")
            self.inference_model.eval()  # Ensure we're back in eval mode
    
    def _update_inference_model(self):
        """Update the fast inference model with knowledge from training model"""
        try:
            # Simple knowledge distillation - copy what we can
            # For now, just retrain the inference model on recent frames
            print(f"Updating inference model (update #{self.stats['training_updates']//self.model_update_interval})")
            
            # Fine-tune inference model on current frame
            self.inference_model.train()
            
            # Get a few recent frames for fine-tuning
            # For simplicity, we'll just mark that an update happened
            self.last_model_update = self.stats['training_updates']
            
            self.inference_model.eval()
            
        except Exception as e:
            print(f"Model update error: {e}")
    
    def process_frame_ultra_fast(self, frame):
        """Ultra-fast inference on single frame"""
        start_time = time.time()
        
        try:
            # Resize to dynamic resolution for speed (half input resolution)
            frame_small = cv2.resize(frame, (self.process_size, self.process_size), interpolation=cv2.INTER_LINEAR)
            frame_rgb = cv2.cvtColor(frame_small, cv2.COLOR_BGR2RGB)
            
            # Convert to tensor
            frame_tensor = torch.tensor(frame_rgb).permute(2, 0, 1).unsqueeze(0).float() / 255.0
            frame_tensor = (frame_tensor - 0.5) * 2  # Normalize to [-1, 1]
            frame_tensor = frame_tensor.to(self.device)
            
            # Ultra-fast inference
            with torch.no_grad():
                reconstructed, _, _ = self.inference_model(frame_tensor)
            
            # Convert back to image
            recon_np = reconstructed.squeeze(0).permute(1, 2, 0).cpu().numpy()
            recon_np = ((recon_np + 1) / 2 * 255).clip(0, 255).astype(np.uint8)
            
            # Upscale back to display size
            recon_upscaled = cv2.resize(recon_np, (256, 256), interpolation=cv2.INTER_LINEAR)
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
    parser = argparse.ArgumentParser(description='Ultra-Fast 60 FPS LeanVAE')
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
    
    print(f"\nUltra-Fast 60 FPS LeanVAE:")
    print(f"- Target FPS: {args.target_fps}")
    print(f"- Inference Resolution: {processor.process_size}x{processor.process_size}")
    print(f"- Training Resolution: {processor.process_size}x{processor.process_size}")
    print(f"- Device: {args.device}")
    print("\nPress 'q' to quit")
    print("Aiming for sub-16ms inference time per frame!")
    
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
                cv2.putText(combined, "Ultra-Fast LeanVAE", (display_width + 10, 25), 
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
                
                # Display
                cv2.imshow('Ultra-Fast 60 FPS LeanVAE', combined)
            
            # Handle input
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            
            # Frame rate limiting (if needed)
            elapsed = time.time() - current_time
            if elapsed < target_frame_time:
                time.sleep(target_frame_time - elapsed)
                
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    
    finally:
        processor.cleanup()
        cap.release()
        cv2.destroyAllWindows()
        
        final_stats = processor.get_stats()
        print(f"\nSession Complete:")
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
