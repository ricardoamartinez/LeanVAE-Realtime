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

class OnlineLearningLeanVAE:
    def __init__(self, device='cpu', learning_rate=1e-4, target_fps=30, buffer_size=5):
        self.device = device
        self.target_fps = target_fps
        self.buffer_size = max(5, buffer_size)
        self.frame_interval = 1.0 / target_fps
        self.learning_rate = learning_rate
        
        # Initialize untrained model
        print("Initializing untrained LeanVAE model...")
        self.model = self._initialize_untrained_model()
        self.model = self.model.to(device)
        self.model.train()  # Keep in training mode
        print(f"Untrained model initialized on {device}")
        
        # Setup optimizer for online learning
        self.optimizer = optim.AdamW(self.model.parameters(), lr=learning_rate, weight_decay=1e-4)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, patience=100, factor=0.8)
        
        # Frame management
        self.frame_buffer = deque(maxlen=buffer_size)
        self.latest_reconstruction = None
        self.is_training = False
        self.frame_lock = threading.Lock()
        
        # Training statistics
        self.training_stats = {
            'total_updates': 0,
            'current_loss': 0.0,
            'avg_loss': 0.0,
            'learning_rate': learning_rate,
            'frames_trained': 0,
            'training_time': 0.0,
            'loss_history': deque(maxlen=1000),
            'reconstruction_quality': 0.0
        }
        
        # Performance statistics
        self.stats = {
            'frames_captured': 0,
            'frames_trained': 0,
            'frames_dropped_input': 0,
            'frames_dropped_training': 0,
            'last_training_time': 0,
            'avg_fps': 0,
            'training_fps': 0
        }
        
        # Timing
        self.last_capture_time = 0
        self.last_training_time = 0
        self.last_stats_time = time.time()
        self.training_interval = 1.0 / 8  # Train at most 8 FPS
        
        # Resolution for speed
        self.process_resolution = 128
        
        # Loss function
        self.mse_loss = nn.MSELoss()
        self.perceptual_weight = 0.1
        
    def _initialize_untrained_model(self):
        """Initialize an untrained LeanVAE model with random weights"""
        # Create args for LeanVAE initialization
        args = argparse.Namespace(
            embedding_dim=512,
            latent_dim=4,
            ista_iter_num=2,
            ista_layer_num=2,
            l_dim=128,
            h_dim=384,
            sep_num_layer=2,
            fusion_num_layer=4,
            use_tile_inference=False,
            chunksize_enc=9,
            chunksize_dec=5
        )
        
        # Create model with same architecture but random weights
        model = LeanVAE(args)
        
        # Initialize weights randomly
        def init_weights(m):
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
        
        model.apply(init_weights)
        return model
    
    def preprocess_frame(self, frame):
        """Preprocess frame for training"""
        # Resize to processing resolution
        frame = cv2.resize(frame, (self.process_resolution, self.process_resolution), 
                          interpolation=cv2.INTER_LINEAR)
        
        # Convert BGR to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return frame
    
    def pad_sequence_for_leanvae(self, frames):
        """Pad sequence to valid LeanVAE length (4n+1)"""
        num_frames = len(frames)
        target_length = ((num_frames - 1) // 4) * 4 + 1
        if target_length < num_frames:
            target_length += 4
        target_length = max(5, target_length)
        
        # Pad by repeating frames
        while len(frames) < target_length:
            frames.append(frames[-1])
            
        return frames[:target_length]
    
    def compute_training_loss(self, original, reconstructed):
        """Compute training loss with multiple components"""
        # MSE reconstruction loss
        mse_loss = self.mse_loss(reconstructed, original)
        
        # Temporal consistency loss (consecutive frame similarity)
        temporal_loss = 0.0
        if original.shape[2] > 1:  # More than 1 frame
            for t in range(1, original.shape[2]):
                temporal_loss += self.mse_loss(
                    reconstructed[:, :, t] - reconstructed[:, :, t-1],
                    original[:, :, t] - original[:, :, t-1]
                )
            temporal_loss /= (original.shape[2] - 1)
        
        # Total loss
        total_loss = mse_loss + 0.1 * temporal_loss
        
        return total_loss, {
            'mse': mse_loss.item(),
            'temporal': temporal_loss.item() if isinstance(temporal_loss, torch.Tensor) else temporal_loss,
            'total': total_loss.item()
        }
    
    def train_on_current_buffer(self):
        """Train model on current frame buffer"""
        if self.is_training or len(self.frame_buffer) < 3:
            return
            
        current_time = time.time()
        if current_time - self.last_training_time < self.training_interval:
            return
            
        self.is_training = True
        
        try:
            with self.frame_lock:
                if len(self.frame_buffer) < 3:
                    self.is_training = False
                    return
                    
                # Get frames and pad for LeanVAE
                frames = list(self.frame_buffer)
                frames = self.pad_sequence_for_leanvae(frames)
            
            start_time = time.time()
            
            # Convert to tensor
            video_np = np.array(frames)
            video_tensor = rearrange(torch.tensor(video_np), 't h w c -> c t h w').unsqueeze(0)
            video_tensor = video_tensor.float() / 255.0
            
            # Normalize to [-0.5, 0.5] for LeanVAE
            video_tensor = video_tensor - 0.5
            video_tensor = video_tensor.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            
            try:
                # Use forward method for training (supports gradients)
                _, reconstructed, _, _, posterior = self.model.forward(video_tensor)
                
                # Compute loss
                loss, loss_components = self.compute_training_loss(video_tensor, reconstructed)
                
                # Add KL divergence loss for VAE training
                kl_loss = posterior.kl()
                total_loss = loss + 0.001 * kl_loss.mean()  # Small KL weight
                
                # Backward pass
                total_loss.backward()
                
                # Gradient clipping for stability
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                # Update weights
                self.optimizer.step()
                
                # Update learning rate based on loss
                self.scheduler.step(loss)
                
                training_time = time.time() - start_time
                self.last_training_time = current_time
                
                # Update statistics
                self.training_stats['total_updates'] += 1
                self.training_stats['current_loss'] = loss.item()
                self.training_stats['loss_history'].append(loss.item())
                self.training_stats['learning_rate'] = self.optimizer.param_groups[0]['lr']
                self.training_stats['frames_trained'] += len(frames)
                self.training_stats['training_time'] = training_time
                
                # Calculate average loss
                if len(self.training_stats['loss_history']) > 0:
                    self.training_stats['avg_loss'] = np.mean(list(self.training_stats['loss_history']))
                
                # Update performance stats
                self.stats['frames_trained'] += 1
                self.stats['last_training_time'] = training_time
                
                # Prepare output for display (latest frame)
                display_reconstructed = reconstructed.detach().squeeze(0).permute(1, 2, 3, 0)  # (T, H, W, C)
                display_reconstructed = torch.clamp(display_reconstructed + 0.5, 0, 1) * 255.0
                display_reconstructed = display_reconstructed.cpu().numpy().astype(np.uint8)
                
                # Take the most recent frame
                latest_frame = display_reconstructed[-1]
                
                # Upscale to display resolution
                latest_frame = cv2.resize(latest_frame, (256, 256), interpolation=cv2.INTER_LINEAR)
                latest_frame = cv2.cvtColor(latest_frame, cv2.COLOR_RGB2BGR)
                
                self.latest_reconstruction = latest_frame
                
                # Calculate reconstruction quality (PSNR)
                mse = np.mean((video_np[-1].astype(float) - display_reconstructed[-1].astype(float))**2)
                if mse > 0:
                    psnr = 20 * np.log10(255.0 / np.sqrt(mse))
                    self.training_stats['reconstruction_quality'] = psnr
                
            except Exception as e:
                print(f"Training forward/backward error: {e}")
                
        except Exception as e:
            print(f"Training error: {e}")
            
        finally:
            self.is_training = False
    
    def add_frame(self, frame):
        """Add frame for training with aggressive dropping"""
        current_time = time.time()
        
        # Skip if training or too frequent
        if self.is_training:
            self.stats['frames_dropped_training'] += 1
            return False
            
        if current_time - self.last_capture_time < self.frame_interval:
            self.stats['frames_dropped_input'] += 1
            return False
        
        self.last_capture_time = current_time
        
        # Preprocess frame
        processed_frame = self.preprocess_frame(frame)
        
        # Add to buffer
        with self.frame_lock:
            self.frame_buffer.append(processed_frame)
        
        self.stats['frames_captured'] += 1
        
        # Trigger training if enough frames
        if len(self.frame_buffer) >= 3:
            self.train_on_current_buffer()
        
        return True
    
    def get_latest_reconstruction(self):
        """Get latest reconstruction result"""
        return self.latest_reconstruction
    
    def get_stats_text(self):
        """Get statistics for display"""
        current_time = time.time()
        time_diff = current_time - self.last_stats_time
        
        if time_diff > 1.0:
            if time_diff > 0:
                self.stats['avg_fps'] = self.stats['frames_captured'] / time_diff
                self.stats['training_fps'] = self.stats['frames_trained'] / time_diff
                
            self.last_stats_time = current_time
            # Reset counters
            self.stats['frames_captured'] = 0
            self.stats['frames_trained'] = 0
        
        stats_text = [
            f"Online Learning Mode",
            f"Capture FPS: {self.stats['avg_fps']:.1f}",
            f"Training FPS: {self.stats['training_fps']:.1f}",
            f"Updates: {self.training_stats['total_updates']}",
            f"Current Loss: {self.training_stats['current_loss']:.4f}",
            f"Avg Loss: {self.training_stats['avg_loss']:.4f}",
            f"LR: {self.training_stats['learning_rate']:.2e}",
            f"PSNR: {self.training_stats['reconstruction_quality']:.1f}dB",
            f"Train Time: {self.stats['last_training_time']*1000:.0f}ms",
            f"Buffer: {len(self.frame_buffer)}/{self.buffer_size}",
            f"Dropped: {self.stats['frames_dropped_input'] + self.stats['frames_dropped_training']}"
        ]
        
        return stats_text

def main():
    parser = argparse.ArgumentParser(description='Real-time Online Learning LeanVAE')
    parser.add_argument('--device', type=str, default='cpu', help='Device (cpu/cuda)')
    parser.add_argument('--camera', type=int, default=0, help='Camera index')
    parser.add_argument('--fps', type=int, default=30, help='Capture FPS')
    parser.add_argument('--train-fps', type=int, default=8, help='Training FPS')
    parser.add_argument('--resolution', type=int, default=128, help='Processing resolution')
    parser.add_argument('--buffer-size', type=int, default=5, help='Frame buffer size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    
    args = parser.parse_args()
    
    # Initialize camera
    print(f"Initializing camera {args.camera}...")
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print("Error: Could not open camera")
        return
    
    # Camera settings for minimal latency
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 60)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    
    print(f"Camera: {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}")
    
    # Initialize online learning processor
    processor = OnlineLearningLeanVAE(
        device=args.device,
        learning_rate=args.lr,
        target_fps=args.fps,
        buffer_size=args.buffer_size
    )
    processor.process_resolution = args.resolution
    processor.training_interval = 1.0 / args.train_fps
    
    print(f"\nOnline Learning LeanVAE:")
    print(f"- Learning Rate: {args.lr}")
    print(f"- Capture FPS: {args.fps}")
    print(f"- Training FPS: {args.train_fps}")
    print(f"- Resolution: {args.resolution}x{args.resolution}")
    print(f"- Buffer Size: {args.buffer_size}")
    print("\nPress 'q' to quit, 's' to save model, 'r' to reset model")
    print("Watch the model learn in real-time!")
    
    frame_count = 0
    save_count = 0
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to read from camera")
                break
            
            frame_count += 1
            
            # Add frame for training
            processor.add_frame(frame)
            
            # Get latest reconstruction
            reconstruction = processor.get_latest_reconstruction()
            
            # Prepare display
            display_frame = frame.copy()
            display_height = 300
            aspect_ratio = frame.shape[1] / frame.shape[0]
            display_width = int(display_height * aspect_ratio)
            display_frame = cv2.resize(display_frame, (display_width, display_height))
            
            # Create display
            if reconstruction is not None:
                reconstruction_display = cv2.resize(reconstruction, (display_width, display_height))
                combined = np.hstack((display_frame, reconstruction_display))
                
                # Labels
                cv2.putText(combined, "Live Camera", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(combined, "Learning LeanVAE", (display_width + 10, 25), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                
                # Show learning progress with color coding
                loss = processor.training_stats['current_loss']
                if loss > 0.1:
                    color = (0, 0, 255)  # Red for high loss
                elif loss > 0.05:
                    color = (0, 165, 255)  # Orange for medium loss
                else:
                    color = (0, 255, 0)  # Green for low loss
                    
                cv2.putText(combined, f"Learning... Loss: {loss:.4f}", 
                           (display_width + 10, display_height - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            else:
                combined = display_frame
                cv2.putText(combined, "Initializing Online Learning...", (10, 25), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # Add compact statistics
            stats = processor.get_stats_text()
            for i, stat in enumerate(stats):
                cv2.putText(combined, stat, (10, 50 + i * 18), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)
            
            # Display
            cv2.imshow('Online Learning LeanVAE', combined)
            
            # Handle input
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                # Save current model state
                save_count += 1
                model_path = f'online_learned_model_{save_count}_{datetime.now().strftime("%H%M%S")}.pth'
                torch.save({
                    'model_state_dict': processor.model.state_dict(),
                    'optimizer_state_dict': processor.optimizer.state_dict(),
                    'training_stats': processor.training_stats,
                    'updates': processor.training_stats['total_updates'],
                    'avg_loss': processor.training_stats['avg_loss']
                }, model_path)
                print(f"Saved model to {model_path}")
            elif key == ord('r'):
                # Reset model to random weights
                processor.model = processor._initialize_untrained_model().to(processor.device)
                processor.optimizer = optim.AdamW(processor.model.parameters(), 
                                                lr=args.lr, weight_decay=1e-4)
                processor.training_stats['total_updates'] = 0
                processor.training_stats['loss_history'].clear()
                print("Model reset to random weights")
                
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    
    finally:
        cap.release()
        cv2.destroyAllWindows()
        
        print(f"\nOnline Learning Session Complete:")
        print(f"- Total frames processed: {frame_count}")
        print(f"- Total training updates: {processor.training_stats['total_updates']}")
        print(f"- Final average loss: {processor.training_stats['avg_loss']:.4f}")
        print(f"- Final PSNR: {processor.training_stats['reconstruction_quality']:.1f}dB")

if __name__ == "__main__":
    main()
