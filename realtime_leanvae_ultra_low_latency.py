import cv2
import torch
import numpy as np
import threading
import time
from collections import deque
from queue import Queue, Empty
import argparse
from LeanVAE import LeanVAE
from einops import rearrange

class UltraLowLatencyLeanVAE:
    def __init__(self, model_path, device='cpu', target_fps=30, buffer_size=5):
        self.device = device
        self.target_fps = target_fps
        self.buffer_size = max(5, buffer_size)  # Minimum 5 frames, much smaller than original 17
        self.frame_interval = 1.0 / target_fps
        
        # Load LeanVAE model
        print(f"Loading LeanVAE model from {model_path}...")
        self.model = LeanVAE.load_from_checkpoint(model_path, strict=False)
        self.model = self.model.to(device)
        self.model.eval()
        print(f"Model loaded on {device}")
        
        # Ultra-aggressive frame management
        self.frame_buffer = deque(maxlen=buffer_size)
        self.latest_reconstruction = None
        self.is_processing = False
        self.frame_lock = threading.Lock()
        
        # Statistics
        self.stats = {
            'frames_captured': 0,
            'frames_processed': 0,
            'frames_dropped_input': 0,
            'frames_dropped_processing': 0,
            'last_process_time': 0,
            'avg_fps': 0,
            'processing_fps': 0
        }
        
        # Timing and control
        self.last_capture_time = 0
        self.last_process_time = 0
        self.last_stats_time = time.time()
        self.processing_interval = 1.0 / 10  # Process at most 10 FPS to reduce latency
        
        # Resolution scaling for speed
        self.process_resolution = 128  # Much smaller resolution for speed
        
    def preprocess_frame(self, frame):
        """Aggressive preprocessing for ultra-low latency"""
        # Resize to very small resolution for speed
        frame = cv2.resize(frame, (self.process_resolution, self.process_resolution), 
                          interpolation=cv2.INTER_LINEAR)
        
        # Convert BGR to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return frame
    
    def pad_sequence_for_leanvae(self, frames):
        """Pad a shorter sequence to work with LeanVAE (needs 4n+1 frames)"""
        num_frames = len(frames)
        
        # Find the nearest valid length (4n+1) that's >= num_frames
        target_length = ((num_frames - 1) // 4) * 4 + 1
        if target_length < num_frames:
            target_length += 4
        
        # Ensure minimum of 5 frames
        target_length = max(5, target_length)
        
        # Pad by repeating the last frame
        while len(frames) < target_length:
            frames.append(frames[-1])
            
        return frames[:target_length]
    
    def process_current_buffer(self):
        """Process current buffer immediately if not already processing"""
        if self.is_processing or len(self.frame_buffer) < 3:
            return
            
        current_time = time.time()
        if current_time - self.last_process_time < self.processing_interval:
            return  # Skip this processing cycle
            
        self.is_processing = True
        
        try:
            with self.frame_lock:
                if len(self.frame_buffer) < 3:
                    self.is_processing = False
                    return
                    
                # Get current frames and pad for LeanVAE
                frames = list(self.frame_buffer)
                frames = self.pad_sequence_for_leanvae(frames)
            
            start_time = time.time()
            
            # Convert to tensor
            video_np = np.array(frames)
            video_tensor = rearrange(torch.tensor(video_np), 't h w c -> c t h w').unsqueeze(0)
            video_tensor = video_tensor.float() / 255.0
            
            # LeanVAE expects [-0.5, 0.5] range
            video_tensor = video_tensor - 0.5
            video_tensor = video_tensor.to(self.device)
            
            # Process with LeanVAE
            with torch.no_grad():
                _, reconstructed = self.model.inference(video_tensor)
            
            # Post-process - take the last frame for lowest latency
            reconstructed = reconstructed.squeeze(0).permute(1, 2, 3, 0)  # (T, H, W, C)
            reconstructed = torch.clamp(reconstructed + 0.5, 0, 1) * 255.0
            reconstructed = reconstructed.cpu().numpy().astype(np.uint8)
            
            # Take the last processed frame (most recent)
            latest_frame = reconstructed[-1]
            
            # Upscale back to display resolution
            latest_frame = cv2.resize(latest_frame, (256, 256), interpolation=cv2.INTER_LINEAR)
            
            # Convert back to BGR for OpenCV display
            latest_frame = cv2.cvtColor(latest_frame, cv2.COLOR_RGB2BGR)
            
            process_time = time.time() - start_time
            self.last_process_time = current_time
            
            # Update statistics
            self.stats['frames_processed'] += 1
            self.stats['last_process_time'] = process_time
            
            # Store result
            self.latest_reconstruction = latest_frame
            
        except Exception as e:
            print(f"Processing error: {e}")
            
        finally:
            self.is_processing = False
    
    def add_frame(self, frame):
        """Ultra-aggressive frame dropping for lowest latency"""
        current_time = time.time()
        
        # Skip frames if we're processing or if adding too quickly
        if self.is_processing:
            self.stats['frames_dropped_processing'] += 1
            return False
            
        # Ultra-aggressive frame rate limiting
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
        
        # Immediately try to process if we have enough frames
        if len(self.frame_buffer) >= 3:
            self.process_current_buffer()
        
        return True
    
    def get_latest_reconstruction(self):
        """Get the latest reconstructed frame"""
        return self.latest_reconstruction
    
    def get_stats_text(self):
        """Get statistics text for display"""
        current_time = time.time()
        time_diff = current_time - self.last_stats_time
        
        if time_diff > 1.0:  # Update stats every second
            if time_diff > 0:
                self.stats['avg_fps'] = self.stats['frames_captured'] / time_diff
                self.stats['processing_fps'] = self.stats['frames_processed'] / time_diff
                
            self.last_stats_time = current_time
            
            # Reset counters for next interval
            captured = self.stats['frames_captured']
            processed = self.stats['frames_processed']
            self.stats['frames_captured'] = 0
            self.stats['frames_processed'] = 0
            
        total_dropped = self.stats['frames_dropped_input'] + self.stats['frames_dropped_processing']
        
        return [
            f"Capture FPS: {self.stats['avg_fps']:.1f}",
            f"Process FPS: {self.stats['processing_fps']:.1f}",
            f"Buffer: {len(self.frame_buffer)}/{self.buffer_size}",
            f"Process Time: {self.stats['last_process_time']*1000:.0f}ms",
            f"Dropped Input: {self.stats['frames_dropped_input']}",
            f"Dropped Process: {self.stats['frames_dropped_processing']}",
            f"Resolution: {self.process_resolution}x{self.process_resolution}"
        ]

def main():
    parser = argparse.ArgumentParser(description='Ultra-Low Latency LeanVAE Camera Processing')
    parser.add_argument('--model', type=str, default='./models/LeanVAE-4ch.ckpt', 
                       help='Path to LeanVAE model checkpoint')
    parser.add_argument('--device', type=str, default='cpu', 
                       help='Device to run model on (cpu/cuda)')
    parser.add_argument('--camera', type=int, default=0, 
                       help='Camera index (usually 0 for default camera)')
    parser.add_argument('--fps', type=int, default=30, 
                       help='Target capture FPS (higher = more responsive)')
    parser.add_argument('--process-fps', type=int, default=5, 
                       help='Target processing FPS (lower = less latency)')
    parser.add_argument('--resolution', type=int, default=128, 
                       help='Processing resolution (lower = faster)')
    parser.add_argument('--buffer-size', type=int, default=5, 
                       help='Frame buffer size (smaller = lower latency)')
    
    args = parser.parse_args()
    
    # Initialize camera with minimal buffering
    print(f"Initializing camera {args.camera}...")
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print("Error: Could not open camera")
        return
    
    # Set camera for lowest latency
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 60)  # High capture rate
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimal buffer
    
    print(f"Camera initialized: {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}")
    
    # Initialize ultra-low latency processor
    processor = UltraLowLatencyLeanVAE(
        args.model, 
        args.device, 
        target_fps=args.fps,
        buffer_size=args.buffer_size
    )
    processor.process_resolution = args.resolution
    processor.processing_interval = 1.0 / args.process_fps
    
    print(f"\nUltra-Low Latency Mode:")
    print(f"- Capture FPS: {args.fps}")
    print(f"- Process FPS: {args.process_fps}")
    print(f"- Process Resolution: {args.resolution}x{args.resolution}")
    print(f"- Buffer Size: {args.buffer_size}")
    print("\nPress 'q' to quit, 's' to save current frame")
    print("Controls: '+'/'-' to adjust process FPS, 'r' to change resolution")
    
    frame_count = 0
    save_count = 0
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to read from camera")
                break
            
            frame_count += 1
            
            # Add frame to processor (with aggressive dropping)
            processor.add_frame(frame)
            
            # Get latest reconstruction
            reconstruction = processor.get_latest_reconstruction()
            
            # Prepare display
            display_frame = frame.copy()
            
            # Resize for display
            display_height = 300
            aspect_ratio = frame.shape[1] / frame.shape[0]
            display_width = int(display_height * aspect_ratio)
            display_frame = cv2.resize(display_frame, (display_width, display_height))
            
            # Create side-by-side display
            if reconstruction is not None:
                # Resize reconstruction to match display size
                reconstruction_display = cv2.resize(reconstruction, (display_width, display_height))
                
                # Combine frames side by side
                combined = np.hstack((display_frame, reconstruction_display))
                
                # Add labels
                cv2.putText(combined, "Live Camera", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(combined, "LeanVAE Ultra-Low Latency", (display_width + 10, 25), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            else:
                combined = display_frame
                cv2.putText(combined, "Warming up...", (10, 25), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # Add statistics (compact)
            stats = processor.get_stats_text()
            for i, stat in enumerate(stats):
                cv2.putText(combined, stat, (10, 50 + i * 20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            
            # Display
            cv2.imshow('Ultra-Low Latency LeanVAE', combined)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s') and reconstruction is not None:
                # Save current frame pair
                save_count += 1
                cv2.imwrite(f'ultra_low_latency_original_{save_count}.png', display_frame)
                cv2.imwrite(f'ultra_low_latency_reconstruction_{save_count}.png', reconstruction_display)
                print(f"Saved ultra-low latency frame pair {save_count}")
            elif key == ord('+'):
                processor.processing_interval = max(0.05, processor.processing_interval - 0.02)
                print(f"Increased processing speed: {1/processor.processing_interval:.1f} FPS")
            elif key == ord('-'):
                processor.processing_interval = min(0.5, processor.processing_interval + 0.02)
                print(f"Decreased processing speed: {1/processor.processing_interval:.1f} FPS")
            elif key == ord('r'):
                # Cycle through resolutions
                resolutions = [64, 96, 128, 160, 192]
                current_idx = resolutions.index(processor.process_resolution) if processor.process_resolution in resolutions else 2
                processor.process_resolution = resolutions[(current_idx + 1) % len(resolutions)]
                print(f"Changed resolution to: {processor.process_resolution}x{processor.process_resolution}")
                
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    
    finally:
        # Cleanup
        cap.release()
        cv2.destroyAllWindows()
        total_dropped = processor.stats['frames_dropped_input'] + processor.stats['frames_dropped_processing']
        print(f"Processed {frame_count} frames total")
        print(f"Dropped {total_dropped} frames for ultra-low latency")

if __name__ == "__main__":
    main()
