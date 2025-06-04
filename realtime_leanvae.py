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

class RealtimeLeanVAE:
    def __init__(self, model_path, device='cpu', target_fps=10, buffer_size=17):
        self.device = device
        self.target_fps = target_fps
        self.buffer_size = buffer_size  # LeanVAE needs 17 frames (4n+1)
        self.frame_interval = 1.0 / target_fps
        
        # Load LeanVAE model
        print(f"Loading LeanVAE model from {model_path}...")
        self.model = LeanVAE.load_from_checkpoint(model_path, strict=False)
        self.model = self.model.to(device)
        self.model.eval()
        print(f"Model loaded on {device}")
        
        # Frame buffers and queues
        self.frame_buffer = deque(maxlen=buffer_size)
        self.processed_frame_queue = Queue(maxsize=5)
        self.latest_reconstruction = None
        
        # Threading control
        self.processing_thread = None
        self.is_running = False
        self.frame_lock = threading.Lock()
        
        # Statistics
        self.stats = {
            'frames_captured': 0,
            'frames_processed': 0,
            'frames_dropped': 0,
            'last_process_time': 0,
            'avg_fps': 0
        }
        
        # Timing
        self.last_capture_time = 0
        self.last_stats_time = time.time()
        
    def preprocess_frame(self, frame):
        """Preprocess frame for LeanVAE"""
        # Resize to 256x256 if needed
        if frame.shape[:2] != (256, 256):
            frame = cv2.resize(frame, (256, 256))
        
        # Convert BGR to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return frame
    
    def process_frame_batch(self):
        """Process a batch of frames with LeanVAE in separate thread"""
        while self.is_running:
            try:
                # Wait until we have enough frames
                with self.frame_lock:
                    if len(self.frame_buffer) < self.buffer_size:
                        time.sleep(0.01)
                        continue
                    
                    # Get current frame buffer
                    frames = list(self.frame_buffer)
                
                start_time = time.time()
                
                # Convert to tensor
                video_np = np.array(frames)
                video_tensor = rearrange(torch.tensor(video_np), 't h w c -> c t h w').unsqueeze(0)
                video_tensor = video_tensor.float() / 255.0  # Normalize to [0, 1]
                
                # LeanVAE expects [-0.5, 0.5] range
                video_tensor = video_tensor - 0.5
                video_tensor = video_tensor.to(self.device)
                
                # Process with LeanVAE
                with torch.no_grad():
                    _, reconstructed = self.model.inference(video_tensor)
                
                # Post-process
                reconstructed = reconstructed.squeeze(0).permute(1, 2, 3, 0)  # (T, H, W, C)
                reconstructed = torch.clamp(reconstructed + 0.5, 0, 1) * 255.0
                reconstructed = reconstructed.cpu().numpy().astype(np.uint8)
                
                # Get the latest frame (middle frame for temporal consistency)
                latest_frame = reconstructed[self.buffer_size // 2]
                
                # Convert back to BGR for OpenCV display
                latest_frame = cv2.cvtColor(latest_frame, cv2.COLOR_RGB2BGR)
                
                process_time = time.time() - start_time
                
                # Update statistics
                self.stats['frames_processed'] += 1
                self.stats['last_process_time'] = process_time
                
                # Store result (non-blocking)
                try:
                    self.processed_frame_queue.put(latest_frame, block=False)
                    self.latest_reconstruction = latest_frame
                except:
                    # Queue full, drop this result
                    self.stats['frames_dropped'] += 1
                
                # Adaptive delay based on processing time
                if process_time < self.frame_interval:
                    time.sleep(self.frame_interval - process_time)
                    
            except Exception as e:
                print(f"Processing error: {e}")
                time.sleep(0.1)
    
    def add_frame(self, frame):
        """Add frame to buffer with frame dropping if needed"""
        current_time = time.time()
        
        # Frame rate limiting
        if current_time - self.last_capture_time < self.frame_interval:
            self.stats['frames_dropped'] += 1
            return False
        
        self.last_capture_time = current_time
        
        # Preprocess frame
        processed_frame = self.preprocess_frame(frame)
        
        # Add to buffer
        with self.frame_lock:
            self.frame_buffer.append(processed_frame)
        
        self.stats['frames_captured'] += 1
        return True
    
    def get_latest_reconstruction(self):
        """Get the latest reconstructed frame"""
        try:
            # Try to get newer result
            while not self.processed_frame_queue.empty():
                self.latest_reconstruction = self.processed_frame_queue.get_nowait()
            return self.latest_reconstruction
        except Empty:
            return self.latest_reconstruction
    
    def start_processing(self):
        """Start the processing thread"""
        self.is_running = True
        self.processing_thread = threading.Thread(target=self.process_frame_batch)
        self.processing_thread.daemon = True
        self.processing_thread.start()
        print("Processing thread started")
    
    def stop_processing(self):
        """Stop the processing thread"""
        self.is_running = False
        if self.processing_thread:
            self.processing_thread.join(timeout=2.0)
        print("Processing thread stopped")
    
    def get_stats_text(self):
        """Get statistics text for display"""
        current_time = time.time()
        time_diff = current_time - self.last_stats_time
        
        if time_diff > 1.0:  # Update stats every second
            if self.stats['frames_captured'] > 0:
                self.stats['avg_fps'] = self.stats['frames_captured'] / time_diff
            self.last_stats_time = current_time
            # Reset counters for next interval
            self.stats['frames_captured'] = 0
            self.stats['frames_processed'] = 0
            
        return [
            f"FPS: {self.stats['avg_fps']:.1f}",
            f"Buffer: {len(self.frame_buffer)}/{self.buffer_size}",
            f"Process Time: {self.stats['last_process_time']*1000:.1f}ms",
            f"Dropped: {self.stats['frames_dropped']}"
        ]

def main():
    parser = argparse.ArgumentParser(description='Real-time LeanVAE Camera Processing')
    parser.add_argument('--model', type=str, default='./models/LeanVAE-4ch.ckpt', 
                       help='Path to LeanVAE model checkpoint')
    parser.add_argument('--device', type=str, default='cpu', 
                       help='Device to run model on (cpu/cuda)')
    parser.add_argument('--camera', type=int, default=0, 
                       help='Camera index (usually 0 for default camera)')
    parser.add_argument('--fps', type=int, default=10, 
                       help='Target FPS for processing')
    parser.add_argument('--width', type=int, default=640, 
                       help='Camera capture width')
    parser.add_argument('--height', type=int, default=480, 
                       help='Camera capture height')
    
    args = parser.parse_args()
    
    # Initialize camera
    print(f"Initializing camera {args.camera}...")
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print("Error: Could not open camera")
        return
    
    # Set camera properties
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    cap.set(cv2.CAP_PROP_FPS, 30)  # High capture FPS, we'll downsample
    
    print(f"Camera initialized: {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}")
    
    # Initialize LeanVAE processor
    processor = RealtimeLeanVAE(args.model, args.device, args.fps)
    processor.start_processing()
    
    print("\nStarting real-time processing...")
    print("Press 'q' to quit, 's' to save current frame")
    
    frame_count = 0
    save_count = 0
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to read from camera")
                break
            
            frame_count += 1
            
            # Add frame to processor
            processor.add_frame(frame)
            
            # Get latest reconstruction
            reconstruction = processor.get_latest_reconstruction()
            
            # Prepare display
            display_frame = frame.copy()
            
            # Resize for display
            display_height = 400
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
                cv2.putText(combined, "Original", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(combined, "LeanVAE Reconstructed", (display_width + 10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            else:
                combined = display_frame
                cv2.putText(combined, "Initializing LeanVAE...", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            # Add statistics
            stats = processor.get_stats_text()
            for i, stat in enumerate(stats):
                cv2.putText(combined, stat, (10, 70 + i * 25), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            
            # Display
            cv2.imshow('Real-time LeanVAE Processing', combined)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s') and reconstruction is not None:
                # Save current frame pair
                save_count += 1
                cv2.imwrite(f'realtime_original_{save_count}.png', display_frame)
                cv2.imwrite(f'realtime_reconstruction_{save_count}.png', reconstruction_display)
                print(f"Saved frame pair {save_count}")
                
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    
    finally:
        # Cleanup
        processor.stop_processing()
        cap.release()
        cv2.destroyAllWindows()
        print(f"Processed {frame_count} frames total")

if __name__ == "__main__":
    main()
