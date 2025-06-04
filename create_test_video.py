import cv2
import numpy as np
import os

def create_test_video():
    # Video parameters
    width, height = 256, 256
    fps = 10
    duration = 2  # seconds
    total_frames = fps * duration
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('input_videos/test_video.mp4', fourcc, fps, (width, height))
    
    # Create frames with moving circle
    for frame_num in range(total_frames):
        # Create a black frame
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Add some background gradient
        for y in range(height):
            for x in range(width):
                frame[y, x] = [
                    int(255 * (x / width)),  # Red gradient
                    int(255 * (y / height)),  # Green gradient
                    int(255 * ((x + y) / (width + height)))  # Blue gradient
                ]
        
        # Add moving circle
        center_x = int(width/4 + (width/2) * (frame_num / total_frames))
        center_y = int(height/4 + (height/2) * (frame_num / total_frames))
        radius = 20 + int(10 * np.sin(frame_num * 0.3))
        
        cv2.circle(frame, (center_x, center_y), radius, (255, 255, 255), -1)
        
        # Add frame number text
        cv2.putText(frame, f'Frame {frame_num+1}/{total_frames}', 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        
        out.write(frame)
    
    out.release()
    print(f"Created test video with {total_frames} frames at {fps} FPS")

if __name__ == "__main__":
    create_test_video()
