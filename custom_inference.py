import os
import argparse
import torch
from LeanVAE import LeanVAE
import cv2
import torch
import os
from einops import rearrange
from torchvision.io import write_video
from torchvision import transforms
import tqdm
import numpy as np
import torch.nn.functional as F

def load_video_opencv(video_path, num_frames=17):
    """Load video using OpenCV instead of decord"""
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    frames = []
    frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    
    for i, frame_idx in enumerate(frame_indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if ret:
            # Convert BGR to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
        else:
            print(f"Warning: Could not read frame {frame_idx}")
    
    cap.release()
    
    if len(frames) < num_frames:
        # Pad with last frame if necessary
        while len(frames) < num_frames:
            frames.append(frames[-1])
    
    video = np.array(frames)
    return video, fps

def save_frames_individually(frames, save_path, video_name, fps):
    """Save each frame as a separate image to see individual frames"""
    base_name = os.path.splitext(video_name)[0]
    frame_dir = os.path.join(save_path, f"{base_name}_frames")
    os.makedirs(frame_dir, exist_ok=True)
    
    for i, frame in enumerate(frames):
        frame_path = os.path.join(frame_dir, f"frame_{i:03d}.png")
        # Convert tensor to numpy if needed
        if isinstance(frame, torch.Tensor):
            frame_np = frame.cpu().numpy()
        else:
            frame_np = frame
        
        # Ensure frame is in the right format (H, W, C) and uint8
        if frame_np.dtype != np.uint8:
            frame_np = frame_np.astype(np.uint8)
        
        # Convert RGB to BGR for OpenCV
        frame_bgr = cv2.cvtColor(frame_np, cv2.COLOR_RGB2BGR)
        cv2.imwrite(frame_path, frame_bgr)
    
    print(f"Saved {len(frames)} individual frames to {frame_dir}")

def save_video_opencv(frames, output_path, fps):
    """Save video using OpenCV"""
    if isinstance(frames, torch.Tensor):
        frames_np = frames.cpu().numpy()
    else:
        frames_np = frames
    
    if len(frames_np.shape) == 4:  # (T, H, W, C)
        height, width = frames_np.shape[1], frames_np.shape[2]
    else:
        height, width = frames_np.shape[0], frames_np.shape[1]
        frames_np = frames_np[None, ...]  # Add batch dimension
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    for frame in frames_np:
        if frame.dtype != np.uint8:
            frame = frame.astype(np.uint8)
        # Convert RGB to BGR for OpenCV
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        out.write(frame_bgr)
    
    out.release()
    print(f"Saved reconstructed video to {output_path}")

def main(args, model, video_path, save_path, video_name):
    use_half = args.fp16
    device = args.device
    num_frames = args.sequence_length

    if args.tile_inference:
        model.set_tile_inference(True)
        model.chunksize_enc = args.chunksize_enc if args.chunksize_enc else 5
        model.chunksize_dec = args.chunksize_dec if args.chunksize_dec else 5

    # Load video using OpenCV
    video_np, fps = load_video_opencv(video_path, num_frames)
    print(f"Loaded video with shape: {video_np.shape}, fps: {fps}")

    video = rearrange(torch.tensor(video_np), 't h w c -> c t h w').unsqueeze(0)
    video = video.half() if use_half else video.float()
    
    regular_size = 2  # input range is [-0.5, 0.5] if regular_size = 2, [-1, 1] if regular_size = 1
    
    with torch.no_grad():
        video = video / (127.5 * regular_size) - (1.0 / regular_size)
        video = video.to(device)
        
        print(f"Input video tensor shape: {video.shape}")
        print(f"Input video range: [{video.min():.3f}, {video.max():.3f}]")
        
        x, x_rec = model.inference(video)
        
        print(f"Reconstructed video tensor shape: {x_rec.shape}")
        print(f"Reconstructed video range: [{x_rec.min():.3f}, {x_rec.max():.3f}]")
        
        # Process reconstructed video
        x_rec = x_rec.squeeze(0).permute(1, 2, 3, 0)  # (T, H, W, C)
        x_rec = (torch.clamp(x_rec, -(1.0 / regular_size), (1.0 / regular_size)) + (1.0 / regular_size)) * (127.5 * regular_size)
        x_rec = x_rec.to('cpu', dtype=torch.uint8)
        
        # Save individual frames to see each frame
        save_frames_individually(x_rec, save_path, video_name, fps)
        
        # Also save original frames for comparison
        original_frames = torch.tensor(video_np, dtype=torch.uint8)
        save_frames_individually(original_frames, save_path, f"original_{video_name}", fps)
        
        # Save reconstructed video using OpenCV
        save_video_opencv(x_rec, os.path.join(save_path, video_name), fps)
        
        print(f"Successfully processed video: {video_name}")
        print(f"Original frames: {len(original_frames)}")
        print(f"Reconstructed frames: {len(x_rec)}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt_path', type=str, default='')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--input_video', type=str, default='./input_videos')
    parser.add_argument('--reconstruct_video', type=str, default='./reconstruct_videos')
    parser.add_argument('--sequence_length', type=int, default=17)
    parser.add_argument('--fp16', action='store_true')
    parser.add_argument('--tile_inference', action='store_true')
    parser.add_argument('--chunksize_enc', type=int, default=None)
    parser.add_argument('--chunksize_dec', type=int, default=None)

    args = parser.parse_args()
    
    # Load model
    print(f"Loading model from: {args.ckpt_path}")
    vae = LeanVAE.load_from_checkpoint(args.ckpt_path, strict=False)
    
    os.makedirs(args.reconstruct_video, exist_ok=True)
    vae = vae.half().to(args.device) if args.fp16 else vae.to(args.device)
    
    print(f"Model loaded and moved to {args.device}")
    
    # Process all videos in input directory
    video_files = [f for f in os.listdir(args.input_video) if f.endswith(('.mp4', '.avi', '.mov', '.mkv'))]
    print(f"Found {len(video_files)} video files: {video_files}")
    
    for vid_name in tqdm.tqdm(video_files):
        video_path = os.path.join(args.input_video, vid_name)
        print(f"\nProcessing: {vid_name}")
        try:
            main(args, vae, video_path, args.reconstruct_video, vid_name)
        except Exception as e:
            print(f"Error processing {vid_name}: {e}")
            import traceback
            traceback.print_exc()
