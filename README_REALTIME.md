# Real-time LeanVAE Video Processing

This repository contains real-time implementations of LeanVAE for live camera processing with ultra-low latency.

## Features

- **Real-time video processing** with LeanVAE pretrained models
- **Ultra-low latency mode** with aggressive frame dropping
- **Live camera feed processing** with side-by-side comparison
- **Adaptive resolution scaling** for performance optimization
- **Frame-by-frame reconstruction** visualization

## Files

### Core Scripts
- `realtime_leanvae.py` - Standard real-time processing with threading
- `realtime_leanvae_ultra_low_latency.py` - Ultra-low latency version with aggressive frame dropping
- `custom_inference.py` - Batch video processing for testing
- `create_test_video.py` - Generate test videos for evaluation
- `view_results.py` - Visualize and analyze results

### Original LeanVAE
- `leanvae_inference.py` - Original inference script
- `leanvAE_train.py` - Training script
- `LeanVAE/` - Core LeanVAE implementation
- `evaluation/` - Evaluation metrics and tools

## Quick Start

### 1. Install Dependencies
```bash
pip install torch torchvision pytorch-lightning opencv-python numpy einops matplotlib tqdm
```

### 2. Download Pretrained Model
The LeanVAE-4ch.ckpt model should be placed in the `models/` directory.

### 3. Run Real-time Processing
```bash
# Standard real-time mode
python realtime_leanvae.py --device cpu --fps 10

# Ultra-low latency mode
python realtime_leanvae_ultra_low_latency.py --device cpu --process-fps 8 --resolution 96
```

## Usage Examples

### Ultra-Low Latency Processing
```bash
python realtime_leanvae_ultra_low_latency.py \
    --device cpu \
    --process-fps 8 \
    --resolution 96 \
    --buffer-size 5
```

**Parameters:**
- `--process-fps`: Lower values = less latency (5-10 recommended)
- `--resolution`: Processing resolution (64, 96, 128, 160, 192)
- `--buffer-size`: Frame buffer size (5 minimum for LeanVAE)
- `--device`: cpu or cuda

### Interactive Controls
While running:
- `q` - Quit
- `s` - Save current frame pair
- `+/-` - Adjust processing speed
- `r` - Cycle through resolutions

### Batch Processing
```bash
python custom_inference.py \
    --ckpt_path "./models/LeanVAE-4ch.ckpt" \
    --input_video "./input_videos" \
    --reconstruct_video "./output" \
    --sequence_length 17
```

## Performance Optimization

### For Lowest Latency:
1. Use smaller buffer size (`--buffer-size 5`)
2. Lower processing resolution (`--resolution 64` or `96`)
3. Reduce processing FPS (`--process-fps 5-8`)
4. Use CPU for consistent timing (GPU can have variable latency)

### For Best Quality:
1. Higher resolution (`--resolution 160` or `192`)
2. Larger buffer for temporal consistency
3. Higher processing FPS if hardware allows

## Technical Details

### Frame Dropping Strategy
The ultra-low latency version implements aggressive frame dropping:
- **Input dropping**: Skip frames that arrive too quickly
- **Processing dropping**: Skip processing if previous operation still running
- **Buffer management**: Use minimal buffer size (5 frames vs original 17)

### LeanVAE Constraints
- Requires `4n+1` frames (5, 9, 13, 17, etc.)
- Automatically pads shorter sequences
- Uses temporal information for reconstruction

### Memory Usage
- Small processing resolution (96x96) uses ~200MB
- Standard resolution (256x256) uses ~500MB
- Model size: ~160MB (39.8M parameters)

## Results

The implementation achieves:
- **Processing time**: 200-800ms per batch (CPU)
- **Effective latency**: 200-400ms with aggressive dropping
- **Frame rate**: 5-10 reconstructed FPS
- **Quality**: High-fidelity reconstruction maintaining temporal consistency

## Troubleshooting

### High Latency
- Reduce `--process-fps`
- Lower `--resolution`
- Decrease `--buffer-size`
- Use CPU instead of GPU for consistent timing

### Poor Quality
- Increase `--resolution`
- Allow larger buffer size
- Reduce frame dropping aggressiveness

### Camera Issues
- Check camera index with `--camera 0` (try 1, 2, etc.)
- Ensure camera permissions are granted
- Try different camera resolutions

## Model Information

**LeanVAE-4ch:**
- Parameters: 39.8M
- Latent channels: 4
- Compression ratio: 8x spatial, 4x temporal
- Input: Video sequences of 4n+1 frames
- Output: Reconstructed video with same temporal structure

## Citation

```bibtex
@misc{cheng2025leanvaeultraefficientreconstructionvae,
    title={LeanVAE: An Ultra-Efficient Reconstruction VAE for Video Diffusion Models}, 
    author={Yu Cheng and Fajie Yuan},
    year={2025},
    eprint={2503.14325},
    archivePrefix={arXiv},
    primaryClass={cs.CV},
    url={https://arxiv.org/abs/2503.14325}, 
}
```

## License

MIT License - see original LeanVAE repository for details.
