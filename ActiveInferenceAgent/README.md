# Active Inference Agent

A revolutionary physics-informed neural architecture that combines cutting-edge concepts from active inference, physics, and machine learning for real-time video processing and understanding.

## 🌟 Key Features

- **Physics-Informed Wavelet Transforms**: Spatiotemporal frequency decomposition with energy conservation
- **Symplectic Hamiltonian Dynamics**: Energy-conserving temporal evolution (experimental)
- **Neural SLAM-inspired Memory**: Topological memory using Skyrmion/Hopfion fields (experimental)
- **Active Inference Framework**: Markov blanket formulation with free energy minimization
- **Real-time Processing**: Optimized for 60+ FPS video inference
- **Collapse Detection & Recovery**: Automatic VAE health monitoring and recovery

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/ActiveInferenceAgent.git
cd ActiveInferenceAgent

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Basic Usage

```python
from active_inference_agent import MicroLeanVAE

# Create model
model = MicroLeanVAE(input_size=128, latent_dim=32)

# Process video frame
import torch
frame = torch.rand(1, 3, 128, 128)  # Batch x Channels x Height x Width
reconstruction, mean, logvar = model(frame)
```

### Real-time Video Processing

```python
from active_inference_agent import UltraFast60FpsLeanVAE
import cv2

# Initialize processor
processor = UltraFast60FpsLeanVAE(device='cuda', input_resolution=(640, 480))

# Process webcam feed
cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    if not ret:
        break
        
    # Process frame
    reconstruction, inference_time, motion_score = processor.process_frame_ultra_fast(frame)
    
    # Display results
    cv2.imshow('Original', frame)
    cv2.imshow('Reconstruction', reconstruction)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
        
cap.release()
cv2.destroyAllWindows()
processor.cleanup()
```

## 🏗️ Architecture

The Active Inference Agent consists of several interconnected components:

### 1. Physics-Informed Wavelet Transform
- 2D spatial wavelet decomposition using Daubechies-4 wavelets
- 3D spatiotemporal wavelet transform for motion analysis
- Energy-conserving inverse transforms based on Parseval's theorem

### 2. Temporal Dynamics
- **TemporalKoopmanOperator**: Stable temporal evolution with EMA smoothing
- **SymplecticKoopmanOperator** (experimental): Energy-conserving Hamiltonian dynamics

### 3. Memory Systems
- **SkyrmionHopfionMemory** (experimental): Topological memory with SLAM-inspired addressing
- Content-based and motion-based memory access patterns
- Spatial locality filtering for flicker prevention

### 4. Active Inference Components
- **MarkovBlanketActiveInference** (experimental): Free energy minimization framework
- **KoopmanDMDOperator** (experimental): Dynamic mode decomposition for linearization
- **PureMathematicalFunctionalOperator** (experimental): Function-to-function neural operators

### 5. System Health Monitoring
- **CollapseDetector**: VAE posterior collapse detection
- **TemporalMemoryManager**: Pattern forgetting and static pixel detection
- **SceneChangeDetector**: Automatic scene transition detection

## 📊 Performance

Current performance metrics on standard hardware:

| Component | Time (ms) | Notes |
|-----------|-----------|-------|
| Wavelet Transform (128x128) | ~5-10 | Highly optimized |
| VAE Inference (64x64) | ~10-15 | Includes all components |
| Full Pipeline (240x240) | ~16-20 | Near 60 FPS |

## 🧪 Testing

Run the comprehensive test suite:

```bash
# Run all tests with coverage
python tests/run_all_tests.py

# Run specific test modules
python -m pytest tests/test_wavelet_transform.py -v
python -m pytest tests/test_temporal_koopman.py -v
python -m pytest tests/test_micro_lean_vae.py -v
```

## 📖 Documentation

### Core Components

- **PhysicsInformedWaveletTransform**: [docs/wavelet_transform.md](docs/wavelet_transform.md)
- **TemporalKoopmanOperator**: [docs/temporal_dynamics.md](docs/temporal_dynamics.md)
- **MicroLeanVAE**: [docs/vae_architecture.md](docs/vae_architecture.md)

### Known Issues

See [docs/issues.md](docs/issues.md) for a comprehensive list of known issues and limitations.

## 🤝 Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

### Development Setup

```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Run tests before submitting PR
python tests/run_all_tests.py

# Format code
black src/ tests/

# Type checking
mypy src/
```

## 📄 License

This project is licensed under the MIT License - see [LICENSE](LICENSE) for details.

## 🙏 Acknowledgments

This work builds upon concepts from:
- Active Inference and the Free Energy Principle (Friston et al.)
- Hamiltonian Neural Networks (Greydanus et al.)
- Neural ODEs and physics-informed neural networks
- Differentiable Neural Computers (DeepMind)
- Koopman operator theory for dynamical systems

## ⚠️ Experimental Features

The following components are experimental and may not be fully functional:
- SymplecticKoopmanOperator (energy conservation issues)
- SkyrmionHopfionMemory (integration incomplete)
- MarkovBlanketActiveInference (not fully integrated)
- PureMathematicalFunctionalOperator (not integrated)
- KoopmanDMDOperator (not integrated)

## 📧 Contact

For questions or collaboration opportunities, please open an issue on GitHub.

---

**Note**: This is an active research project. APIs and functionality may change rapidly. Use tagged releases for stability.