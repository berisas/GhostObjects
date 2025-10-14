# Ghost2Real - Neural Mesh Super-Resolution

A deep learning system that enhances low-poly 3D meshes to high-poly quality using a custom TensorFlow neural network. Ghost2Real learns geometric details from simplified meshes and reconstructs high-resolution vertices with impressive accuracy.

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.13+-orange.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

## Features

- **Neural Mesh Enhancement**: Train a custom deep learning model to upscale mesh resolution
- **Multi-Level LOD System**: Automatic generation of multiple quality levels for comparison
- **Dual Interface**: Both GUI and console modes supported
- **Visual Comparison**: Side-by-side visualization of all quality levels
- **Advanced Architecture**: Encoder-decoder network with local/global feature learning
- **Quality Metrics**: Detailed reconstruction error analysis and training statistics

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/GhostObjects.git
cd GhostObjects
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Quick Start (Console Mode)

```bash
python Ghost2Real2.py
```

The application will:
1. Load the tennis ball mesh
2. Create multiple LOD variants
3. Train the ML model (150 epochs)
4. Display visual results with quality metrics

### GUI Mode

If tkinter is available, the application automatically launches a GUI:

1. Click "Train ML Model" to start training
2. Wait for training to complete (progress shown in console)
3. Click "Show Results" to visualize the comparison

### Custom Mesh

```python
from Ghost2Real2 import MLLODSystem

# Load your own mesh
lod_system = MLLODSystem("path/to/your/mesh.obj")

# Train with custom settings
lod_system.train_ml_model(epochs=300)

# Visualize
lod_system.show_visual_results()
```

## How It Works

### 1. Data Preparation
- Loads original high-poly mesh (target)
- Creates simplified low-poly mesh (30% face reduction)
- Normalizes geometry and computes interpolation weights

### 2. Neural Network Training
- **Local Encoder**: Extracts geometric features from low-poly vertices
- **Global Encoder**: Captures overall mesh shape statistics
- **Position Decoder**: Predicts high-quality vertex positions

### 3. Multi-Phase Training
- Position learning phase (high regularization)
- Refinement phase (medium regularization)
- Fine-tuning phase (low regularization)

### 4. Visualization
Displays 5 meshes side-by-side:
- **Red**: Ultra Low (5% faces)
- **Orange**: Low (12.5% faces)
- **Yellow**: Medium Base (25% faces) - ML input
- **Green**: ML Enhanced (100% faces) - Your model's output!
- **Blue**: Original (100% faces) - Ground truth

## Project Structure

```
GhostObjects/
├── Ghost2Real2.py         # Production version with LOD & GUI
├── tennis_ball.obj        # Test mesh
├── requirements.txt       # Python dependencies
├── CLAUDE.md             # Detailed technical documentation
└── README.md             # This file
```

## Results

Expected performance on tennis ball mesh:
- **Mean Reconstruction Error**: < 0.01
- **Training Loss**: Converges below 0.001
- **Visual Quality**: Green mesh nearly identical to Blue

## Technical Details

### Model Architecture
- Hidden dimension: 512
- Encoder: 3-layer dense network with LayerNormalization
- Decoder: 5-layer network with residual connections
- Learnable scaling parameters for adaptive refinement

### Loss Functions
1. Position Loss: MSE between predicted and target
2. Smoothness Loss: Regularizes displacement magnitude
3. Consistency Loss: Enforces neighbor coherence
4. Scale Loss: Prevents extreme scaling

### Training Configuration
- Optimizer: Adam with exponential learning rate decay
- Initial LR: 0.001, decay: 0.95 every 100 steps
- Gradient clipping: L2 norm max 1.0
- Early stopping: Patience 100 epochs

## Requirements

- Python 3.8+
- TensorFlow 2.13+
- NumPy 1.24+
- Trimesh 4.0+
- SciPy 1.11+
- Matplotlib 3.7+ (for GUI)
- Tkinter (optional, for GUI mode)

## Limitations

- Fixed topology: Same face connectivity required for input/output
- Single mesh training: Model doesn't generalize across different objects
- Memory intensive: Large meshes (>50k vertices) benefit from GPU
- No checkpoint saving: Model retrains each session

## Future Improvements

- [ ] Multi-mesh training for generalized models
- [ ] Model checkpoint saving/loading
- [ ] GPU acceleration with CUDA
- [ ] Support for texture and normal map enhancement
- [ ] Batch processing for multiple meshes
- [ ] Real-time preview during training

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Built with TensorFlow and Trimesh libraries
- Inspired by mesh super-resolution research in computer graphics
- Tennis ball model used for testing and demonstration

## Contact

For questions or feedback, please open an issue on GitHub.

---

**Made with machine learning and passion for 3D graphics**
