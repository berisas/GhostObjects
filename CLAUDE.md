# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Ghost2Real** is a neural mesh super-resolution system that uses deep learning to enhance low-poly 3D meshes to high-poly quality. The project implements a custom TensorFlow neural network trained to learn geometric details and reconstruct high-resolution mesh vertices from simplified inputs.

This project is part of the **GhostObjects** repository, which combines Unity 3D development with machine learning for mesh enhancement.

**Core Component**:
- `Ghost2Real2.py` - Production version with LOD system, GUI interface, and complete ML pipeline

## Running the Application

### Console Mode
```bash
python Ghost2Real2.py
```

### GUI Mode (if tkinter available)
```bash
python Ghost2Real2.py
```
The application will automatically detect if GUI libraries are available and launch the appropriate interface.

### Command Line with Custom Mesh
```python
from Ghost2Real2 import MLLODSystem

# Initialize with custom mesh file
lod_system = MLLODSystem("path/to/your/mesh.obj")

# Train the model
lod_system.train_ml_model(epochs=200)

# Visualize results
lod_system.show_visual_results()
```

## Architecture

### Neural Network Model (`MeshSuperResNet`)

The model uses a sophisticated encoder-decoder architecture:

1. **Local Encoder** (512-dim hidden)
   - 3-layer dense network with LayerNorm and Dropout
   - Extracts local geometric features from low-poly vertices
   - Scaled by learnable `feature_scale` parameter

2. **Global Encoder** (256-dim output)
   - Processes max, mean, and std statistics
   - Captures overall mesh shape and structure
   - Provides context for local refinement

3. **Position Decoder** (5-layer network)
   - Combines local features + global context + query positions
   - Predicts displacement vectors for vertex refinement
   - Outputs 3D positions with tanh activation and learnable scaling

### Training Pipeline

1. **Data Preparation**
   - Loads original high-poly mesh (target)
   - Creates simplified low-poly mesh (30% reduction by default)
   - Normalizes vertices around centroid with uniform scale
   - Computes k-NN interpolation weights (k=8 neighbors, Gaussian weighting)

2. **Template Generation**
   - Creates initial guess with minimal noise (2% of std)
   - Used as starting point for vertex prediction

3. **Training Loop**
   - MSE loss between predicted and target vertices
   - Adam optimizer with exponential learning rate decay
   - Gradient clipping (max norm 1.0) for stability
   - Early stopping available via patience parameter

4. **LOD System** (`MLLODSystem`)
   - Creates multiple mesh quality levels:
     - Ultra Low: ~5% of original faces
     - Low: ~12.5% of original faces
     - Medium Base: ~25% of original faces (ML training input)
     - ML Enhanced: Full face count with ML-predicted vertices
     - High: Original mesh (target quality)

## Data Format

**Input**: OBJ mesh file (tested with `tennis_ball.obj`)
- Must contain valid vertices and faces
- Recommended: 1000+ faces for good results
- Larger meshes work better (more training data)

**Output**: Trimesh scene with visual comparison
- Red/Orange/Yellow: Traditional LOD levels
- Green: ML-enhanced mesh (reconstructed)
- Blue: Original high-poly target

## Key Design Decisions

- **Interpolation Strategy**: k-NN with Gaussian weighting (k=8) provides smooth feature transfer
- **Reduction Ratio**: 30% reduction (keeping 70% of faces) balances training speed vs. quality
- **Hidden Dimension**: 512-dim provides sufficient capacity without overfitting
- **Learnable Parameters**: `displacement_scale` and `feature_scale` adapt to mesh characteristics
- **Gradient Clipping**: Norm clipping (max=1.0) prevents training instability
- **Learning Rate Schedule**: Exponential decay (0.95 every 100 steps) improves convergence

## Dependencies

Install via requirements.txt:
```bash
pip install -r requirements.txt
```

Required packages:
- `tensorflow` - Neural network framework
- `numpy` - Numerical computations
- `trimesh` - Mesh loading and visualization
- `scipy` - Spatial data structures (k-NN search)
- `matplotlib` - Plotting (optional for GUI)

Optional for GUI:
- `tkinter` - GUI framework (usually bundled with Python)

## Integration with Unity

The GhostObjects project includes both Unity development and ML mesh enhancement:
- Unity project files are in the root directory
- ML scripts are in the same directory as `tennis_ball.obj`
- Enhanced meshes can be exported and imported into Unity scenes

## Known Limitations

1. **Fixed Topology**: Model requires same face connectivity for input/output
2. **Single Mesh Training**: Model trains on one mesh at a time (not generalized)
3. **Memory Usage**: Large meshes (>50k vertices) may require GPU
4. **Default Path**: Default mesh path is `C:\Users\Ber\GhostObjects\tennis_ball.obj`
5. **No Model Saving**: Model is retrained each run (no checkpoint persistence)

## Performance Metrics

Expected results on tennis ball mesh:
- **Reconstruction Error (Mean)**: < 0.01 for good quality
- **Reconstruction Error (Max)**: < 0.05 acceptable
- **Training Loss**: Should converge below 0.001
- **Visual Quality**: Green mesh should closely match Blue mesh

## Troubleshooting

**Issue**: "Invalid mesh file - no vertices found"
- **Solution**: Ensure OBJ file is valid, check file path

**Issue**: Training loss not decreasing
- **Solution**: Increase epochs, adjust learning rate, or try different reduction_ratio

**Issue**: High reconstruction error
- **Solution**: Train longer, increase hidden_dim, use less aggressive reduction

**Issue**: GUI not available
- **Solution**: Install tkinter or use console mode (works automatically)
