# SQPnP v2.0.0 - Python Bindings Release

## 🎉 What's New

This major release adds comprehensive Python bindings to the SQPnP library, making it accessible to a wider audience of researchers and developers.

### ✨ New Features

#### Python Bindings
- **Easy-to-use Python API**: Simple interface with NumPy arrays
- **Multiple camera models**: Support for pinhole, fisheye, and distortion models
- **Integrated RANSAC**: Robust estimation with outlier rejection
- **High performance**: Direct C++ bindings with minimal overhead
- **Real-time demo**: Live head pose estimation with webcam

#### Documentation & Examples
- **Comprehensive README**: Updated with both C++ and Python usage
- **Detailed API reference**: Complete documentation for all features
- **Installation scripts**: Automated setup for different platforms
- **Real-time demo**: Head pose estimation with MediaPipe integration
- **Performance benchmarks**: Comprehensive testing and optimization

#### Developer Experience
- **Automated installation**: Cross-platform install script
- **Comprehensive testing**: Unit tests and integration tests
- **Error handling**: Robust validation and error reporting
- **Performance optimization**: Optimized for real-time applications

### 🔧 Installation

#### C++ (Existing)
```bash
mkdir build && cd build
cmake ..
make -j4
```

#### Python (New)
```bash
cd python_bindings
pip install -e .
```

### 📖 Quick Start

#### Python Usage
```python
import numpy as np
import sqpnp_python

# Create solver and solve PnP
solver = sqpnp_python.SQPnPSolver()
result = solver.solve_pinhole(points_3d, points_2d, camera_params)

if result.success:
    print(f"Rotation: {result.rotation}")
    print(f"Translation: {result.translation}")
```

#### Real-time Demo
```bash
python adapter.py
```

### 🎮 Demo Features

- Real-time face detection using MediaPipe
- 3D head pose estimation using SQPnP
- Visual axes overlay on the face
- Performance metrics (FPS, CPU usage)
- Support for different landmark subsets

### 📊 Performance

- **C++**: Optimized for embedded devices and real-time applications
- **Python**: Minimal overhead with direct C++ bindings
- **RANSAC**: Integrated outlier rejection for robust estimation
- **Multi-threading**: Support for parallel processing

### 🔄 Migration Guide

No breaking changes for existing C++ users. Python users can start with the new bindings immediately.

### 📚 Documentation

- [Main README](README.md) - Comprehensive project overview
- [C++ API Documentation](CLEAN_README.md) - Detailed C++ usage
- [Python API Documentation](python_bindings/README.md) - Python bindings guide
- [Real-time Demo](adapter.py) - Live head pose estimation

### 🙏 Acknowledgments

- Original SQPnP implementation by G. Terzakis and M. Lourakis
- RANSAC implementation from RansacLib
- Python bindings using pybind11
- Real-time demo using MediaPipe Face Mesh

### 📄 License

Same as the original SQPnP project.

---

**Note**: This release maintains full backward compatibility with existing C++ code while adding powerful new Python capabilities.
