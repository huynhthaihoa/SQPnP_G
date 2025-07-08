# SQPnP - Efficient Perspective-n-Point Algorithm

A high-performance, robust implementation of SQPnP (Semi-Quadratic Programming Perspective-n-Point) with integrated RANSAC for camera pose estimation. Features both C++ and Python interfaces, supporting multiple camera models including pinhole and fisheye cameras.

## 🎯 Features

- **Unified API**: Single namespace `sqpnp::` for all C++ functionality
- **Python Bindings**: Easy-to-use Python interface with NumPy arrays
- **Integrated RANSAC**: No external dependencies beyond Eigen3
- **Multi-Camera Support**: Pinhole (with/without distortion) + Fisheye (KB model)
- **Command-Line Interface**: Professional CLI with comprehensive options
- **Automatic Normalization**: Converts pixel coordinates to normalized coordinates
- **Real Data Support**: Tested with original SQPnP datasets (891 correspondences)
- **High Performance**: Optimized for embedded devices and real-time applications
- **Robust**: Handles outliers automatically
- **Real-time Demo**: Live head pose estimation with webcam

## 📁 Project Structure

```
.
├── sqpnp/                          # Core C++ implementation
│   ├── sqpnp.h                     # Main SQPnP header
│   ├── sqpnp.cpp                   # SQPnP implementation
│   ├── types.h                     # Type definitions
│   ├── unified_sqpnp.h             # Unified API header
│   ├── RansacLib/                  # RANSAC library
│   └── CMakeLists.txt              # Library build configuration
├── python_bindings/                # Python interface
│   ├── sqpnp_bindings.cpp          # pybind11 bindings
│   ├── setup.py                    # Python package setup
│   ├── requirements.txt            # Python dependencies
│   ├── example_usage.py            # Python usage examples
│   ├── test_sqpnp.py               # Python tests
│   └── README.md                   # Python-specific documentation
├── assests/                        # Demo screenshots and assets
│   ├── front.png                   # Front view demo
│   ├── right-view.png              # Right view demo
│   └── top_left.png                # Top-left view demo
├── data/                           # Test datasets
├── adapter.py                      # Real-time head pose estimation demo
├── test_adapter.py                 # Adapter testing script
├── face_model_all.npy              # 3D face model for head pose
├── clean_example.cpp               # C++ example with hardcoded data
├── clean_example_cli.cpp           # Command-line interface
├── CMakeLists.txt                  # Root build configuration
├── CLEAN_README.md                 # Detailed C++ documentation
└── README.md                       # This file
```

## 🚀 Quick Start

### Prerequisites

- **C++**: CMake 3.10+, C++14 compiler, Eigen3 library
- **Python**: Python 3.7+, C++ compiler with C++17 support, pybind11, NumPy

### C++ Installation

```bash
# Install Eigen3 (Ubuntu/Debian)
sudo apt-get install libeigen3-dev

# Build C++ library
mkdir build && cd build
cmake ..
make -j4

# Run examples
./clean_example                    # Original with hardcoded data
./sqpnp_cli --demo                # CLI with demo data
./sqpnp_cli --help                # Show CLI help
```

### Python Installation

```bash
# Install Python dependencies
pip install pybind11 numpy opencv-python mediapipe

# Install Eigen3 (Ubuntu/Debian)
sudo apt-get install libeigen3-dev

# Build and install Python bindings
cd python_bindings
pip install -e .

# Test installation
python test_sqpnp.py
```

## 📖 Usage Examples

### C++ Usage

```cpp
#include "sqpnp/unified_sqpnp.h"

// Prepare data
std::vector<sqpnp::_Point> points3d = /* your 3D points */;
std::vector<sqpnp::_Projection> points2d = /* your 2D projections */;

// Solve with RANSAC
sqpnp::UnifiedPoseResult result = sqpnp::solveRobustPnP(points3d, points2d);

if (result.success) {
    Eigen::Matrix3d rotation = result.rotation;
    Eigen::Vector3d translation = result.translation;
    double error = result.reprojection_error;
}
```

### Python Usage

```python
import numpy as np
import sqpnp_python

# Create 3D points and 2D projections
points_3d = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float64)
points_2d = np.array([[320, 240], [420, 240], [320, 340]], dtype=np.float64)
camera_params = np.array([500.0, 500.0, 320.0, 240.0], dtype=np.float64)

# Solve PnP
solver = sqpnp_python.SQPnPSolver()
result = solver.solve_pinhole(points_3d, points_2d, camera_params)

if result.success:
    print(f"Rotation: {result.rotation}")
    print(f"Translation: {result.translation}")
    print(f"Error: {result.error}")
```

### Real-time Head Pose Estimation

```bash
# Run the real-time head pose estimation demo
python adapter.py

# Or with custom parameters
python adapter.py --use-all-points --measure-cpu --show-fps
```

## 🎮 Real-time Demo

The repository includes a real-time head pose estimation demo using your webcam:

- **File**: `adapter.py`
- **Features**: 
  - Real-time face detection using MediaPipe
  - 3D head pose estimation using SQPnP
  - Visual axes overlay on the face
  - Performance metrics (FPS, CPU usage)
  - Support for different landmark subsets

### Demo Screenshots

Check the `assests/` folder for demo screenshots showing:
- Front view with pose axes
- Right view with pose axes  
- Top-left view with pose axes

## 📖 API Reference

### C++ API

See `CLEAN_README.md` for detailed C++ API documentation.

### Python API

See `python_bindings/README.md` for detailed Python API documentation.

## 🔧 Command-Line Interface

```bash
# Demo mode (no files needed)
./sqpnp_cli --demo

# Real data with simple pinhole camera
./sqpnp_cli -d ../data/data.txt -c ../data/camera_pinhole_simple.txt

# Real data with RANSAC
./sqpnp_cli -d ../data/data.txt -c ../data/camera.txt -r -i 2000 -t 0.15

# Verbose output
./sqpnp_cli -d ../data/data.txt -c ../data/camera.txt -v
```

## 🧪 Testing

### C++ Tests
```bash
cd build
./clean_example
./sqpnp_cli --demo
```

### Python Tests
```bash
cd python_bindings
python test_sqpnp.py
```

### Real-time Demo Test
```bash
python test_adapter.py
```

## 📊 Performance

- **C++**: Optimized for embedded devices and real-time applications
- **Python**: Minimal overhead with direct C++ bindings
- **RANSAC**: Integrated outlier rejection for robust estimation
- **Multi-threading**: Support for parallel processing

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the same license as the original SQPnP project.

## 🙏 Acknowledgments

- Original SQPnP implementation by G. Terzakis and M. Lourakis
- RANSAC implementation from RansacLib
- Python bindings using pybind11
- Real-time demo using MediaPipe Face Mesh

## 📚 References

- **Paper**: "A Consistently Fast and Globally Optimal Solution to the Perspective-n-Point Problem" by G. Terzakis and M. Lourakis (ECCV 2020)
- **Paper URL**: https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123460460.pdf
- **Supplementary**: https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123460460-supp.pdf
- **OriginalImplementation**: https://github.com/terzakig/sqpnp/tree/master 

## 🆘 Support

For issues and questions:
1. Check the documentation in `CLEAN_README.md` and `python_bindings/README.md`
2. Run the test scripts to verify your installation
3. Open an issue on GitHub with detailed information about your problem 