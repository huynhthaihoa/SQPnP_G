# SQPnP Python Bindings

Python bindings for SQPnP (Efficient Perspective-n-Point Algorithm) using pybind11.

## Features

- **Fast PnP solving**: Efficient implementation of the Perspective-n-Point algorithm
- **Multiple camera models**: Support for pinhole, fisheye, and distortion models
- **RANSAC integration**: Robust estimation with outlier rejection
- **Easy-to-use API**: Simple Python interface with NumPy arrays
- **High performance**: Direct C++ bindings with minimal overhead
- **Real-time demo**: Live head pose estimation with webcam

## Installation

### Prerequisites

- Python 3.7 or higher
- C++ compiler with C++17 support
- Eigen3 library
- pybind11

### Install Dependencies

#### Ubuntu/Debian
```bash
# Install system dependencies
sudo apt-get update
sudo apt-get install build-essential cmake libeigen3-dev

# Install Python dependencies
pip install pybind11 numpy opencv-python mediapipe
```

#### macOS
```bash
# Install system dependencies
brew install eigen cmake

# Install Python dependencies
pip install pybind11 numpy opencv-python mediapipe
```

#### Windows
```bash
# Install Visual Studio Build Tools or Visual Studio Community
# Install Eigen3 (via vcpkg or download from eigen.tuxfamily.org)

# Install Python dependencies
pip install pybind11 numpy opencv-python mediapipe
```

### Build and Install

#### Method 1: Using pip (Recommended)

```bash
cd python_bindings
pip install -e .
```

#### Method 2: Using the install script

```bash
cd python_bindings
chmod +x install.sh
./install.sh
```

#### Method 3: Using CMake

```bash
cd python_bindings
mkdir build && cd build
cmake ..
make -j4
```

#### Method 4: Direct build

```bash
cd python_bindings
python setup.py build_ext --inplace
```

### Verify Installation

```bash
cd python_bindings
python test_sqpnp.py
```

## Usage

### Basic Example

```python
import numpy as np
import sqpnp_python

# Create 3D points (world coordinates)
points_3d = np.array([
    [0, 0, 0],
    [1, 0, 0],
    [0, 1, 0],
    [1, 1, 0],
    [0, 0, 1],
    [1, 0, 1],
    [0, 1, 1],
    [1, 1, 1]
], dtype=np.float64)

# Create 2D projections (image coordinates)
points_2d = np.array([
    [320, 240],
    [420, 240],
    [320, 340],
    [420, 340],
    [320, 140],
    [420, 140],
    [320, 440],
    [420, 440]
], dtype=np.float64)

# Camera parameters [fx, fy, cx, cy]
camera_params = np.array([500.0, 500.0, 320.0, 240.0], dtype=np.float64)

# Create solver and solve PnP
solver = sqpnp_python.SQPnPSolver()
result = solver.solve_pinhole(points_3d, points_2d, camera_params)

if result.success:
    print(f"Rotation matrix:\n{result.rotation}")
    print(f"Translation vector: {result.translation}")
    print(f"Reprojection error: {result.error}")
    print(f"Number of solutions: {result.num_solutions}")
else:
    print("PnP solving failed")
```

### RANSAC Example

```python
# Solve with RANSAC for robust estimation
ransac_result = solver.solve_ransac(
    points_3d, points_2d, camera_params,
    max_iterations=1000,
    threshold=2.0,
    confidence=0.99
)

if ransac_result.success:
    print(f"RANSAC rotation matrix:\n{ransac_result.rotation}")
    print(f"RANSAC translation vector: {ransac_result.translation}")
    print(f"RANSAC reprojection error: {ransac_result.error}")
    print(f"Number of inliers: {ransac_result.num_inliers}")
    print(f"Number of outliers: {ransac_result.num_outliers}")
```

### Different Camera Models

```python
# Pinhole camera [fx, fy, cx, cy]
pinhole_params = np.array([500.0, 500.0, 320.0, 240.0], dtype=np.float64)

# Fisheye camera [fx, fy, cx, cy, k1, k2, k3, k4]
fisheye_params = np.array([500.0, 500.0, 320.0, 240.0, 0.1, 0.01, 0.001, 0.0001], dtype=np.float64)

# Pinhole with distortion [fx, fy, cx, cy, k1, k2, k3, k4, p1, p2]
distortion_params = np.array([500.0, 500.0, 320.0, 240.0, 0.1, 0.01, 0.001, 0.0001, 0.001, 0.001], dtype=np.float64)

# Detect camera type
camera_type = solver.get_camera_type(pinhole_params)
print(f"Camera type: {camera_type}")  # Output: "pinhole"

# Solve with different camera models
result_pinhole = solver.solve_pinhole(points_3d, points_2d, pinhole_params)
result_fisheye = solver.solve_fisheye(points_3d, points_2d, fisheye_params)
result_distortion = solver.solve_distortion(points_3d, points_2d, distortion_params)
```

### Convenience Functions

```python
# Use convenience functions for quick solving
result = sqpnp_python.solve_pnp(points_3d, points_2d, camera_params)
ransac_result = sqpnp_python.solve_pnp_ransac(points_3d, points_2d, camera_params)
```

### Real-time Head Pose Estimation

The repository includes a real-time head pose estimation demo:

```bash
# From the root directory
python adapter.py

# With custom parameters
python adapter.py --use-all-points --measure-cpu --show-fps
```

## API Reference

### SQPnPSolver

Main solver class for PnP problems.

#### Constructor
```python
solver = sqpnp_python.SQPnPSolver()
```

#### Methods

- `solve_pinhole(points_3d, points_2d, camera_params)` - Solve PnP with pinhole camera model
- `solve_fisheye(points_3d, points_2d, camera_params)` - Solve PnP with fisheye camera model
- `solve_distortion(points_3d, points_2d, camera_params)` - Solve PnP with pinhole + distortion model
- `solve_ransac(points_3d, points_2d, camera_params, max_iterations=1000, threshold=2.0, confidence=0.99)` - Solve PnP with RANSAC
- `get_camera_type(camera_params)` - Detect camera type from parameters

### SQPnPResult

Result object containing the solution.

#### Attributes

- `success` (bool) - Whether the solution was successful
- `rotation` (numpy.ndarray) - 3x3 rotation matrix
- `translation` (numpy.ndarray) - 3D translation vector
- `error` (float) - Average squared reprojection error
- `num_solutions` (int) - Number of solutions found
- `num_inliers` (int) - Number of inliers (RANSAC only)
- `num_outliers` (int) - Number of outliers (RANSAC only)
- `execution_time_us` (float) - Execution time in microseconds

### Camera Parameters

The system automatically detects camera type based on the number of parameters:

- **4 parameters**: Pinhole camera `[fx, fy, cx, cy]`
- **8 parameters**: Fisheye camera `[fx, fy, cx, cy, k1, k2, k3, k4]`
- **10 parameters**: Pinhole with distortion `[fx, fy, cx, cy, k1, k2, k3, k4, p1, p2]`

## Testing

Run the test script to verify the installation:

```bash
cd python_bindings
python test_sqpnp.py
```

## Performance Tips

1. **Use float64**: Always use `dtype=np.float64` for input arrays
2. **Batch processing**: For multiple PnP problems, reuse the solver instance
3. **RANSAC parameters**: Adjust `threshold` and `max_iterations` based on your data quality
4. **Camera calibration**: Use properly calibrated camera parameters for best results

## Troubleshooting

### Common Issues

1. **Import Error**: Make sure you've built the extension correctly
   ```bash
   cd python_bindings
   pip install -e .
   ```

2. **Eigen3 not found**: Install Eigen3 development headers
   ```bash
   # Ubuntu/Debian
   sudo apt-get install libeigen3-dev
   
   # macOS
   brew install eigen
   ```

3. **Compiler errors**: Ensure your compiler supports C++17
   ```bash
   # Check compiler version
   g++ --version
   ```

4. **pybind11 not found**: Install pybind11
   ```bash
   pip install pybind11
   ```

5. **Permission denied**: Make sure you have write permissions
   ```bash
   sudo chown -R $USER:$USER python_bindings/
   ```

### Build Issues

If you encounter build issues, try:

```bash
# Clean build
cd python_bindings
rm -rf build/ *.so *.pyd dist/ *.egg-info/

# Reinstall
pip install -e .
```

### Runtime Issues

1. **Segmentation fault**: Usually indicates incompatible Eigen3 version
2. **Wrong results**: Check camera parameters and coordinate system
3. **Slow performance**: Ensure you're using optimized builds (`-O3` flag)

### Platform-Specific Issues

#### Windows
- Install Visual Studio Build Tools
- Use vcpkg for Eigen3: `vcpkg install eigen3`
- Set environment variables for Eigen3 path

#### macOS
- Install Xcode Command Line Tools: `xcode-select --install`
- Use Homebrew for dependencies: `brew install eigen cmake`

#### Linux
- Install build essentials: `sudo apt-get install build-essential`
- Install Eigen3: `sudo apt-get install libeigen3-dev`

## Examples

See `example_usage.py` for comprehensive usage examples including:
- Basic PnP solving
- RANSAC with different parameters
- Multiple camera models
- Performance benchmarking
- Real-time head pose estimation

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

Same as the main SQPnP project. 