#!/usr/bin/env python3
"""
SQPnP Python Bindings - Comprehensive Usage Examples

This script demonstrates various ways to use the SQPnP Python bindings
for camera pose estimation from 2D-3D correspondences.

Features demonstrated:
- Basic PnP solving with different camera models
- RANSAC for robust estimation
- Performance benchmarking
- Real-time head pose estimation
- Error handling and validation
"""

import numpy as np
import time
import sys
import os

# Add the parent directory to the path to import sqpnp_python
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import sqpnp_python

    print("✅ SQPnP Python bindings imported successfully")
except ImportError as e:
    print(f"❌ Failed to import SQPnP Python bindings: {e}")
    print("Please install the bindings first: cd python_bindings && pip install -e .")
    sys.exit(1)


def create_test_data():
    """Create synthetic test data for PnP solving."""
    print("\n=== Creating Test Data ===")

    # Create 3D points (world coordinates) - a cube
    points_3d = np.array(
        [
            [0, 0, 0],  # Bottom front left
            [1, 0, 0],  # Bottom front right
            [0, 1, 0],  # Bottom back left
            [1, 1, 0],  # Bottom back right
            [0, 0, 1],  # Top front left
            [1, 0, 1],  # Top front right
            [0, 1, 1],  # Top back left
            [1, 1, 1],  # Top back right
        ],
        dtype=np.float64,
    )

    # Create 2D projections (image coordinates)
    points_2d = np.array(
        [
            [320, 240],  # Bottom front left
            [420, 240],  # Bottom front right
            [320, 340],  # Bottom back left
            [420, 340],  # Bottom back right
            [320, 140],  # Top front left
            [420, 140],  # Top front right
            [320, 440],  # Top back left
            [420, 440],  # Top back right
        ],
        dtype=np.float64,
    )

    # Camera parameters [fx, fy, cx, cy]
    camera_params = np.array([500.0, 500.0, 320.0, 240.0], dtype=np.float64)

    print(f"Created {len(points_3d)} 3D-2D correspondences")
    print(f"Camera parameters: {camera_params}")

    return points_3d, points_2d, camera_params


def test_basic_pnp():
    """Test basic PnP solving without RANSAC."""
    print("\n=== Testing Basic PnP ===")

    points_3d, points_2d, camera_params = create_test_data()

    # Create solver
    solver = sqpnp_python.SQPnPSolver()

    # Solve PnP
    start_time = time.time()
    result = solver.solve_pinhole(points_3d, points_2d, camera_params)
    end_time = time.time()

    print(f"Solving time: {(end_time - start_time) * 1000:.2f} ms")

    if result.success:
        print("✅ PnP solving successful!")
        print(f"Rotation matrix:\n{result.rotation}")
        print(f"Translation vector: {result.translation}")
        print(f"Reprojection error: {result.error:.6f}")
        print(f"Number of solutions: {result.num_solutions}")
    else:
        print("❌ PnP solving failed")

    return result


def test_ransac_pnp():
    """Test PnP solving with RANSAC for robust estimation."""
    print("\n=== Testing RANSAC PnP ===")

    points_3d, points_2d, camera_params = create_test_data()

    # Add some outliers to test RANSAC
    points_2d_with_outliers = points_2d.copy()
    points_2d_with_outliers[0] = [100, 100]  # Add outlier
    points_2d_with_outliers[1] = [600, 600]  # Add outlier

    print("Added 2 outliers to test RANSAC robustness")

    # Create solver
    solver = sqpnp_python.SQPnPSolver()

    # Solve with RANSAC
    start_time = time.time()
    result = solver.solve_ransac(
        points_3d,
        points_2d_with_outliers,
        camera_params,
        max_iterations=1000,
        threshold=2.0,
        confidence=0.99,
    )
    end_time = time.time()

    print(f"RANSAC solving time: {(end_time - start_time) * 1000:.2f} ms")

    if result.success:
        print("✅ RANSAC PnP solving successful!")
        print(f"Rotation matrix:\n{result.rotation}")
        print(f"Translation vector: {result.translation}")
        print(f"Reprojection error: {result.error:.6f}")
        print(f"Number of inliers: {result.num_inliers}")
        print(f"Number of outliers: {result.num_outliers}")
        print(f"Execution time: {result.execution_time_us:.2f} μs")
    else:
        print("❌ RANSAC PnP solving failed")

    return result


def test_different_camera_models():
    """Test PnP solving with different camera models."""
    print("\n=== Testing Different Camera Models ===")

    points_3d, points_2d, _ = create_test_data()

    # Create solver
    solver = sqpnp_python.SQPnPSolver()

    # Test pinhole camera
    pinhole_params = np.array([500.0, 500.0, 320.0, 240.0], dtype=np.float64)
    print(f"\nTesting pinhole camera: {pinhole_params}")
    result_pinhole = solver.solve_pinhole(points_3d, points_2d, pinhole_params)
    if result_pinhole.success:
        print(f"✅ Pinhole: Error = {result_pinhole.error:.6f}")

    # Test fisheye camera
    fisheye_params = np.array(
        [500.0, 500.0, 320.0, 240.0, 0.1, 0.01, 0.001, 0.0001], dtype=np.float64
    )
    print(f"\nTesting fisheye camera: {fisheye_params}")
    result_fisheye = solver.solve_fisheye(points_3d, points_2d, fisheye_params)
    if result_fisheye.success:
        print(f"✅ Fisheye: Error = {result_fisheye.error:.6f}")

    # Test pinhole with distortion
    distortion_params = np.array(
        [500.0, 500.0, 320.0, 240.0, 0.1, 0.01, 0.001, 0.0001, 0.001, 0.001],
        dtype=np.float64,
    )
    print(f"\nTesting pinhole with distortion: {distortion_params}")
    result_distortion = solver.solve_distortion(points_3d, points_2d, distortion_params)
    if result_distortion.success:
        print(f"✅ Distortion: Error = {result_distortion.error:.6f}")


def test_camera_type_detection():
    """Test automatic camera type detection."""
    print("\n=== Testing Camera Type Detection ===")

    solver = sqpnp_python.SQPnPSolver()

    # Test different camera parameter arrays
    test_cameras = [
        (np.array([500.0, 500.0, 320.0, 240.0], dtype=np.float64), "pinhole"),
        (
            np.array(
                [500.0, 500.0, 320.0, 240.0, 0.1, 0.01, 0.001, 0.0001], dtype=np.float64
            ),
            "fisheye",
        ),
        (
            np.array(
                [500.0, 500.0, 320.0, 240.0, 0.1, 0.01, 0.001, 0.0001, 0.001, 0.001],
                dtype=np.float64,
            ),
            "distortion",
        ),
    ]

    for params, expected_type in test_cameras:
        detected_type = solver.get_camera_type(params)
        status = "✅" if detected_type == expected_type else "❌"
        print(
            f"{status} {len(params)} params -> {detected_type} (expected: {expected_type})"
        )


def test_convenience_functions():
    """Test convenience functions for quick PnP solving."""
    print("\n=== Testing Convenience Functions ===")

    points_3d, points_2d, camera_params = create_test_data()

    # Test basic convenience function
    print("Testing sqpnp_python.solve_pnp()...")
    result = sqpnp_python.solve_pnp(points_3d, points_2d, camera_params)
    if result.success:
        print(f"✅ Convenience function: Error = {result.error:.6f}")

    # Test RANSAC convenience function
    print("Testing sqpnp_python.solve_pnp_ransac()...")
    ransac_result = sqpnp_python.solve_pnp_ransac(points_3d, points_2d, camera_params)
    if ransac_result.success:
        print(f"✅ RANSAC convenience function: Error = {ransac_result.error:.6f}")


def benchmark_performance():
    """Benchmark the performance of different solving methods."""
    print("\n=== Performance Benchmarking ===")

    points_3d, points_2d, camera_params = create_test_data()
    solver = sqpnp_python.SQPnPSolver()

    # Benchmark basic PnP
    num_runs = 100
    times_basic = []

    for _ in range(num_runs):
        start_time = time.time()
        result = solver.solve_pinhole(points_3d, points_2d, camera_params)
        end_time = time.time()
        times_basic.append((end_time - start_time) * 1000)

    avg_time_basic = np.mean(times_basic)
    std_time_basic = np.std(times_basic)

    print(f"Basic PnP ({num_runs} runs):")
    print(f"  Average time: {avg_time_basic:.3f} ± {std_time_basic:.3f} ms")
    print(f"  Min time: {np.min(times_basic):.3f} ms")
    print(f"  Max time: {np.max(times_basic):.3f} ms")

    # Benchmark RANSAC PnP
    times_ransac = []

    for _ in range(10):  # Fewer runs for RANSAC as it's slower
        start_time = time.time()
        result = solver.solve_ransac(
            points_3d, points_2d, camera_params, max_iterations=100
        )
        end_time = time.time()
        times_ransac.append((end_time - start_time) * 1000)

    avg_time_ransac = np.mean(times_ransac)
    std_time_ransac = np.std(times_ransac)

    print(f"\nRANSAC PnP (10 runs):")
    print(f"  Average time: {avg_time_ransac:.3f} ± {std_time_ransac:.3f} ms")
    print(f"  Min time: {np.min(times_ransac):.3f} ms")
    print(f"  Max time: {np.max(times_ransac):.3f} ms")


def test_error_handling():
    """Test error handling with invalid inputs."""
    print("\n=== Testing Error Handling ===")

    solver = sqpnp_python.SQPnPSolver()

    # Test with insufficient points
    print("Testing with insufficient points...")
    points_3d_few = np.array([[0, 0, 0], [1, 0, 0]], dtype=np.float64)
    points_2d_few = np.array([[320, 240], [420, 240]], dtype=np.float64)
    camera_params = np.array([500.0, 500.0, 320.0, 240.0], dtype=np.float64)

    result = solver.solve_pinhole(points_3d_few, points_2d_few, camera_params)
    if not result.success:
        print("✅ Correctly handled insufficient points")
    else:
        print("❌ Should have failed with insufficient points")

    # Test with mismatched array sizes
    print("Testing with mismatched array sizes...")
    points_3d_mismatch = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float64)
    points_2d_mismatch = np.array([[320, 240], [420, 240]], dtype=np.float64)

    try:
        result = solver.solve_pinhole(
            points_3d_mismatch, points_2d_mismatch, camera_params
        )
        print("❌ Should have raised an error for mismatched sizes")
    except Exception as e:
        print(f"✅ Correctly handled mismatched sizes: {e}")


def main():
    """Main function to run all tests."""
    print("🚀 SQPnP Python Bindings - Comprehensive Usage Examples")
    print("=" * 60)

    try:
        # Run all tests
        test_basic_pnp()
        test_ransac_pnp()
        test_different_camera_models()
        test_camera_type_detection()
        test_convenience_functions()
        benchmark_performance()
        test_error_handling()

        print("\n" + "=" * 60)
        print("✅ All tests completed successfully!")
        print("\n🎯 Next steps:")
        print("1. Try the real-time demo: python adapter.py")
        print("2. Check the documentation in README.md")
        print("3. Explore the API reference for more features")

    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback

        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
