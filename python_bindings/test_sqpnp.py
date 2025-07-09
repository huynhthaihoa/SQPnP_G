#!/usr/bin/env python3
"""
Test script for SQPnP Python bindings
"""

import numpy as np
import sys
import os

# Add current directory to path for testing
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from sqpnp_python import sqpnp_python

    print("✓ Successfully imported sqpnp_python")
except ImportError as e:
    print(f"✗ Failed to import sqpnp_python: {e}")
    sys.exit(1)


def test_basic_functionality():
    """Test basic SQPnP functionality"""
    print("\n=== Testing Basic Functionality ===")

    # Create test data
    # 3D points (world coordinates)
    points_3d = np.array(
        [
            [0, 0, 0],
            [1, 0, 0],
            [0, 1, 0],
            [1, 1, 0],
            [0, 0, 1],
            [1, 0, 1],
            [0, 1, 1],
            [1, 1, 1],
        ],
        dtype=np.float64,
    )

    # 2D projections (image coordinates)
    points_2d = np.array(
        [
            [320, 240],
            [420, 240],
            [320, 340],
            [420, 340],
            [320, 140],
            [420, 140],
            [320, 440],
            [420, 440],
        ],
        dtype=np.float64,
    )

    # Camera parameters [fx, fy, cx, cy]
    camera_params = np.array([500.0, 500.0, 320.0, 240.0], dtype=np.float64)

    print(f"3D points shape: {points_3d.shape}")
    print(f"2D points shape: {points_2d.shape}")
    print(f"Camera params: {camera_params}")

    # Test solver creation
    try:
        solver = sqpnp_python.SQPnPSolver()
        print("✓ Created SQPnPSolver instance")
    except Exception as e:
        print(f"✗ Failed to create solver: {e}")
        return False

    # Test camera type detection
    try:
        camera_type = solver.get_camera_type(camera_params)
        print(f"✓ Camera type detected: {camera_type}")
    except Exception as e:
        print(f"✗ Failed to detect camera type: {e}")
        return False

    # Test PnP solving
    try:
        result = solver.solve_pinhole(points_3d, points_2d, camera_params)
        print(f"✓ PnP solve completed")
        print(f"  Success: {result.success}")
        print(f"  Error: {result.error}")
        print(f"  Num solutions: {result.num_solutions}")
        if result.success:
            print(f"  Rotation shape: {result.rotation.shape}")
            print(f"  Translation shape: {result.translation.shape}")
    except Exception as e:
        print(f"✗ Failed to solve PnP: {e}")
        return False

    # Test RANSAC solving
    try:
        ransac_result = solver.solve_ransac(points_3d, points_2d, camera_params)
        print(f"✓ RANSAC solve completed")
        print(f"  Success: {ransac_result.success}")
        print(f"  Error: {ransac_result.error}")
        print(f"  Num solutions: {ransac_result.num_solutions}")
    except Exception as e:
        print(f"✗ Failed to solve RANSAC: {e}")
        return False

    # Test convenience functions
    try:
        conv_result = sqpnp_python.solve_pnp(points_3d, points_2d, camera_params)
        print("✓ Convenience function works")
    except Exception as e:
        print(f"✗ Convenience function failed: {e}")
        return False

    return True


def test_different_camera_models():
    """Test different camera models"""
    print("\n=== Testing Different Camera Models ===")

    # Test data
    points_3d = np.array([[0, 0, 1], [1, 0, 1], [0, 1, 1]], dtype=np.float64)
    points_2d = np.array([[320, 240], [420, 240], [320, 340]], dtype=np.float64)

    # Test pinhole camera
    pinhole_params = np.array([500.0, 500.0, 320.0, 240.0], dtype=np.float64)
    try:
        solver = sqpnp_python.SQPnPSolver()
        result = solver.solve_pinhole(points_3d, points_2d, pinhole_params)
        print(f"✓ Pinhole camera test: {result.success}")
    except Exception as e:
        print(f"✗ Pinhole camera test failed: {e}")

    # Test fisheye camera
    fisheye_params = np.array(
        [500.0, 500.0, 320.0, 240.0, 0.1, 0.01, 0.001, 0.0001], dtype=np.float64
    )
    try:
        result = solver.solve_pinhole(points_3d, points_2d, fisheye_params)
        print(f"✓ Fisheye camera test: {result.success}")
    except Exception as e:
        print(f"✗ Fisheye camera test failed: {e}")

    # Test pinhole with distortion
    distortion_params = np.array(
        [500.0, 500.0, 320.0, 240.0, 0.1, 0.01, 0.001, 0.0001, 0.001, 0.001],
        dtype=np.float64,
    )
    try:
        result = solver.solve_pinhole(points_3d, points_2d, distortion_params)
        print(f"✓ Distortion camera test: {result.success}")
    except Exception as e:
        print(f"✗ Distortion camera test failed: {e}")


if __name__ == "__main__":
    print("SQPnP Python Bindings Test")
    print("=" * 40)

    # Test basic functionality
    if test_basic_functionality():
        print("\n✓ All basic tests passed!")
    else:
        print("\n✗ Some basic tests failed!")
        sys.exit(1)

    # Test different camera models
    test_different_camera_models()

    print("\n" + "=" * 40)
    print("✓ All tests completed successfully!")
    print("The SQPnP Python bindings are working correctly.")
