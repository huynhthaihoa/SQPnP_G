#!/usr/bin/env python3
"""
Test script to verify the adapter.py integration with SQPnP
"""

import numpy as np
import sys
import os

# Add the python_bindings directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "python_bindings"))

try:
    from sqpnp_python import sqpnp_python

    print("✓ Successfully imported sqpnp_python")
except ImportError as e:
    print(f"✗ Failed to import sqpnp_python: {e}")
    print("Make sure you've built the SQPnP bindings first!")
    sys.exit(1)


# Test the camera parameters extraction (from adapter.py)
def test_camera_params():
    """Test camera parameters extraction"""
    print("\n=== Testing Camera Parameters ===")

    # Simulate the camera info from adapter.py
    cam_info = {
        "K": np.array(
            [
                [601.80834207, 0.0, 318.07937938],
                [0.0, 604.8713534, 225.3853271],
                [0.0, 0.0, 1.0],
            ]
        ),
        "dist": np.array(
            [[-0.21991499, 0.81145141, -0.00738429, 0.00400082, -1.14865218]]
        ),
    }

    # Extract camera parameters for SQPnP
    fx, fy = cam_info["K"][0, 0], cam_info["K"][1, 1]
    cx, cy = cam_info["K"][0, 2], cam_info["K"][1, 2]

    # Handle distortion coefficients - OpenCV format is [k1, k2, p1, p2, k3]
    # But SQPnP expects [k1, k2, k3, k4, p1, p2]
    dist_coeffs = cam_info["dist"][0]  # [k1, k2, p1, p2, k3]

    # Reorder to SQPnP format: [k1, k2, k3, k4, p1, p2]
    k1, k2 = dist_coeffs[0], dist_coeffs[1]
    p1, p2 = dist_coeffs[2], dist_coeffs[3]
    k3 = dist_coeffs[4]
    k4 = 0.0  # OpenCV doesn't use k4, set to 0

    # Create camera parameters array (10 parameters for distortion model)
    camera_params = np.array([fx, fy, cx, cy, k1, k2, k3, k4, p1, p2], dtype=np.float64)

    print(f"Camera parameters shape: {camera_params.shape}")
    print(f"Camera parameters: {camera_params}")
    print(f"  fx, fy: {fx:.2f}, {fy:.2f}")
    print(f"  cx, cy: {cx:.2f}, {cy:.2f}")
    print(f"  k1, k2, k3, k4: {k1:.4f}, {k2:.4f}, {k3:.4f}, {k4:.4f}")
    print(f"  p1, p2: {p1:.4f}, {p2:.4f}")

    # Test camera type detection
    solver = sqpnp_python.SQPnPSolver()
    camera_type = solver.get_camera_type(camera_params)
    print(f"Camera type detected: {camera_type}")

    return camera_params


def test_face_model():
    """Test face model loading and processing"""
    print("\n=== Testing Face Model ===")

    # Try to load the actual face model file
    face_model_path = "face_model_all.npy"
    if os.path.exists(face_model_path):
        face_model_all = np.load(face_model_path)
        print(f"✓ Loaded face model from {face_model_path}")
    else:
        # Create a mock face model with 468 points (MediaPipe standard)
        print(f"⚠ Face model file {face_model_path} not found, creating mock model")
        face_model_all = np.random.rand(468, 3).astype(np.float64)
        # Set some realistic face-like structure
        face_model_all[:, 2] = 0.1  # All points slightly in front

    print(f"Full face model shape: {face_model_all.shape}")

    # Select landmark indices for PnP (like in adapter.py)
    landmarks_ids_pnp = [33, 263, 61, 291, 199, 1]

    # Check if all indices are valid
    max_idx = face_model_all.shape[0] - 1
    valid_indices = [i for i in landmarks_ids_pnp if 0 <= i <= max_idx]

    if len(valid_indices) != len(landmarks_ids_pnp):
        print(
            f"⚠ Some landmark indices are out of bounds. Using valid ones: {valid_indices}"
        )
        landmarks_ids_pnp = valid_indices

    face_model_pnp = np.asarray(
        [face_model_all[i] for i in landmarks_ids_pnp], dtype=np.float64
    )

    print(f"PnP face model shape: {face_model_pnp.shape}")
    print(f"Selected landmark indices: {landmarks_ids_pnp}")

    return face_model_pnp


def test_pnp_solving():
    """Test PnP solving with realistic data"""
    print("\n=== Testing PnP Solving ===")

    # Get camera parameters and face model
    camera_params = test_camera_params()
    face_model_pnp = test_face_model()

    # Create realistic 2D projections
    # Simulate a face at a reasonable distance and orientation
    landmarks_2d_pnp = np.array(
        [
            [320, 240],  # Point 33
            [420, 240],  # Point 263
            [320, 340],  # Point 61
            [420, 340],  # Point 291
            [320, 140],  # Point 199
            [370, 200],  # Point 1 (nose tip)
        ],
        dtype=np.float64,
    )

    print(f"2D landmarks shape: {landmarks_2d_pnp.shape}")

    # Test SQPnP solving
    solver = sqpnp_python.SQPnPSolver()

    # Test simple PnP
    result = solver.solve_pinhole(face_model_pnp, landmarks_2d_pnp, camera_params)
    print(f"Simple PnP success: {result.success}")
    if result.success:
        print(f"  Error: {result.error:.4f}")
        print(f"  Distance: {np.linalg.norm(result.translation):.2f}")

    # Test RANSAC PnP
    ransac_result = solver.solve_ransac(
        face_model_pnp,
        landmarks_2d_pnp,
        camera_params,
        max_iterations=1000,
        threshold=2.0,
        confidence=0.99,
    )
    print(f"RANSAC PnP success: {ransac_result.success}")
    if ransac_result.success:
        print(f"  Error: {ransac_result.error:.4f}")
        print(f"  Distance: {np.linalg.norm(ransac_result.translation):.2f}")

    return result.success and ransac_result.success


if __name__ == "__main__":
    print("SQPnP Adapter Integration Test")
    print("=" * 50)

    try:
        # Test all components
        test_camera_params()
        test_face_model()
        success = test_pnp_solving()

        if success:
            print("\n" + "=" * 50)
            print("✓ All tests passed! The adapter.py should work with SQPnP.")
            print("You can now run: python adapter.py")
        else:
            print("\n" + "=" * 50)
            print("✗ Some tests failed. Check the output above.")

    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback

        traceback.print_exc()
