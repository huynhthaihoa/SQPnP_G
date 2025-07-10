import numpy as np
import cv2
import mediapipe as mp
import sys
import os
import time
import argparse
import psutil
import threading

# Add the python_bindings directory to path so we can import sqpnp_python
# sys.path.insert(0, os.path.join(os.path.dirname(__file__), "python_bindings"))

from sqpnp_python import sqpnp_python  # Import the new SQPnP bindings

# Parse command line arguments
parser = argparse.ArgumentParser(description="Head Pose Estimation with SQPnP")
parser.add_argument(
    "--use-all-points",
    action="store_true",
    help="Use all 468 MediaPipe face landmarks instead of subset",
)
parser.add_argument(
    "--measure-cpu", action="store_true", help="Measure CPU cycles for SQPnP operations"
)
parser.add_argument(
    "--show-fps",
    action="store_true",
    default=True,
    help="Show FPS counter (default: True)",
)
args = parser.parse_args()

cam_info = {
    "rms": 0.34175056443810903,
    "K": np.array(
        [
            [601.80834207, 0.0, 318.07937938],
            [0.0, 604.8713534, 225.3853271],
            [0.0, 0.0, 1.0],
        ]
    ),
    "dist": np.array([[-0.21991499, 0.81145141, -0.00738429, 0.00400082, -1.14865218]]),
}

FACE_MODEL_NUMPY = "face_model_all.npy"

face_model_all = np.load(FACE_MODEL_NUMPY)

# Select landmark indices for PnP
if args.use_all_points:
    # Use all 468 MediaPipe landmarks
    landmarks_ids_pnp = list(range(468))
    print(f"Using all {len(landmarks_ids_pnp)} MediaPipe landmarks")
else:
    # Use subset for stability and performance
    landmarks_ids_pnp = [
        33,
        263,  # Outer eye corners
        61,
        291,  # Mouth corners
        199,  # Chin
        1,  # Nose tip
    ]
    print(f"Using subset of {len(landmarks_ids_pnp)} landmarks: {landmarks_ids_pnp}")

face_model_pnp = np.asarray(
    [face_model_all[i] for i in landmarks_ids_pnp], dtype=np.float64
)

# Initialize SQPnP solver once
sqpnp_solver = sqpnp_python.SQPnPSolver()

# Extract camera parameters for SQPnP from your calibration
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

# Performance monitoring variables
fps_counter = 0
fps_start_time = time.time()
fps = 0.0
sqpnp_times = []
cpu_usage = []


def get_cpu_cycles():
    """Get current CPU cycles (approximate)"""
    try:
        # Read CPU cycles from /proc/stat (Linux only)
        with open("/proc/stat", "r") as f:
            lines = f.readlines()
            cpu_line = lines[0].split()
            # Sum of all CPU times
            total_cycles = sum(int(x) for x in cpu_line[1:])
            return total_cycles
    except:
        return 0


def measure_sqpnp_performance(func, measure_cpu=False, *func_args, **func_kwargs):
    """Measure SQPnP performance with CPU cycles and time"""
    if measure_cpu:
        cpu_start = get_cpu_cycles()
        time_start = time.perf_counter()

        result = func(*func_args, **func_kwargs)

        time_end = time.perf_counter()
        cpu_end = get_cpu_cycles()

        elapsed_time = (time_end - time_start) * 1000  # Convert to milliseconds
        cpu_cycles = cpu_end - cpu_start

        sqpnp_times.append(elapsed_time)
        cpu_usage.append(cpu_cycles)

        return result, elapsed_time, cpu_cycles
    else:
        return func(*func_args, **func_kwargs), None, None


def define_custom_head_frame(canonical_face_model: np.ndarray):
    """
    Computes the static rotation (R) and translation (T) of a custom-defined
    head coordinate system relative to the MediaPipe canonical model's frame.
    """
    # Key landmark indices from the canonical model (subject's perspective)
    left_eye_corner_idx = 362
    right_eye_corner_idx = 133
    left_mouth_corner_idx = 291
    right_mouth_corner_idx = 61

    left_eye_pt = canonical_face_model[left_eye_corner_idx]
    right_eye_pt = canonical_face_model[right_eye_corner_idx]
    left_mouth_pt = canonical_face_model[left_mouth_corner_idx]
    right_mouth_pt = canonical_face_model[right_mouth_corner_idx]

    # Calculate origin and points for axis definition
    eye_center = (left_eye_pt + right_eye_pt) / 2.0
    mouth_center = (left_mouth_pt + right_mouth_pt) / 2.0

    # Origin (Translation vector T)
    T_custom_to_mp = eye_center.reshape(3, 1)

    # Define axes according to your specification
    x_vec = right_eye_pt - left_eye_pt
    x_axis = x_vec / np.linalg.norm(x_vec)

    y_vec_initial = mouth_center - eye_center
    y_axis_initial = y_vec_initial / np.linalg.norm(y_vec_initial)

    z_vec = np.cross(x_axis, y_axis_initial)
    z_axis = z_vec / np.linalg.norm(z_vec)

    # Recompute Y to ensure a perfect right-hand orthogonal system
    y_vec_final = np.cross(z_axis, x_axis)
    y_axis = y_vec_final / np.linalg.norm(y_vec_final)

    # Construct Rotation Matrix R
    R_custom_to_mp = np.stack([x_axis, y_axis, z_axis], axis=1)

    return R_custom_to_mp, T_custom_to_mp


def draw_axes(img, rvec, tvec, K, D, axis_length=50.0):
    """Draws the 3D coordinate axes on the image."""
    axis_points_3d = np.float32(
        [[0, 0, 0], [axis_length, 0, 0], [0, axis_length, 0], [0, 0, axis_length]]
    )
    imgpts, _ = cv2.projectPoints(axis_points_3d, rvec, tvec, K, D)

    imgpts = np.int32(imgpts).reshape(-1, 2)

    origin = tuple(imgpts[0])
    cv2.line(img, origin, tuple(imgpts[1]), (0, 0, 255), 3)  # X-axis: Red
    cv2.line(img, origin, tuple(imgpts[2]), (0, 255, 0), 3)  # Y-axis: Green
    cv2.line(img, origin, tuple(imgpts[3]), (255, 0, 0), 3)  # Z-axis: Blue


def draw_performance_info(img, fps, sqpnp_time=None, cpu_cycles=None):
    """Draw performance information on the image"""
    y_offset = 30
    line_height = 25

    # FPS
    cv2.putText(
        img,
        f"FPS: {fps:.1f}",
        (10, y_offset),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2,
    )
    y_offset += line_height

    # SQPnP time
    if sqpnp_time is not None:
        cv2.putText(
            img,
            f"SQPnP: {sqpnp_time:.2f}ms",
            (10, y_offset),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 0),
            2,
        )
        y_offset += line_height

    # CPU cycles
    if cpu_cycles is not None:
        cv2.putText(
            img,
            f"CPU Cycles: {cpu_cycles:,}",
            (10, y_offset),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 0),
            2,
        )
        y_offset += line_height

    # Number of points used
    cv2.putText(
        img,
        f"Points: {len(landmarks_ids_pnp)}",
        (10, y_offset),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 0),
        2,
    )


def main():
    """
    Main function to run the real-time custom head pose estimation.
    """
    global fps_counter, fps_start_time, fps

    # --- 1. Initialization and Pre-computation ---

    # Camera Intrinsics (use your calibrated values if available)
    # If not, these are reasonable generic values for a standard webcam.
    cam_matrix = cam_info["K"]
    dist_coeffs = cam_info["dist"]

    # Compute the static transformation from our custom frame to MediaPipe's frame
    R_custom_to_mp, T_custom_to_mp = define_custom_head_frame(face_model_all)

    # Initialize MediaPipe Face Mesh and webcam
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    cap = cv2.VideoCapture(0)

    print(f"Starting head pose estimation with SQPnP")
    print(f"Using {'all' if args.use_all_points else 'subset'} landmarks")
    print(f"CPU measurement: {'enabled' if args.measure_cpu else 'disabled'}")
    print(f"FPS display: {'enabled' if args.show_fps else 'disabled'}")

    # --- 2. Real-time Loop ---

    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue

        # Flip for selfie view, and convert BGR to RGB for MediaPipe
        image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
        height, width, _ = image.shape
        image.flags.writeable = False  # Performance optimization
        results = face_mesh.process(image)

        # Prepare image for drawing
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if results.multi_face_landmarks:
            face_landmarks = results.multi_face_landmarks[0]

            # --- 3. Get MediaPipe Pose using SQPnP ---
            landmarks_2d = np.array(
                [(lm.x * width, lm.y * height) for lm in face_landmarks.landmark],
                dtype=np.float64,  # Changed to float64 for SQPnP
            )
            landmarks_2d_pnp = np.asarray(
                [landmarks_2d[i] for i in landmarks_ids_pnp], dtype=np.float64
            )

            # Use SQPnP with RANSAC for robust pose estimation
            sqpnp_result, sqpnp_time, cpu_cycles = measure_sqpnp_performance(
                sqpnp_solver.solve_ransac,
                args.measure_cpu,
                face_model_pnp,
                landmarks_2d_pnp,
                camera_params,
                max_iterations=1000,
                threshold=2.0,
                confidence=0.99,
            )

            if sqpnp_result.success:
                # Extract rotation and translation from SQPnP result
                R_mp_to_cam = sqpnp_result.rotation  # Already a 3x3 matrix
                tvec_mp = sqpnp_result.translation.reshape(3, 1)  # Reshape to 3x1

                print(f"Face Distance: {np.linalg.norm(tvec_mp):.2f} units")
                print(f"SQPnP Error: {sqpnp_result.error:.4f}")
                if sqpnp_time is not None:
                    print(f"SQPnP Time: {sqpnp_time:.2f}ms")
                if cpu_cycles is not None:
                    print(f"CPU Cycles: {cpu_cycles:,}")

                # --- 4. Calculate the Final Pose of the Custom Frame ---
                # Combine rotations to get the final rotation of our custom frame
                R_final = R_mp_to_cam @ R_custom_to_mp

                # Combine translations to get the final origin of our custom frame
                t_final = tvec_mp + R_mp_to_cam @ T_custom_to_mp

                # Convert rotation matrix to rotation vector for OpenCV drawing
                rvec_final, _ = cv2.Rodrigues(R_final)

                # --- 5. Draw the custom axes on the image ---
                draw_axes(image, rvec_final, t_final, cam_matrix, dist_coeffs)
            else:
                print("SQPnP failed to find a solution")

        # Update FPS
        fps_counter += 1
        if time.time() - fps_start_time >= 1.0:
            fps = fps_counter / (time.time() - fps_start_time)
            fps_counter = 0
            fps_start_time = time.time()

        # Draw performance information
        if args.show_fps:
            draw_performance_info(image, fps, sqpnp_time, cpu_cycles)

        cv2.imshow("Custom Head Pose Estimation (SQPnP)", image)
        if cv2.waitKey(5) & 0xFF == 27:  # Press ESC to exit
            break

    # --- 6. Cleanup and Performance Summary ---
    face_mesh.close()
    cap.release()
    cv2.destroyAllWindows()

    # Print performance summary
    if args.measure_cpu and sqpnp_times:
        print(f"\n=== Performance Summary ===")
        print(
            f"Average SQPnP time: {np.mean(sqpnp_times):.2f}ms ± {np.std(sqpnp_times):.2f}ms"
        )
        print(f"Min SQPnP time: {np.min(sqpnp_times):.2f}ms")
        print(f"Max SQPnP time: {np.max(sqpnp_times):.2f}ms")
        if cpu_usage:
            print(
                f"Average CPU cycles: {np.mean(cpu_usage):,.0f} ± {np.std(cpu_usage):,.0f}"
            )
            print(f"Min CPU cycles: {np.min(cpu_usage):,}")
            print(f"Max CPU cycles: {np.max(cpu_usage):,}")


if __name__ == "__main__":
    main()
