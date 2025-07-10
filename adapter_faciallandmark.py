import math
import numpy as np
import cv2
import mediapipe as mp
import sys
import os
import time
import argparse
import psutil
import threading

from nets.yolox_detection_model import DetectionModel
from nets.faciallandmark2d_model import FacialLandmark2DModel#, estimate_headpose_eareye, estimate_headpose_eyenose, estimate_headpose_earnose, draw_headpose

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
    "rms": 0.38494676143048673,
    "K": np.array(
        [
            [595.8036047830891, 0.0, 924.65430795264774],
            [0.0, 598.33827405037948, 580.41919770485049],
            [0.0, 0.0, 1.0],
        ]
    ),
    "dist": np.array([-0.015349419086740696, -0.0536764772521049, 0.061315407683887907,
    -0.026142516909791854])   #np.zeros(4),
}

FACE_MODEL_NUMPY = "face_model_all.npy"

face_model_all = np.load(FACE_MODEL_NUMPY)

# Select landmark indices for PnP
# if args.use_all_points:
#     # Use all 468 MediaPipe landmarks
#     landmarks_ids_pnp = list(range(468))
#     print(f"Using all {len(landmarks_ids_pnp)} MediaPipe landmarks")
# else:
#     # Use subset for stability and performance
#     landmarks_ids_pnp = [
#         33,
#         263,  # Outer eye corners
#         61,
#         291,  # Mouth corners
#         199,  # Chin
#         1,  # Nose tip
#     ]
#     print(f"Using subset of {len(landmarks_ids_pnp)} landmarks: {landmarks_ids_pnp}")

landmarks3d_ids_pnp = [
    33,   # Right outer eye
    263,  # Lef outer eye
    61,   # Right outer mouth
    291,  # Left outer mouth
    199,  # Chin
    1,    # Nose tip
    # 133,  # Right inner eye
    # 362,  # Left inner eye
]

face_model_pnp = np.asarray(
    [face_model_all[i] for i in landmarks3d_ids_pnp], dtype=np.float64
)

landmarks_ids_pnp = [
    1,   # Right outer eye
    11,  # Lef outer eye
    15,   # Right outer mouth
    17,  # Left outer mouth
    23,  # Chin
    0,    # Nose tip
    # 4,  # Right inner eye
    # 8,  # Left inner eye
]

# Initialize SQPnP solver once
sqpnp_solver = sqpnp_python.SQPnPSolver()

# Extract camera parameters for SQPnP from your calibration
fx, fy = cam_info["K"][0, 0], cam_info["K"][1, 1]
cx, cy = cam_info["K"][0, 2], cam_info["K"][1, 2]

# Handle distortion coefficients - OpenCV format is [k1, k2, p1, p2, k3]
# But SQPnP expects [k1, k2, k3, k4, p1, p2]
dist_coeffs = cam_info["dist"]#[0]  # [k1, k2, p1, p2, k3]

# Reorder to SQPnP format: [k1, k2, k3, k4, p1, p2]
k1, k2 = dist_coeffs[0], dist_coeffs[1]
p1, p2 = dist_coeffs[2], dist_coeffs[3]
k3 = 0.0 #dist_coeffs[4]
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


def draw_axes(img, rvec, tvec, K, D,roll=0, pitch=0, yaw=0, axis_length=50.0):
    """Draws the 3D coordinate axes on the image."""
    axis_points_3d = np.float32(
        [[0, 0, 0], [axis_length, 0, 0], [0, axis_length, 0], [0, 0, axis_length]]
    )
    imgpts, _ = cv2.projectPoints(axis_points_3d, rvec, tvec, K, D)

    imgpts = np.int32(imgpts).reshape(-1, 2)
    
    print("3d:", axis_points_3d.reshape(-1, 3))
    print("2d:", imgpts)
    print("==========")

    origin = tuple(imgpts[0])
    cv2.line(img, origin, tuple(imgpts[1]), (0, 0, 255), 3)  # X-axis: Red
    cv2.line(img, origin, tuple(imgpts[2]), (0, 255, 0), 3)  # Y-axis: Green
    cv2.line(img, origin, tuple(imgpts[3]), (255, 0, 0), 3)  # Z-axis: Blue
    cv2.putText(img, f"R: {roll * 180 / math.pi:.2f}, P: {pitch * 180 / math.pi:.2f}, Y: {yaw * 180 / math.pi:.2f}", (origin[0] + 10, origin[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

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
    
    # print("R_custom_to_mp:\n", R_custom_to_mp)
    # print("T_custom_to_mp:\n", T_custom_to_mp)
    # exit(0)

    detection_model = DetectionModel(model_path="models/detection/2025_01_03_03.onnx", 
                        device="cuda", 
                        classes=[
                            'human',
                            'face',
                            'seatbelt_on',
                            'seatbelt_off',
                            'hod_on',
                            'hod_off',
                            'phone',
                            'child',
                            'child_face'], 
                        conf_threshold=0.6,
                        class_agnostic=0.7,
                        use_min_ratio=True) 
    face_id = 1
    
    faciallandmark_model = FacialLandmark2DModel(model_path="models/facial_landmark_2d/best_nme.onnx",
                                                 device="cuda",
                                                 apply_all_optim=True,
                                                 scale=1.2)

    # Initialize MediaPipe Face Mesh and webcam
    # mp_face_mesh = mp.solutions.face_mesh
    # face_mesh = mp_face_mesh.FaceMesh(
    #     max_num_faces=1,
    #     refine_landmarks=True,
    #     min_detection_confidence=0.5,
    #     min_tracking_confidence=0.5,
    # )
    video_path = "head_pose.mp4"
    
    output_path = "head_pose_output_faciallandmark.mp4"
    writer = None
    
    cap = cv2.VideoCapture(video_path)

    print(f"Starting head pose estimation with SQPnP")
    print(f"Using {'all' if args.use_all_points else 'subset'} landmarks")
    print(f"CPU measurement: {'enabled' if args.measure_cpu else 'disabled'}")
    print(f"FPS display: {'enabled' if args.show_fps else 'disabled'}")

    # --- 2. Real-time Loop ---

    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            break
            # continue
            
        height, width, _ = image.shape

        # Flip for selfie view, and convert BGR to RGB for MediaPipe
        image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
        bbox_dict = detection_model.pipeline(image)
        if face_id in bbox_dict.keys():
            face_bboxes = bbox_dict[face_id]
            
            min_idx = -1
            if len(face_bboxes) > 0:
                for i, bbox in enumerate(face_bboxes):
                    if min_idx == -1 or bbox[0] < face_bboxes[min_idx][0]:
                        min_idx = i
                        
            if min_idx != -1:
                original_frame = image.copy()
                bbox = face_bboxes[min_idx]
                x_min, y_min, x_max, y_max, _ = bbox
                x_min = int(x_min)
                y_min = int(y_min)
                x_max = int(x_max)
                y_max = int(y_max)
                landmarks_2D = faciallandmark_model.pipeline(original_frame, bbox)
                landmarks_2d_pnp = []
                
                for i, landmark in enumerate(landmarks_2D):
                    if i in landmarks_ids_pnp:
                        landmarks_2d_pnp.append(landmark[:2])
                        cv2.circle(image, (int(landmark[0]), int(landmark[1])), 3, (0, 255, 0), -1)
                
                landmarks_2d_pnp = np.asarray(landmarks_2d_pnp, dtype=np.float32)

                if len(landmarks_2d_pnp) > 0:

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
                        
                        # print(R_final.shape, rvec_final.shape, t_final.shape)
                        # exit(0)

                        sy = math.sqrt(R_final[0][0] * R_final[0][0] + R_final[1][0] * R_final[1][0])
                        pitch = math.atan2(-R_final[2][0], sy)#;
                        if (sy >= 1e-6):
        
                            roll = math.atan2(R_final[2][1], R_final[2][2])
                            yaw = math.atan2(R_final[1][0], R_final[0][0])
                        else:
                            roll = math.atan2(-R_final[1][2], R_final[1][1])
                            yaw = 0


                        # --- 5. Draw the custom axes on the image ---
                        draw_axes(image, rvec_final, t_final, cam_matrix, dist_coeffs, roll, pitch, yaw)
                    else:
                        print("SQPnP failed to find a solution")
                else:
                    print("No face landmarks detected")
            
        # Update FPS
        fps_counter += 1
        if time.time() - fps_start_time >= 1.0:
            fps = fps_counter / (time.time() - fps_start_time)
            fps_counter = 0
            fps_start_time = time.time()

        cv2.putText(image, "FPS: %d" % fps, (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        # Draw performance information
        # if args.show_fps:
        #     draw_performance_info(image, fps, sqpnp_time, cpu_cycles)

        if writer is None:
            # Initialize video writer if not already done
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(output_path, fourcc, 30.0, (width, height))
        # Write the frame to the output video
        writer.write(image)
        
        # cv2.imshow("Custom Head Pose Estimation (SQPnP)", image)
        # if cv2.waitKey(5) & 0xFF == 27:  # Press ESC to exit
        #     break

    # --- 6. Cleanup and Performance Summary ---
    # face_mesh.close()
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
            
    print("Done! Output video saved to:", output_path)

if __name__ == "__main__":
    main()
