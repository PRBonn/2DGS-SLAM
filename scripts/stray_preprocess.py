"""Preprocess StrayScanner (iPhone RGB-D) recordings for 2DGS-SLAM.

A StrayScanner recording directory typically contains:
    rgb.mp4            – HEVC RGB video (e.g. 1920x1280)
    depth/             – 16-bit PNG depth maps in millimeters (256x192)
    confidence/        – per-pixel confidence (0/1/2), optional
    odometry.csv       – ARKit poses + per-frame intrinsics
    camera_matrix.csv  – 3x3 intrinsic matrix (last frame, at RGB resolution)

This script converts a recording into the format expected by 2DGS-SLAM:
    <output>/rgb/frame_XXXXX.png
    <output>/depth/frame_XXXXX.TIFF   (float32 meters)
    <output>/gt_pose.txt              (TUM-style: ts tx ty tz qx qy qz qw)
    <output>/intrinsic.txt            (fx fy cx cy at output resolution)

Usage:
    python scripts/stray_preprocess.py --input <stray_scene> --output <processed>
    python scripts/stray_preprocess.py --input <stray_scene> --output <processed> --resolution 640 480
    python scripts/stray_preprocess.py --input <stray_scene> --output <processed> --confidence-threshold 1
"""

import argparse
import csv
import os
import sys

import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm


def read_camera_matrix(path):
    """Read 3x3 intrinsic matrix from camera_matrix.csv."""
    return np.loadtxt(path, delimiter=",")


def read_odometry(path):
    """Read odometry.csv and return list of dicts with parsed fields.

    Expected header (may vary slightly):
        timestamp, frame, x, y, z, qx, qy, qz, qw, fx, fy, cx, cy, ...
    """
    rows = []
    with open(path, "r") as f:
        reader = csv.reader(f)
        header = [h.strip() for h in next(reader)]
        for line in reader:
            if len(line) < 9:
                continue
            entry = {}
            for i, col in enumerate(header):
                if i < len(line):
                    entry[col] = line[i].strip()
            rows.append(entry)
    return header, rows


def build_pose_matrix(row):
    """Build a 4x4 camera-to-world matrix from an odometry row."""
    x, y, z = float(row["x"]), float(row["y"]), float(row["z"])
    qx, qy, qz, qw = (
        float(row["qx"]),
        float(row["qy"]),
        float(row["qz"]),
        float(row["qw"]),
    )
    T = np.eye(4)
    T[:3, :3] = R.from_quat([qx, qy, qz, qw]).as_matrix()
    T[:3, 3] = [x, y, z]
    return T


def main():
    parser = argparse.ArgumentParser(description="Preprocess StrayScanner recording for 2DGS-SLAM")
    parser.add_argument("--input", "-i", required=True, help="Path to StrayScanner recording directory")
    parser.add_argument("--output", "-o", required=True, help="Path to output processed directory")
    parser.add_argument(
        "--resolution",
        nargs=2,
        type=int,
        default=None,
        metavar=("W", "H"),
        help="Output resolution (width height). Default: use depth resolution (256x192)",
    )
    parser.add_argument(
        "--confidence-threshold",
        type=int,
        default=1,
        choices=[0, 1, 2],
        help="Minimum confidence to keep a depth pixel (0=keep all, 1=medium+high, 2=high only). Default: 1",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Maximum number of frames to extract (default: all)",
    )
    parser.add_argument(
        "--frame-step",
        type=int,
        default=1,
        help="Extract every N-th frame (default: 1 = all frames)",
    )
    args = parser.parse_args()

    input_dir = args.input
    output_dir = args.output

    # --- Validate inputs ---
    video_path = os.path.join(input_dir, "rgb.mp4")
    depth_dir = os.path.join(input_dir, "depth")
    confidence_dir = os.path.join(input_dir, "confidence")
    odometry_path = os.path.join(input_dir, "odometry.csv")
    camera_matrix_path = os.path.join(input_dir, "camera_matrix.csv")

    if not os.path.isfile(video_path):
        print(f"Error: rgb.mp4 not found at {video_path}", file=sys.stderr)
        sys.exit(1)
    if not os.path.isdir(depth_dir):
        print(f"Error: depth/ directory not found at {depth_dir}", file=sys.stderr)
        sys.exit(1)
    if not os.path.isfile(odometry_path):
        print(f"Error: odometry.csv not found at {odometry_path}", file=sys.stderr)
        sys.exit(1)

    has_confidence = os.path.isdir(confidence_dir)
    has_camera_matrix = os.path.isfile(camera_matrix_path)

    # --- Read odometry ---
    header, odom_rows = read_odometry(odometry_path)
    print(f"Odometry: {len(odom_rows)} entries")

    # --- Read depth file list ---
    depth_files = sorted(
        [f for f in os.listdir(depth_dir) if f.endswith(".png")],
        key=lambda x: int(os.path.splitext(x)[0]),
    )
    print(f"Depth frames: {len(depth_files)}")

    # --- Open video ---
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: cannot open {video_path}", file=sys.stderr)
        sys.exit(1)

    video_fps = cap.get(cv2.CAP_PROP_FPS)
    video_frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    video_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Video: {video_w}x{video_h} @ {video_fps:.1f} fps, {video_frame_count} frames")

    # --- Determine number of usable frames ---
    n_frames = min(len(odom_rows), len(depth_files), video_frame_count)
    if args.max_frames is not None:
        n_frames = min(n_frames, args.max_frames * args.frame_step)
    print(f"Usable frames: {n_frames}")

    # --- Determine output resolution ---
    sample_depth = cv2.imread(
        os.path.join(depth_dir, depth_files[0]), cv2.IMREAD_UNCHANGED
    )
    depth_h, depth_w = sample_depth.shape[:2]
    print(f"Native depth resolution: {depth_w}x{depth_h}")

    if args.resolution is not None:
        out_w, out_h = args.resolution
    else:
        out_w, out_h = depth_w, depth_h
    print(f"Output resolution: {out_w}x{out_h}")

    # --- Determine intrinsics ---
    # Prefer per-frame intrinsics from odometry.csv if available
    has_per_frame_intrinsics = all(
        k in header for k in ["fx", "fy", "cx", "cy"]
    )
    if has_per_frame_intrinsics:
        print("Using per-frame intrinsics from odometry.csv")
        # Use first frame's intrinsics as reference (they're at RGB resolution)
        ref_fx = float(odom_rows[0]["fx"])
        ref_fy = float(odom_rows[0]["fy"])
        ref_cx = float(odom_rows[0]["cx"])
        ref_cy = float(odom_rows[0]["cy"])
    elif has_camera_matrix:
        print("Using intrinsics from camera_matrix.csv")
        K = read_camera_matrix(camera_matrix_path)
        ref_fx, ref_fy = K[0, 0], K[1, 1]
        ref_cx, ref_cy = K[0, 2], K[1, 2]
    else:
        print("Error: no intrinsics source found", file=sys.stderr)
        sys.exit(1)

    # Scale intrinsics from RGB resolution to output resolution
    scale_x = out_w / video_w
    scale_y = out_h / video_h
    out_fx = ref_fx * scale_x
    out_fy = ref_fy * scale_y
    out_cx = ref_cx * scale_x
    out_cy = ref_cy * scale_y
    print(f"Scaled intrinsics: fx={out_fx:.2f} fy={out_fy:.2f} cx={out_cx:.2f} cy={out_cy:.2f}")

    # --- Create output directories ---
    os.makedirs(os.path.join(output_dir, "rgb"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "depth"), exist_ok=True)

    # --- Process frames ---
    fake_fps = 20  # fake timestamp spacing for offline SLAM
    initial_timestamp = 0.0
    frame_idx = 0

    gt_pose_path = os.path.join(output_dir, "gt_pose.txt")
    with open(gt_pose_path, "w") as f_pose:
        f_pose.write("# timestamp tx ty tz qx qy qz qw\n")

    for raw_idx in tqdm(range(0, n_frames, args.frame_step), desc="Processing"):
        # Read video frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, raw_idx)
        ret, rgb_frame = cap.read()
        if not ret:
            print(f"Warning: failed to read video frame {raw_idx}, skipping")
            continue

        # Read depth
        depth_path = os.path.join(depth_dir, depth_files[raw_idx])
        depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
        if depth is None:
            print(f"Warning: failed to read depth {depth_files[raw_idx]}, skipping")
            continue

        # Apply confidence filter
        if has_confidence and args.confidence_threshold > 0:
            conf_name = depth_files[raw_idx]  # same naming convention
            conf_path = os.path.join(confidence_dir, conf_name)
            if os.path.isfile(conf_path):
                conf = cv2.imread(conf_path, cv2.IMREAD_UNCHANGED)
                if conf is not None and conf.shape == depth.shape:
                    depth[conf < args.confidence_threshold] = 0

        # Convert depth to float32 meters
        depth_m = depth.astype(np.float32) / 1000.0

        # Resize RGB to output resolution
        if (rgb_frame.shape[1], rgb_frame.shape[0]) != (out_w, out_h):
            rgb_frame = cv2.resize(
                rgb_frame, (out_w, out_h), interpolation=cv2.INTER_CUBIC
            )

        # Resize depth to output resolution (nearest-neighbor to avoid interpolation artifacts)
        if (depth_m.shape[1], depth_m.shape[0]) != (out_w, out_h):
            depth_m = cv2.resize(
                depth_m, (out_w, out_h), interpolation=cv2.INTER_NEAREST
            )

        # Get pose from odometry
        row = odom_rows[raw_idx]
        T_wc = build_pose_matrix(row)

        # Write RGB
        rgb_out = os.path.join(
            output_dir, "rgb", f"frame_{str(frame_idx).zfill(5)}.png"
        )
        cv2.imwrite(rgb_out, rgb_frame)

        # Write depth as float32 TIFF (meters)
        depth_out = os.path.join(
            output_dir, "depth", f"frame_{str(frame_idx).zfill(5)}.TIFF"
        )
        cv2.imwrite(depth_out, depth_m)

        # Write pose in TUM format: timestamp tx ty tz qx qy qz qw
        timestamp = initial_timestamp + frame_idx / fake_fps
        trans = T_wc[:3, 3]
        quat = R.from_matrix(T_wc[:3, :3]).as_quat()  # [qx, qy, qz, qw]
        pose_line = f"{timestamp:.4f}"
        for t in trans:
            pose_line += f" {t:.9f}"
        for q in quat:
            pose_line += f" {q:.9f}"

        with open(gt_pose_path, "a") as f_pose:
            f_pose.write(pose_line + "\n")

        frame_idx += 1

    cap.release()

    # --- Write intrinsics file ---
    intrinsic_path = os.path.join(output_dir, "intrinsic.txt")
    with open(intrinsic_path, "w") as f:
        f.write(f"{out_fx:.6f} 0.0 {out_cx:.6f} 0.0\n")
        f.write(f"0.0 {out_fy:.6f} {out_cy:.6f} 0.0\n")
        f.write(f"0.0 0.0 1.0 0.0\n")
        f.write(f"0.0 0.0 0.0 1.0\n")

    print(f"\nDone! Processed {frame_idx} frames → {output_dir}")
    print(f"Resolution: {out_w}x{out_h}")
    print(f"Intrinsics: fx={out_fx:.2f} fy={out_fy:.2f} cx={out_cx:.2f} cy={out_cy:.2f}")
    print(f"\nNext steps:")
    print(f"  1. Create a config YAML (see configs/stray/base_config.yaml)")
    print(f"  2. Set Dataset.dataset_path to: {os.path.abspath(output_dir)}")
    print(f"  3. Set Calibration values: fx={out_fx:.2f} fy={out_fy:.2f} cx={out_cx:.2f} cy={out_cy:.2f}")
    print(f"     width={out_w} height={out_h} depth_scale=1.0")
    print(f"  4. Run: python slam.py --config configs/stray/<your_scene>.yaml -v")


if __name__ == "__main__":
    main()
