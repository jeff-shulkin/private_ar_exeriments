#!/usr/bin/env python3
"""
pii_detect.py
Iterates through CMU Panoptic experiment directories, reads Kinect RGB video and
depthData.dat streams (per node), and displays them side by side.
Optimized to stream large depth files frame-by-frame to avoid OOM.
"""

import cv2
import numpy as np
from pathlib import Path
import argparse

# ---------------------------------------------------------------------
# Kinect depth resolution constants
# (for Kinect v2 used in CMU Panoptic dataset)
# ---------------------------------------------------------------------
DEPTH_W, DEPTH_H = 512, 424
FRAME_SIZE = DEPTH_W * DEPTH_H * 2  # uint16 -> 2 bytes per pixel

# ---------------------------------------------------------------------
# Utility: load directories and match depth/video data
# ---------------------------------------------------------------------
def get_kinect_nodes(experiment_path):
    depth_root = experiment_path / "kinect_shared_depth"
    video_root = experiment_path / "kinectVideos"

    if not depth_root.exists() or not video_root.exists():
        print(f"[WARN] Missing Kinect folders in {experiment_path}")
        return []

    nodes = []
    for node_dir in sorted(depth_root.iterdir()):
        if not node_dir.is_dir():
            continue
        node_name = node_dir.name
        depth_path = node_dir / "depthData.dat"

        rgb_candidates = list(video_root.glob(f"*_{int(node_name[-1]):02d}.mp4"))
        if not rgb_candidates:
            print(f"[WARN] No matching RGB video for {node_name}")
            continue
        rgb_path = rgb_candidates[0]

        nodes.append((node_name, rgb_path, depth_path))
    return nodes

# ---------------------------------------------------------------------
# Streaming depth reader utilities
# ---------------------------------------------------------------------
def load_depth_stream(depth_path):
    """Open a depth .dat file for streaming"""
    if not depth_path.exists():
        raise FileNotFoundError(f"Depth file not found: {depth_path}")
    return open(depth_path, "rb")

def read_next_depth_frame(f):
    """Read the next 512x424 uint16 frame; return None if EOF"""
    frame_bytes = f.read(FRAME_SIZE)
    if len(frame_bytes) < FRAME_SIZE:
        return None
    depth_frame = np.frombuffer(frame_bytes, dtype=np.uint16).reshape((DEPTH_H, DEPTH_W))
    return depth_frame

# ---------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------
def display_pair(rgb_path, depth_path, node_name):
    """Display RGB and corresponding depth frames side by side (streaming mode)"""
    print(f"[INFO] Displaying {node_name}:")
    cap = cv2.VideoCapture(str(rgb_path))
    depth_file = load_depth_stream(depth_path)

    while True:
        ret, rgb_frame = cap.read()
        if not ret:
            break

        depth_frame = read_next_depth_frame(depth_file)
        if depth_frame is None:
            break

        # Normalize and colorize depth
        max_depth = np.max(depth_frame)
        if max_depth > 0:
            depth_vis = (depth_frame / max_depth * 255).astype(np.uint8)
        else:
            depth_vis = np.zeros_like(depth_frame, dtype=np.uint8)
        depth_vis = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)

        # Resize RGB to match depth
        rgb_frame_resized = cv2.resize(rgb_frame, (DEPTH_W, DEPTH_H))
        combined = np.hstack((rgb_frame_resized, depth_vis))

        cv2.imshow(f"{node_name} RGB + Depth", combined)
        key = cv2.waitKey(10) & 0xFF
        if key == ord("q"):
            break

    cap.release()
    depth_file.close()
    cv2.destroyAllWindows()

# ---------------------------------------------------------------------
# Experiment folder processing
# ---------------------------------------------------------------------
def process_experiment_folder(experiment_path, target_node=None):
    print(f"\n=== Processing: {experiment_path.name} ===")
    kinect_nodes = get_kinect_nodes(experiment_path)
    if not kinect_nodes:
        print(f"[WARN] No Kinect nodes found in {experiment_path}")
        return

    # Filter to a specific node if requested
    if target_node:
        kinect_nodes = [n for n in kinect_nodes if n[0] == target_node]
        if not kinect_nodes:
            print(f"[ERROR] Node {target_node} not found in {experiment_path}")
            return

    for node_name, rgb_path, depth_path in kinect_nodes:
        display_pair(rgb_path, depth_path, node_name)

# ---------------------------------------------------------------------
# Main driver
# ---------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Display RGB + Depth pairs for CMU Panoptic Kinect nodes.")
    parser.add_argument(
        "experiment_path",
        type=str,
        nargs="?",
        default=None,
        help="Optional: specific experiment folder (e.g., videos/171204_pose1)"
    )
    parser.add_argument(
        "--node",
        type=str,
        default=None,
        help="Optional: specify Kinect node (e.g., KINECTNODE1)"
    )
    args = parser.parse_args()

    base_dir = Path("videos")

    # If a specific experiment is provided
    if args.experiment_path:
        experiment_path = Path(args.experiment_path)
        if not experiment_path.exists():
            print(f"[ERROR] Experiment path not found: {experiment_path}")
            return
        process_experiment_folder(experiment_path, target_node=args.node)

    # Otherwise, iterate through all experiments in videos/
    else:
        if not base_dir.exists():
            print("[ERROR] 'videos' directory not found!")
            return
        for experiment_path in sorted(base_dir.iterdir()):
            if experiment_path.is_dir():
                process_experiment_folder(experiment_path, target_node=args.node)


if __name__ == "__main__":
    main()
