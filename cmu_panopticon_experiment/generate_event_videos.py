#!/usr/bin/env python3
"""
Recursive Batch Event Generation for Kinect MP4 Videos

Automatically finds all MP4 files under videos/*subset*/kinectVideos
and runs event_generator_pqdm.py on each.

Features:
- Automatically detects CPU cores for parallel processing
- FPS configurable via command-line argument
- Outputs organized by video folder

Example usage:
    python batch_event_gen_recursive.py --fps 30
"""

import os
import cv2
import pandas as pd
import subprocess
from pathlib import Path
import argparse

# ---------------------- ARGPARSE ---------------------- #
parser = argparse.ArgumentParser(description="Batch event generation for all Kinect MP4s")
parser.add_argument("--fps", type=float, default=30.0, help="Output FPS for frame timestamps")
args = parser.parse_args()

FPS = args.fps
WORKERS = os.cpu_count()  # automatically use all available cores
GROUP_SIZE = 500

EVENT_GENERATOR_SCRIPT = Path("./event_simulator/event_generator_pqdm.py")
VIDEOS_ROOT = Path("./videos")

if not EVENT_GENERATOR_SCRIPT.exists():
    raise RuntimeError(f"event_generator_pqdm.py not found at {EVENT_GENERATOR_SCRIPT}")

# ---------------------- FIND ALL kinectVideos ---------------------- #
kinect_video_folders = sorted(VIDEOS_ROOT.glob("*/kinectVideos"))
if not kinect_video_folders:
    raise RuntimeError("No kinectVideos folders found under videos/*/*")

# ---------------------- PROCESS EACH FOLDER ---------------------- #
for kv_folder in kinect_video_folders:
    subset_folder = kv_folder.parent
    print(f"\nProcessing kinectVideos folder: {kv_folder} (subset: {subset_folder.name})")
    mp4_files = sorted(kv_folder.glob("*.mp4"))
    if not mp4_files:
        print(f"No MP4 files in {kv_folder}, skipping...")
        continue

    for mp4_file in mp4_files:
        video_name = mp4_file.stem
        print(f"\nProcessing video: {mp4_file.name}")

        video_output_dir = subset_folder / "kinectEventVideos" / video_name
        frames_dir = video_output_dir / "frames"
        events_output_dir = video_output_dir / "events"
        frames_dir.mkdir(parents=True, exist_ok=True)
        events_output_dir.mkdir(parents=True, exist_ok=True)

        # ------------------ Extract frames ------------------ #
        cap = cv2.VideoCapture(str(mp4_file))
        video_fps = cap.get(cv2.CAP_PROP_FPS) or FPS
        frame_idx = 0
        timestamps = []
        frame_files = []

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame_file = frames_dir / f"{frame_idx:06d}.png"
            cv2.imwrite(str(frame_file), gray)
            frame_files.append(frame_file.name)
            timestamps.append(frame_idx / video_fps)
            frame_idx += 1
        cap.release()
        print(f"Extracted {frame_idx} frames from {mp4_file.name}")

        # ------------------ Generate timestamps.csv ------------------ #
        df = pd.DataFrame({
            "png_filename": frame_files,
            "timestamp": timestamps
        })
        csv_path = frames_dir / "timestamps.csv"
        df.to_csv(csv_path, index=False)
        print(f"Saved timestamps.csv at {csv_path}")

        # ------------------ Run event_generator_pqdm.py ------------------ #
        cmd = [
            "python", str(EVENT_GENERATOR_SCRIPT),
            str(frames_dir), str(events_output_dir),
            "--group_size", str(GROUP_SIZE),
            "--workers", str(WORKERS),
            "--out_fps", str(FPS),
            "--mp4_output", str(mp4_file.name)
        ]
        print(f"Running event generator for {video_name} with {WORKERS} workers...")
        subprocess.run(cmd, check=True)
        print(f"Events saved in {events_output_dir}")

