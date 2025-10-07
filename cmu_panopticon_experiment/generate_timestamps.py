#!/usr/bin/env python3
"""
Recursive Batch Event Generation for Kinect Images

Automatically finds all kinectImgs folders under videos/*subset*/kinectImgs
and runs event_generator_pqdm.py on each. If timestamps.csv is missing,
it uses slomo_generator.py to generate it first.

Features:
- Automatically detects CPU cores for parallel processing
- FPS configurable via command-line argument
- Outputs organized by video folder
"""

import os
import subprocess
from pathlib import Path
import argparse

# ---------------------- ARGPARSE ---------------------- #
parser = argparse.ArgumentParser(description="Batch event generation for Kinect images")
parser.add_argument("--fps", type=float, default=30.0, help="Output FPS for frame timestamps")
args = parser.parse_args()

FPS = args.fps
WORKERS = os.cpu_count()  # automatically use all available cores
GROUP_SIZE = 500

EVENT_GENERATOR_SCRIPT = Path("./event_simulator/event_generator_pqdm.py")
SLOMO_GENERATOR_SCRIPT = Path("./event_simulator/slomo_generator.py")
VIDEOS_ROOT = Path("./videos")

if not EVENT_GENERATOR_SCRIPT.exists():
    raise RuntimeError(f"event_generator_pqdm.py not found at {EVENT_GENERATOR_SCRIPT}")
if not SLOMO_GENERATOR_SCRIPT.exists():
    raise RuntimeError(f"slomo_generator.py not found at {SLOMO_GENERATOR_SCRIPT}")

# ---------------------- FIND ALL kinectImgs ---------------------- #
kinect_img_folders = sorted(VIDEOS_ROOT.glob("*/kinectImgs/*"))
if not kinect_img_folders:
    raise RuntimeError("No kinectImgs folders found under videos/*/*")

# ---------------------- PROCESS EACH FOLDER ---------------------- #
for img_folder in kinect_img_folders:
    subset_folder = img_folder.parent.parent
    video_name = img_folder.name
    print(f"\nProcessing folder: {img_folder} (subset: {subset_folder.name})")

    # Ensure timestamps.csv exists
    timestamps_csv = img_folder / "timestamps.csv"
    if not timestamps_csv.exists():
        print(f"timestamps.csv not found in {img_folder}, generating...")
        cmd_ts = [
            "python", str(SLOMO_GENERATOR_SCRIPT),
            str(img_folder),
            str(img_folder),   # output dir same as input for timestamps
            "dummy.mp4",      # mp4 arg required but ignored in timestamps-only mode
            "--timestamps_only"
        ]
        subprocess.run(cmd_ts, check=True)
        print(f"Generated timestamps.csv at {timestamps_csv}")

    # Setup event output directories
    video_output_dir = subset_folder / "kinectEventVideos" / video_name
    events_output_dir = video_output_dir / "events"
    events_output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------ Run event_generator_pqdm.py ------------------ #
    cmd = [
        "python", str(EVENT_GENERATOR_SCRIPT),
        str(img_folder), str(events_output_dir),
        "--group_size", str(GROUP_SIZE),
        "--workers", str(WORKERS),
        "--out_fps", str(FPS),
        "--mp4_output", f"{video_name}.mp4"
    ]
    print(f"Running event generator for {video_name} with {WORKERS} workers...")
    subprocess.run(cmd, check=True)
    print(f"Events saved in {events_output_dir}")
