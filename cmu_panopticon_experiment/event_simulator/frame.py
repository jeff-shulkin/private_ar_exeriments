import glob
import os
import cv2
from tqdm import tqdm
import numpy as np
from pqdm.processes import pqdm
import csv
from typing import Optional, Tuple

ORIG_SHAPE = (1408, 1408)
SEG_SHAPE = (512, 512)
EVT_SHAPE = (320, 320)

def convert_frame(frame: np.ndarray, shape=(1408, 1408)) -> np.ndarray:
    frame = np.frombuffer(frame, dtype=np.uint8)
    img = cv2.imdecode(frame, cv2.IMREAD_COLOR)
    img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    img = cv2.resize(img, shape, interpolation=cv2.INTER_LINEAR)
    img = img.astype(np.uint8)
    return img

def process_frames(path: str) -> None:
    shape = SEG_SHAPE
    vrs_path = os.path.join(path, 'recording_head', 'data', 'camera-rgb.vrs')
    output_dir = os.path.join(path, f'camera-rgb-{shape[0]}x{shape[1]}')

    if not os.path.exists(vrs_path):
        raise FileNotFoundError(f'VRS file not found: {vrs_path}')
    
    # check if output directory exists, if so return early
    if os.path.exists(output_dir):
        return 

    os.makedirs(output_dir, exist_ok=True)
    
    from pyvrs.reader import SyncVRSReader
    reader = SyncVRSReader(vrs_path).filtered_by_fields(stream_ids='214-1', record_types='data')
    
    csv_path = os.path.join(output_dir, 'timestamps.csv')
    with open(csv_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['timestamp', 'png_filename'])

        for i, record in enumerate(tqdm(reader, desc="Processing frames")):
            frame = convert_frame(record.image_blocks[0], shape)
            timestamp = record.timestamp
            
            png_filename = f'{i:05d}.png'
            png_path = os.path.join(output_dir, png_filename)
            
            cv2.imwrite(png_path, frame)
            writer.writerow([timestamp, png_filename])
    
    reader.close()
    del reader


def process_frames_all(ds_path: str, use_pqdm: bool = False, n_jobs: int = 12) -> None:
    all_paths = glob.glob(os.path.join(ds_path, "*"))
    if use_pqdm:
        pqdm(all_paths, process_frames, n_jobs=n_jobs)
    else:
        for path in tqdm(all_paths):
            try:
                process_frames(path)
            except FileNotFoundError:
                print(f"File not found for path: {path}")
    

def write_video(path: str, out_path: str) -> bool:
    """
    Write a video from a folder containing PNG files and timestamps.csv
    
    Args:
        path: Path to the folder containing PNG files and timestamps.csv
        out_path: Path where the output MP4 video will be saved
        
    Returns:
        bool: True if video was successfully created, False otherwise
    """
    csv_path = os.path.join(path, 'timestamps.csv')
    
    if not os.path.exists(csv_path):
        print(f"timestamps.csv not found in {path}")
        return False
    
    try:
        timestamps = []
        filenames = []
        
        with open(csv_path, 'r') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                timestamps.append(float(row['timestamp']))
                filenames.append(row['png_filename'])
        
        if len(timestamps) < 2:
            print("Not enough frames to create video")
            return False
        
        duration = timestamps[-1] - timestamps[0]
        if duration <= 0:
            print("Invalid duration")
            return False
        
        fps = round(len(timestamps) / duration)
        print(f"Creating video with {len(filenames)} frames at {fps} FPS")
        
        # Read first frame to get dimensions
        first_frame_path = os.path.join(path, filenames[0])
        first_frame = cv2.imread(first_frame_path)
        if first_frame is None:
            print(f"Could not read first frame: {first_frame_path}")
            return False
        
        h, w = first_frame.shape[:2]
        
        # Create output directory if needed
        os.makedirs(os.path.dirname(out_path) if os.path.dirname(out_path) else '.', exist_ok=True)
        
        # Initialize video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))
        
        if not writer.isOpened():
            print("Failed to open video writer")
            return False
        
        # Write first frame
        writer.write(first_frame)
        
        # Process and write remaining frames
        for filename in tqdm(filenames[1:], desc="Writing video"):
            frame_path = os.path.join(path, filename)
            frame = cv2.imread(frame_path)
            
            if frame is not None:
                writer.write(frame)
            else:
                print(f"Warning: Could not read frame {frame_path}")
        
        writer.release()
        
        if os.path.exists(out_path):
            print(f"Video successfully created: {out_path}")
            return True
        else:
            print("Failed to create video file")
            return False
            
    except Exception as e:
        print(f"Error creating video: {e}")
        return False


def load_video_frames(path: str, start: float = 0.0, end: Optional[float] = None, grayscale: bool = False, target_fs: int = 30, target_shape: Tuple[int, int] = (320, 320)) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Load and decode video frames from PNG files within a specified time range.
    
    Args:
        path: Path to the directory containing PNG files and timestamps.csv
        start: Start time in seconds (relative to first frame, default: 0.0)
        end: End time in seconds (relative to first frame, default: None for all frames)
        
    Returns:
        Tuple containing:
        - frames: Array of decoded frames (N, 320, 320) or (N, 320, 320, 3)
        - selected_ts: Relative timestamps for selected frames
        - t0: Absolute timestamp of the first frame
    """
    csv_path = os.path.join(path, 'timestamps.csv')
    
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f'timestamps.csv not found: {csv_path}')
    
    timestamps = []
    filenames = []
    
    with open(csv_path, 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            timestamps.append(float(row['timestamp']))
            filenames.append(row['png_filename'])
    
    timestamps = np.array(timestamps)
    t0 = timestamps[0]
    rel_ts = timestamps - t0
    
    if end is None:
        end = rel_ts[-1]
    
    mask = (rel_ts >= start) & (rel_ts <= end)
    selected_ts = rel_ts[mask]
    selected_files = [filenames[i] for i, m in enumerate(mask) if m]
    
    fs = 1 / np.mean(np.diff(selected_ts))
    downsample_factor = int(np.round(fs / target_fs))
    
    selected_ts = selected_ts[::downsample_factor]
    selected_files = selected_files[::downsample_factor]
    
    print(f"Loading {len(selected_files)} frames from {path} with target fs={target_fs} (original fs={fs})")
    
    frames = []
    for i, filename in tqdm(enumerate(selected_files), desc="Loading frames"):
        img_path = os.path.join(path, filename)
        img = cv2.imread(img_path)
        img = cv2.resize(img, target_shape)
        if grayscale:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        frames.append(img)
    
    return np.array(frames), selected_ts, t0


def main():
    process_frames_all('dataset/nymeria', use_pqdm=True, n_jobs=2)
    # process_frames('dataset/nymeria/20230929_s1_alan_burns_act4_zch0c7')

if __name__ == "__main__":
    main()