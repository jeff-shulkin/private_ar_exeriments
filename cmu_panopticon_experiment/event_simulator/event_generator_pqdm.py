#!/usr/bin/env python3
"""
Parallel DVS Event Generator with IIR Low Pass Filtering using PQDM

A high-performance parallel version of the DVS event generator that uses pqdm to process
frame groups in parallel for improved efficiency. This version splits input frames into
groups and processes them concurrently across multiple CPU cores.

Key Features:
    - Parallel processing using pqdm for frame groups
    - IIR low pass filtering for realistic photoreceptor simulation
    - Intensity-dependent temporal filtering for natural response
    - Advanced event filtering (AFK, STC, ERC)
    - Configurable group size for parallel processing
    - Efficient PNG/CSV input/output with event visualization

Basic Usage:
    python event_generator_pqdm.py input_dir output_dir

Advanced Usage:
    python event_generator_pqdm.py input_dir output_dir --mp4_output output.mp4 --group_size 1000 --workers 8
"""

import argparse
import math
from collections import deque, defaultdict
import os
from pathlib import Path
from typing import List, Tuple, Dict, Any
import pickle

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
from pqdm.processes import pqdm

# Import shared utilities from the original event_generator
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from event_generator import (
    lin_log, rescale_intensity_frame, low_pass_filter,
    ERCFilter, STCFilter, AFKFilter,
    EventGenerator,
    render_event_frames_and_csv,
    print_filter_params
)


# ================================================================================================
# PARALLEL PROCESSING FUNCTIONS
# ================================================================================================

def process_frame_group(args: Dict[str, Any]) -> Tuple[List[List], List[int], int, int]:
    """
    Process a group of frames in parallel.
    
    This function is designed to be called by pqdm workers. Each worker processes
    a group of frames independently and returns the events and statistics.
    
    Args:
        args: Dictionary containing:
            - frame_paths: List of paths to PNG files
            - timestamps: Corresponding timestamps
            - group_idx: Index of this group
            - start_idx: Starting frame index in the overall sequence
            - end_idx: Ending frame index in the overall sequence
            - generator_params: Parameters for EventGenerator
            - output_frame_count: Total number of output frames
            - frames_per_bin: Frames per output bin
            - out_fps: Output FPS
            - base_log_frame: Initial reference frame (for continuity)
            - lp_log_frame: Initial filtered frame (for continuity)
    
    Returns:
        Tuple of (event_bins, raw_event_counts, start_idx, end_idx)
    """
    # Unpack arguments
    frame_paths = args['frame_paths']
    timestamps = args['timestamps']
    group_idx = args['group_idx']
    start_idx = args['start_idx']
    end_idx = args['end_idx']
    gen_params = args['generator_params']
    output_frame_count = args['output_frame_count']
    frames_per_bin = args['frames_per_bin']
    out_fps = args['out_fps']
    base_log_frame = args.get('base_log_frame')
    lp_log_frame = args.get('lp_log_frame')
    
    # Create local event generator
    generator = EventGenerator(
        pos_thres=gen_params['pos_thres'],
        neg_thres=gen_params['neg_thres'],
        enable_erc=gen_params['enable_erc'],
        enable_stc=gen_params['enable_stc'],
        enable_afk=gen_params['enable_afk'],
        erc_max_rate=gen_params['erc_max_rate'],
        erc_window_size=gen_params['erc_window_size'],
        stc_mode=gen_params['stc_mode'],
        stc_threshold_us=gen_params['stc_threshold_us'],
        afk_patch=gen_params['afk_patch'],
        afk_low_freq=gen_params['afk_low_freq'],
        afk_high_freq=gen_params['afk_high_freq'],
        afk_diff_thresh_s=gen_params['afk_diff_thresh_s'],
        cutoff_hz=gen_params['cutoff_hz'],
        intensity_dependent=gen_params['intensity_dependent']
    )
    
    # Initialize generator state if provided (for continuity between groups)
    if base_log_frame is not None:
        generator.base_log_frame = base_log_frame.copy()
        generator.lp_log_frame = lp_log_frame.copy() if lp_log_frame is not None else base_log_frame.copy()
        generator.frame_count = start_idx
    
    # Initialize local event bins and counts
    event_bins = [[] for _ in range(output_frame_count)]
    raw_event_counts = [0 for _ in range(output_frame_count)]
    
    # Process frames in this group
    for i, (frame_path, timestamp) in enumerate(zip(frame_paths, timestamps)):
        frame_idx = start_idx + i
        
        # Load frame
        frame = cv2.imread(str(frame_path), cv2.IMREAD_GRAYSCALE)
        if frame is None:
            continue
        
        # Determine output bin
        bin_idx = int(np.ceil(frame_idx / frames_per_bin))
        if bin_idx >= output_frame_count:
            break
        
        # Generate events
        result_timestamp = bin_idx / out_fps
        events, raw_count = generator.generate_events(frame, result_timestamp)
        
        # Store events and counts
        if events:
            event_bins[bin_idx].extend(events)
        raw_event_counts[bin_idx] += raw_count
    
    return event_bins, raw_event_counts, start_idx, end_idx


def merge_results(results: List[Tuple], output_frame_count: int) -> Tuple[List[List], List[int]]:
    """
    Merge results from parallel processing.
    
    Args:
        results: List of tuples (event_bins, raw_event_counts, start_idx, end_idx) from workers
        output_frame_count: Total number of output frames
    
    Returns:
        Tuple of merged (event_bins, raw_event_counts)
    """
    # Initialize merged containers
    merged_event_bins = [[] for _ in range(output_frame_count)]
    merged_raw_counts = [0 for _ in range(output_frame_count)]
    
    # Sort results by start_idx to maintain order
    results.sort(key=lambda x: x[2])
    
    # Merge results
    for event_bins, raw_counts, start_idx, end_idx in results:
        for bin_idx in range(output_frame_count):
            if event_bins[bin_idx]:
                merged_event_bins[bin_idx].extend(event_bins[bin_idx])
            merged_raw_counts[bin_idx] += raw_counts[bin_idx]
    
    # Sort events within each bin by timestamp
    for bin_idx in range(output_frame_count):
        if merged_event_bins[bin_idx]:
            merged_event_bins[bin_idx].sort(key=lambda x: x[0])
    
    return merged_event_bins, merged_raw_counts


# ================================================================================================
# MAIN FUNCTION WITH PARALLEL PROCESSING
# ================================================================================================

def main():
    """
    Main processing pipeline with parallel frame group processing:
    1. Parse arguments and validate input
    2. Configure event generation and filtering
    3. Split frames into groups for parallel processing
    4. Process groups in parallel using pqdm
    5. Merge results and render output
    """
    parser = argparse.ArgumentParser(
        description='Parallel DVS Event Generator with PQDM (based on v2e)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Basic usage with parallel processing:
    python event_generator_pqdm.py input_dir output_dir --group_size 1000 --workers 8
    
  High sensitivity with custom slowdown:
    python event_generator_pqdm.py input_dir output_dir --threshold_pos 0.1 --threshold_neg 0.1 --in_slowdown 5
    
  Fast mode (no filters) with parallel processing:
    python event_generator_pqdm.py input_dir output_dir --fast_mode --workers 12
    
  With MP4 video output:
    python event_generator_pqdm.py input_dir output_dir --mp4_output output.mp4 --group_size 500
        """)
    
    # Required Arguments
    parser.add_argument('input_dir',
                       help='Input directory with PNG frames and timestamps.csv')
    parser.add_argument('output_dir',
                       help='Output directory for event PNG frames and timestamps.csv')
    
    # Parallel Processing Options
    parallel_group = parser.add_argument_group('Parallel Processing Options')
    parallel_group.add_argument('--group_size', type=int, default=1000,
                               help='Number of frames per processing group (default: 1000)')
    parallel_group.add_argument('--workers', type=int, default=8,
                               help='Number of parallel workers (default: 8)')
    
    # Output Options
    output_group = parser.add_argument_group('Output Options')
    output_group.add_argument('--mp4_output', type=str,
                             help='Optional MP4 output video file for visualization')
    
    # Core Processing Parameters
    core_group = parser.add_argument_group('Core Processing Parameters')
    core_group.add_argument('--out_fps', type=float, default=30,
                           help='Output video frame rate (default: 30)')
    core_group.add_argument('--in_slowdown', type=float, default=10.0,
                           help='Input frame grouping factor (default: 10.0)')
    core_group.add_argument('--threshold_pos', type=float, default=0.25,
                           help='Positive event threshold (default: 0.25)')
    core_group.add_argument('--threshold_neg', type=float, default=0.25,
                           help='Negative event threshold (default: 0.25)')
    core_group.add_argument('--contrast', type=float, default=16,
                           help='Event visualization contrast (default: 16)')
    
    # IIR Low Pass Filter Parameters
    iir_group = parser.add_argument_group('IIR Low Pass Filter Parameters')
    iir_group.add_argument('--cutoff_hz', type=float, default=0.0,
                          help='IIR low pass filter cutoff frequency (0 = disabled)')
    iir_group.add_argument('--intensity_dependent', action='store_true',
                          help='Use intensity-dependent time constants')
    
    # Filter Control
    filter_group = parser.add_argument_group('Filter Control')
    filter_group.add_argument('--disable_erc', action='store_true',
                             help='Disable Event Rate Controller')
    filter_group.add_argument('--disable_stc', action='store_true',
                             help='Disable Spatio-Temporal Correlation Filter')
    filter_group.add_argument('--disable_afk', action='store_true',
                             help='Disable Anti-Flicker Filter')
    filter_group.add_argument('--fast_mode', action='store_true',
                             help='Disable all filters for maximum speed')
    
    # Event Rate Controller (ERC) Parameters
    erc_group = parser.add_argument_group('Event Rate Controller (ERC) Parameters')
    erc_group.add_argument('--erc_max_rate', type=float, default=1e6,
                          help='Maximum allowed event rate (default: 1,000,000)')
    erc_group.add_argument('--erc_window_size', type=float, default=0.01,
                          help='ERC time window size in seconds (default: 0.01)')
    
    # Spatio-Temporal Correlation (STC) Parameters
    stc_group = parser.add_argument_group('Spatio-Temporal Correlation (STC) Parameters')
    stc_group.add_argument('--stc_mode', type=str, default='STC_CUT_TRAIL',
                          choices=['STC_CUT_TRAIL', 'STC_KEEP_TRAIL', 'TRAIL'],
                          help='STC filter mode (default: STC_CUT_TRAIL)')
    stc_group.add_argument('--stc_threshold_us', type=int, default=1000,
                          help='STC burst detection threshold in microseconds (default: 1000)')
    
    # Anti-Flicker (AFK) Parameters
    afk_group = parser.add_argument_group('Anti-Flicker (AFK) Parameters')
    afk_group.add_argument('--afk_patch', type=int, default=4,
                          help='AFK spatial patch size (default: 4)')
    afk_group.add_argument('--afk_low_freq', type=float, default=49.0,
                          help='AFK lower bound of flicker band (default: 49.0)')
    afk_group.add_argument('--afk_high_freq', type=float, default=51.0,
                          help='AFK upper bound of flicker band (default: 51.0)')
    afk_group.add_argument('--afk_diff_thresh_s', type=float, default=0.002,
                          help='AFK period variation threshold (default: 0.002)')
    
    args = parser.parse_args()
    
    # Load input data
    input_path = Path(args.input_dir)
    if not input_path.exists() or not input_path.is_dir():
        raise ValueError(f"Input directory not found: {args.input_dir}")
    
    csv_path = input_path / "timestamps.csv"
    if not csv_path.exists():
        raise ValueError(f"timestamps.csv not found in {args.input_dir}")
    
    # Load timestamps and filenames
    df = pd.read_csv(csv_path)
    timestamps = df['timestamp'].values
    png_filenames = df['png_filename'].values
    total_frames = len(timestamps)
    first_frame_timestamp = timestamps[0]
    
    # Get frame dimensions
    first_png_path = input_path / png_filenames[0]
    if not first_png_path.exists():
        raise ValueError(f"First PNG file not found: {first_png_path}")
    
    first_frame = cv2.imread(str(first_png_path), cv2.IMREAD_GRAYSCALE)
    if first_frame is None:
        raise ValueError(f"Could not load first PNG file: {first_png_path}")
    
    height, width = first_frame.shape
    
    # Calculate parameters
    if total_frames > 1:
        total_duration = timestamps[-1] - timestamps[0]
        input_fps = (total_frames - 1) / total_duration if total_duration > 0 else 30.0
    else:
        input_fps = 30.0
        total_duration = 0
    
    frames_per_bin = int(args.in_slowdown)
    output_frame_count = int(np.ceil(total_frames / frames_per_bin))
    output_duration = (output_frame_count - 1) / args.out_fps
    
    print("\n" + "="*60)
    print("PARALLEL DVS EVENT GENERATOR WITH PQDM")
    print("="*60)
    print(f"Input: {width}x{height}, {total_frames} frames at {input_fps:.1f} FPS")
    print(f"Time range: {timestamps[0]:.3f}s to {timestamps[-1]:.3f}s ({total_duration:.3f}s)")
    print(f"Parallel processing: {args.workers} workers, {args.group_size} frames/group")
    print(f"Output: {output_frame_count} frames at {args.out_fps} FPS")
    print(f"Event thresholds: +{args.threshold_pos}, -{args.threshold_neg}")
    
    # Configure filters
    if args.fast_mode:
        print("\nFAST MODE ENABLED - All filters disabled")
        enable_erc = enable_stc = enable_afk = False
    else:
        enable_erc = not args.disable_erc
        enable_stc = not args.disable_stc
        enable_afk = not args.disable_afk
    
    print_filter_params(args, enable_erc, enable_stc, enable_afk)
    
    # Prepare generator parameters
    generator_params = {
        'pos_thres': args.threshold_pos,
        'neg_thres': args.threshold_neg,
        'enable_erc': enable_erc,
        'enable_stc': enable_stc,
        'enable_afk': enable_afk,
        'erc_max_rate': args.erc_max_rate,
        'erc_window_size': args.erc_window_size,
        'stc_mode': args.stc_mode,
        'stc_threshold_us': args.stc_threshold_us,
        'afk_patch': args.afk_patch,
        'afk_low_freq': args.afk_low_freq,
        'afk_high_freq': args.afk_high_freq,
        'afk_diff_thresh_s': args.afk_diff_thresh_s,
        'cutoff_hz': args.cutoff_hz,
        'intensity_dependent': args.intensity_dependent
    }
    
    # Split frames into groups for parallel processing
    all_frame_paths = [input_path / filename for filename in png_filenames]
    group_size = args.group_size
    num_groups = int(np.ceil(total_frames / group_size))
    
    print(f"\nSplitting {total_frames} frames into {num_groups} groups...")
    
    # Prepare arguments for each worker
    worker_args = []
    
    # Process first frame separately to get initial state
    print("Initializing with first frame...")
    init_generator = EventGenerator(**generator_params)
    init_frame = cv2.imread(str(all_frame_paths[0]), cv2.IMREAD_GRAYSCALE)
    init_generator.generate_events(init_frame, 0.0)  # Initialize state
    base_log_frame = init_generator.base_log_frame
    lp_log_frame = init_generator.lp_log_frame
    
    for group_idx in range(num_groups):
        start_idx = group_idx * group_size
        end_idx = min(start_idx + group_size, total_frames)
        
        # For first group, skip the first frame since we already processed it
        if group_idx == 0:
            start_idx = 1
        
        group_args = {
            'frame_paths': all_frame_paths[start_idx:end_idx],
            'timestamps': timestamps[start_idx:end_idx],
            'group_idx': group_idx,
            'start_idx': start_idx,
            'end_idx': end_idx,
            'generator_params': generator_params,
            'output_frame_count': output_frame_count,
            'frames_per_bin': frames_per_bin,
            'out_fps': args.out_fps,
            'base_log_frame': base_log_frame if group_idx == 0 else None,
            'lp_log_frame': lp_log_frame if group_idx == 0 else None
        }
        worker_args.append(group_args)
    
    # Process groups in parallel
    print(f"Processing {num_groups} groups in parallel with {args.workers} workers...")
    results = pqdm(worker_args, process_frame_group, n_jobs=args.workers, 
                   desc="Processing frame groups")
    
    # Merge results
    print("Merging results from parallel processing...")
    event_bins, raw_event_counts = merge_results(results, output_frame_count)
    
    # Calculate statistics
    total_events = sum(len(bin_events) for bin_events in event_bins)
    total_raw_events = sum(raw_event_counts)
    
    print(f"\n" + "="*60)
    print("PROCESSING COMPLETE")
    print("="*60)
    print(f"Generated {total_events:,} events from {total_frames:,} frames")
    print(f"Raw events (before filtering): {total_raw_events:,}")
    print(f"Filtering removed: {total_raw_events - total_events:,} events ({100*(total_raw_events - total_events)/max(total_raw_events, 1):.1f}%)")
    print(f"Average: {total_events / max(total_frames, 1):.1f} events per frame")
    
    if total_events > 0:
        # Render output
        render_event_frames_and_csv(event_bins, width, height, args.output_dir,
                                   args.mp4_output, args.out_fps, args.contrast,
                                   first_frame_timestamp, raw_event_counts)
        print(f"SUCCESS! Event frames saved to: {args.output_dir}")
        if args.mp4_output:
            print(f"MP4 video saved to: {args.mp4_output}")
        print("="*60)
        return 0
    else:
        print("WARNING: No events were generated.")
        print("Try lowering the thresholds or disabling filters.")
        print("="*60)
        return 1


if __name__ == "__main__":
    main()