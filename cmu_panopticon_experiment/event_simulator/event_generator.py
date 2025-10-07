#!/usr/bin/env python3
"""
DVS Event Generator with IIR Low Pass Filtering

A high-performance Dynamic Vision Sensor (DVS) event generator that converts standard video frames
into DVS events with advanced filtering capabilities. DVS sensors respond to brightness changes
rather than absolute intensity levels, making them ideal for high-speed vision applications.

Key Features:
    - IIR low pass filtering for realistic photoreceptor simulation
    - Intensity-dependent temporal filtering for natural response
    - Advanced event filtering:
        * Anti-Flicker Filter (AFK) - Removes artificial lighting artifacts
        * Spatio-Temporal Correlation (STC) - Reduces redundant events
        * Event Rate Controller (ERC) - Manages event throughput
    - Configurable thresholds and parameters
    - Efficient PNG/CSV input/output with event visualization
    - Output event frames and timestamps as PNG files with CSV

Basic Usage:
    python event_generator.py input_dir output_dir

Advanced Usage:
    python event_generator.py input_dir output_dir --mp4_output output.mp4 --threshold_pos 0.25 --threshold_neg 0.25
"""

import argparse
import math
from collections import deque, defaultdict
import os
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm


# ================================================================================================
# PHOTORECEPTOR AND FILTERING UTILITIES
# ================================================================================================

def lin_log(frame: np.ndarray, threshold: float = 20, eps: float = 1e-6) -> np.ndarray:
    """
    Convert linear intensity to log space, mimicking biological photoreceptor response.
    Uses a piecewise function: linear mapping below threshold and logarithmic above.
    
    This implementation ensures smooth transition between linear and log regions while
    maintaining numerical stability for event generation.
    
    Args:
        frame: Input intensity frame (0-255)
        threshold: Linear/log transition point
        eps: Small value to prevent log(0)
    
    Returns:
        Converted frame in log space with same shape as input
    
    Note:
        The output is rounded to avoid floating point precision issues that could
        prevent proper OFF events after ON events.
    """
    frame = frame.astype(np.float64)
    
    # Compute linear scaling factor for smooth transition
    f = (1.0 / threshold) * math.log(threshold)
    
    # Ensure no zero values for log operation
    frame_safe = np.maximum(frame, eps)
    
    # Apply piecewise linear-log transformation
    log_frame = np.where(frame <= threshold, frame * f, np.log(frame_safe))
    
    # Round to avoid floating point precision issues
    rounding = 1e8
    log_frame = np.round(log_frame * rounding) / rounding
    
    return log_frame.astype(np.float32)


def rescale_intensity_frame(frame: np.ndarray) -> np.ndarray:
    """
    Rescale intensity frame for computing time constants in IIR filtering.
    
    Args:
        frame: Input intensity frame (0-255)
    
    Returns:
        Normalized intensity frame (0-1) with offset to ensure non-zero time constants
    """
    return (frame + 20) / 275.0


def low_pass_filter(log_new_frame: np.ndarray, 
                   lp_log_frame: np.ndarray,
                   inten01: np.ndarray | None,
                   delta_time: float,
                   cutoff_hz: float = 0) -> np.ndarray:
    """
    Apply intensity-dependent IIR low pass filter to simulate photoreceptor dynamics.
    Brighter regions have faster response times, matching biological behmp4or.
    
    Args:
        log_new_frame: New frame in log space
        lp_log_frame: Previous filtered frame
        inten01: Normalized intensity for time constant scaling (None for uniform)
        delta_time: Time step in seconds
        cutoff_hz: 3dB cutoff frequency (0 to disable)
    
    Returns:
        Filtered frame with same shape as input
    
    Note:
        The filter update factor (eps) is clamped to maintain stability.
        For intensity-dependent filtering, brighter regions have larger eps values.
    """
    if cutoff_hz <= 0:
        return log_new_frame
    
    # Compute base time constant from cutoff frequency
    tau = 1 / (math.pi * 2 * cutoff_hz)
    
    # Compute update factor (mixing coefficient)
    if inten01 is not None:
        # Intensity-dependent time constants
        eps = inten01 * (delta_time / tau)
        eps = np.clip(eps, 0, 1)  # Clamp for stability
    else:
        # Uniform time constant
        eps = min(delta_time / tau, 1.0)
    
    # First-order IIR update: y[n] = (1-eps)*y[n-1] + eps*x[n]
    return (1 - eps) * lp_log_frame + eps * log_new_frame


# ================================================================================================
# EVENT FILTER CLASSES
# ================================================================================================

class ERCFilter:
    """
    Event Rate Controller (ERC) - Manages event throughput using a sliding window approach.
    
    The ERC filter ensures the event rate stays below a specified maximum by probabilistically
    dropping events when the rate exceeds the threshold. This is particularly useful for:
        - Preventing event buffer overflow in hardware
        - Maintaining consistent event rates
        - Reducing computational load in downstream processing
    
    The filter maintains a sliding window of event timestamps and drops new events when
    the window becomes full. This provides smooth rate control without sharp cutoffs.
    """

    def __init__(self, max_rate: float = 1e6, window_size: float = 0.01):
        """
        Initialize the ERC filter.
        
        Args:
            max_rate: Maximum allowed events per second
            window_size: Time window size in seconds for rate control (e.g., 0.01 = 10ms)
        """
        self.max_rate = max_rate
        self.window_size = window_size
        self.max_events_per_window = int(max_rate * window_size)
        
        self.event_timestamps = deque()  # Sliding window of timestamps
        self.dropped_count = 0           # Total events dropped
        self.frame_dropped_count = 0     # Events dropped in current frame
        

    def should_emit_event(self, timestamp: float) -> bool:
        """
        Determine if a new event should be emitted based on current rate.
        
        Args:
            timestamp: Event timestamp in seconds
        
        Returns:
            True if event should be kept, False if it should be dropped
        """
        # Remove events outside the current window
        window_start = timestamp - self.window_size
        while self.event_timestamps and self.event_timestamps[0] < window_start:
            self.event_timestamps.popleft()
        
        # Accept event if below rate limit
        if len(self.event_timestamps) < self.max_events_per_window:
            self.event_timestamps.append(timestamp)
            return True
        
        # Drop event if rate limit exceeded
        self.dropped_count += 1
        self.frame_dropped_count += 1
        return False

    def reset_frame_stats(self):
        """Reset the per-frame event drop counter."""
        self.frame_dropped_count = 0

    def get_current_rate(self, current_timestamp: float) -> float:
        """
        Calculate the current event rate within the window.
        
        Args:
            current_timestamp: Current time in seconds
            
        Returns:
            Current event rate in events/second
        """
        # Clean up old events
        window_start = current_timestamp - self.window_size
        while self.event_timestamps and self.event_timestamps[0] < window_start:
            self.event_timestamps.popleft()
        
        return len(self.event_timestamps) / self.window_size


class STCFilter:
    """
    Spatio-Temporal Correlation Filter - Reduces redundant events from rapid intensity changes.
    
    This filter implements the event burst detection algorithm used in advanced DVS sensors
    like the IMX636 and GenX320. It identifies and filters redundant events that occur in
    quick succession at the same pixel location.
    
    Supported Modes:
        - STC_CUT_TRAIL: Keep second event of burst, remove trailing events
        - STC_KEEP_TRAIL: Keep second event of burst and all trailing events
        - TRAIL: Keep first event after polarity change, remove same-polarity events
    
    The filter helps reduce data rate while preserving important temporal information
    about intensity changes.
    """
    
    def __init__(self, mode: str = "STC_CUT_TRAIL", threshold_us: int = 10000):
        """
        Initialize the STC filter.
        
        Args:
            mode: Filter operation mode ("STC_CUT_TRAIL", "STC_KEEP_TRAIL", or "TRAIL")
            threshold_us: Burst detection threshold in microseconds (1000-100000)
        """
        valid_modes = ["STC_CUT_TRAIL", "STC_KEEP_TRAIL", "TRAIL"]
        if mode not in valid_modes:
            raise ValueError(f"Mode must be one of: {', '.join(valid_modes)}")
            
        if not (1000 <= threshold_us <= 100000):
            raise ValueError("Threshold must be between 1ms and 100ms (1000-100000 us)")
            
        self.mode = mode
        self.threshold_us = threshold_us
        self.pixel_states = {}  # (x,y) → EventState
        self.dropped_count = 0
        self.frame_dropped_count = 0

    class EventState:
        """Tracks the temporal state of events at a single pixel location."""
        def __init__(self):
            self.last_timestamp_us: int | None = None    # Time of last event
            self.last_polarity: int | None = None        # Polarity of last event
            self.burst_start_time_us: int | None = None  # Start time of current burst
            self.burst_polarity: int | None = None       # Polarity of current burst
            self.burst_event_count: int = 0              # Events in current burst
            self.second_event_emitted: bool = False      # Whether second event was output
    
    def should_emit_event(self, x: int, y: int, timestamp: float, polarity: int) -> bool:
        """
        Determine if an event should be emitted based on burst detection logic.
        
        Args:
            x, y: Pixel coordinates
            timestamp: Event timestamp in seconds
            polarity: Event polarity (+1 or -1)
            
        Returns:
            True if event should be kept, False if filtered
        """
        timestamp_us = int(timestamp * 1e6)
        pixel_key = (x, y)
        
        # Get or create pixel state
        if pixel_key not in self.pixel_states:
            self.pixel_states[pixel_key] = self.EventState()
        state = self.pixel_states[pixel_key]
        
        # Check event timing and polarity
        polarity_changed = (state.last_polarity is not None and 
                          state.last_polarity != polarity)
        in_burst = (state.last_timestamp_us is not None and
                   state.last_polarity == polarity and
                   (timestamp_us - state.last_timestamp_us) <= self.threshold_us)
        
        # Apply filter logic and update state
        decision = self._apply_filter_logic(state, timestamp_us, polarity, 
                                          polarity_changed, in_burst)
        state.last_timestamp_us = timestamp_us
        state.last_polarity = polarity
        
        if not decision:
            self.dropped_count += 1
            self.frame_dropped_count += 1
        return decision
    
    def _apply_filter_logic(self, state: EventState, timestamp_us: int,
                          polarity: int, polarity_changed: bool, 
                          in_burst: bool) -> bool:
        """Apply the specific filter logic based on selected mode."""
        if self.mode == "TRAIL":
            return self._apply_trail_logic(state, timestamp_us, polarity, 
                                         polarity_changed, in_burst)
        else:  # STC modes
            return self._apply_stc_logic(state, timestamp_us, polarity, 
                                       polarity_changed, in_burst)
    
    def _apply_trail_logic(self, state: EventState, timestamp_us: int,
                          polarity: int, polarity_changed: bool, 
                          in_burst: bool) -> bool:
        """
        TRAIL mode logic: Keep first event after polarity change.
        Removes events of same polarity within threshold period.
        """
        if polarity_changed or state.last_timestamp_us is None:
            # First event or polarity transition - emit
            state.burst_start_time_us = timestamp_us
            state.burst_polarity = polarity
            return True
        elif in_burst:
            # Same polarity within threshold - filter
            return False
        else:
            # Same polarity but outside threshold - new burst
            state.burst_start_time_us = timestamp_us
            state.burst_polarity = polarity
            return True
    
    def _apply_stc_logic(self, state: EventState, timestamp_us: int,
                        polarity: int, polarity_changed: bool, 
                        in_burst: bool) -> bool:
        """
        STC mode logic: Keep second event of burst, then handle trail based on mode.
        Filters out isolated events (single events without follow-up).
        """
        if polarity_changed or state.last_timestamp_us is None:
            # Start new sequence - don't emit yet
            state.burst_start_time_us = timestamp_us
            state.burst_polarity = polarity
            state.burst_event_count = 1
            state.second_event_emitted = False
            return False
        elif in_burst:
            # Continue burst
            state.burst_event_count += 1
            
            if state.burst_event_count == 2:
                # Second event - always emit
                state.second_event_emitted = True
                return True
            elif state.burst_event_count > 2:
                # Trail events
                return self.mode == "STC_KEEP_TRAIL"
            return False
        else:
            # Outside threshold - start new potential burst
            state.burst_start_time_us = timestamp_us
            state.burst_polarity = polarity
            state.burst_event_count = 1
            state.second_event_emitted = False
            return False

    def reset_frame_stats(self):
        """Reset the per-frame event drop counter."""
        self.frame_dropped_count = 0


class AFKFilter:
    """
    Anti-Flicker Filter - Removes events caused by artificial lighting flicker.
    
    This filter identifies and removes events that occur at frequencies matching typical
    artificial lighting (e.g., 50/60 Hz mains frequency). It operates by:
        1. Grouping pixels into patches to detect spatial correlation
        2. Tracking event timing to identify periodic patterns
        3. Filtering events that match the flicker frequency band
    
    The filter is particularly effective at removing unwanted events in indoor environments
    while preserving genuine motion-induced events.
    """

    class _State:
        """Tracks temporal state for a patch of pixels."""
        __slots__ = ("last_ts_us", "last_delta_us")
        def __init__(self):
            self.last_ts_us: int | None = None      # Last event timestamp
            self.last_delta_us: int | None = None   # Last inter-event interval

    def __init__(self,
                 patch: int = 4,
                 low_freq: float = 49.0,
                 high_freq: float = 51.0,
                 diff_thresh_s: float = 0.002):
        """
        Initialize the AFK filter.
        
        Args:
            patch: Size of pixel patch for spatial correlation
            low_freq: Lower bound of flicker frequency band (Hz)
            high_freq: Upper bound of flicker frequency band (Hz)
            diff_thresh_s: Maximum allowed period variation (seconds)
        """
        self.patch = patch
        self.low_freq = low_freq
        self.high_freq = high_freq
        self.diff_thresh_us = int(diff_thresh_s * 1e6)
        self._states = defaultdict(AFKFilter._State)
        self.dropped_count = 0
        self.frame_dropped_count = 0

    def should_emit_event(self, x: int, y: int, timestamp_s: float, polarity: int = 0) -> bool:
        """
        Determine if an event should be emitted based on flicker detection.
        
        Args:
            x, y: Pixel coordinates
            timestamp_s: Event timestamp in seconds
            polarity: Event polarity (ignored - all polarities treated same)
            
        Returns:
            True if event should be kept, False if identified as flicker
        """
        ts_us = int(timestamp_s * 1e6)
        key = (x // self.patch, y // self.patch)
        st = self._states[key]

        emit = True
        if st.last_ts_us is not None:
            dt = ts_us - st.last_ts_us
            if dt > 0:
                f = 1e6 / dt
                # Check if frequency matches flicker band and period is consistent
                if self.low_freq <= f <= self.high_freq:
                    if st.last_delta_us is not None and abs(dt - st.last_delta_us) <= self.diff_thresh_us:
                        emit = False

        if not emit:
            self.dropped_count += 1
            self.frame_dropped_count += 1
        else:
            # Update state only for kept events
            if st.last_ts_us is not None:
                st.last_delta_us = ts_us - st.last_ts_us
            st.last_ts_us = ts_us

        return emit

    def reset_frame_stats(self):
        """Reset the per-frame event drop counter."""
        self.frame_dropped_count = 0

# ================================================================================================
# CORE EVENT GENERATOR WITH IIR FILTERING
# ================================================================================================

class EventGenerator:
    """
    Core DVS event generator with IIR filtering and advanced event processing.
    
    This class implements a high-performance Dynamic Vision Sensor (DVS) event generator
    that converts standard video frames into DVS events. It simulates the key properties
    of biological vision and hardware DVS sensors:
    
    1. Logarithmic Intensity Response:
       - Converts linear intensity to log space
       - Matches biological photoreceptor response
       - Provides better dynamic range handling
    
    2. Temporal Filtering:
       - IIR low-pass filtering simulates photoreceptor dynamics
       - Optional intensity-dependent time constants
       - Reduces noise while preserving temporal information
    
    3. Event Generation:
       - Generates events when brightness changes exceed thresholds
       - Separate thresholds for ON (brightness increase) and OFF (decrease) events
       - Maintains per-pixel reference levels for accurate change detection
    
    4. Advanced Filtering:
       - Anti-Flicker Filter (AFK) removes artificial lighting artifacts
       - Spatio-Temporal Correlation (STC) reduces redundant events
       - Event Rate Controller (ERC) manages output data rate
    
    The generator is designed for both real-time processing and offline video conversion,
    with configurable parameters to match different application requirements.
    """
    
    def __init__(self, 
                 pos_thres: float = 0.25,
                 neg_thres: float = 0.25, 
                 enable_erc: bool = True,
                 enable_stc: bool = True,
                 enable_afk: bool = True,
                 erc_max_rate: float = 1e5,
                 erc_window_size: float = 0.01,
                 stc_mode: str = "STC_CUT_TRAIL", 
                 stc_threshold_us: int = 10000,
                 afk_patch: int = 4, 
                 afk_low_freq: float = 49.0,
                 afk_high_freq: float = 51.0,
                 afk_diff_thresh_s: float = 0.002,
                 cutoff_hz: float = 0.0,
                 intensity_dependent: bool = True):
        """
        Initialize the event generator with specified parameters.
        
        Args:
            pos_thres: Threshold for positive (ON) events
            neg_thres: Threshold for negative (OFF) events
            enable_erc: Enable Event Rate Controller
            enable_stc: Enable Spatio-Temporal Correlation filter
            enable_afk: Enable Anti-Flicker filter
            erc_max_rate: Maximum event rate (events/second)
            erc_window_size: ERC window size (seconds)
            stc_mode: STC filter mode ("STC_CUT_TRAIL", "STC_KEEP_TRAIL", "TRAIL")
            stc_threshold_us: STC burst detection threshold (microseconds)
            afk_patch: AFK patch size for spatial correlation
            afk_low_freq: AFK lower flicker frequency (Hz)
            afk_high_freq: AFK upper flicker frequency (Hz)
            afk_diff_thresh_s: AFK period variation threshold (seconds)
            cutoff_hz: IIR filter cutoff frequency (0 to disable)
            intensity_dependent: Use intensity-dependent time constants
        """
        # Event thresholds
        self.pos_thres = pos_thres
        self.neg_thres = neg_thres
        
        # IIR filter parameters
        self.cutoff_hz = cutoff_hz
        self.intensity_dependent = intensity_dependent
        
        # Initialize event filters
        self.erc = ERCFilter(max_rate=erc_max_rate, 
                           window_size=erc_window_size) if enable_erc else None
        self.stc = STCFilter(mode=stc_mode, 
                           threshold_us=stc_threshold_us) if enable_stc else None
        self.afk = AFKFilter(patch=afk_patch,
                           low_freq=afk_low_freq,
                           high_freq=afk_high_freq,
                           diff_thresh_s=afk_diff_thresh_s) if enable_afk else None
        
        # Processing state
        self.base_log_frame = None    # Reference frame in log space
        self.lp_log_frame = None      # Low-pass filtered frame
        self.frame_count = 0          # Number of frames processed
        self.last_timestamp = 0.0     # Last frame timestamp
    
    def reset_frame_stats(self):
        """Reset per-frame statistics for all active filters."""
        if self.erc: self.erc.reset_frame_stats()
        if self.stc: self.stc.reset_frame_stats()
        if self.afk: self.afk.reset_frame_stats()
    
    def generate_events(self, new_frame: np.ndarray, timestamp: float) -> tuple[list, int]:
        """
        Generate DVS events from a new input frame with IIR filtering.
        
        This method implements the core event generation pipeline:
        1. Convert input to log intensity space
        2. Apply IIR filtering if enabled
        3. Compare with reference frame to detect changes
        4. Generate events when changes exceed thresholds
        5. Apply event filters (AFK → STC → ERC)
        6. Update reference frame
        
        Args:
            new_frame: Input frame (grayscale, uint8)
            timestamp: Frame timestamp in seconds
            
        Returns:
            Tuple of (filtered_events, raw_event_count) where:
            - filtered_events: List of events after filtering, each as [timestamp, x, y, polarity]
            - raw_event_count: Number of events before any filtering
        
        Note:
            The first frame is used to initialize the reference frame and
            will not generate any events.
        """
        # Ensure consistent input format
        if new_frame.dtype != np.uint8:
            new_frame = new_frame.astype(np.uint8)
        
        self.reset_frame_stats()
        
        # Convert to log intensity space
        log_frame = lin_log(new_frame)
        
        # Initialize on first frame
        if self.base_log_frame is None:
            self.base_log_frame = log_frame.copy()
            self.lp_log_frame = log_frame.copy()
            self.last_timestamp = timestamp
            self.frame_count += 1
            return [], 0
        
        # Compute time step
        delta_time = max(timestamp - self.last_timestamp, 1e-6)
        
        # Apply IIR filtering if enabled
        if self.cutoff_hz > 0:
            # Compute intensity scaling for time constants
            inten01 = rescale_intensity_frame(new_frame) if self.intensity_dependent else None
            
            # Apply low-pass filter
            self.lp_log_frame = low_pass_filter(
                log_new_frame=log_frame,
                lp_log_frame=self.lp_log_frame,
                inten01=inten01,
                delta_time=delta_time,
                cutoff_hz=self.cutoff_hz
            )
            photoreceptor_output = self.lp_log_frame
        else:
            photoreceptor_output = log_frame
        
        # Compute brightness changes
        diff_frame = photoreceptor_output - self.base_log_frame
        events = []
        
        # Track raw event count before any filtering
        raw_event_count = 0
        
        # Generate ON events (brightness increase)
        pos_mask = diff_frame > self.pos_thres
        if np.any(pos_mask):
            pos_y, pos_x = np.where(pos_mask)
            pos_values = diff_frame[pos_mask]
            pos_num_events = np.minimum(
                np.floor(pos_values / self.pos_thres).astype(int), 2)
            
            for i in range(len(pos_y)):
                if pos_num_events[i] > 0:
                    y, x = pos_y[i], pos_x[i]
                    # Count raw events before filtering
                    raw_event_count += pos_num_events[i]
                    
                    if self._should_emit_event(x, y, timestamp, 1):
                        # Generate multiple events for large changes
                        for j in range(pos_num_events[i]):
                            events.append([timestamp + j * 1e-6, x, y, 1])
                    # Update reference level
                    self.base_log_frame[y, x] += pos_num_events[i] * self.pos_thres
        
        # Generate OFF events (brightness decrease)
        neg_mask = diff_frame < -self.neg_thres
        if np.any(neg_mask):
            neg_y, neg_x = np.where(neg_mask)
            neg_values = -diff_frame[neg_mask]
            neg_num_events = np.minimum(
                np.floor(neg_values / self.neg_thres).astype(int), 2)
            
            for i in range(len(neg_y)):
                if neg_num_events[i] > 0:
                    y, x = neg_y[i], neg_x[i]
                    # Count raw events before filtering
                    raw_event_count += neg_num_events[i]
                    
                    if self._should_emit_event(x, y, timestamp, -1):
                        # Generate multiple events for large changes
                        for j in range(neg_num_events[i]):
                            events.append([timestamp + j * 1e-6, x, y, -1])
                    # Update reference level
                    self.base_log_frame[y, x] -= neg_num_events[i] * self.neg_thres
        
        self.last_timestamp = timestamp
        self.frame_count += 1
        return events, raw_event_count
    
    def _should_emit_event(self, x: int, y: int, timestamp: float, polarity: int) -> bool:
        """
        Apply event filters to determine if an event should be emitted.
        
        Filters are applied in sequence: AFK → STC → ERC
        The event is dropped if any filter rejects it.
        
        Args:
            x, y: Pixel coordinates
            timestamp: Event timestamp in seconds
            polarity: Event polarity (+1 or -1)
            
        Returns:
            True if event passes all filters, False if dropped
        """
        if self.afk and not self.afk.should_emit_event(x, y, timestamp, polarity):
            return False
        if self.stc and not self.stc.should_emit_event(x, y, timestamp, polarity):
            return False
        if self.erc and not self.erc.should_emit_event(timestamp):
            return False
        return True

    def get_frame_filter_stats(self) -> tuple:
        """
        Get per-frame filter drop counts for progress display.
        
        Returns:
            Tuple of (erc_dropped, stc_dropped, afk_dropped) counts
        """
        return (
            self.erc.frame_dropped_count if self.erc else 0,
            self.stc.frame_dropped_count if self.stc else 0,
            self.afk.frame_dropped_count if self.afk else 0
        )


# ================================================================================================
# VIDEO RENDERING AND UTILITIES
# ================================================================================================

def render_event_frames_and_csv(event_bins: list,
                                width: int,
                                height: int,
                                output_dir: str,
                                output_video_file: str | None,
                                fps: float,
                                contrast: float,
                                first_frame_timestamp: float,
                                raw_event_counts: list[int] | None = None) -> None:
    """
    Render DVS events to PNG frames with CSV timestamps, and optionally create video.
    
    Args:
        event_bins: List of event lists, one per output frame
        width: Output video width in pixels
        height: Output video height in pixels
        output_dir: Directory to save PNG files and timestamps.csv
        output_video_file: Path to save output video (None to skip video)
        fps: Output video frame rate
        contrast: Event visualization contrast factor
        first_frame_timestamp: Absolute timestamp of first input frame
        raw_event_counts: List of raw event counts before filtering (optional)
    """
    if not event_bins:
        print("No events to render")
        return

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Initialize video writer if requested
    out = None
    if output_video_file:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_video_file, fourcc, fps, (width, height), False)

    # Buffers for PNG and CSV output
    frame_timestamps: list[float] = []
    png_filenames: list[str] = []

    print(f"Rendering {len(event_bins)} frames at {fps} FPS...")

    for bin_idx, events in enumerate(tqdm(event_bins, desc="Rendering")):
        # Start with neutral gray background for event visualization
        frame = np.zeros((height, width), dtype=np.float32)

        if events:
            events_array = np.array(events)
            x_coords = events_array[:, 1].astype(int)
            y_coords = events_array[:, 2].astype(int)
            polarities = events_array[:, 3]

            valid_mask = ((x_coords >= 0) & (x_coords < width) &
                          (y_coords >= 0) & (y_coords < height))
            if np.any(valid_mask):
                x_valid = x_coords[valid_mask]
                y_valid = y_coords[valid_mask]
                pol_valid = polarities[valid_mask]
                np.add.at(frame, (y_valid, x_valid), pol_valid * contrast)

        output_frame = np.clip(frame + 127, 0, 255).astype(np.uint8)
        absolute_timestamp = first_frame_timestamp + (bin_idx + 1) / fps
        
        # Save PNG file (without timestamp overlay)
        num_digits = max(6, len(str(len(event_bins) - 1)))
        filename = f"{bin_idx:0{num_digits}d}.png"
        png_path = output_path / filename
        cv2.imwrite(str(png_path), output_frame)
        
        png_filenames.append(filename)
        frame_timestamps.append(absolute_timestamp)
        
        # Write to video if requested (with timestamp overlay)
        if out:
            video_frame = output_frame.copy()
            
            # === Add timestamp overlay for video only ===
            timestamp_text = f"{absolute_timestamp:.3f} s"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6
            font_thickness = 2
            text_color = 255  # white
            margin = 8
            # Position: bottom-left corner
            text_size, _ = cv2.getTextSize(timestamp_text, font, font_scale, font_thickness)
            text_x = margin
            text_y = height - margin
            cv2.putText(video_frame, timestamp_text, (text_x, text_y), font, font_scale, text_color, font_thickness, cv2.LINE_AA)
            # === End timestamp overlay ===
            
            out.write(video_frame)

    if out:
        out.release()
        print(f"Video saved: {output_video_file}")

    # Save CSV file with timestamps and raw event counts
    csv_path = output_path / "timestamps.csv"
    csv_data = {
        'timestamp': frame_timestamps,
        'png_filename': png_filenames
    }
    
    # Add raw event count column if provided
    if raw_event_counts is not None:
        csv_data['raw_event_count'] = raw_event_counts
    
    df = pd.DataFrame(csv_data)
    df.to_csv(csv_path, index=False)
    
    print(f"Event frames saved: {output_dir} ({len(png_filenames)} PNGs)")
    print(f"Timestamps saved: {csv_path}")


def print_filter_params(args: argparse.Namespace,
                       enable_erc: bool,
                       enable_stc: bool,
                       enable_afk: bool) -> None:
    """
    Display the current event filter configuration.
    
    Prints a formatted summary of all filter parameters including:
    - IIR low-pass filter settings
    - Anti-Flicker Filter (AFK) configuration
    - Spatio-Temporal Correlation (STC) settings
    - Event Rate Controller (ERC) parameters
    
    Args:
        args: Parsed command line arguments
        enable_erc: Whether ERC is enabled
        enable_stc: Whether STC is enabled
        enable_afk: Whether AFK is enabled
    """
    print("\n" + "="*50)
    print("EVENT FILTER CONFIGURATION")
    print("="*50)
    
    # Show IIR low-pass filter settings
    if args.cutoff_hz > 0:
        print(f"IIR Low Pass Filter: ENABLED")
        print(f"  • Cutoff frequency: {args.cutoff_hz} Hz")
        print(f"  • Time constant: {1/(2*math.pi*args.cutoff_hz)*1000:.1f} ms")
        print(f"  • Intensity-dependent: {'YES' if args.intensity_dependent else 'NO'}")
    else:
        print("IIR Low Pass Filter: DISABLED")
    
    print()
    if enable_afk:
        print(f"AFK (Anti-Flicker Filter): ENABLED")
        print(f"  • Patch size: {args.afk_patch}x{args.afk_patch}")
        print(f"  • Flicker band: {args.afk_low_freq}-{args.afk_high_freq} Hz")
        print(f"  • Period tolerance: {args.afk_diff_thresh_s} s")
    else:
        print("AFK (Anti-Flicker Filter): DISABLED")
    
    print()
    if enable_stc:
        print(f"STC (Spatio-Temporal Filter): ENABLED")
        print(f"  • Mode: {args.stc_mode}")
        print(f"  • Threshold: {args.stc_threshold_us/1000:.1f} ms")
        if args.stc_mode.startswith('STC'):
            print(f"  • Behmp4or: Retains second event of burst")
            if args.stc_mode == 'STC_CUT_TRAIL':
                print(f"  • Trail handling: CUT (removes trail events)")
            else:
                print(f"  • Trail handling: KEEP (retains trail events)")
        else:  # TRAIL
            print(f"  • Behmp4or: Retains first event after polarity transition")
    else:
        print("STC (Spatio-Temporal Filter): DISABLED")
    
    print()
    if enable_erc:
        print(f"ERC (Event Rate Controller): ENABLED")
        print(f"  • Max rate: {args.erc_max_rate:.0f} events/s")
        print(f"  • Window size: {args.erc_window_size*1000:.1f} ms")
        print(f"  • Max events per window: {int(args.erc_max_rate * args.erc_window_size)}")
        print(f"  • Method: Sliding window algorithm")
    else:
        print("ERC (Event Rate Controller): DISABLED")
    
    print("="*50 + "\n")


# ================================================================================================
# MAIN FUNCTION AND ARGUMENT PARSING
# ================================================================================================

def main():
    """
    Main processing pipeline:
    1. Parse arguments and validate input
    2. Configure event generation and filtering
    3. Process H5 frames to events
    4. Render events to output video and H5
    """
    parser = argparse.ArgumentParser(
        description='Generate DVS events from PNG directory with advanced IIR filtering (based on v2e)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Basic usage with photoreceptor simulation:
    python event_generator.py input_dir output_dir --cutoff_hz 200
    
  High sensitivity with custom slowdown:
    python event_generator.py input_dir output_dir --threshold_pos 0.1 --threshold_neg 0.1 --in_slowdown 5
    
  Fast mode (no filters):
    python event_generator.py input_dir output_dir --fast_mode
    
  With MP4 video output:
    python event_generator.py input_dir output_dir --mp4_output output.mp4
    
  Complete pipeline with all options:
    python event_generator.py input_dir output_dir --mp4_output output.mp4 --cutoff_hz 300 --contrast 20
        """)
    
    # Required Arguments
    parser.add_argument('input_dir',
                       help='Input directory with PNG frames and timestamps.csv (from slomo_generator.py)')
    parser.add_argument('output_dir',
                       help='Output directory for event PNG frames and timestamps.csv')
    
    # Output Options
    output_group = parser.add_argument_group('Output Options')
    output_group.add_argument('--mp4_output', type=str,
                             help='Optional MP4 output video file for visualization')
    
    # Core Processing Parameters
    core_group = parser.add_argument_group('Core Processing Parameters')
    core_group.add_argument('--out_fps', type=float, default=30,
                           help='Output video frame rate (default: 30)')
    core_group.add_argument('--in_slowdown', type=float, default=10.0,
                           help='Input frame grouping factor - higher values = slower, more detail '
                                '(default: 30.0), input 30 fps with 10 slowdown / 30.0 -> 30 fps output')
    core_group.add_argument('--threshold_pos', type=float, default=0.25,
                           help='Positive event threshold - lower = more sensitive (default: 0.25)')
    core_group.add_argument('--threshold_neg', type=float, default=0.25,
                           help='Negative event threshold - lower = more sensitive (default: 0.25)')
    core_group.add_argument('--contrast', type=float, default=16,
                           help='Event visualization contrast in output video (default: 16)')
    
    # IIR Low Pass Filter Parameters (Photoreceptor Simulation)
    iir_group = parser.add_argument_group('IIR Low Pass Filter Parameters (Photoreceptor Simulation)')
    iir_group.add_argument('--cutoff_hz', type=float, default=0.0,
                          help='IIR low pass filter 3dB cutoff frequency in Hz '
                               '(0 = disabled, typical: 100-300)')
    iir_group.add_argument('--intensity_dependent', action='store_true',
                          help='Use intensity-dependent time constants '
                               '(brighter = faster response)')
    
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
                          help='Maximum allowed event rate in events/second (default: 1,000,000)')
    erc_group.add_argument('--erc_window_size', type=float, default=0.01,
                          help='ERC time window size in seconds for rate control '
                               '(default: 0.01 = 10ms)')
    
    # Spatio-Temporal Correlation (STC) Parameters
    stc_group = parser.add_argument_group('Spatio-Temporal Correlation (STC) Parameters')
    stc_group.add_argument('--stc_mode', type=str, default='STC_CUT_TRAIL',
                          choices=['STC_CUT_TRAIL', 'STC_KEEP_TRAIL', 'TRAIL'],
                          help='STC filter mode (default: STC_CUT_TRAIL):\n'
                               '  - STC_CUT_TRAIL: Retains second event of burst, removes trail after\n'
                               '  - STC_KEEP_TRAIL: Retains second event of burst, keeps trail after\n'
                               '  - TRAIL: Retains first event after polarity transition')
    stc_group.add_argument('--stc_threshold_us', type=int, default=1000,
                          help='STC burst detection threshold in microseconds '
                               '(1000-100000, default: 1000)')
    
    # Anti-Flicker (AFK) Parameters
    afk_group = parser.add_argument_group('Anti-Flicker (AFK) Parameters')
    afk_group.add_argument('--afk_patch', type=int, default=4,
                          help='AFK spatial patch size for grouping pixels (default: 4)')
    afk_group.add_argument('--afk_low_freq', type=float, default=49.0,
                          help='AFK lower bound of flicker band in Hz (default: 49.0)')
    afk_group.add_argument('--afk_high_freq', type=float, default=51.0,
                          help='AFK upper bound of flicker band in Hz (default: 51.0)')
    afk_group.add_argument('--afk_diff_thresh_s', type=float, default=0.002,
                          help='AFK minimum allowed change in period in seconds '
                               '(default: 0.002)')

    
    args = parser.parse_args()
    
    # Load input data from PNG directory and CSV
    input_path = Path(args.input_dir)
    if not input_path.exists() or not input_path.is_dir():
        raise ValueError(f"Input directory not found: {args.input_dir}")
    
    csv_path = input_path / "timestamps.csv"
    if not csv_path.exists():
        raise ValueError(f"timestamps.csv not found in {args.input_dir}")
    
    # Load timestamps and filenames from CSV
    df = pd.read_csv(csv_path)
    timestamps = df['timestamp'].values
    png_filenames = df['png_filename'].values
    total_frames = len(timestamps)
    first_frame_timestamp = timestamps[0]  # Store for absolute timestamp calculation
    
    # Get frame dimensions from first PNG file
    first_png_path = input_path / png_filenames[0]
    if not first_png_path.exists():
        raise ValueError(f"First PNG file not found: {first_png_path}")
    
    first_frame = cv2.imread(str(first_png_path), cv2.IMREAD_GRAYSCALE)
    if first_frame is None:
        raise ValueError(f"Could not load first PNG file: {first_png_path}")
    
    height, width = first_frame.shape
        
    # Calculate effective FPS from timestamps
    if total_frames > 1:
        total_duration = timestamps[-1] - timestamps[0]
        input_fps = (total_frames - 1) / total_duration if total_duration > 0 else 30.0
    else:
        input_fps = 30.0
        
    print("\n" + "="*60)
    print("DVS EVENT GENERATOR WITH IIR FILTERING")
    print("="*60)
    print(f"Input dir: {width}x{height}, {total_frames} frames at {input_fps:.1f} FPS")
    print(f"Time range: {timestamps[0]:.3f}s to {timestamps[-1]:.3f}s ({total_duration:.3f}s)")
    print(f"Slowdown factor: {args.in_slowdown}x")
    print(f"Event thresholds: +{args.threshold_pos}, -{args.threshold_neg}")
    print(f"Visualization contrast: {args.contrast}")
        
    # Calculate output parameters correctly
    frames_per_bin = int(args.in_slowdown)
    output_frame_count = int(np.ceil(total_frames / frames_per_bin))  # Number of bins we can make
    output_duration = (output_frame_count - 1) / args.out_fps  # Duration at target FPS
    
    print(f"Original duration: {total_duration:.2f}s")
    print(f"Processing: {frames_per_bin} input frames → 1 output frame")
    print(f"Output: {output_frame_count} frames at {args.out_fps} FPS")
    print(f"Output duration: {output_duration:.2f}s")
            
    
    # Configure filters based on command line arguments
    if args.fast_mode:
        print("\nFAST MODE ENABLED - All filters disabled for maximum speed")
        enable_erc = enable_stc = enable_afk = False
    else:
        enable_erc = not args.disable_erc
        enable_stc = not args.disable_stc
        enable_afk = not args.disable_afk
    
    # Display filter configuration
    print_filter_params(args, enable_erc, enable_stc, enable_afk)
    
    # Initialize event generator with configured parameters
    generator = EventGenerator(
        pos_thres=args.threshold_pos,
        neg_thres=args.threshold_neg,
        enable_erc=enable_erc,
        enable_stc=enable_stc,
        enable_afk=enable_afk,
        erc_max_rate=args.erc_max_rate,
        erc_window_size=args.erc_window_size,
        stc_mode=args.stc_mode,
        stc_threshold_us=args.stc_threshold_us,
        afk_patch=args.afk_patch,
        afk_low_freq=args.afk_low_freq,
        afk_high_freq=args.afk_high_freq,
        afk_diff_thresh_s=args.afk_diff_thresh_s,
        cutoff_hz=args.cutoff_hz,
        intensity_dependent=args.intensity_dependent
    )
    
    # Process H5 frames and generate events
    event_bins: list[list] = [[] for _ in range(output_frame_count)]
    raw_event_counts: list[int] = [0 for _ in range(output_frame_count)]
    frame_count = 0
    
    print("Loading PNG frames...")
    
    # Load all frame paths for processing
    all_frame_paths = [input_path / filename for filename in png_filenames]
    all_timestamps = timestamps
    
    print(f"Found {len(all_frame_paths)} PNG frames. Processing...")
    
    # Process frames from memory with progress tracking
    pbar = tqdm(total=total_frames, desc="Processing",
                bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] {postfix}')
        
    for frame_idx in range(total_frames):
        # Load frame from PNG file
        frame_path = all_frame_paths[frame_idx]
        frame = cv2.imread(str(frame_path), cv2.IMREAD_GRAYSCALE)
        if frame is None:
            print(f"Warning: Could not load frame {frame_path}, skipping")
            continue
        
        timestamp = all_timestamps[frame_idx]
        
        # Determine which output bin this frame contributes to
        bin_idx = int(np.ceil(frame_count / frames_per_bin))
        if bin_idx >= output_frame_count:
            break
        
        # Generate events for this frame using actual timestamp
        result_timestamp = bin_idx / args.out_fps
        events, raw_count = generator.generate_events(frame, result_timestamp)
        
        # Add events to the appropriate output bin
        if events:
            event_bins[bin_idx].extend(events)
        
        # Accumulate raw event count for this bin
        raw_event_counts[bin_idx] += raw_count
        
        # Update progress with filter statistics
        erc_dropped, stc_dropped, afk_dropped = generator.get_frame_filter_stats()
        
        # Add IIR status if enabled
        status = {
            'Events': f"{len(events) if events else 0:5d}",
            'AFK': f"{afk_dropped:5d}",
            'STC': f"{stc_dropped:5d}",
            'ERC': f"{erc_dropped:5d}"
        }
        if args.cutoff_hz > 0:
            status['IIR'] = 'ON'
        
        pbar.set_postfix(status, refresh=True)
        
        frame_count += 1
        pbar.update(1)
    
    pbar.close()
    
    # Generate output and display results
    total_events = sum(len(bin_events) for bin_events in event_bins)
    print(f"\n" + "="*60)
    print("PROCESSING COMPLETE")
    print("="*60)
    print(f"Generated {total_events:,} events from {frame_count:,} input frames")
    print(f"Average: {total_events / max(frame_count, 1):.1f} events per input frame")
    
    if args.cutoff_hz > 0:
        print(f"IIR filtering applied at {args.cutoff_hz} Hz")
        print(f"Intensity-dependent time constants: {'YES' if args.intensity_dependent else 'NO'}")
    
    if total_events > 0:
        # Render events to PNG frames and CSV, optionally create video
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
