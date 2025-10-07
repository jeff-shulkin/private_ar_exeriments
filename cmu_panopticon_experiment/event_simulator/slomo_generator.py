#!/usr/bin/env python3
"""
Standalone Super Slow Motion Video Generator

Based on the v2e implementation by Zhe He (hezhehz@live.cn)
Modified for standalone use without v2e dependencies.

Requirements:
- PyTorch
- OpenCV
- PIL (Pillow)
- NumPy
- tqdm

The SuperSloMo model checkpoint (SuperSloMo39.ckpt) can be downloaded from:
https://github.com/yuhuixu1993/Adobe240fps/releases/download/v1.0/SuperSloMo39.ckpt
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torchvision.transforms as transforms

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm
import logging
import warnings
import os
import sys
import argparse
import pandas as pd
from pathlib import Path

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.frame import load_video_frames

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

warnings.filterwarnings("ignore", category=UserWarning, module="torch.nn.functional")

logger = logging.getLogger(__name__)

class down(nn.Module):
    """UNet downsampling block: AvgPool -> Conv+LeakyReLU -> Conv+LeakyReLU"""
    def __init__(self, inChannels, outChannels, filterSize):
        super(down, self).__init__()
        self.conv1 = nn.Conv2d(inChannels, outChannels, filterSize, 
                              stride=1, padding=int((filterSize - 1) / 2))
        self.conv2 = nn.Conv2d(outChannels, outChannels, filterSize,
                              stride=1, padding=int((filterSize - 1) / 2))

    def forward(self, x):
        x = F.avg_pool2d(x, 2)
        x = F.leaky_relu(self.conv1(x), negative_slope=0.1)
        x = F.leaky_relu(self.conv2(x), negative_slope=0.1)
        return x

class up(nn.Module):
    """UNet upsampling block: Bilinear interpolation -> Conv+LeakyReLU -> Conv+LeakyReLU"""
    def __init__(self, inChannels, outChannels):
        super(up, self).__init__()
        self.conv1 = nn.Conv2d(inChannels, outChannels, 3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(2 * outChannels, outChannels, 3, stride=1, padding=1)

    def forward(self, x, skpCn):
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        x = F.leaky_relu(self.conv1(x), negative_slope=0.1)
        x = F.leaky_relu(self.conv2(torch.cat((x, skpCn), 1)), negative_slope=0.1)
        return x

class UNet(nn.Module):
    """UNet architecture for Super SloMo"""
    def __init__(self, inChannels, outChannels):
        super(UNet, self).__init__()
        self.conv1 = nn.Conv2d(inChannels, 32, 7, stride=1, padding=3)
        self.conv2 = nn.Conv2d(32, 32, 7, stride=1, padding=3)
        self.down1 = down(32, 64, 5)
        self.down2 = down(64, 128, 3)
        self.down3 = down(128, 256, 3)
        self.down4 = down(256, 512, 3)
        self.down5 = down(512, 512, 3)
        self.up1 = up(512, 512)
        self.up2 = up(512, 256)
        self.up3 = up(256, 128)
        self.up4 = up(128, 64)
        self.up5 = up(64, 32)
        self.conv3 = nn.Conv2d(32, outChannels, 3, stride=1, padding=1)

    def forward(self, x):
        x = F.leaky_relu(self.conv1(x), negative_slope=0.1)
        s1 = F.leaky_relu(self.conv2(x), negative_slope=0.1)
        s2 = self.down1(s1)
        s3 = self.down2(s2)
        s4 = self.down3(s3)
        s5 = self.down4(s4)
        x = self.down5(s5)
        x = self.up1(x, s5)
        x = self.up2(x, s4)
        x = self.up3(x, s3)
        x = self.up4(x, s2)
        x = self.up5(x, s1)
        x = F.leaky_relu(self.conv3(x), negative_slope=0.1)
        return x

class backWarp(nn.Module):
    """Backwarping block for frame interpolation using optical flow"""
    def __init__(self, W, H, device):
        super(backWarp, self).__init__()
        self.W = W
        self.H = H
        self.device = device
        
        # Create mesh grid
        gridX, gridY = np.meshgrid(np.arange(W), np.arange(H))
        self.gridX = torch.tensor(gridX, requires_grad=False).to(device)
        self.gridY = torch.tensor(gridY, requires_grad=False).to(device)
        
    def forward(self, img, flow):
        u = flow[:, 0, :, :]
        v = flow[:, 1, :, :]
        
        x = self.gridX.unsqueeze(0).expand_as(u).float() + u
        y = self.gridY.unsqueeze(0).expand_as(v).float() + v
        
        # normalize coordinates to [-1, 1] for grid_sample
        x = 2*(x/self.W - 0.5)
        y = 2*(y/self.H - 0.5)
        
        # stacking X and Y
        grid = torch.stack((x,y), dim=3)
        
        # Sample pixels using bilinear interpolation
        return F.grid_sample(img, grid, align_corners=True)

class FramesDataset(data.Dataset):
    """Dataset for loading and preprocessing frames from directory with PNGs and timestamps.csv"""
    def __init__(self, input_dir, frame_size=(320, 320), transform=None, start_time=None, end_time=None):
        self.transform = transform
        self.input_dir = input_dir
        self.frame_size = frame_size or (320, 320)
        
        frames, self.timestamps, t0 = load_video_frames(
            input_dir, start=start_time, end=end_time, grayscale=True, target_fs=30, target_shape=(320, 320)
        )

        self.timestamps += t0
            
        self.processed_frames = []
        
        for frame in tqdm(frames, desc="Preprocessing dataset"):
            frame_tensor = torch.from_numpy(frame.astype(np.float32) / 255.0).unsqueeze(0)
            if self.transform:
                frame_pil = Image.fromarray((frame_tensor.squeeze(0) * 255).numpy().astype(np.uint8))
                frame_tensor = self.transform(frame_pil)
            self.processed_frames.append(frame_tensor)
        self.total_frames = len(self.processed_frames)

    def __len__(self):
        return self.total_frames

    def __getitem__(self, idx):
        if idx >= len(self.processed_frames):
            raise IndexError(f"Index {idx} out of range")
        return self.processed_frames[idx]
    
    def get_timestamp(self, idx):
        """Get timestamp for a specific frame index."""
        return self.timestamps[idx]
    
    def get_frame_and_timestamp(self, idx):
        """Get both frame and timestamp for a specific index."""
        return self.processed_frames[idx], self.timestamps[idx]

class SuperSloMo:
    """Super SloMo class for video frame interpolation"""
    def __init__(self, model_path, device=None, batch_size=1, frame_size=(320, 320)):
        """
        Initialize SuperSloMo.
        
        Args:
            model_path: Path to the SuperSloMo model checkpoint
            device: Device to run on ('cpu', 'cuda', 'auto')
            batch_size: Batch size for processing
            frame_size: Target frame size (width, height) tuple (any size allowed)
        """
        if device is None:
            self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        self.model_path = model_path
        self.batch_size = batch_size
        self.frame_size = frame_size
        
        # Initialize transforms
        mean = [0.428]
        std = [1.0]
        
        if self.device == "cpu":
            self.transform = transforms.Compose([
                transforms.ToTensor()
            ])
            self.reverse_transform = transforms.Compose([
                transforms.ToPILImage()
            ])
        else:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std)
            ])
            self.reverse_transform = transforms.Compose([
                transforms.Normalize(mean=[-m for m in mean], std=std),
                transforms.ToPILImage()
            ])
        
    def load_model(self, frame_size):
        """Load the SuperSloMo model."""
        self.flow_estimator = UNet(2, 4).to(self.device)
        self.interpolator = UNet(12, 5).to(self.device)
        
        checkpoint = torch.load(self.model_path, map_location=self.device, weights_only=False)
        self.flow_estimator.load_state_dict(checkpoint['state_dictFC'])
        self.interpolator.load_state_dict(checkpoint['state_dictAT'])
        
        self.flow_estimator.eval()
        self.interpolator.eval()
        self.warper = backWarp(frame_size[0], frame_size[1], self.device)
        
    def interpolate_frames(self, frame0, frame1, num_interp):
        """
        Interpolate between two frames.
        
        Args:
            frame0: First frame tensor
            frame1: Second frame tensor
            num_interp: Number of frames to interpolate
            
        Returns:
            List of interpolated frame tensors
        """
        frame0 = frame0.to(self.device)
        frame1 = frame1.to(self.device)
        
        # Calculate flow
        flows = self.flow_estimator(torch.cat((frame0, frame1), dim=1))
        flow01 = flows[:, :2]  # F_0_1
        flow10 = flows[:, 2:]  # F_1_0
        
        frames = []
        for i in range(1, num_interp + 1):
            t = (i + 0.5) / (num_interp + 1)  # Use v2e's time calculation
            
            # Calculate intermediate flows (following v2e's formulation)
            temp = -t * (1 - t)
            fCoeff = [temp, t * t, (1 - t) * (1 - t), temp]
            
            flow_t0 = fCoeff[0] * flow01 + fCoeff[1] * flow10  # F_t_0
            flow_t1 = fCoeff[2] * flow01 + fCoeff[3] * flow10  # F_t_1
            
            # Warp frames
            frame_t0 = self.warper(frame0, flow_t0)  # g_I0_F_t_0
            frame_t1 = self.warper(frame1, flow_t1)  # g_I1_F_t_1
            
            # Generate interpolated frame using v2e's exact concatenation order
            intrp_out = self.interpolator(torch.cat([
                frame0, frame1, flow01, flow10,
                flow_t1, flow_t0, frame_t1, frame_t0
            ], dim=1))
            
            # Extract refined flows and visibility map
            flow_t0_refined = intrp_out[:, :2] + flow_t0
            flow_t1_refined = intrp_out[:, 2:4] + flow_t1
            visibility_t0 = torch.sigmoid(intrp_out[:, 4:5])
            visibility_t1 = 1 - visibility_t0
            
            # Warp with refined flows
            frame_t0_refined = self.warper(frame0, flow_t0_refined)
            frame_t1_refined = self.warper(frame1, flow_t1_refined)
            
            # Blend frames using visibility weights
            wCoeff = [1 - t, t]
            frame_final = (wCoeff[0] * visibility_t0 * frame_t0_refined +
                          wCoeff[1] * visibility_t1 * frame_t1_refined) / \
                         (wCoeff[0] * visibility_t0 + wCoeff[1] * visibility_t1)
            
            frames.append(frame_final)
            
        return frames
        
    def generate_slow_motion(self, input_dir, output_dir, mp4_output_path, slowdown_factor, fps=30, start_time=None, end_time=None):
        """
        Generate slow motion video from input directory with PNGs and timestamps.csv.
        
        Args:
            input_dir: Path to input directory with PNGs and timestamps.csv
            output_dir: Path to output directory for interpolated PNGs and timestamps.csv
            mp4_output_path: Path to output demo mp4 video
            slowdown_factor: Factor to slow down the video
            fps: Output video frame rate
            start_time: Start time offset in seconds from first frame's timestamp
            end_time: End time offset in seconds from first frame's timestamp
        """
        dataset = FramesDataset(input_dir, self.frame_size, self.transform, start_time, end_time)
        frame_size = dataset.frame_size

        print(f"Processing {dataset.total_frames} frames {frame_size} on {self.device}")

        self.load_model(frame_size)
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(mp4_output_path, fourcc, fps, frame_size, False)
        
        # Buffer to store exact frames written to video
        video_frames_buffer = []
        interpolated_timestamps = []
        
        frames_loader = data.DataLoader(dataset, batch_size=1, shuffle=False)
        
        with torch.no_grad():
            for i, frame0 in enumerate(tqdm(frames_loader, desc="Interpolating frames")):
                if i == len(frames_loader) - 1:
                    break
                    
                frame1 = dataset[i + 1].unsqueeze(0)
                timestamp0 = dataset.get_timestamp(i)
                timestamp1 = dataset.get_timestamp(i + 1)
                
                # Process first frame and write to video
                frame_pil = self.reverse_transform(frame0[0])
                video_frame = np.array(frame_pil)
                out.write(video_frame)
                # Store exactly what was written to video
                video_frames_buffer.append(video_frame.copy())
                interpolated_timestamps.append(timestamp0)
                
                interp_frames = self.interpolate_frames(frame0.to(self.device), frame1.to(self.device), slowdown_factor - 1)
                
                time_diff = timestamp1 - timestamp0
                for j, frame in enumerate(interp_frames):
                    t = (j + 1.0) / slowdown_factor
                    interp_timestamp = timestamp0 + t * time_diff
                    
                    # Process interpolated frame and write to video
                    frame_pil = self.reverse_transform(frame[0].cpu())
                    video_frame = np.array(frame_pil)
                    out.write(video_frame)
                    # Store exactly what was written to video
                    video_frames_buffer.append(video_frame.copy())
                    interpolated_timestamps.append(interp_timestamp)
            
            # Process last frame and write to video
            last_frame, last_timestamp = dataset.get_frame_and_timestamp(dataset.total_frames - 1)
            frame_pil = self.reverse_transform(last_frame)
            video_frame = np.array(frame_pil)
            out.write(video_frame)
            # Store exactly what was written to video
            video_frames_buffer.append(video_frame.copy())
            interpolated_timestamps.append(last_timestamp)
                    
        out.release()
        
        # Save as PNGs and CSV
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Determine number of digits needed for filenames
        num_frames = len(video_frames_buffer)
        num_digits = max(6, len(str(num_frames - 1)))
        
        # Save PNG files and prepare CSV data
        png_filenames = []
        timestamps_array = np.array(interpolated_timestamps)
        
        print(f"Saving {num_frames} frames as PNGs ({timestamps_array[0]:.3f}s-{timestamps_array[-1]:.3f}s)")
        
        for i, (frame, timestamp) in enumerate(tqdm(zip(video_frames_buffer, interpolated_timestamps), 
                                                   total=num_frames, desc="Saving PNGs")):
            # Create filename with appropriate zero padding
            filename = f"{i:0{num_digits}d}.png"
            filepath = output_path / filename
            
            # Save frame as PNG
            cv2.imwrite(str(filepath), frame)
            png_filenames.append(filename)
        
        # Save timestamps CSV
        csv_path = output_path / "timestamps.csv"
        df = pd.DataFrame({
            'timestamp': interpolated_timestamps,
            'png_filename': png_filenames
        })
        df.to_csv(csv_path, index=False)
        
        print(f"Saved: {output_dir} ({num_frames} PNGs), {mp4_output_path} ({num_frames / dataset.total_frames:.1f}x interpolation)")


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Generate slow motion video using SuperSloMo")
    parser.add_argument("input_dir", help="Path to input directory with PNGs and timestamps.csv")
    parser.add_argument("output_dir", help="Path to output directory for interpolated PNGs and timestamps.csv")
    parser.add_argument("mp4_output_path", help="Path to output demo mp4 video file")
    parser.add_argument("--model", default="simulator/SuperSloMo39.ckpt",
                      help="Path to SuperSloMo model checkpoint")
    parser.add_argument("--slowdown_factor", type=int, default=10,
                      help="Slowdown factor (default: 10)")
    parser.add_argument("--fps", type=int, default=30,
                      help="Output video frame rate (default: 30)")
    parser.add_argument("--start_time", type=float,
                      help="Start time offset in seconds from first frame's timestamp")
    parser.add_argument("--end_time", type=float,
                      help="End time offset in seconds from first frame's timestamp")
    parser.add_argument("--batch_size", type=int, default=128,
                      help="Batch size for processing (default: 128)")

    args = parser.parse_args()
    
    
    slomo = SuperSloMo(args.model, DEVICE, args.batch_size, frame_size=(320, 320))
    slomo.generate_slow_motion(args.input_dir, args.output_dir, args.mp4_output_path,
                             args.slowdown_factor, args.fps, args.start_time, args.end_time)