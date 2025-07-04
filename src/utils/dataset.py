"""
Dataset utilities for QualiVision video quality assessment.

This module provides dataset classes and data loading utilities for the
VQualA 2025 challenge dataset.
"""

import os
import gc
import torch
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.transforms.functional import convert_image_dtype
import decord
from decord import VideoReader
import cv2
from PIL import Image
from sentence_transformers import SentenceTransformer
from transformers import AutoVideoProcessor

# Set decord bridge to torch for better integration
decord.bridge.set_bridge('torch')


class TaobaoVDDataset(Dataset):
    """
    Dataset class for TaobaoVD-GC video quality assessment dataset.
    
    This dataset handles video loading, frame sampling, and preprocessing
    for both DOVER++ (640x640) and V-JEPA2 (384x384) models.
    """
    
    MOS_COLS = ['Traditional_MOS', 'Alignment_MOS', 'Aesthetic_MOS', 'Temporal_MOS', 'Overall_MOS']
    
    def __init__(self, 
                 csv_file: str,
                 video_dir: str,
                 num_frames: int = 64,
                 resolution: int = 640,
                 mode: str = 'train',
                 video_processor: Optional[AutoVideoProcessor] = None):
        """
        Initialize TaobaoVD dataset.
        
        Args:
            csv_file: Path to CSV file with labels
            video_dir: Directory containing video files
            num_frames: Number of frames to sample from each video
            resolution: Target resolution for videos
            mode: Dataset mode ('train', 'val', 'test')
            video_processor: Optional video processor for V-JEPA2
        """
        self.df = pd.read_csv(csv_file)
        self.video_dir = Path(video_dir)
        self.num_frames = num_frames
        self.resolution = resolution
        self.mode = mode
        self.video_processor = video_processor
        
        # Video transforms - no normalization for quality assessment
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((resolution, resolution)),
            transforms.ToTensor(),  # Converts to [0,1] range
        ])
        
        # Check if we have ground truth labels
        self.has_labels = all(col in self.df.columns for col in self.MOS_COLS)
        
        print(f"Dataset initialized:")
        print(f"  Mode: {mode}")
        print(f"  Has labels: {self.has_labels}")
        print(f"  Samples: {len(self.df)}")
        print(f"  Resolution: {resolution}x{resolution}")
        print(f"  Frames per video: {num_frames}")
        print(f"  Video processor: {'V-JEPA2' if video_processor else 'Manual'}")
    
    def _sample_frames_manual(self, video_path: Path) -> torch.Tensor:
        """
        Sample frames from video manually and resize to target resolution.
        Used for DOVER++ model.
        """
        try:
            vr = VideoReader(str(video_path))
            total_frames = len(vr)
            
            # Sample frames uniformly
            if total_frames <= self.num_frames:
                indices = np.linspace(0, total_frames - 1, self.num_frames).astype(int)
            else:
                indices = np.linspace(0, total_frames - 1, self.num_frames).astype(int)
            
            indices = np.clip(indices, 0, total_frames - 1)
            frames = vr.get_batch(indices)  # Shape: (T, H, W, C)
            
            # Transform each frame
            transformed_frames = []
            for i in range(frames.shape[0]):
                frame = frames[i].numpy().astype(np.uint8)
                transformed_frame = self.transform(frame)
                transformed_frames.append(transformed_frame)
            
            # Stack frames: (T, C, H, W)
            video_tensor = torch.stack(transformed_frames)
            
            # Rearrange to (C, T, H, W) for 3D CNN
            video_tensor = video_tensor.permute(1, 0, 2, 3)
            
            return video_tensor
            
        except Exception as e:
            print(f"Error reading video {video_path}: {e}")
            # Return dummy tensor with correct shape
            return torch.zeros((3, self.num_frames, self.resolution, self.resolution), dtype=torch.float32)
    
    def _sample_frames_processor(self, video_path: Path) -> torch.Tensor:
        """
        Sample frames from video using video processor.
        Used for V-JEPA2 model.
        """
        try:
            vr = VideoReader(str(video_path))
            total_frames = len(vr)
            
            # Sample frames uniformly
            if total_frames <= self.num_frames:
                indices = np.linspace(0, total_frames - 1, self.num_frames).astype(int)
            else:
                indices = np.linspace(0, total_frames - 1, self.num_frames).astype(int)
            
            indices = np.clip(indices, 0, total_frames - 1)
            frames = vr.get_batch(indices)  # Shape: (T, H, W, C)
            
            # Convert to numpy and process
            frames_np = frames.numpy()
            
            # Process with video processor
            processed = self.video_processor(
                list(frames_np),
                return_tensors="pt"
            )
            
            return processed["pixel_values"][0]  # Remove batch dimension
            
        except Exception as e:
            print(f"Error reading video {video_path}: {e}")
            # Return dummy tensor with correct shape
            return torch.zeros((3, self.num_frames, self.resolution, self.resolution), dtype=torch.float32)
    
    def __len__(self) -> int:
        return len(self.df)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get a single sample from the dataset.
        
        Args:
            idx: Sample index
            
        Returns:
            Dictionary containing frames, prompt, video_name, and optional labels
        """
        row = self.df.iloc[idx]
        video_path = self.video_dir / row["video_name"]
        
        # Sample and transform frames
        if self.video_processor is not None:
            frames = self._sample_frames_processor(video_path)
        else:
            frames = self._sample_frames_manual(video_path)
        
        result = {
            "frames": frames,
            "prompt": row["Prompt"],
            "video_name": row["video_name"]
        }
        
        if self.has_labels:
            # Training/validation mode - include labels
            labels = pd.to_numeric(row[self.MOS_COLS], errors="coerce").fillna(3.0).astype(np.float32).values
            result["labels"] = torch.tensor(labels, dtype=torch.float32)
        else:
            # Test mode - no labels available
            result["labels"] = torch.zeros(5, dtype=torch.float32)
        
        return result


class OptimizedGPUCollate:
    """
    Optimized collate function for GPU processing with text encoding.
    
    This class handles batching of video data and text encoding,
    optimizing for GPU memory usage and processing speed.
    """
    
    def __init__(self, 
                 video_processor: Optional[AutoVideoProcessor] = None,
                 text_encoder: Optional[SentenceTransformer] = None,
                 device: str = 'cuda',
                 max_frames: int = 64):
        """
        Initialize the collate function.
        
        Args:
            video_processor: Optional video processor for V-JEPA2
            text_encoder: Text encoder for prompt processing
            device: Device to place tensors on
            max_frames: Maximum number of frames per video
        """
        self.video_processor = video_processor
        self.text_encoder = text_encoder
        self.device = device
        self.max_frames = max_frames
        
        print(f"OptimizedGPUCollate initialized:")
        print(f"  Device: {device}")
        print(f"  Max frames: {max_frames}")
        print(f"  Video processor: {'V-JEPA2' if video_processor else 'Manual'}")
        print(f"  Text encoder: {'Available' if text_encoder else 'None'}")
    
    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """
        Collate a batch of samples.
        
        Args:
            batch: List of samples from dataset
            
        Returns:
            Batched data dictionary
        """
        # Extract components
        frames_list = [item["frames"] for item in batch]
        prompts = [item["prompt"] for item in batch]
        video_names = [item["video_name"] for item in batch]
        labels_list = [item["labels"] for item in batch]
        
        # Stack frames
        frames = torch.stack(frames_list, dim=0)  # (B, C, T, H, W) or (B, T, C, H, W)
        
        # Handle different frame formats
        if frames.dim() == 5:
            if frames.shape[1] == 3:  # (B, C, T, H, W) - DOVER format
                pixel_values_videos = frames
            else:  # (B, T, C, H, W) - V-JEPA2 format
                pixel_values_videos = frames.permute(0, 2, 1, 3, 4)  # (B, C, T, H, W)
        else:
            raise ValueError(f"Unexpected frame tensor shape: {frames.shape}")
        
        # Move to device
        pixel_values_videos = pixel_values_videos.to(self.device)
        
        # Encode text
        text_emb = None
        if self.text_encoder is not None:
            try:
                with torch.no_grad():
                    text_emb = self.text_encoder.encode(
                        prompts,
                        convert_to_tensor=True,
                        normalize_embeddings=True,
                        device=self.device,
                        batch_size=len(prompts)
                    )
            except Exception as e:
                print(f"Warning: Text encoding failed: {e}")
                # Create dummy text embeddings
                text_dim = 768  # Default BGE dimension
                text_emb = torch.zeros(len(prompts), text_dim, device=self.device)
        
        # Stack labels
        labels = torch.stack(labels_list, dim=0).to(self.device)
        
        # Clean up memory
        del frames_list
        torch.cuda.empty_cache()
        
        return {
            "pixel_values_videos": pixel_values_videos,
            "text_emb": text_emb,
            "labels": labels,
            "video_names": video_names,
            "prompts": prompts  # Keep original prompts for debugging
        }


def create_data_loaders(train_csv: str,
                       val_csv: str,
                       train_video_dir: str,
                       val_video_dir: str,
                       batch_size: int = 4,
                       num_frames: int = 64,
                       resolution: int = 640,
                       video_processor: Optional[AutoVideoProcessor] = None,
                       text_encoder: Optional[SentenceTransformer] = None,
                       device: str = 'cuda',
                       num_workers: int = 4) -> Tuple[DataLoader, DataLoader]:
    """
    Create train and validation data loaders.
    
    Args:
        train_csv: Path to training CSV file
        val_csv: Path to validation CSV file
        train_video_dir: Directory containing training videos
        val_video_dir: Directory containing validation videos
        batch_size: Batch size for data loading
        num_frames: Number of frames per video
        resolution: Target resolution
        video_processor: Optional video processor
        text_encoder: Optional text encoder
        device: Device for processing
        num_workers: Number of worker processes
        
    Returns:
        Tuple of (train_loader, val_loader)
    """
    # Create datasets
    train_dataset = TaobaoVDDataset(
        csv_file=train_csv,
        video_dir=train_video_dir,
        num_frames=num_frames,
        resolution=resolution,
        mode='train',
        video_processor=video_processor
    )
    
    val_dataset = TaobaoVDDataset(
        csv_file=val_csv,
        video_dir=val_video_dir,
        num_frames=num_frames,
        resolution=resolution,
        mode='val',
        video_processor=video_processor
    )
    
    # Create collate function
    collate_fn = OptimizedGPUCollate(
        video_processor=video_processor,
        text_encoder=text_encoder,
        device=device,
        max_frames=num_frames
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True if num_workers > 0 else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True if num_workers > 0 else False
    )
    
    print(f"Data loaders created:")
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val batches: {len(val_loader)}")
    print(f"  Batch size: {batch_size}")
    print(f"  Num workers: {num_workers}")
    
    return train_loader, val_loader


def create_test_loader(test_csv: str,
                      test_video_dir: str,
                      batch_size: int = 1,
                      num_frames: int = 64,
                      resolution: int = 640,
                      video_processor: Optional[AutoVideoProcessor] = None,
                      text_encoder: Optional[SentenceTransformer] = None,
                      device: str = 'cuda') -> DataLoader:
    """
    Create test data loader.
    
    Args:
        test_csv: Path to test CSV file
        test_video_dir: Directory containing test videos
        batch_size: Batch size for data loading
        num_frames: Number of frames per video
        resolution: Target resolution
        video_processor: Optional video processor
        text_encoder: Optional text encoder
        device: Device for processing
        
    Returns:
        Test data loader
    """
    # Create test dataset
    test_dataset = TaobaoVDDataset(
        csv_file=test_csv,
        video_dir=test_video_dir,
        num_frames=num_frames,
        resolution=resolution,
        mode='test',
        video_processor=video_processor
    )
    
    # Create collate function
    collate_fn = OptimizedGPUCollate(
        video_processor=video_processor,
        text_encoder=text_encoder,
        device=device,
        max_frames=num_frames
    )
    
    # Create test loader
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,  # Use single worker for test
        pin_memory=True
    )
    
    print(f"Test loader created:")
    print(f"  Test batches: {len(test_loader)}")
    print(f"  Batch size: {batch_size}")
    
    return test_loader 