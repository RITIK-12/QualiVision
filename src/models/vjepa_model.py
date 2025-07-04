"""
V-JEPA2 Model Implementation for Video Quality Assessment

This module implements the V-JEPA2 (Video Joint Embedding Predictive Architecture) 
with discriminative learning and strategic layer freezing for video quality assessment.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
from transformers import AutoModel, AutoVideoProcessor
from sentence_transformers import SentenceTransformer


class OptimizedMOSHead(nn.Module):
    """
    Optimized MOS prediction head that combines video and text features.
    """
    
    def __init__(self, dv: int, dt: int, h: int = 512):
        """
        Initialize MOS prediction head.
        
        Args:
            dv: Video feature dimension
            dt: Text feature dimension  
            h: Hidden dimension
        """
        super().__init__()
        
        self.net = nn.Sequential(
            nn.LayerNorm(dv + dt),
            nn.Linear(dv + dt, h), 
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(h, h // 2), 
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(h // 2, 5)  # 4 sub-MOS + Overall
        )
        
        print(f"✓ OptimizedMOSHead initialized: Video={dv}, Text={dt}, Hidden={h}")
    
    def forward(self, v: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Forward pass combining video and text features.
        
        Args:
            v: Video features (B, dv)
            t: Text features (B, dt)
            
        Returns:
            MOS predictions (B, 5)
        """
        return self.net(torch.cat([v, t], dim=-1))


class VJEPAModel(nn.Module):
    """
    V-JEPA2 model with discriminative learning and strategic layer freezing.
    
    This model implements:
    - V-JEPA2 ViT-Giant video encoder
    - Strategic layer freezing (bottom 85% frozen)
    - BGE-Large text encoder
    - Discriminative learning rates
    - Optimized MOS prediction head
    """
    
    def __init__(self, 
                 vjepa_model_id: str = "facebook/vjepa2-vitg-fpc64-384-ssv2",
                 text_model_id: str = "BAAI/bge-large-en-v1.5",
                 freeze_ratio: float = 0.85,
                 device: str = 'cuda'):
        """
        Initialize V-JEPA2 model.
        
        Args:
            vjepa_model_id: HuggingFace model ID for V-JEPA2
            text_model_id: HuggingFace model ID for text encoder
            freeze_ratio: Ratio of layers to freeze (0.85 = freeze bottom 85%)
            device: Device to place model on
        """
        super().__init__()
        
        self.device = device
        self.freeze_ratio = freeze_ratio
        
        print(f"Initializing V-JEPA2 model with {freeze_ratio*100:.0f}% layer freezing...")
        
        # Video processor
        self.vproc = AutoVideoProcessor.from_pretrained(vjepa_model_id)
        
        # Video encoder - Use FP32 for stable gradients
        self.venc = AutoModel.from_pretrained(
            vjepa_model_id,
            torch_dtype=torch.float32,
            output_hidden_states=True,
            attn_implementation="sdpa",  # Use scaled dot-product attention
        )
        
        # Apply strategic layer freezing
        self._apply_strategic_freezing()
        
        # Disable gradient checkpointing for better gradient flow
        if hasattr(self.venc, 'gradient_checkpointing_enable'):
            self.venc.gradient_checkpointing_disable()
        
        # Text encoder
        self.tenc = SentenceTransformer(text_model_id, device=device)
        
        # Get dimensions
        dv = self.venc.config.hidden_size
        dt = self.tenc.get_sentence_embedding_dimension()
        
        # Prediction head
        self.head = OptimizedMOSHead(dv, dt, h=512)
        
        # Calculate and print model statistics
        self._print_model_stats()
        
        print(f"✓ V-JEPA2 model initialized successfully")
        print(f"  Video encoder: {vjepa_model_id}")
        print(f"  Text encoder: {text_model_id}")
        print(f"  Video features: {dv}")
        print(f"  Text features: {dt}")
    
    def _apply_strategic_freezing(self):
        """
        Apply strategic layer freezing to reduce memory usage and improve training.
        Freezes bottom 85% of transformer layers + embeddings + pooler.
        """
        frozen_count = 0
        trainable_count = 0
        total_layers = 0
        
        # Count total layers first
        for name, p in self.venc.named_parameters():
            if "encoder.layer." in name:
                layer_match = name.split("encoder.layer.")[1].split(".")[0]
                if layer_match.isdigit():
                    total_layers = max(total_layers, int(layer_match) + 1)
        
        print(f"Total transformer layers detected: {total_layers}")
        
        # Freeze bottom layers based on freeze_ratio
        freeze_until_layer = int(total_layers * self.freeze_ratio)
        print(f"Freezing layers 0-{freeze_until_layer-1}, training layers {freeze_until_layer}-{total_layers-1}")
        
        for name, p in self.venc.named_parameters():
            should_freeze = False
            
            # Always freeze embeddings and pooler
            if "embeddings" in name or "pooler" in name:
                should_freeze = True
            
            # Freeze bottom layers
            elif "encoder.layer." in name:
                layer_match = name.split("encoder.layer.")[1].split(".")[0]
                if layer_match.isdigit():
                    layer_num = int(layer_match)
                    if layer_num < freeze_until_layer:
                        should_freeze = True
            
            # Apply freezing
            if should_freeze:
                p.requires_grad = False
                frozen_count += 1
            else:
                p.requires_grad = True
                trainable_count += 1
        
        print(f"Layer freezing applied:")
        print(f"  Frozen parameters: {frozen_count:,}")
        print(f"  Trainable parameters: {trainable_count:,}")
        print(f"  Trainable ratio: {trainable_count/(frozen_count+trainable_count)*100:.1f}%")
        print(f"  Memory savings: ~{(frozen_count/(frozen_count+trainable_count))*100:.0f}% reduction in gradient computation")
    
    def _print_model_stats(self):
        """Print model statistics."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        print(f"Model statistics:")
        print(f"  Total parameters: {total_params:,}")
        print(f"  Trainable parameters: {trainable_params:,}")
        print(f"  Frozen parameters: {total_params - trainable_params:,}")
        print(f"  Model size: ~{total_params * 4 / 1024**2:.1f} MB")
        print(f"  Trainable size: ~{trainable_params * 4 / 1024**2:.1f} MB")
    
    def forward(self, pixel_values_videos: torch.Tensor, text_emb: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through V-JEPA2 model.
        
        Args:
            pixel_values_videos: Video tensor (B, C, T, H, W)
            text_emb: Text embeddings (B, text_dim)
            
        Returns:
            MOS predictions (B, 5) - [Traditional, Alignment, Aesthetic, Temporal, Overall]
        """
        # Ensure FP32 for stable gradients
        pixel_values_videos = pixel_values_videos.to(self.venc.device, dtype=torch.float32)
        
        # Forward pass through video encoder
        outputs = self.venc(pixel_values_videos=pixel_values_videos, output_hidden_states=True)
        
        # Get CLS token (first token)
        cls_token = outputs.last_hidden_state[:, 0]  # (B, hidden_size)
        
        # Ensure text embeddings match video features
        text_emb = text_emb.to(cls_token.device, dtype=cls_token.dtype)
        
        # MOS prediction
        mos_scores = self.head(cls_token, text_emb)
        
        return mos_scores
    
    def get_discriminative_params(self) -> List[Dict[str, Any]]:
        """
        Get parameter groups for discriminative learning rates.
        
        Returns:
            List of parameter groups with different learning rates
        """
        # Text encoder parameters (lowest lr)
        text_params = list(self.tenc.parameters())
        
        # Video encoder parameters (medium lr)
        video_params = [p for p in self.venc.parameters() if p.requires_grad]
        
        # Prediction head parameters (highest lr)
        head_params = list(self.head.parameters())
        
        param_groups = [
            {'params': text_params, 'name': 'text_encoder'},
            {'params': video_params, 'name': 'video_encoder'},
            {'params': head_params, 'name': 'prediction_head'}
        ]
        
        return param_groups
    
    def encode_text(self, prompts: List[str]) -> torch.Tensor:
        """
        Encode text prompts using the text encoder.
        
        Args:
            prompts: List of text prompts
            
        Returns:
            Text embeddings (B, text_dim)
        """
        with torch.no_grad():
            text_emb = self.tenc.encode(
                prompts,
                convert_to_tensor=True,
                normalize_embeddings=True,
                device=self.device
            )
        return text_emb
    
    def encode_video(self, pixel_values_videos: torch.Tensor) -> torch.Tensor:
        """
        Encode video using the video encoder.
        
        Args:
            pixel_values_videos: Video tensor (B, C, T, H, W)
            
        Returns:
            Video features (B, hidden_size)
        """
        pixel_values_videos = pixel_values_videos.to(self.venc.device, dtype=torch.float32)
        
        with torch.no_grad():
            outputs = self.venc(pixel_values_videos=pixel_values_videos, output_hidden_states=True)
            cls_token = outputs.last_hidden_state[:, 0]
        
        return cls_token
    
    def extract_features(self, pixel_values_videos: torch.Tensor, prompts: List[str]) -> Dict[str, torch.Tensor]:
        """
        Extract features from video and text without MOS prediction.
        
        Args:
            pixel_values_videos: Video tensor (B, C, T, H, W)
            prompts: List of text prompts
            
        Returns:
            Dictionary with video and text features
        """
        # Extract video features
        video_features = self.encode_video(pixel_values_videos)
        
        # Extract text features
        text_features = self.encode_text(prompts)
        
        return {
            'video_features': video_features,
            'text_features': text_features
        }
    
    def predict_with_features(self, video_features: torch.Tensor, text_features: torch.Tensor) -> torch.Tensor:
        """
        Predict MOS scores from pre-extracted features.
        
        Args:
            video_features: Video features (B, video_dim)
            text_features: Text features (B, text_dim)
            
        Returns:
            MOS predictions (B, 5)
        """
        return self.head(video_features, text_features)
    
    def freeze_video_encoder(self):
        """Freeze the entire video encoder."""
        for param in self.venc.parameters():
            param.requires_grad = False
        print("✓ Video encoder frozen")
    
    def unfreeze_video_encoder(self):
        """Unfreeze the video encoder (respecting strategic freezing)."""
        self._apply_strategic_freezing()
        print("✓ Video encoder unfrozen (with strategic freezing)")
    
    def freeze_text_encoder(self):
        """Freeze the text encoder."""
        for param in self.tenc.parameters():
            param.requires_grad = False
        print("✓ Text encoder frozen")
    
    def unfreeze_text_encoder(self):
        """Unfreeze the text encoder."""
        for param in self.tenc.parameters():
            param.requires_grad = True
        print("✓ Text encoder unfrozen")


def create_vjepa_model(config: Dict[str, Any]) -> VJEPAModel:
    """
    Create V-JEPA2 model from configuration.
    
    Args:
        config: Model configuration dictionary
        
    Returns:
        Initialized V-JEPA2 model
    """
    return VJEPAModel(
        vjepa_model_id=config.get('video_encoder', 'facebook/vjepa2-vitg-fpc64-384-ssv2'),
        text_model_id=config.get('text_encoder', 'BAAI/bge-large-en-v1.5'),
        freeze_ratio=config.get('freeze_ratio', 0.85),
        device=config.get('device', 'cuda')
    ) 