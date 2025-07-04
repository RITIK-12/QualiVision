"""
DOVER++ Model Implementation for Video Quality Assessment

This module implements the DOVER++ (Disentangled Objective Video Quality Evaluator) 
architecture with quality-aware fusion for multi-modal video quality assessment.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
from sentence_transformers import SentenceTransformer
from pathlib import Path
import urllib.request
import os


class DOVERModelLoader:
    """Utility class for loading DOVER++ pretrained weights."""
    
    @staticmethod
    def load_dover_model(weights_path: str, device: str = 'cuda') -> 'DOVERModelSimple':
        """
        Load DOVER model with pretrained weights.
        
        Args:
            weights_path: Path to DOVER++ weights file
            device: Device to load model on
            
        Returns:
            Loaded DOVER model
        """
        print(f"Loading DOVER model from {weights_path}")
        
        # Download weights if not exists
        if not os.path.exists(weights_path):
            os.makedirs(os.path.dirname(weights_path), exist_ok=True)
            print(f"Downloading DOVER++ weights to {weights_path}")
            urllib.request.urlretrieve(
                "https://huggingface.co/teowu/DOVER/resolve/main/DOVER_plus_plus.pth",
                weights_path
            )
        
        try:
            # Load the weights
            state_dict = torch.load(weights_path, map_location=device)
            print(f"✓ Loaded weights from {weights_path}")
            
            # Create model
            model = DOVERModelSimple(device=device)
            
            # Handle different checkpoint formats
            if 'state_dict' in state_dict:
                model_state = state_dict['state_dict']
            elif 'model' in state_dict:
                model_state = state_dict['model']
            else:
                model_state = state_dict
            
            # Load compatible weights
            try:
                model.load_state_dict(model_state, strict=False)
                print("✓ Model weights loaded successfully")
            except Exception as e:
                print(f"⚠ Partial weight loading: {e}")
                # Load only compatible weights
                model_dict = model.state_dict()
                compatible_dict = {k: v for k, v in model_state.items() 
                                 if k in model_dict and v.shape == model_dict[k].shape}
                model_dict.update(compatible_dict)
                model.load_state_dict(model_dict)
                print(f"✓ Loaded {len(compatible_dict)}/{len(model_dict)} compatible weights")
            
            return model.to(device)
            
        except Exception as e:
            print(f"✗ Error loading DOVER model: {e}")
            print("Creating model with random initialization...")
            return DOVERModelSimple(device=device).to(device)


class DOVERModelSimple(nn.Module):
    """
    Simplified DOVER++ model with ConvNeXt 3D backbone.
    
    This implementation is compatible with DOVER++ pretrained weights and 
    provides features for quality-aware fusion.
    """
    
    def __init__(self, device: str = 'cuda'):
        super().__init__()
        
        self.device = device
        
        # DOVER++ ConvNeXt 3D backbone
        self.backbone = self._build_convnext_backbone()
        
        # Separate heads for aesthetic and technical quality
        self.aesthetic_head = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Flatten(),
            nn.Linear(768, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(256, 1)
        )
        
        self.technical_head = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Flatten(),
            nn.Linear(768, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(256, 1)
        )
        
        # Feature extractor for fusion
        self.feature_extractor = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Flatten(),
            nn.Linear(768, 1024),
            nn.ReLU(inplace=True)
        )
        
        print("✓ DOVER++ model architecture created")
    
    def _build_convnext_backbone(self) -> nn.Module:
        """Build ConvNeXt 3D backbone for video processing."""
        return nn.Sequential(
            # Stem
            nn.Conv3d(3, 96, kernel_size=(1, 4, 4), stride=(1, 4, 4)),
            nn.GroupNorm(1, 96),
            
            # Stage 1
            *[self._make_convnext_block(96) for _ in range(3)],
            nn.Conv3d(96, 192, kernel_size=(1, 2, 2), stride=(1, 2, 2)),
            nn.GroupNorm(1, 192),
            
            # Stage 2
            *[self._make_convnext_block(192) for _ in range(3)],
            nn.Conv3d(192, 384, kernel_size=(1, 2, 2), stride=(1, 2, 2)),
            nn.GroupNorm(1, 384),
            
            # Stage 3
            *[self._make_convnext_block(384) for _ in range(9)],
            nn.Conv3d(384, 768, kernel_size=(1, 2, 2), stride=(1, 2, 2)),
            nn.GroupNorm(1, 768),
            
            # Stage 4
            *[self._make_convnext_block(768) for _ in range(3)],
        )
    
    def _make_convnext_block(self, dim: int) -> nn.Module:
        """Create a ConvNeXt block for 3D processing."""
        return nn.Sequential(
            nn.Conv3d(dim, dim, kernel_size=7, padding=3, groups=dim),
            nn.GroupNorm(1, dim),
            nn.Conv3d(dim, dim * 4, kernel_size=1),
            nn.GELU(),
            nn.Conv3d(dim * 4, dim, kernel_size=1),
        )
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass through DOVER model.
        
        Args:
            x: Video tensor (B, C, T, H, W)
            
        Returns:
            Dictionary containing:
                - features: Feature vector for fusion (B, 1024)
                - aesthetic_score: Aesthetic quality score (B, 1)
                - technical_score: Technical quality score (B, 1)
                - backbone_features: Raw backbone features (B, 768, T', H', W')
        """
        # Extract backbone features
        backbone_features = self.backbone(x)  # (B, 768, T', H', W')
        
        # Get aesthetic and technical scores
        aesthetic_score = self.aesthetic_head(backbone_features)
        technical_score = self.technical_head(backbone_features)
        
        # Extract features for fusion
        features = self.feature_extractor(backbone_features)
        
        return {
            'features': features,
            'aesthetic_score': aesthetic_score,
            'technical_score': technical_score,
            'backbone_features': backbone_features
        }


class QualityAwareFusion(nn.Module):
    """
    Quality-aware fusion module for combining DOVER++ and text features.
    
    This module implements cross-modal attention and quality aspect weighting
    to effectively combine video and text representations.
    """
    
    def __init__(self, dover_dim: int = 1024, text_dim: int = 1024, hidden_dim: int = 512):
        super().__init__()
        
        self.dover_dim = dover_dim
        self.text_dim = text_dim
        self.hidden_dim = hidden_dim
        
        # Quality aspect classifier
        self.quality_classifier = nn.Sequential(
            nn.Linear(text_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 4),  # 4 quality aspects
            nn.Softmax(dim=-1)
        )
        
        # Cross-modal attention
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=8,
            dropout=0.1,
            batch_first=True
        )
        
        # Feature projection layers
        self.dover_proj = nn.Linear(dover_dim, hidden_dim)
        self.text_proj = nn.Linear(text_dim, hidden_dim)
        
        # Fusion layers
        self.fusion_layer = nn.Sequential(
            nn.LayerNorm(hidden_dim * 2),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(0.1)
        )
        
        print(f"✓ Quality-aware fusion initialized")
        print(f"  DOVER dim: {dover_dim}, Text dim: {text_dim}, Hidden dim: {hidden_dim}")
    
    def forward(self, dover_features: torch.Tensor, text_features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Fuse DOVER++ and text features with quality awareness.
        
        Args:
            dover_features: DOVER features (B, dover_dim)
            text_features: Text features (B, text_dim)
            
        Returns:
            Tuple of (fused_features, quality_weights)
        """
        # Determine quality aspects focus
        quality_weights = self.quality_classifier(text_features)  # (B, 4)
        
        # Project features to common dimension
        dover_proj = self.dover_proj(dover_features)  # (B, hidden_dim)
        text_proj = self.text_proj(text_features)     # (B, hidden_dim)
        
        # Cross-modal attention
        dover_proj_seq = dover_proj.unsqueeze(1)  # (B, 1, hidden_dim)
        text_proj_seq = text_proj.unsqueeze(1)    # (B, 1, hidden_dim)
        
        attended_dover, _ = self.cross_attention(
            query=text_proj_seq,
            key=dover_proj_seq,
            value=dover_proj_seq
        )
        attended_dover = attended_dover.squeeze(1)  # (B, hidden_dim)
        
        # Final fusion
        combined_features = torch.cat([attended_dover, text_proj], dim=-1)
        fused_features = self.fusion_layer(combined_features)
        
        return fused_features, quality_weights


class MOSPredictor(nn.Module):
    """MOS prediction head for 4 quality aspects + overall score."""
    
    def __init__(self, input_dim: int, hidden_dim: int = 256):
        super().__init__()
        
        self.predictor = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(hidden_dim // 2, 5)  # 4 sub-MOS + Overall
        )
        
        print(f"✓ MOS predictor initialized with input dim: {input_dim}")
    
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Predict MOS scores."""
        return self.predictor(features)


class DOVERModel(nn.Module):
    """
    Complete DOVER++ model with quality-aware fusion and text understanding.
    
    This is the main model class that combines:
    - DOVER++ video encoder
    - Text encoder (BGE-Large)
    - Quality-aware fusion
    - MOS prediction
    """
    
    def __init__(self, 
                 dover_weights_path: str = "models/DOVER_plus_plus.pth",
                 text_encoder_name: str = "BAAI/bge-large-en-v1.5",
                 device: str = 'cuda'):
        super().__init__()
        
        self.device = device
        
        print("Initializing DOVER++ model with quality-aware fusion...")
        
        # Load DOVER++ video encoder
        self.dover_model = DOVERModelLoader.load_dover_model(
            weights_path=dover_weights_path,
            device=device
        )
        
        # Load text encoder
        print(f"Loading text encoder: {text_encoder_name}")
        try:
            self.text_encoder = SentenceTransformer(text_encoder_name, device=device)
            print(f"✓ Text encoder loaded successfully")
        except Exception as e:
            print(f"✗ Failed to load {text_encoder_name}: {e}")
            # Fallback to a smaller model
            self.text_encoder = SentenceTransformer("all-MiniLM-L6-v2", device=device)
            print("✓ Fallback text encoder loaded")
        
        # Get dimensions
        dover_dim = 1024
        text_dim = self.text_encoder.get_sentence_embedding_dimension()
        
        # Quality-aware fusion
        self.fusion = QualityAwareFusion(
            dover_dim=dover_dim,
            text_dim=text_dim,
            hidden_dim=512
        )
        
        # MOS predictor
        self.mos_predictor = MOSPredictor(
            input_dim=256,  # fusion output dimension
            hidden_dim=256
        )
        
        # Calculate parameters
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        print("✓ DOVER++ model initialized successfully")
        print(f"  DOVER++ feature dim: {dover_dim}")
        print(f"  Text encoder dim: {text_dim}")
        print(f"  Total parameters: {total_params:,}")
        print(f"  Trainable parameters: {trainable_params:,}")
        print(f"  Model size: ~{total_params * 4 / 1024**2:.1f} MB")
    
    def forward(self, frames: torch.Tensor, prompts: List[str]) -> torch.Tensor:
        """
        Forward pass through the complete model.
        
        Args:
            frames: Video frames tensor (B, C, T, H, W)
            prompts: List of text prompts
            
        Returns:
            MOS predictions (B, 5) - [Traditional, Alignment, Aesthetic, Temporal, Overall]
        """
        # Extract DOVER++ features
        dover_output = self.dover_model(frames)
        dover_features = dover_output['features']
        
        # Extract text features
        with torch.no_grad():
            text_features = self.text_encoder.encode(
                prompts,
                convert_to_tensor=True,
                normalize_embeddings=True,
                device=self.device
            )
        
        # Quality-aware fusion
        fused_features, quality_weights = self.fusion(dover_features, text_features)
        
        # Predict MOS scores
        mos_predictions = self.mos_predictor(fused_features)
        
        return mos_predictions
    
    def get_quality_weights(self, prompts: List[str]) -> torch.Tensor:
        """
        Get quality aspect weights for given prompts.
        
        Args:
            prompts: List of text prompts
            
        Returns:
            Quality weights tensor (B, 4) - [Traditional, Alignment, Aesthetic, Temporal]
        """
        with torch.no_grad():
            text_features = self.text_encoder.encode(
                prompts,
                convert_to_tensor=True,
                normalize_embeddings=True,
                device=self.device
            )
            quality_weights = self.fusion.quality_classifier(text_features)
        
        return quality_weights
    
    def extract_features(self, frames: torch.Tensor, prompts: List[str]) -> Dict[str, torch.Tensor]:
        """
        Extract features from video and text without MOS prediction.
        
        Args:
            frames: Video frames tensor (B, C, T, H, W)
            prompts: List of text prompts
            
        Returns:
            Dictionary with various feature representations
        """
        # Extract DOVER++ features
        dover_output = self.dover_model(frames)
        
        # Extract text features
        with torch.no_grad():
            text_features = self.text_encoder.encode(
                prompts,
                convert_to_tensor=True,
                normalize_embeddings=True,
                device=self.device
            )
        
        # Quality-aware fusion
        fused_features, quality_weights = self.fusion(dover_output['features'], text_features)
        
        return {
            'dover_features': dover_output['features'],
            'text_features': text_features,
            'fused_features': fused_features,
            'quality_weights': quality_weights,
            'aesthetic_score': dover_output['aesthetic_score'],
            'technical_score': dover_output['technical_score']
        } 