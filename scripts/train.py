#!/usr/bin/env python3
"""
QualiVision Model Training Script

This script provides training capabilities for both DOVER++ and V-JEPA2 models.
"""

import argparse
import os
import sys
from pathlib import Path
from datetime import datetime

import torch
import wandb
from torch.cuda.amp import GradScaler

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.models.dover_model import DOVERModel
from src.models.vjepa_model import VJEPAModel
from src.utils.dataset import create_data_loaders
from src.utils.training import HybridLossFunction, train_epoch, evaluate
from src.utils.metrics import compute_vquala_score
from src.utils.memory import ultra_memory_cleanup
from src.config.config import *


def create_model(model_type: str, device: str = 'cuda'):
    """Create model based on type."""
    if model_type == 'dover':
        config = DOVER_CONFIG
        model = DOVERModel(
            dover_weights_path=str(MODELS_DIR / "DOVER_plus_plus.pth"),
            text_encoder_name=config["text_encoder"],
            device=device
        )
    elif model_type == 'vjepa':
        config = VJEPA_CONFIG
        model = VJEPAModel(
            vjepa_model_id=config["video_encoder"],
            text_model_id=config["text_encoder"],
            freeze_ratio=config["freeze_ratio"],
            device=device
        )
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    
    return model, config


def main():
    parser = argparse.ArgumentParser(description="Train QualiVision models")
    parser.add_argument('--model', choices=['dover', 'vjepa'], required=True)
    parser.add_argument('--data', type=str, required=True, help='Path to data directory')
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--batch-size', type=int, default=None)
    parser.add_argument('--lr', type=float, default=None)
    parser.add_argument('--output', type=str, default='models')
    parser.add_argument('--resume', type=str, help='Resume from checkpoint')
    parser.add_argument('--wandb', action='store_true', help='Use W&B logging')
    
    args = parser.parse_args()
    
    # Create model and get config
    model, config = create_model(args.model)
    
    # Override config with command line args
    if args.batch_size:
        config["batch_size"] = args.batch_size
    if args.lr:
        config["learning_rate"] = args.lr
    
    print(f"Training {args.model.upper()} model")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {config['batch_size']}")
    print(f"Learning rate: {config['learning_rate']}")
    
    # Create data loaders
    train_csv = Path(args.data) / "train" / "labels" / "train_labels.csv"
    val_csv = Path(args.data) / "val" / "labels" / "val_labels.csv"
    train_video_dir = Path(args.data) / "train" / "videos"
    val_video_dir = Path(args.data) / "val" / "videos"
    
    train_loader, val_loader = create_data_loaders(
        train_csv=str(train_csv),
        val_csv=str(val_csv),
        train_video_dir=str(train_video_dir),
        val_video_dir=str(val_video_dir),
        batch_size=config["batch_size"],
        num_frames=config["num_frames"],
        resolution=config["video_resolution"][0]
    )
    
    # Training setup
    loss_fn = HybridLossFunction()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = GradScaler()
    
    # Initialize W&B if requested
    if args.wandb:
        wandb.init(project="QualiVision", config=config)
    
    best_score = 0.0
    
    # Training loop
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        
        # Train
        train_metrics = train_epoch(
            model=model,
            train_loader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            loss_fn=loss_fn,
            epoch=epoch
        )
        
        # Validate
        val_metrics, predictions, targets = evaluate(
            model=model,
            val_loader=val_loader,
            loss_fn=loss_fn
        )
        
        # Compute VQualA score
        vquala_score = compute_vquala_score(predictions, targets)
        
        print(f"Epoch {epoch+1} Results:")
        print(f"  Train Loss: {train_metrics['train_loss']:.4f}")
        print(f"  Val Loss: {val_metrics['val_loss']:.4f}")
        print(f"  VQualA Score: {vquala_score:.4f}")
        
        # Save best model
        if vquala_score > best_score:
            best_score = vquala_score
            os.makedirs(args.output, exist_ok=True)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_score': best_score,
                'config': config
            }, f"{args.output}/{args.model}_best.pt")
            print(f"  New best model saved! Score: {best_score:.4f}")
        
        # Log to W&B
        if args.wandb:
            wandb.log({
                'epoch': epoch,
                'train_loss': train_metrics['train_loss'],
                'val_loss': val_metrics['val_loss'],
                'vquala_score': vquala_score,
                'best_score': best_score
            })
        
        ultra_memory_cleanup()
    
    print(f"\nTraining completed! Best VQualA score: {best_score:.4f}")


if __name__ == '__main__':
    main() 