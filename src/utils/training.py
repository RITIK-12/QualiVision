"""
Training utilities for QualiVision models.

This module provides training functions, loss functions, and optimization utilities
for the VQualA 2025 challenge.
"""

import gc
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from tqdm import tqdm


def ultra_memory_cleanup():
    """Aggressive GPU memory cleanup."""
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()


class AdaptiveLossManager:
    """
    Adaptive loss weight manager that dynamically adjusts loss component weights
    during training based on their relative performance.
    """
    
    def __init__(self, initial_alpha: float = 0.7, initial_beta: float = 0.3, adaptation_rate: float = 0.1):
        """
        Initialize adaptive loss manager.
        
        Args:
            initial_alpha: Initial weight for smooth L1 loss
            initial_beta: Initial weight for ranking loss  
            adaptation_rate: Rate of adaptation for weight updates
        """
        self.alpha = initial_alpha
        self.beta = initial_beta
        self.mae_history = []
        self.ranking_history = []
        self.adaptation_rate = adaptation_rate
        
        print(f"AdaptiveLossManager initialized:")
        print(f"  Initial alpha (smooth_l1): {initial_alpha}")
        print(f"  Initial beta (ranking): {initial_beta}")
        print(f"  Adaptation rate: {adaptation_rate}")
    
    def update_weights(self, mae_loss: float, ranking_loss: float):
        """
        Update loss weights based on recent loss trends.
        
        Args:
            mae_loss: Current MAE loss value
            ranking_loss: Current ranking loss value
        """
        self.mae_history.append(mae_loss)
        self.ranking_history.append(ranking_loss)
        
        # Keep only recent history
        if len(self.mae_history) > 10:
            self.mae_history = self.mae_history[-10:]
            self.ranking_history = self.ranking_history[-10:]
        
        # Adapt weights based on trends
        if len(self.mae_history) >= 6:
            mae_trend = np.mean(self.mae_history[-3:]) / np.mean(self.mae_history[-6:-3])
            ranking_trend = np.mean(self.ranking_history[-3:]) / np.mean(self.ranking_history[-6:-3])
            
            # If MAE is getting worse but ranking is improving, increase ranking weight
            if mae_trend > 1.1 and ranking_trend < 0.9:
                self.alpha = max(0.5, self.alpha - self.adaptation_rate)
                self.beta = min(0.5, self.beta + self.adaptation_rate)
            # If ranking is getting worse but MAE is improving, increase MAE weight
            elif ranking_trend > 1.1 and mae_trend < 0.9:
                self.alpha = min(0.8, self.alpha + self.adaptation_rate)
                self.beta = max(0.2, self.beta - self.adaptation_rate)
        
        # Normalize weights
        total = self.alpha + self.beta
        self.alpha = self.alpha / total
        self.beta = self.beta / total
    
    def get_weights(self) -> Tuple[float, float]:
        """Get current loss weights."""
        return self.alpha, self.beta


class HybridLossFunction:
    """
    Hybrid loss function combining multiple loss components for robust MOS prediction.
    
    Components:
    - Smooth L1 loss for basic regression
    - Ranking loss for preserving relative order
    - Scale-aware loss for emphasizing extreme quality values
    """
    
    def __init__(self, 
                 smooth_l1_beta: float = 0.1,
                 ranking_margin: float = 0.2,
                 scale_weights: Dict[str, float] = None,
                 use_adaptive_weighting: bool = True):
        """
        Initialize hybrid loss function.
        
        Args:
            smooth_l1_beta: Beta parameter for smooth L1 loss
            ranking_margin: Margin for ranking loss
            scale_weights: Weights for different quality ranges
            use_adaptive_weighting: Whether to use adaptive loss weighting
        """
        self.smooth_l1_beta = smooth_l1_beta
        self.ranking_margin = ranking_margin
        self.use_adaptive_weighting = use_adaptive_weighting
        
        if scale_weights is None:
            scale_weights = {'low_quality': 1.5, 'high_quality': 1.5, 'normal': 1.0}
        self.scale_weights = scale_weights
        
        if use_adaptive_weighting:
            self.loss_manager = AdaptiveLossManager()
        else:
            self.loss_manager = None
        
        print(f"HybridLossFunction initialized:")
        print(f"  Smooth L1 beta: {smooth_l1_beta}")
        print(f"  Ranking margin: {ranking_margin}")
        print(f"  Scale weights: {scale_weights}")
        print(f"  Adaptive weighting: {use_adaptive_weighting}")
    
    def __call__(self, pred: torch.Tensor, target: torch.Tensor, epoch: int = 0) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute hybrid loss.
        
        Args:
            pred: Predicted MOS scores (B, 5)
            target: Target MOS scores (B, 5)
            epoch: Current training epoch
            
        Returns:
            Tuple of (total_loss, loss_components_dict)
        """
        device = pred.device
        batch_size = pred.shape[0]
        
        # Smooth L1 loss component
        smooth_l1_loss = F.smooth_l1_loss(pred, target, beta=self.smooth_l1_beta)
        
        # Ranking loss component
        ranking_loss = torch.tensor(0.0, device=device)
        
        if batch_size > 1:
            total_pairs = 0
            for i in range(batch_size):
                for j in range(i + 1, batch_size):
                    for dim in range(pred.shape[1]):
                        pred_diff = pred[i, dim] - pred[j, dim]
                        target_diff = target[i, dim] - target[j, dim]
                        
                        # Only apply ranking loss for significant differences
                        if target_diff > 0.1:
                            ranking_loss += torch.clamp(self.ranking_margin - pred_diff, min=0)
                        elif target_diff < -0.1:
                            ranking_loss += torch.clamp(self.ranking_margin + pred_diff, min=0)
                        
                        total_pairs += 1
            
            if total_pairs > 0:
                ranking_loss = ranking_loss / total_pairs
        
        # Scale-aware loss component
        scale_weights_tensor = torch.where(
            target < 2.5, 
            self.scale_weights['low_quality'],
            torch.where(
                target > 4.0, 
                self.scale_weights['high_quality'], 
                self.scale_weights['normal']
            )
        )
        scale_loss = F.mse_loss(pred * scale_weights_tensor, target * scale_weights_tensor)
        
        # Get loss weights
        if self.loss_manager is not None:
            alpha, beta = self.loss_manager.get_weights()
            self.loss_manager.update_weights(smooth_l1_loss.item(), ranking_loss.item())
        else:
            alpha, beta = 0.7, 0.3
        
        gamma = 0.1  # Fixed weight for scale loss
        
        # Combine losses
        total_loss = alpha * smooth_l1_loss + beta * ranking_loss + gamma * scale_loss
        
        loss_components = {
            'total_loss': total_loss.item(),
            'smooth_l1_loss': smooth_l1_loss.item(),
            'ranking_loss': ranking_loss.item(),
            'scale_loss': scale_loss.item(),
            'alpha': alpha,
            'beta': beta,
            'gamma': gamma
        }
        
        return total_loss, loss_components


def train_epoch(model: nn.Module,
                train_loader: torch.utils.data.DataLoader,
                optimizer: torch.optim.Optimizer,
                scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
                scaler: torch.cuda.amp.GradScaler,
                loss_fn: HybridLossFunction,
                accumulation_steps: int = 8,
                epoch: int = 0,
                device: str = 'cuda',
                max_grad_norm: float = 1.0,
                log_interval: int = 50) -> Dict[str, float]:
    """
    Train model for one epoch.
    
    Args:
        model: Model to train
        train_loader: Training data loader
        optimizer: Optimizer
        scheduler: Learning rate scheduler
        scaler: Gradient scaler for mixed precision
        loss_fn: Loss function
        accumulation_steps: Gradient accumulation steps
        epoch: Current epoch number
        device: Device to use
        max_grad_norm: Maximum gradient norm for clipping
        log_interval: Logging interval in batches
        
    Returns:
        Dictionary with training metrics
    """
    model.train()
    
    total_loss = 0.0
    total_smooth_l1 = 0.0
    total_ranking = 0.0
    total_scale = 0.0
    num_batches = 0
    
    optimizer.zero_grad(set_to_none=True)
    
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
    
    for i, batch in enumerate(progress_bar):
        try:
            # Forward pass with mixed precision
            with torch.cuda.amp.autocast():
                if isinstance(model, torch.nn.DataParallel):
                    outputs = model(batch['pixel_values_videos'], batch['prompts'])
                else:
                    # Handle different model interfaces
                    if hasattr(model, 'forward') and 'text_emb' in batch:
                        outputs = model(batch['pixel_values_videos'], batch['text_emb'])
                    else:
                        outputs = model(batch['pixel_values_videos'], batch['prompts'])
                
                loss, loss_components = loss_fn(outputs, batch['labels'].to(outputs.device), epoch)
                loss = loss / accumulation_steps
            
            # Backward pass
            scaler.scale(loss).backward()
            
            # Gradient accumulation
            if (i + 1) % accumulation_steps == 0:
                # Gradient clipping
                scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
                
                # Optimizer step
                scaler.step(optimizer)
                scaler.update()
                
                if scheduler is not None:
                    scheduler.step()
                
                optimizer.zero_grad(set_to_none=True)
            
            # Update metrics
            total_loss += loss_components['total_loss']
            total_smooth_l1 += loss_components['smooth_l1_loss']
            total_ranking += loss_components['ranking_loss']
            total_scale += loss_components['scale_loss']
            num_batches += 1
            
            # Update progress bar
            if num_batches > 0:
                avg_loss = total_loss / num_batches
                progress_bar.set_postfix({
                    'Loss': f'{avg_loss:.4f}',
                    'LR': f'{optimizer.param_groups[0]["lr"]:.2e}'
                })
            
            # Memory cleanup
            del outputs, loss
            if i % 10 == 0:
                ultra_memory_cleanup()
            
            # Logging
            if i % log_interval == 0 and i > 0:
                allocated = torch.cuda.memory_allocated() / 1e9
                print(f"    Batch {i}/{len(train_loader)}, "
                      f"Loss: {total_loss/num_batches:.4f}, "
                      f"Memory: {allocated:.1f}GB")
                
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"⚠ OOM at batch {i}, skipping...")
                optimizer.zero_grad(set_to_none=True)
                ultra_memory_cleanup()
                continue
            else:
                raise e
    
    # Handle remaining gradients
    if num_batches % accumulation_steps != 0:
        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
        scaler.step(optimizer)
        scaler.update()
        if scheduler is not None:
            scheduler.step()
        optimizer.zero_grad(set_to_none=True)
    
    ultra_memory_cleanup()
    
    # Return metrics
    metrics = {
        'train_loss': total_loss / max(num_batches, 1),
        'train_smooth_l1': total_smooth_l1 / max(num_batches, 1),
        'train_ranking': total_ranking / max(num_batches, 1),
        'train_scale': total_scale / max(num_batches, 1),
        'num_batches': num_batches
    }
    
    return metrics


def evaluate(model: nn.Module,
             val_loader: torch.utils.data.DataLoader,
             loss_fn: HybridLossFunction,
             device: str = 'cuda') -> Tuple[Dict[str, float], List[float], List[float]]:
    """
    Evaluate model on validation set.
    
    Args:
        model: Model to evaluate
        val_loader: Validation data loader
        loss_fn: Loss function
        device: Device to use
        
    Returns:
        Tuple of (metrics_dict, predictions, targets)
    """
    model.eval()
    
    total_loss = 0.0
    total_smooth_l1 = 0.0
    total_ranking = 0.0
    total_scale = 0.0
    num_batches = 0
    
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        progress_bar = tqdm(val_loader, desc="Validation")
        
        for batch in progress_bar:
            try:
                # Forward pass
                if isinstance(model, torch.nn.DataParallel):
                    outputs = model(batch['pixel_values_videos'], batch['prompts'])
                else:
                    # Handle different model interfaces
                    if hasattr(model, 'forward') and 'text_emb' in batch:
                        outputs = model(batch['pixel_values_videos'], batch['text_emb'])
                    else:
                        outputs = model(batch['pixel_values_videos'], batch['prompts'])
                
                loss, loss_components = loss_fn(outputs, batch['labels'].to(outputs.device))
                
                # Update metrics
                total_loss += loss_components['total_loss']
                total_smooth_l1 += loss_components['smooth_l1_loss']
                total_ranking += loss_components['ranking_loss']
                total_scale += loss_components['scale_loss']
                num_batches += 1
                
                # Collect predictions and targets (overall MOS only)
                pred_overall = outputs[:, -1].cpu().tolist()  # Last column is overall MOS
                target_overall = batch['labels'][:, -1].cpu().tolist()
                
                all_predictions.extend(pred_overall)
                all_targets.extend(target_overall)
                
                # Update progress bar
                avg_loss = total_loss / num_batches
                progress_bar.set_postfix({'Val Loss': f'{avg_loss:.4f}'})
                
                # Memory cleanup
                del outputs, loss
                ultra_memory_cleanup()
                
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    print(f"⚠ OOM during validation, skipping batch...")
                    ultra_memory_cleanup()
                    continue
                else:
                    raise e
    
    # Calculate metrics
    metrics = {
        'val_loss': total_loss / max(num_batches, 1),
        'val_smooth_l1': total_smooth_l1 / max(num_batches, 1),
        'val_ranking': total_ranking / max(num_batches, 1),
        'val_scale': total_scale / max(num_batches, 1),
        'num_val_batches': num_batches
    }
    
    return metrics, all_predictions, all_targets


def create_optimizer(model: nn.Module,
                    learning_rate: float = 1e-4,
                    weight_decay: float = 1e-2,
                    discriminative_lr: Optional[Dict[str, float]] = None) -> torch.optim.Optimizer:
    """
    Create optimizer with optional discriminative learning rates.
    
    Args:
        model: Model to optimize
        learning_rate: Base learning rate
        weight_decay: Weight decay
        discriminative_lr: Dictionary with component-specific LR multipliers
        
    Returns:
        Configured optimizer
    """
    if discriminative_lr is not None and hasattr(model, 'get_discriminative_params'):
        # Use discriminative learning rates for V-JEPA2 model
        param_groups = model.get_discriminative_params()
        
        for i, group in enumerate(param_groups):
            component_name = group['name']
            if 'text' in component_name:
                group['lr'] = learning_rate * discriminative_lr.get('text', 1.0)
            elif 'video' in component_name:
                group['lr'] = learning_rate * discriminative_lr.get('video', 1.0)
            elif 'head' in component_name:
                group['lr'] = learning_rate * discriminative_lr.get('head', 1.0)
            else:
                group['lr'] = learning_rate
            
            group['weight_decay'] = weight_decay
        
        optimizer = torch.optim.AdamW(param_groups)
        
        print(f"Discriminative optimizer created:")
        for group in param_groups:
            print(f"  {group['name']}: LR={group['lr']:.2e}")
    
    else:
        # Standard optimizer
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            betas=(0.9, 0.999),
            eps=1e-8
        )
        
        print(f"Standard optimizer created: LR={learning_rate:.2e}, WD={weight_decay:.2e}")
    
    return optimizer


def create_scheduler(optimizer: torch.optim.Optimizer,
                    num_training_steps: int,
                    warmup_steps: int = 100,
                    scheduler_type: str = 'cosine') -> torch.optim.lr_scheduler._LRScheduler:
    """
    Create learning rate scheduler.
    
    Args:
        optimizer: Optimizer to schedule
        num_training_steps: Total number of training steps
        warmup_steps: Number of warmup steps
        scheduler_type: Type of scheduler ('cosine', 'linear', 'constant')
        
    Returns:
        Configured scheduler
    """
    if scheduler_type == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=num_training_steps, eta_min=1e-6
        )
    elif scheduler_type == 'linear':
        scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=0.1, end_factor=1.0, total_iters=num_training_steps
        )
    else:  # constant
        scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer, factor=1.0)
    
    print(f"Scheduler created: {scheduler_type}, steps={num_training_steps}, warmup={warmup_steps}")
    
    return scheduler 