"""
Metrics utilities for QualiVision video quality assessment.

This module provides evaluation metrics including correlation coefficients
and VQualA challenge specific metrics.
"""

import numpy as np
from typing import List, Tuple, Dict, Any
from scipy.stats import spearmanr, pearsonr


def rank_corr(predictions: List[float], targets: List[float]) -> Tuple[float, float]:
    """
    Compute Spearman and Pearson correlation coefficients.
    
    Args:
        predictions: Predicted MOS scores
        targets: Ground truth MOS scores
        
    Returns:
        Tuple of (spearman_corr, pearson_corr)
    """
    if len(predictions) == 0 or len(targets) == 0:
        return 0.0, 0.0
    
    try:
        spearman_corr = spearmanr(predictions, targets).correlation
        pearson_corr = pearsonr(predictions, targets)[0]
        
        # Handle NaN values
        if np.isnan(spearman_corr):
            spearman_corr = 0.0
        if np.isnan(pearson_corr):
            pearson_corr = 0.0
            
        return spearman_corr, pearson_corr
    
    except Exception as e:
        print(f"Error computing correlations: {e}")
        return 0.0, 0.0


def compute_vquala_score(predictions: List[float], targets: List[float]) -> float:
    """
    Compute VQualA challenge score: (SROCC + PLCC) / 2
    
    Args:
        predictions: Predicted overall MOS scores
        targets: Ground truth overall MOS scores
        
    Returns:
        VQualA challenge score
    """
    srocc, plcc = rank_corr(predictions, targets)
    return (srocc + plcc) / 2.0


def compute_metrics(predictions: List[float], 
                   targets: List[float],
                   metric_names: List[str] = None) -> Dict[str, float]:
    """
    Compute comprehensive evaluation metrics.
    
    Args:
        predictions: Predicted MOS scores
        targets: Ground truth MOS scores
        metric_names: List of metrics to compute
        
    Returns:
        Dictionary of computed metrics
    """
    if metric_names is None:
        metric_names = ['spearman', 'pearson', 'vquala_score', 'mae', 'mse', 'rmse']
    
    metrics = {}
    
    predictions = np.array(predictions)
    targets = np.array(targets)
    
    # Correlation metrics
    if 'spearman' in metric_names or 'pearson' in metric_names or 'vquala_score' in metric_names:
        srocc, plcc = rank_corr(predictions.tolist(), targets.tolist())
        
        if 'spearman' in metric_names:
            metrics['spearman'] = srocc
        if 'pearson' in metric_names:
            metrics['pearson'] = plcc
        if 'vquala_score' in metric_names:
            metrics['vquala_score'] = (srocc + plcc) / 2.0
    
    # Error metrics
    if 'mae' in metric_names:
        metrics['mae'] = np.mean(np.abs(predictions - targets))
    
    if 'mse' in metric_names:
        metrics['mse'] = np.mean((predictions - targets) ** 2)
    
    if 'rmse' in metric_names:
        metrics['rmse'] = np.sqrt(np.mean((predictions - targets) ** 2))
    
    # Additional metrics
    if 'std_pred' in metric_names:
        metrics['std_pred'] = np.std(predictions)
    
    if 'std_target' in metric_names:
        metrics['std_target'] = np.std(targets)
    
    if 'mean_pred' in metric_names:
        metrics['mean_pred'] = np.mean(predictions)
    
    if 'mean_target' in metric_names:
        metrics['mean_target'] = np.mean(targets)
    
    return metrics


def print_metrics(metrics: Dict[str, float], title: str = "Metrics"):
    """
    Pretty print metrics dictionary.
    
    Args:
        metrics: Dictionary of metrics
        title: Title for the metrics display
    """
    print(f"\n{title}:")
    print("=" * (len(title) + 1))
    
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"  {key:>12}: {value:.4f}")
        else:
            print(f"  {key:>12}: {value}")
    print()


def evaluate_all_dimensions(predictions: np.ndarray, 
                          targets: np.ndarray,
                          dimension_names: List[str] = None) -> Dict[str, Dict[str, float]]:
    """
    Evaluate all MOS dimensions separately.
    
    Args:
        predictions: Predicted MOS scores (N, 5)
        targets: Ground truth MOS scores (N, 5)
        dimension_names: Names for each dimension
        
    Returns:
        Dictionary with metrics for each dimension
    """
    if dimension_names is None:
        dimension_names = ['Traditional', 'Alignment', 'Aesthetic', 'Temporal', 'Overall']
    
    results = {}
    
    for i, dim_name in enumerate(dimension_names):
        dim_predictions = predictions[:, i]
        dim_targets = targets[:, i]
        
        dim_metrics = compute_metrics(
            dim_predictions.tolist(),
            dim_targets.tolist(),
            ['spearman', 'pearson', 'vquala_score', 'mae', 'rmse']
        )
        
        results[dim_name] = dim_metrics
    
    return results 