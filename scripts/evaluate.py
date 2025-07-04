#!/usr/bin/env python3
"""
QualiVision Model Evaluation Script

This script allows users to evaluate trained QualiVision models on test data.
It supports both DOVER++ and V-JEPA2 models and can generate predictions
in various formats.

Usage:
    python scripts/evaluate.py --model dover --checkpoint models/dover_best.pt --data data/test
    python scripts/evaluate.py --model vjepa --checkpoint models/vjepa_best.pt --data data/test
"""

import argparse
import os
import sys
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional

import torch
import pandas as pd
import numpy as np
from tqdm import tqdm

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.models.dover_model import DOVERModel
from src.models.vjepa_model import VJEPAModel
from src.utils.dataset import create_test_loader
from src.utils.metrics import compute_metrics, compute_vquala_score, print_metrics
from src.utils.memory import ultra_memory_cleanup, print_gpu_memory
from src.config.config import *


class ModelEvaluator:
    """Main evaluator class for QualiVision models."""
    
    def __init__(self, 
                 model_type: str,
                 checkpoint_path: str,
                 device: str = 'cuda'):
        """
        Initialize the evaluator.
        
        Args:
            model_type: Type of model ('dover' or 'vjepa')
            checkpoint_path: Path to model checkpoint
            device: Device to use for evaluation
        """
        self.model_type = model_type.lower()
        self.checkpoint_path = checkpoint_path
        self.device = device
        
        print(f"Initializing {model_type.upper()} Model Evaluator")
        print(f"Checkpoint: {checkpoint_path}")
        print(f"Device: {device}")
        
        # Load model
        self.model = self._load_model()
        self.model.eval()
        
        print("✓ Model loaded successfully")
        print_gpu_memory()
    
    def _load_model(self) -> torch.nn.Module:
        """Load the specified model with checkpoint."""
        if self.model_type == 'dover':
            return self._load_dover_model()
        elif self.model_type == 'vjepa':
            return self._load_vjepa_model()
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")
    
    def _load_dover_model(self) -> DOVERModel:
        """Load DOVER++ model."""
        model = DOVERModel(
            dover_weights_path=str(MODELS_DIR / "DOVER_plus_plus.pth"),
            text_encoder_name=DOVER_CONFIG["text_encoder"],
            device=self.device
        )
        
        # Load checkpoint if provided
        if os.path.exists(self.checkpoint_path):
            print(f"Loading DOVER checkpoint: {self.checkpoint_path}")
            checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
            
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)
            
            print("✓ DOVER checkpoint loaded")
        else:
            print(f"⚠ Checkpoint not found: {self.checkpoint_path}")
            print("Using model with pre-trained weights only")
        
        return model.to(self.device)
    
    def _load_vjepa_model(self) -> VJEPAModel:
        """Load V-JEPA2 model."""
        model = VJEPAModel(
            vjepa_model_id=VJEPA_CONFIG["video_encoder"],
            text_model_id=VJEPA_CONFIG["text_encoder"],
            freeze_ratio=VJEPA_CONFIG["freeze_ratio"],
            device=self.device
        )
        
        # Load checkpoint if provided
        if os.path.exists(self.checkpoint_path):
            print(f"Loading V-JEPA2 checkpoint: {self.checkpoint_path}")
            checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
            
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)
            
            print("✓ V-JEPA2 checkpoint loaded")
        else:
            print(f"⚠ Checkpoint not found: {self.checkpoint_path}")
            print("Using model with random initialization")
        
        return model.to(self.device)
    
    def evaluate_dataset(self, 
                        test_csv: str,
                        test_video_dir: str,
                        output_dir: str = "results",
                        batch_size: int = 1) -> Dict[str, Any]:
        """
        Evaluate model on test dataset.
        
        Args:
            test_csv: Path to test CSV file
            test_video_dir: Directory containing test videos
            output_dir: Directory to save results
            batch_size: Batch size for evaluation
            
        Returns:
            Dictionary with evaluation results
        """
        print(f"\nEvaluating on test dataset:")
        print(f"  CSV: {test_csv}")
        print(f"  Videos: {test_video_dir}")
        print(f"  Batch size: {batch_size}")
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Get model configuration
        config = DOVER_CONFIG if self.model_type == 'dover' else VJEPA_CONFIG
        
        # Create test loader
        test_loader = create_test_loader(
            test_csv=test_csv,
            test_video_dir=test_video_dir,
            batch_size=batch_size,
            num_frames=config["num_frames"],
            resolution=config["video_resolution"][0],
            video_processor=getattr(self.model, 'vproc', None),
            text_encoder=getattr(self.model, 'text_encoder', None) or getattr(self.model, 'tenc', None),
            device=self.device
        )
        
        # Run evaluation
        results = self._predict_on_loader(test_loader)
        
        # Check if we have ground truth labels
        test_df = pd.read_csv(test_csv)
        has_labels = 'Overall_MOS' in test_df.columns
        
        if has_labels:
            print("✓ Ground truth labels found, computing metrics")
            metrics = self._compute_metrics(results)
            results['metrics'] = metrics
        else:
            print("⚠ No ground truth labels found, skipping metrics computation")
            results['metrics'] = {}
        
        # Save results
        self._save_results(results, output_dir, config)
        
        return results
    
    def _predict_on_loader(self, test_loader) -> Dict[str, Any]:
        """Run predictions on data loader."""
        predictions = []
        targets = []
        video_names = []
        
        print("\nGenerating predictions...")
        
        with torch.no_grad():
            for i, batch in enumerate(tqdm(test_loader, desc="Predicting")):
                try:
                    # Forward pass
                    if self.model_type == 'dover':
                        outputs = self.model(batch['pixel_values_videos'], batch['prompts'])
                    else:  # vjepa
                        outputs = self.model(batch['pixel_values_videos'], batch['text_emb'])
                    
                    # Extract predictions
                    batch_predictions = outputs.cpu().numpy()
                    batch_targets = batch['labels'].cpu().numpy()
                    
                    predictions.append(batch_predictions)
                    targets.append(batch_targets)
                    video_names.extend(batch['video_names'])
                    
                    # Memory cleanup
                    del outputs
                    if i % 10 == 0:
                        ultra_memory_cleanup()
                
                except Exception as e:
                    print(f"⚠ Error processing batch {i}: {e}")
                    # Add dummy predictions for failed batch
                    batch_size = len(batch['video_names'])
                    dummy_preds = np.full((batch_size, 5), 3.0)
                    dummy_targets = batch['labels'].cpu().numpy()
                    
                    predictions.append(dummy_preds)
                    targets.append(dummy_targets)
                    video_names.extend(batch['video_names'])
        
        # Concatenate all predictions
        predictions = np.concatenate(predictions, axis=0)
        targets = np.concatenate(targets, axis=0)
        
        print(f"✓ Generated predictions for {len(predictions)} samples")
        
        return {
            'predictions': predictions,
            'targets': targets,
            'video_names': video_names
        }
    
    def _compute_metrics(self, results: Dict[str, Any]) -> Dict[str, float]:
        """Compute evaluation metrics."""
        predictions = results['predictions']
        targets = results['targets']
        
        # Overall MOS metrics (last column)
        overall_pred = predictions[:, -1].tolist()
        overall_target = targets[:, -1].tolist()
        
        # Remove dummy targets (zeros) if present
        valid_indices = [i for i, t in enumerate(overall_target) if t > 0]
        if len(valid_indices) < len(overall_target):
            print(f"⚠ Found {len(overall_target) - len(valid_indices)} samples without ground truth")
            overall_pred = [overall_pred[i] for i in valid_indices]
            overall_target = [overall_target[i] for i in valid_indices]
        
        if len(overall_target) == 0:
            print("⚠ No valid ground truth labels found")
            return {}
        
        # Compute metrics
        metrics = compute_metrics(overall_pred, overall_target)
        vquala_score = compute_vquala_score(overall_pred, overall_target)
        metrics['vquala_score'] = vquala_score
        
        print_metrics(metrics, "Evaluation Results")
        
        return metrics
    
    def _save_results(self, results: Dict[str, Any], output_dir: str, config: Dict[str, Any]):
        """Save evaluation results to files."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name = self.model_type.upper()
        
        # Create predictions DataFrame
        predictions_df = pd.DataFrame({
            'video_name': results['video_names'],
            'Traditional_MOS': results['predictions'][:, 0],
            'Alignment_MOS': results['predictions'][:, 1],
            'Aesthetic_MOS': results['predictions'][:, 2],
            'Temporal_MOS': results['predictions'][:, 3],
            'Overall_MOS': results['predictions'][:, 4]
        })
        
        # Save predictions
        pred_csv = f"{output_dir}/predictions_{model_name}_{timestamp}.csv"
        pred_xlsx = f"{output_dir}/predictions_{model_name}_{timestamp}.xlsx"
        
        predictions_df.to_csv(pred_csv, index=False)
        predictions_df.to_excel(pred_xlsx, index=False)
        
        print(f"✓ Predictions saved:")
        print(f"  CSV: {pred_csv}")
        print(f"  Excel: {pred_xlsx}")
        
        # Save detailed results
        results_file = f"{output_dir}/results_{model_name}_{timestamp}.json"
        
        # Prepare results for JSON serialization
        json_results = {
            'model_type': self.model_type,
            'checkpoint_path': self.checkpoint_path,
            'timestamp': timestamp,
            'num_samples': len(results['video_names']),
            'config': config,
            'metrics': results.get('metrics', {}),
            'prediction_stats': {
                'min': float(np.min(results['predictions'][:, -1])),
                'max': float(np.max(results['predictions'][:, -1])),
                'mean': float(np.mean(results['predictions'][:, -1])),
                'std': float(np.std(results['predictions'][:, -1]))
            }
        }
        
        with open(results_file, 'w') as f:
            json.dump(json_results, f, indent=2)
        
        print(f"✓ Results saved: {results_file}")
        
        # Create summary report
        self._create_summary_report(json_results, f"{output_dir}/report_{model_name}_{timestamp}.txt")
    
    def _create_summary_report(self, results: Dict[str, Any], report_path: str):
        """Create a human-readable summary report."""
        with open(report_path, 'w') as f:
            f.write(f"QualiVision Model Evaluation Report\n")
            f.write(f"===================================\n\n")
            f.write(f"Model: {results['model_type'].upper()}\n")
            f.write(f"Checkpoint: {results['checkpoint_path']}\n")
            f.write(f"Timestamp: {results['timestamp']}\n")
            f.write(f"Samples: {results['num_samples']}\n\n")
            
            # Model configuration
            f.write(f"Model Configuration:\n")
            f.write(f"-------------------\n")
            config = results['config']
            for key, value in config.items():
                f.write(f"  {key}: {value}\n")
            f.write(f"\n")
            
            # Metrics
            if results['metrics']:
                f.write(f"Evaluation Metrics:\n")
                f.write(f"------------------\n")
                for key, value in results['metrics'].items():
                    if isinstance(value, float):
                        f.write(f"  {key}: {value:.4f}\n")
                    else:
                        f.write(f"  {key}: {value}\n")
                f.write(f"\n")
            
            # Prediction statistics
            f.write(f"Prediction Statistics:\n")
            f.write(f"---------------------\n")
            stats = results['prediction_stats']
            f.write(f"  Min: {stats['min']:.4f}\n")
            f.write(f"  Max: {stats['max']:.4f}\n")
            f.write(f"  Mean: {stats['mean']:.4f}\n")
            f.write(f"  Std: {stats['std']:.4f}\n")
        
        print(f"✓ Summary report saved: {report_path}")


def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(
        description="Evaluate QualiVision models on test data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Evaluate DOVER++ model
  python scripts/evaluate.py --model dover --checkpoint models/dover_best.pt --data data/test

  # Evaluate V-JEPA2 model
  python scripts/evaluate.py --model vjepa --checkpoint models/vjepa_best.pt --data data/test
  
  # Evaluate with custom output directory
  python scripts/evaluate.py --model dover --checkpoint models/dover_best.pt --data data/test --output results/dover
        """
    )
    
    parser.add_argument('--model', 
                       choices=['dover', 'vjepa'], 
                       required=True,
                       help='Model type to evaluate')
    
    parser.add_argument('--checkpoint', 
                       type=str, 
                       required=True,
                       help='Path to model checkpoint file')
    
    parser.add_argument('--data', 
                       type=str, 
                       required=True,
                       help='Path to test data directory')
    
    parser.add_argument('--output', 
                       type=str, 
                       default='results',
                       help='Output directory for results (default: results)')
    
    parser.add_argument('--batch-size', 
                       type=int, 
                       default=1,
                       help='Batch size for evaluation (default: 1)')
    
    parser.add_argument('--device', 
                       type=str, 
                       default='cuda',
                       help='Device to use (default: cuda)')
    
    parser.add_argument('--csv-name', 
                       type=str, 
                       default='test_labels.csv',
                       help='Name of test CSV file (default: test_labels.csv)')
    
    parser.add_argument('--video-dir', 
                       type=str, 
                       default='videos',
                       help='Name of video directory (default: videos)')
    
    args = parser.parse_args()
    
    # Validate inputs
    data_path = Path(args.data)
    if not data_path.exists():
        print(f"Error: Data directory not found: {data_path}")
        sys.exit(1)
    
    test_csv = data_path / args.csv_name
    test_video_dir = data_path / args.video_dir
    
    if not test_csv.exists():
        print(f"Error: Test CSV not found: {test_csv}")
        sys.exit(1)
    
    if not test_video_dir.exists():
        print(f"Error: Test video directory not found: {test_video_dir}")
        sys.exit(1)
    
    if not os.path.exists(args.checkpoint):
        print(f"Error: Checkpoint not found: {args.checkpoint}")
        sys.exit(1)
    
    # Check device
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("Warning: CUDA not available, using CPU")
        args.device = 'cpu'
    
    print(f"QualiVision Model Evaluation")
    print(f"============================")
    print(f"Model: {args.model.upper()}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Test CSV: {test_csv}")
    print(f"Test videos: {test_video_dir}")
    print(f"Output: {args.output}")
    print(f"Device: {args.device}")
    print()
    
    # Create evaluator and run evaluation
    try:
        evaluator = ModelEvaluator(
            model_type=args.model,
            checkpoint_path=args.checkpoint,
            device=args.device
        )
        
        results = evaluator.evaluate_dataset(
            test_csv=str(test_csv),
            test_video_dir=str(test_video_dir),
            output_dir=args.output,
            batch_size=args.batch_size
        )
        
        print("\n✓ Evaluation completed successfully!")
        
        if results.get('metrics'):
            vquala_score = results['metrics'].get('vquala_score', 0.0)
            print(f"Final VQualA Score: {vquala_score:.4f}")
        
    except Exception as e:
        print(f"\n✗ Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main() 