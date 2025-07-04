"""
Configuration file for QualiVision models and training parameters.
"""

import os
from pathlib import Path

# Base paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"
DOCS_DIR = PROJECT_ROOT / "docs"

# Data paths
TRAIN_DATA_PATH = DATA_DIR / "train"
VAL_DATA_PATH = DATA_DIR / "val"
TEST_DATA_PATH = DATA_DIR / "test"

# Model configurations
DOVER_CONFIG = {
    "model_name": "DOVER++",
    "video_resolution": (640, 640),
    "num_frames": 64,
    "text_encoder": "BAAI/bge-large-en-v1.5",
    "dover_dim": 1024,
    "text_dim": 1024,
    "hidden_dim": 512,
    "pretrained_weights": "https://huggingface.co/teowu/DOVER/resolve/main/DOVER_plus_plus.pth",
    "batch_size": 4,
    "learning_rate": 1e-4,
    "epochs": 5,
    "gradient_accumulation_steps": 8,
    "effective_batch_size": 32
}

VJEPA_CONFIG = {
    "model_name": "V-JEPA2",
    "video_resolution": (384, 384),
    "num_frames": 64,
    "text_encoder": "BAAI/bge-large-en-v1.5",
    "video_encoder": "facebook/vjepa-vit-giant-p16",
    "freeze_ratio": 0.85,  # Freeze bottom 85% of layers
    "video_dim": 1408,
    "text_dim": 768,
    "hidden_dim": 512,
    "batch_size": 6,
    "learning_rate": 2e-4,
    "epochs": 10,
    "gradient_accumulation_steps": 32,
    "effective_batch_size": 192,
    "discriminative_lr": {
        "text": 0.1,
        "video": 0.5,
        "head": 2.0
    }
}

# Training configurations
TRAINING_CONFIG = {
    "device": "cuda",
    "mixed_precision": True,
    "gradient_clipping": 1.0,
    "warmup_steps": 100,
    "logging_steps": 50,
    "save_steps": 500,
    "eval_steps": 500,
    "max_grad_norm": 1.0,
    "weight_decay": 1e-2,
    "adam_epsilon": 1e-8,
    "adam_betas": (0.9, 0.999),
    "scheduler": "cosine",
    "num_workers": 4,
    "pin_memory": True,
    "persistent_workers": True
}

# Loss function configurations
LOSS_CONFIG = {
    "smooth_l1_beta": 0.1,
    "ranking_margin": 0.2,
    "scale_weights": {
        "low_quality": 1.5,  # < 2.5
        "high_quality": 1.5,  # > 4.0
        "normal": 1.0
    },
    "loss_weights": {
        "alpha": 0.7,  # smooth_l1 weight
        "beta": 0.3,   # ranking weight
        "gamma": 0.1   # scale weight
    },
    "adaptive_weighting": True,
    "adaptation_rate": 0.1
}

# Dataset configurations
DATASET_CONFIG = {
    "mos_columns": ["Traditional_MOS", "Alignment_MOS", "Aesthetic_MOS", "Temporal_MOS", "Overall_MOS"],
    "text_column": "Prompt",
    "video_column": "video_name",
    "train_split": 0.8,
    "val_split": 0.2,
    "seed": 42,
    "max_text_length": 512,
    "video_extensions": [".mp4", ".avi", ".mov", ".mkv"]
}

# Evaluation configurations
EVAL_CONFIG = {
    "metrics": ["spearman", "pearson"],
    "batch_size": 1,
    "num_workers": 0,
    "save_predictions": True,
    "output_format": ["csv", "xlsx"],
    "generate_report": True
}

# Model checkpoint configurations
CHECKPOINT_CONFIG = {
    "save_dir": MODELS_DIR,
    "save_top_k": 3,
    "monitor": "val_score",
    "mode": "max",
    "save_last": True,
    "filename": "model_epoch_{epoch:02d}_score_{val_score:.4f}",
    "auto_insert_metric_name": False
}

# Logging configurations
LOGGING_CONFIG = {
    "project_name": "QualiVision-VQualA2025",
    "entity": "your_wandb_entity",
    "log_dir": PROJECT_ROOT / "logs",
    "log_level": "INFO",
    "log_format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
}

# GPU and memory configurations
GPU_CONFIG = {
    "memory_fraction": 0.9,
    "allow_growth": True,
    "mixed_precision": True,
    "gradient_checkpointing": False,  # Disabled for V-JEPA2
    "dataloader_pin_memory": True,
    "cleanup_frequency": 50  # Clean GPU memory every N batches
}

# Ensure directories exist
for directory in [DATA_DIR, MODELS_DIR, LOGGING_CONFIG["log_dir"]]:
    directory.mkdir(parents=True, exist_ok=True) 