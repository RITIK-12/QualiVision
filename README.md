# QualiVision - Video Quality Assessment for AI-Generated Content

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat&logo=PyTorch&logoColor=white)](https://pytorch.org/)

**QualiVision** is a comprehensive framework for video quality assessment specifically designed for AI-generated content. This repository contains our submission for the **VQualA 2025 Challenge**, featuring two state-of-the-art multi-modal architectures: **DOVER++** and **V-JEPA2**.

## 🎯 Overview

Our approach addresses four critical quality dimensions for AI-generated videos:
- **Temporal Consistency**: Coherence across frames
- **Image Fidelity**: Visual quality and sharpness  
- **Aesthetic Appeal**: Artistic and visual attractiveness
- **Text-Video Alignment**: Correspondence between prompt and content

## 📊 Data and Preparation

### Dataset Structure
We utilize the **TaobaoVD-GC** dataset provided by the VQualA 2025 Challenge, containing AI-generated videos with comprehensive quality annotations across multiple dimensions.

**Expected Data Directory Structure:**
```
data/
├── train/
│   ├── labels/
│   │   └── train_labels.csv
│   └── videos/
│       ├── video001.mp4
│       ├── video002.mp4
│       └── ...
├── val/
│   ├── labels/
│   │   └── val_labels.csv
│   └── videos/
│       ├── val_video001.mp4
│       └── ...
└── test/
    ├── test_labels.csv
    └── videos/
        ├── test_video001.mp4
        └── ...
```

**CSV Label Format:**
```csv
video_name,Prompt,Traditional_MOS,Alignment_MOS,Aesthetic_MOS,Temporal_MOS,Overall_MOS
video001.mp4,"A cat playing piano",3.2,4.1,3.8,3.5,3.65
video002.mp4,"Sunset over mountains",4.5,4.2,4.8,4.1,4.4
```

### Data Preprocessing
- **Frame Sampling**: Uniform temporal sampling of 64 frames per video
- **Resolution Adaptation**: 640×640 for DOVER++, 384×384 for V-JEPA2
- **Text Processing**: BGE-Large embeddings for prompt understanding
- **Quality Normalization**: MOS scores standardized to 1-5 scale

## 🏗️ Our Approach

### DOVER++ Model
- **Architecture**: ConvNeXt 3D backbone with quality-aware fusion
- **Innovation**: Cross-modal attention between video and text features
- **Strengths**: Robust quality assessment with aesthetic/technical separation
- **Input**: 640×640 resolution, 64 frames

### V-JEPA2 Model  
- **Architecture**: Vision-JEPA2 ViT-Giant with strategic layer freezing
- **Innovation**: Discriminative learning rates and efficient fine-tuning
- **Strengths**: Strong video representation with memory efficiency
- **Input**: 384×384 resolution, 64 frames

### Key Technical Contributions
1. **Quality-Aware Fusion**: Dynamic attention weighting based on text content
2. **Hybrid Loss Function**: Combines smooth L1, ranking, and scale-aware losses
3. **Strategic Freezing**: Freeze 85% of V-JEPA2 layers for efficient training
4. **Adaptive Loss Weighting**: Dynamic adjustment during training

## 🚀 Quick Setup and Evaluation

### Installation
```bash
# Clone the repository
git clone https://github.com/RITIK-12/QualiVision.git
cd QualiVision

# Install dependencies
pip install -r requirements.txt
```

### Direct Evaluation (Pre-trained Models)
We provide pre-trained model weights for immediate evaluation:

```bash
# Download and evaluate DOVER++ model
python scripts/evaluate.py --model dover --checkpoint models/dover_best.pt --data path/to/test/data

# Download and evaluate V-JEPA2 model  
python scripts/evaluate.py --model vjepa --checkpoint models/vjepa_best.pt --data path/to/test/data
```

**Note**: Model weights will be automatically downloaded on first use.

## 🔧 Fine-tuning on Custom Data

### Data Preparation
1. **Organize your data** following the structure shown above
2. **Prepare CSV files** with the required columns
3. **Update data paths** in the configuration

### Training Commands

**Train DOVER++ Model:**
```bash
python scripts/train.py \
    --model dover \
    --data path/to/your/data \
    --epochs 5 \
    --batch-size 4 \
    --lr 1e-4 \
    --output models/ \
    --wandb
```

**Train V-JEPA2 Model:**
```bash
python scripts/train.py \
    --model vjepa \
    --data path/to/your/data \
    --epochs 10 \
    --batch-size 6 \
    --lr 2e-4 \
    --output models/ \
    --wandb
```

### Evaluation on Custom Models
```bash
# Evaluate your trained model
python scripts/evaluate.py \
    --model dover \
    --checkpoint path/to/your/checkpoint.pt \
    --data path/to/test/data \
    --output results/
```

## ⚙️ Configuration

### Model Parameters
Key configurations can be found in `src/config/config.py`:

```python
DOVER_CONFIG = {
    "video_resolution": (640, 640),
    "num_frames": 64,
    "batch_size": 4,
    "learning_rate": 1e-4,
    "text_encoder": "BAAI/bge-large-en-v1.5"
}

VJEPA_CONFIG = {
    "video_resolution": (384, 384), 
    "num_frames": 64,
    "freeze_ratio": 0.85,
    "batch_size": 6,
    "learning_rate": 2e-4,
    "video_encoder": "facebook/vjepa2-vitg-fpc64-384-ssv2"
}
```

### Memory Optimization
For limited GPU memory:
```bash
# Reduce batch size and use gradient accumulation
python scripts/train.py --model vjepa --batch-size 2 --data path/to/data
```


## 📊 Results

Our models achieve competitive performance on the VQualA 2025 Challenge:

| Model    | SROCC | PLCC | VQualA Score | Parameters | Memory |
|----------|-------|------|--------------|------------|--------|
| DOVER++  | TBA   | TBA  | TBA          | ~120M      | ~12GB  |
| V-JEPA2  | TBA   | TBA  | TBA          | ~1.1B      | ~16GB  |

## 🔬 Research Notebooks

Explore our development process:
- `notebooks/Dover.ipynb`: DOVER++ model development and experiments
- `notebooks/VJEPA.ipynb`: V-JEPA2 model development and experiments

## 🤝 Contributing

We welcome contributions! Please:
1. Fork the repository
2. Create a feature branch
3. Submit a pull request with clear description

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

We would like to thank:

- **VQualA 2025 Challenge Organizers** for providing the TaobaoVD-GC dataset and establishing this important benchmark for AI-generated video quality assessment

- **DOVER Team** ([VQAssessment/DOVER](https://github.com/VQAssessment/DOVER)) for their foundational work on video quality assessment and making their pre-trained models available

- **V-JEPA Team** ([Meta Research](https://github.com/facebookresearch/jepa)) for their innovative video representation learning approach and open-source implementation

- **BGE Team** ([BAAI](https://huggingface.co/BAAI/bge-large-en-v1.5)) for providing high-quality text embeddings

- **Northeastern University** for providing the computational resources that made this research possible


## 🔗 Citation

If you use this code in your research, please cite:

```bibtex
@misc{qualivision2025,
  title={QualiVision: Multi-Modal Video Quality Assessment for AI-Generated Content},
  author={Ritik Bompilwar, Saurabh Koshatwar},
  year={2025},
  url={https://github.com/RITIK-12/QualiVision}
}
```

---

**Built for the VQualA 2025 Challenge** 🎯