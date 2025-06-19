# UDA-Cycle-Self-Training

A PyTorch implementation of Unsupervised Domain Adaptation using Cycle Self-Training (CST) for computer vision tasks. This project implements the CST method for domain adaptation and aims to extend it to audio scene classification using the DCASE TAU 2020 dataset with PaSST feature extractors.

## Overview

This repository contains an implementation of Cycle Self-Training (CST) for Unsupervised Domain Adaptation. The method combines:
- **Tsallis Entropy** for domain adaptation
- **FixMatch-style augmentation** for consistency regularization  
- **Cycle Self-Training** mechanism for improved pseudo-labeling
- **SAM (Sharpness-Aware Minimization)** optimizer for better generalization

## Project Status

🚧 **Work in Progress** - Currently tested on SVHN→MNIST adaptation. The final goal is to implement this approach on the DCASE TAU 2020 dataset using PaSST (Patchout Audio Spectrogram Transformer) feature extractors for acoustic scene classification.

## Repository Structure

```
├── .gitignore
├── cst_cv.ipynb             # Main implementation notebook
├── common/                  # Common utilities and modules
│   ├── loss/                # Loss functions
│   ├── modules/             # Model architectures
│   │   ├── classifier.py
│   │   └── regressor.py
│   ├── tools/               # Training utilities
│   │   ├── fix_utils.py
│   │   ├── randaugment.py
│   │   └── sam.py           # SAM optimizer
│   ├── utils/               # General utilities
│   │   ├── data.py
│   │   ├── logger.py
│   │   └── meter.py
│   └── vision/              # Vision-specific modules
├── dalib/                   # Domain adaptation library
│   ├── adaptation/          # Adaptation algorithms
│   ├── modules/             # Core modules
│   └── translation/         # Translation utilities
└── logs/                    # Training logs and checkpoints
```

## Current Implementation

### Tested Configuration
- **Source Domain**: SVHN (Street View House Numbers)
- **Target Domain**: MNIST
- **Architecture**: ResNet-18 with bottleneck dimension 256
- **Training**: 50 epochs with 1000 iterations per epoch

### Key Features
1. **Cycle Self-Training**: Implements bidirectional consistency between source and target domains
2. **Adaptive Thresholding**: Uses confidence-based pseudo-labeling with threshold 0.97
3. **Multi-loss Training**: Combines classification loss, transfer loss, CST loss, and FixMatch loss
4. **SAM Optimization**: Uses Sharpness-Aware Minimization for improved generalization

### Training Results
The current implementation shows progressive improvement in classification accuracy over epochs, with detailed logging of:
- Classification Loss
- Transfer Loss (Tsallis entropy)
- CST Loss (Cycle Self-Training)
- FixMatch Loss
- Classification Accuracy

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/UDA-Cycle-Self-Training.git
cd UDA-Cycle-Self-Training
```

2. Install required dependencies:
```bash
pip install torch torchvision
pip install numpy pandas matplotlib
pip install jupyter notebook
```

## Usage

### Quick Start
Run the main implementation notebook:
```bash
jupyter notebook cst_cv.ipynb
```

### Configuration
Key hyperparameters can be modified in the notebook:
```python
args = SimpleNamespace(
    root='./data',
    data='SVHN',
    source='svhn',
    target='mnist',
    arch='resnet18',
    bottleneck_dim=256,
    temperature=2.0,
    alpha=1.9,
    trade_off=0.08,      # Tsallis entropy weight
    trade_off1=0.5,      # CST loss weight  
    trade_off3=0.5,      # FixMatch loss weight
    threshold=0.97,      # Pseudo-label threshold
    batch_size=28,
    lr=0.005,
    epochs=50
)
```

## Future Work

### DCASE TAU 2020 Integration
The next phase involves adapting this framework for audio scene classification:

1. **Dataset Integration**: Implement DCASE TAU 2020 dataset loading
2. **PaSST Feature Extraction**: Integrate Patchout faSt Spectrogram Transformer
3. **Audio-specific Augmentations**: Adapt FixMatch augmentations for audio spectrograms
4. **Domain Adaptation**: Apply CST to different acoustic environments

### Planned Features
- [ ] DCASE TAU 2020 dataset integration
- [ ] PaSST feature extractor implementation
- [ ] Audio-specific data augmentation strategies
- [ ] Cross-domain audio scene classification
- [ ] Comprehensive evaluation metrics
- [ ] Visualization tools for audio domain adaptation

## Method Details

### Cycle Self-Training Loss
The CST loss encourages consistency between source and target domain predictions:
```
L_CST = MSE(f_target→source, one_hot(y_source))
```

### Combined Loss Function
```
L_total = L_cls + λ₁ * L_transfer + λ₂ * L_CST + λ₃ * L_FixMatch
```

Where:
- `L_cls`: Cross-entropy loss on source domain
- `L_transfer`: Tsallis entropy on target domain
- `L_CST`: Cycle self-training loss
- `L_FixMatch`: Consistency regularization loss

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Original CST paper and methodology
- PyTorch Transfer Learning Library
- DCASE challenge organizers
- PaSST model authors

## Contact

For questions and collaborations, please open an issue or contact the maintainers.

---

**Note**: This repository is part of ongoing research. The implementation is subject to changes as we progress toward the final goal of audio scene classification on DCASE TAU 2020.