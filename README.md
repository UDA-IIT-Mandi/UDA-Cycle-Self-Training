# UDA-Cycle-Self-Training

A PyTorch implementation of Unsupervised Domain Adaptation using Cycle Self-Training (CST) adapted for computer vision tasks, with future extensions planned for audio scene classification using the DCASE TAU 2020 dataset with PaSST feature extractors.

## Overview

This repository is based on the Cycle Self-Training (CST) method from the paper ["Cycle Self-Training for Domain Adaptation"](https://arxiv.org/abs/2103.03571) by Liu et al. Our implementation adapts the original [CST codebase](https://github.com/Liuhong99/CST) to work with SVHN→MNIST domain adaptation and extends it for future audio applications.

The method combines:
- **Tsallis Entropy** for domain adaptation
- **FixMatch-style augmentation** for consistency regularization  
- **Cycle Self-Training** mechanism for improved pseudo-labeling
- **SAM (Sharpness-Aware Minimization)** optimizer for better generalization

## Project Status

🚧 **Work in Progress** - Currently adapted and tested on SVHN→MNIST adaptation. The final goal is to implement this approach on the DCASE TAU 2020 dataset using PaSST (Patchout Audio Spectrogram Transformer) feature extractors for acoustic scene classification.

## Repository Structure

```
├── .gitignore
├── README.md
├── cst_cv.ipynb             # Main implementation notebook (adapted from original CST)
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
├── dalib/                   # Domain adaptation library (from original CST)
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
5. **Automatic Dataset Download**: SVHN and MNIST datasets are automatically downloaded when running the code

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

**Note**: The SVHN and MNIST datasets will be automatically downloaded to the `./data` directory when you first run the code.

## Usage

### Quick Start
Run the main implementation notebook:
```bash
jupyter notebook cst_cv.ipynb
```

The notebook will automatically:
- Download SVHN and MNIST datasets (if not already present)
- Set up the CST training pipeline
- Train the model with the configured hyperparameters
- Log training progress and results

### Configuration
Key hyperparameters can be modified in the notebook:
```python
args = SimpleNamespace(
    root='./data',           # Data directory (datasets auto-downloaded here)
    data='SVHN',
    source='svhn',
    target='mnist',
    arch='resnet18',
    bottleneck_dim=256,
    temperature=2.0,
    alpha=1.9,
    trade_off=0.08,          # Tsallis entropy weight
    trade_off1=0.5,          # CST loss weight  
    trade_off3=0.5,          # FixMatch loss weight
    threshold=0.97,          # Pseudo-label threshold
    batch_size=28,
    lr=0.005,
    epochs=50
)
```

## Contributions and Adaptations

### Our Contributions
- **SVHN/MNIST Adaptation**: Modified the original CST code to work with SVHN→MNIST domain adaptation
- **Notebook Implementation**: Created a comprehensive Jupyter notebook implementation
- **Audio Domain Planning**: Designed the framework extension for audio scene classification
- **Documentation**: Enhanced documentation and usage examples

### Future Work - Audio Domain Adaptation

The next phase involves adapting this framework for audio scene classification:

1. **Dataset Integration**: Implement DCASE TAU 2020 dataset loading and preprocessing
2. **PaSST Integration**: Integrate Patchout faSt Spectrogram Transformer for audio feature extraction
3. **Audio-specific Augmentations**: Develop FixMatch-style augmentations for audio spectrograms
4. **Cross-domain Audio**: Apply CST to different acoustic environments and recording conditions

### Planned Features
- [ ] DCASE TAU 2020 dataset integration
- [ ] PaSST feature extractor implementation
- [ ] Audio-specific data augmentation strategies
- [ ] Cross-domain audio scene classification
- [ ] Comprehensive evaluation metrics for audio tasks
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

## Citation

If you use this code, please cite the original CST paper:

```bibtex
@article{liu2021cycle,
  title={Cycle Self-Training for Domain Adaptation},
  author={Liu, Hong and Long, Mingsheng and Wang, Jianmin and Jordan, Michael I},
  journal={arXiv preprint arXiv:2103.03571},
  year={2021}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- **Original CST Authors**: Liu et al. for the Cycle Self-Training methodology and [original implementation](https://github.com/Liuhong99/CST)
- **CST Paper**: ["Cycle Self-Training for Domain Adaptation"](https://arxiv.org/abs/2103.03571)
- **PyTorch Transfer Learning Library**: For domain adaptation utilities
- **DCASE Challenge**: For the audio scene classification dataset and benchmarks
- **PaSST Authors**: For the Patchout faSt Spectrogram Transformer model

## Contact

For questions about our adaptations and future audio extensions, please open an issue or contact the maintainers.

---

**Note**: This repository adapts the original CST method for SVHN→MNIST and extends it toward audio applications. The original CST implementation and methodology are credited to Liu et al. Our contributions focus on the planned audio domain extensions.