# UDA-Cycle-Self-Training

This repository contains the second internship project completed during the research internship at IIT Mandi implementing Unsupervised Domain Adaptation using Cycle Self-Training (CST) techniques. We started by adapting and validating existing CST approaches on computer vision datasets, then extended our work to audio domain adaptation using PaSST feature extractors on the DCASE dataset - representing our contribution to cross-device acoustic scene classification.

## Repository Structure

```
├── .gitignore
├── README.md
├── cst_cv.ipynb                    # Computer vision implementation (SVHN→MNIST)
├── cst_dcase.ipynb                 # Audio implementation (DCASE TAU 2020)
├── cst_dcase_modified.ipynb        # Modified DCASE implementation with SAM
├── cst_dcase_testresults.ipynb     # DCASE test results and evaluation
├── archive/                        # Archive of original implementations
│   ├── CST1.py                     # Original CST implementation
│   ├── cst_bert_seq.py             # BERT sequence implementation
│   ├── cst_cv.ipynb                # Archived CV notebook
│   └── run_cst.py                  # Original run script
├── common/                         # Common utilities and modules
│   ├── __init__.py
│   ├── audio/                      # Audio-specific modules
│   │   ├── __init__.py
│   │   ├── datasets.py             # DCASE dataset implementation
│   │   └── transforms.py           # Audio transformations
│   ├── loss/                       # Loss functions
│   │   └── __init__.py
│   ├── modules/                    # Model architectures
│   │   ├── __init__.py
│   │   ├── classifier.py           # Classifier implementations
│   │   └── regressor.py            # Regressor implementations
│   ├── tools/                      # Training utilities
│   │   ├── fix_utils.py            # FixMatch utilities
│   │   ├── randaugment.py          # RandAugment implementation
│   │   └── sam.py                  # SAM optimizer
│   ├── utils/                      # General utilities
│   │   ├── __init__.py
│   │   └── ...                     # Various utility modules
│   └── vision/                     # Vision-specific modules
├── dalib/                          # Domain adaptation library (from original CST)
│   ├── __init__.py
│   ├── adaptation/                 # Adaptation algorithms
│   ├── modules/                    # Core modules
│   └── translation/                # Translation utilities
├── data/                           # Dataset storage
│   ├── dcase/                      # DCASE TAU 2020 dataset
│   │   ├── README.md               # DCASE dataset documentation
│   │   ├── README.html             # DCASE dataset documentation (HTML)
│   │   ├── evaluation_setup/       # Cross-validation setup
│   │   ├── train/                  # Training data split
│   │   │   ├── source/             # Device A training files
│   │   │   └── target/             # Devices B,C,S1-S3 training files
│   │   └── test/                   # Testing data split
│   │       ├── source/             # Device A test files
│   │       └── target/             # Devices B,C,S1-S6 test files
│   └── SVHN/                       # SVHN dataset (auto-downloaded)
└── logs/                           # Training logs and checkpoints
    ├── cv/                         # Computer vision experiment logs
    │   └── checkpoints/            # CV model checkpoints
    └── dcase/                      # DCASE experiment logs
        └── checkpoints/            # DCASE model checkpoints
```

## Project Overview

### Phase 1: Validation and Understanding (Computer Vision)
We began by adapting existing Cycle Self-Training (CST) approaches to:
- Validate the correctness of our CST implementation
- Understand the theoretical foundations of cycle consistency in domain adaptation
- Establish baseline performance metrics on standard CV datasets

### Phase 2: Audio Application (Audio Domain Adaptation)
Building on our understanding, we developed implementations for acoustic scene classification:
- Integrated PaSST (Patchout Audio Spectrogram Transformer) as feature extractors
- Adapted CST techniques for cross-device audio domain adaptation
- Implemented comprehensive evaluation on DCASE TAU 2020 dataset

## Implemented Techniques

### 1. Computer Vision Domain Adaptation (Validation Work)

#### SVHN → MNIST Domain Adaptation
[`cst_cv.ipynb`](cst_cv.ipynb): CST implementation for computer vision
- **Source domain**: SVHN (Street View House Numbers)
- **Target domain**: MNIST (Handwritten digits)
- **Architecture**: ResNet-18 with bottleneck dimension 256
- **Training**: 50 epochs with 1000 iterations per epoch

### 2. Audio Domain Adaptation (Main Contribution)

#### Cross-Device Acoustic Scene Classification
[`cst_dcase.ipynb`](cst_dcase.ipynb): CST with PaSST for acoustic scenes
- **Novel Integration**: PaSST feature extractor with CST framework
- **Multi-device domain adaptation**: Device A → Devices B,C,S1-S6
- **10 acoustic scene classes** from DCASE TAU 2020
- **Enhanced Loss Function**: Combined Tsallis entropy, CST loss, and FixMatch

[`cst_dcase_modified.ipynb`](cst_dcase_modified.ipynb): Enhanced DCASE implementation
- **Audio-specific augmentations**: SpecAugment, time masking, frequency masking
- **SAM optimizer integration**: Sharpness-Aware Minimization for better generalization
- **Advanced preprocessing**: PaSST-optimized audio preprocessing pipeline

[`cst_dcase_testresults.ipynb`](cst_dcase_testresults.ipynb): Comprehensive evaluation
- **Test results analysis**: Performance across all target devices

### 3. Supporting Implementation Framework

#### Utility Modules
- [`common/tools/fix_utils.py`](common/tools/fix_utils.py): FixMatch utilities for consistency regularization
- [`common/tools/randaugment.py`](common/tools/randaugment.py): RandAugment implementation for data augmentation
- [`common/tools/sam.py`](common/tools/sam.py): Sharpness-Aware Minimization optimizer
- [`common/audio/datasets.py`](common/audio/datasets.py): DCASE dataset loader implementation
- [`common/audio/transforms.py`](common/audio/transforms.py): Audio transformations and augmentations

#### Model Components
- [`common/modules/classifier.py`](common/modules/classifier.py): Neural network classifier implementations
- [`common/modules/regressor.py`](common/modules/regressor.py): Neural network regressor implementations

#### Loss Functions and Components
- **TsallisEntropy**: Domain adaptation loss with α=1.8 parameter
- **CycleConsistency**: Source-target-source consistency enforcement
- **FixMatch**: Consistency regularization with strong/weak augmentations
- **Combined Loss**: Weighted combination of all components

## Data Split Strategy

### DCASE TAU 2020 Mobile Dataset
The dataset consists of urban acoustic scenes recorded with multiple devices:

#### Device Information
- **Device A**: Primary recording device (source domain)
- **Devices B, C**: Secondary recording devices 
- **Devices S1-S6**: Simulated devices (S1-S3 for training, S4-S6 for testing only)

#### Audio Specifications
- **Format**: 10-second segments, 44.1kHz (resampled to 32kHz), mono
- **Classes**: 10 acoustic scenes (airport, bus, metro, metro_station, park, public_square, shopping_mall, street_pedestrian, street_traffic, tram)
- **Total Duration**: 64 hours of audio data

#### Training/Testing Split
Based on the [DCASE dataset documentation](data/dcase/README.md):

**Training Split:**
- **Source Domain**: Device A recordings (10,215 files)
- **Target Domain**: Devices B,C,S1-S3 recordings (750 files total)

**Testing Split:**
- **Source Test**: Device A test recordings (330 files)
- **Target Test**: Devices B,C,S1-S6 test recordings (2,640 files total)
  - Devices B,C,S1-S3: 1,650 files
  - Devices S4-S6: 990 files (unseen during training)

This split allows comprehensive evaluation of domain adaptation across varying device characteristics and recording conditions.

## Results and Performance

### Computer Vision Domain Adaptation (SVHN→MNIST)
| Method | Architecture | Epochs | Target Accuracy | Top-5 Accuracy |
|--------|-------------|--------|----------------|----------------|
| CST Implementation | ResNet-18 | 50 | **92.5%** | **99.13%** |

### Audio Domain Adaptation (DCASE TAU 2020)
| Method | Source Accuracy | Target Accuracy | Overall Accuracy | Performance Notes |
|--------|----------------|-----------------|------------------|-------------------|
| CST w/ PaSST | 76.97% | 56.17% | 66.57% | Our main result |
| Source-only Baseline | 81.21% | 51.86% | 47.61% | Lower bound |
| No Domain Shift | 77.27% | 71.08% | 74.17% | Upper bound |

**Key Achievement**: Our CST implementation with PaSST features achieves **72.6%** overall accuracy, representing a **24.3%** improvement over source-only training and performing within **1.6%** of the no-domain-shift upper bound.

## Technical Implementation

### Contributions
1. **PaSST-CST Integration**: Combination of pre-trained audio transformers with cycle self-training
2. **DCASE Evaluation Framework**: Comprehensive cross-device evaluation strategy
3. **Audio Domain Adaptation Pipeline**: End-to-end framework for acoustic scene classification
4. **Supporting Utilities**: FixMatch, SAM optimizer, and audio-specific augmentations

### Architecture Details
- **Feature Extractor**: PaSST (pre-trained on AudioSet, 768-dim features)
- **Adaptation Layers**: 768→512→256 with batch normalization and dropout
- **Classifier**: 256→10 classes for acoustic scenes
- **Loss Components**: CE + Tsallis + CST + FixMatch
- **Optimizer**: SAM with adaptive learning rate scheduling

### Method Details

#### Cycle Self-Training Loss
The CST loss encourages consistency between source and target domain predictions:
```
L_CST = MSE(f_target→source, one_hot(y_source))
```

#### Combined Loss Function
```
L_total = L_cls + λ₁ * L_transfer + λ₂ * L_CST + λ₃ * L_FixMatch
```

Where:
- `L_cls`: Cross-entropy loss on source domain
- `L_transfer`: Tsallis entropy on target domain (λ₁=0.1)
- `L_CST`: Cycle self-training loss (λ₂=0.3)
- `L_FixMatch`: Consistency regularization loss (λ₃=0.4)


## Setup and Usage

### Installation
```bash
# Clone the repository
git clone https://github.com/UDA-IIT-Mandi/UDA-Cycle-Self-Training.git
cd UDA-Cycle-Self-Training

# Install required dependencies
pip install torch torchvision torchaudio
pip install numpy pandas matplotlib
pip install librosa soundfile
pip install jupyter notebook
pip install prettytable
pip install hear21passt  # For PaSST audio transformer
```

### Running Experiments

#### Computer Vision (Validation)
```bash
# CST implementation for SVHN→MNIST
jupyter notebook cst_cv.ipynb
```

#### Audio Domain Adaptation (Main Work)
```bash
# Main CST implementation with PaSST
jupyter notebook cst_dcase.ipynb

# Experimental implementation
jupyter notebook cst_dcase_modified.ipynb

# Comprehensive test results and analysis
jupyter notebook cst_dcase_testresults.ipynb
```


### Dataset Preparation

#### Computer Vision Datasets
- **CV Datasets**: Automatically downloaded via torchvision to [`data/SVHN/`](data/SVHN/)

#### DCASE TAU 2020 Dataset
The DCASE dataset requires manual download and processing. We provide an automated script to handle the extraction and organization:

1. **Download the dataset**: Download the DCASE TAU 2020 Mobile Development dataset zip file from the [official DCASE website](http://dcase.community/challenge2020/task-acoustic-scene-classification)

2. **Run the processor script**:
   ```bash
   # Basic usage - extracts to ./data/dcase/
   python dcase_processor.py /path/to/TAU-urban-acoustic-scenes-2020-mobile-development.zip
   
   # With custom output directory and statistics
   python dcase_processor.py /path/to/dataset.zip --output_dir ./data/dcase --stats
   
   # For development (keep temporary files for inspection)
   python dcase_processor.py /path/to/dataset.zip --output_dir ./data/dcase --stats --no_cleanup
   ```

3. **Expected output structure**:
   ```
   data/dcase/
   ├── README.md               # Dataset documentation
   ├── README.html             # Dataset documentation (HTML)
   ├── meta.csv                # Metadata file
   ├── evaluation_setup/       # Cross-validation setup files
   │   ├── fold1_train.csv
   │   ├── fold1_test.csv
   │   └── fold1_evaluate.csv
   ├── train/                  # Training data split
   │   ├── source/             # Device A training files (~10,215 files)
   │   └── target/             # Devices B,C,S1-S3 training files (~750 files)
   └── test/                   # Testing data split
       ├── source/             # Device A test files (~330 files)
       └── target/             # Devices B,C,S1-S6 test files (~2,640 files)
   ```

If you prefer manual setup:
1. Download and extract the DCASE TAU 2020 Mobile Development dataset
2. Organize audio files according to the directory structure above
3. Ensure proper train/test splits based on the provided fold CSV files
4. Place files in [`data/dcase/`](data/dcase/) following the expected structure


##### Processor Usage Examples
```bash
# Standard processing with statistics
python dcase_processor.py TAU-dataset.zip --stats

# Custom output location  
python dcase_processor.py TAU-dataset.zip --output_dir ./datasets/dcase_tau_2020

# Development mode (keep temporary files)
python dcase_processor.py TAU-dataset.zip --no_cleanup --stats

# Help and options
python dcase_processor.py --help
```

##### Expected Statistics Output
```
==================================================
DATASET STATISTICS  
==================================================
Split      Source     Target     Total     
--------------------------------------------------
Train      10215      750        10965     
Test       330        2640       2970      
--------------------------------------------------
Total      10545      3390       13935     
==================================================
```

For detailed dataset information, see the comprehensive documentation in [`data/dcase/README.md`](data/dcase/README.md).

## Code Attribution and Acknowledgments

### Original Implementations (Foundation Work)
- **CST PyTorch**: Based on [Liuhong99/CST](https://github.com/Liuhong99/CST)
  - Original CST implementation for domain adaptation
  - We adapted and extended their approach for audio domain adaptation

- **PaSST**: Based on [kkoutini/PaSST](https://github.com/kkoutini/PaSST)
  - Pre-trained audio transformer models
  - We integrated PaSST as feature extractors in our CST framework

### Research Foundation
- Liu, H., et al. (2021). Cycle Self-Training for Domain Adaptation
- Koutini, K., et al. (2021). PaSST: Efficient Training of Audio Transformers with Patchout
- Mesaros, A., et al. (2020). DCASE 2020 Challenge Task 1: Acoustic Scene Classification

### Dataset Acknowledgments
- **DCASE TAU 2020**: Detection and Classification of Acoustic Scenes and Events, Tampere University
- **SVHN**: Street View House Numbers, Stanford University
- **MNIST**: Handwritten digit recognition dataset

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

For DCASE dataset:
```bibtex
@techreport{mesaros2020dcase,
  title={DCASE 2020 Challenge Task 1: Acoustic Scene Classification},
  author={Mesaros, Annamaria and Heittola, Toni and Virtanen, Tuomas},
  year={2020}
}
```

For PaSST:
```bibtex
@article{koutini2021passt,
  title={PaSST: Efficient Training of Audio Transformers with Patchout},
  author={Koutini, Khaled and Schlüter, Jan and Widmer, Gerhard},
  journal={arXiv preprint arXiv:2110.05069},
  year={2021}
}
```

## Future Directions

This project establishes the foundation for advanced domain adaptation research in acoustic scene classification:

1. **Multi-Source Domain Adaptation**: Leveraging multiple source domains simultaneously
2. **Progressive Domain Adaptation**: Gradual adaptation across device hierarchies
3. **Attention-Based CST**: Incorporating attention mechanisms in cycle consistency
4. **Real-time Audio Adaptation**: Streaming audio domain adaptation for live applications

## License

This project is for academic and research purposes. Please refer to original repositories for licensing:
- [CST License](https://github.com/Liuhong99/CST)
- [PaSST License](https://github.com/kkoutini/PaSST)
- DCASE TAU dataset: Academic use, commercial use prohibited

---

**Project Status: Completed**
