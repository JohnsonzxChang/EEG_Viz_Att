# RSVP-COCO: Multi-Label EEG Classification (80 Categories)

## Paradigm Overview

Rapid Serial Visual Presentation (RSVP) paradigm for EEG-based brain-computer interface. Subjects view rapid streams of MS-COCO images while 32-channel EEG is recorded. Each image may contain multiple COCO object categories (multi-label), yielding an 80-class multi-label classification problem.

**Key challenge**: Decoding which object categories a subject perceived from short EEG epochs (~700ms post-stimulus), where each image may have 1-15 simultaneous labels.

## Data Format

- **EEG epochs**: MNE `.fif` files, stored at `<data_root>/data/udf_viz/<subject>/erp.fif`
- **COCO annotations**: `instances_train2017.json`, `captions_train2017.json`
- **Subjects**: `zfn-erp-1128`, `zx-1122`, etc.
- **Channels**: 32 EEG electrodes, 1000 Hz sampling rate
- **Epoch window**: -200ms to +500ms relative to stimulus onset
- **Labels**: 80 COCO object categories (person class removed due to label imbalance)

## Architecture

### Encoder Zoo (`encoder/`)

| `config.model` | Class | Description |
|----------------|-------|-------------|
| `'ATM'` | `ATM_Encoder` | Adaptive Thinking Mapper (primary): channel-wise transformer + temporal-spatial conv + MLP projector |
| `'CNN'` | `CNN_Encoder` | 2D convolutional baseline |
| `'TF'` | `TFEncoder` | Standard transformer encoder |
| `'EEGNet'` | `EEGNet` | Compact CNN designed for EEG |
| `'FAPeM'` | `fapem_encoder` | Feature-Aware Positional Encoding Model |
| `'LSTM'` | `lstm_encoder` | Bidirectional LSTM |
| `'Mamba'` | `HHMambaEncoder` | State-space model |

### Config Hierarchy (`conf/`)

```
ModelConfig (model_config.py)
  neural network hyperparams: heads, d_ff, layers

BaseConfig (base.py)
  training: optimizer, loss, device, data key
  |
  +-- BaseConfigViz (base_cla_viz.py)
  |     80 classes, data='udf_viz_m'
  |     |
  |     +-- BaseConfigCircle (base_cla_circle.py)
  |     |     ASL + Circle Loss, gamma/margin/lambda
  |     |
  |     +-- BaseConfigEnhanced (base_cla_enhanced.py)
  |     |     + Proxy-Anchor + Memory Queue
  |     |
  |     +-- BaseConfigClipFused (base_cla_clip_fused.py)
  |           ERP-averaged + CLIP gated fusion
  |
  +-- BaseConfigVizCoCo (base_cla_viz_coco.py)
  |     EEG-image retrieval, contrastive learning
  |
  +-- BaseConfigRSVP (base_rsvp.py)
        12-class internal RSVP with CLIP
```

### Loss Functions (`losses/`)

- **BCEWithLogitsLoss**: Standard multi-label binary cross-entropy (baseline)
- **ASL (Asymmetric Loss)**: Down-weights easy negatives, implemented in task layer
- **Circle Loss**: Margin-based metric learning for multi-label, with optional Jaccard weighting
- **Proxy-Anchor Loss**: Proxy-based metric learning alternative

### Training Scripts

| Script | Loss | Description |
|--------|------|-------------|
| `main_pure_claM.py` | BCE | Baseline multi-label classification |
| `main_circle.py` | ASL + Circle | Contrastive metric learning |
| `main_enhanced.py` | ASL + Circle + Proxy-Anchor | Full contrastive suite with memory queue |
| `main_clip_fused.py` | BCE + CLIP | ERP-averaged with gated CLIP fusion |
| `main_retrieval.py` | InfoNCE | EEG-to-image contrastive retrieval |
| `main_rsvp_clip_v3.py` | CE + CLIP | Mini-ERP augmented RSVP with CLIP prototypes |

### Evaluation Metrics

- **Top-k Accuracy** (k=1, 3, 5): At least k correct labels in top predictions
- **Micro/Macro F1**: Multi-label F1 scores
- **mAP**: Mean Average Precision across 80 categories
- **Retrieval**: R@K, MRR for EEG-to-image retrieval

## Usage

### Prerequisites

```bash
conda activate VIZ
pip install torch mne numpy scikit-learn matplotlib
# For CLIP features: pip install transformers
```

### Quick Start

```bash
# 1. Set data paths in conf/base_cla_viz.py:
#    config.data_root = '/path/to/data'

# 2. Baseline classification
python main_pure_claM.py

# 3. Circle Loss training
python main_circle.py

# 4. EEG-image retrieval
python main_retrieval.py

# Or use run.bat:
run.bat classify
run.bat circle
run.bat retrieval
```

### Data Path Configuration

Edit `conf/base_cla_viz.py` to set your paths:

```python
config.data_root = '/your/path/to/data'     # contains udf_viz/<subject>/erp.fif
config.coco_root = '/your/path/to/coco'     # contains annotations/ and train2017/
```

### Key Hyperparameters

```python
# Circle Loss (conf/base_cla_circle.py)
config.circle_gamma = 80      # scale factor
config.circle_margin = 0.25   # separation margin
config.circle_lambda = 0.5    # L_total = L_ASL + lambda * L_circle
config.circle_jaccard = True  # weight by label set similarity

# Training
config.learning_rate = 3e-4
config.train_epochs = 100
config.patience = 20
config.batch_size = 64
```

## Output

- **Logs**: `./logs_Visual/classification/<run_name>/`
- **Checkpoints**: `checkpoint.pth` in run directory
- **Config**: `dict.json` serialized configuration
- **TensorBoard**: Use `LoggerFile` wrapper for training curves

## File Summary

```
rsvp_coco/
  encoder/              # Neural network architectures (ATM, CNN, EEGNet, etc.)
  conf/                 # Configuration hierarchy
  utils/                # Data loaders, augmentation, sampling, logging
  losses/               # Circle Loss, Proxy-Anchor Loss
  tasks/                # Training loops (classification, retrieval, tracking)
  contrast/             # CLIP integration, retrieval metrics
  main_pure_claM.py     # Baseline multi-label (BCE)
  main_circle.py        # ASL + Circle Loss
  main_enhanced.py      # Full contrastive suite
  main_clip_fused.py    # CLIP-fused ERP classification
  main_retrieval.py     # EEG-image retrieval
  main_rsvp_clip_v3.py  # Mini-ERP + CLIP prototypes
  run.bat               # Convenience launcher
```

## References

- Li et al., "Visual Decoding and Reconstruction via EEG Embeddings with Guided Diffusion," NeurIPS 2024 (ATM encoder)
- Sun et al., "Circle Loss: A Unified Perspective of Pair Similarity Optimization," CVPR 2020
- Lin et al., "Microsoft COCO: Common Objects in Context," ECCV 2014
