# Speech Keyword Detection Training Pipeline

## Overview

This document describes the final training pipeline for the speech keyword detection model, implemented in `finetuning_pretrained_pipeline_with_google_speech.py`. The pipeline trains a custom MobileNetV2 model on a merged dataset combining local speech keywords with Google Speech Commands data.

## Dataset Composition

### Input Dataset
- **Local Classes**: 13 custom keyword classes from `/data/audio_chunks/`
- **Google Speech Data**: 3,000 background speech samples (class ID: 13)
- **Total Classes**: 14
- **Class Mapping**: Each local class is assigned an ID (0-12) in alphabetical order, with Google Speech always as class 13

### Class Mapping Example
```
local_classes (0-12): sorted alphabetically from dataset_path
google_speech (13):   background/negative class
```

## Audio Input Specifications

### Maximum Input Vector Length
- **Max Length**: **53,248 samples**
- **Sample Rate**: 16,000 Hz
- **Duration**: ~3.33 seconds of audio (53,248 / 16,000 ≈ 3.33s)
- **Padding/Truncation**: All audio is standardized to exactly this length:
  - Shorter audio is **zero-padded** at the end
  - Longer audio is **truncated** to this length

### Mel-Spectrogram Configuration
- **N Mels**: 64 frequency bins
- **FFT Size**: 512
- **Hop Length**: 160 samples (69% overlap / 10ms step size)
- **Output Shape**: `[1, 64, ~334]` (Channels, Mels, TimeFrames)
  - Note: ~334 time frames from 53,248 samples with hop_length=160

## Training Pipeline Overview

### Phase 1: Classifier Head Training (Frozen Base)
**Objective**: Train only the classification head on the pretrained MobileNetV2 base features

**Configuration**:
- Model: MobileNetV2Custom with `freeze_base=True`
- Learning Rate: `1e-3` (Adam optimizer)
- Epochs: 10
- Batch Size: 32
- Scheduler: StepLR with step_size=5, gamma=0.5 (reduce LR by 50% every 5 epochs)
- Number of Classes: 14
- Save Path: `models/best_keyword_model.pth`

**Process**:
1. Load pretrained ImageNet weights (frozen)
2. Train custom classifier head on merged dataset
3. Save best model based on validation accuracy
4. Validate on 20% of the data (stratified split)

### Phase 2: Fine-Tuning the Entire Network
**Objective**: Fine-tune all network weights (including pretrained base) with a microscopic learning rate

**Configuration**:
- Model: MobileNetV2Custom with `freeze_base=False`
- Learning Rate: `1e-5` (microscopic to protect ImageNet filters)
- Epochs: 60
- Batch Size: 32
- Scheduler: StepLR with step_size=35, gamma=0.5
- Number of Classes: 14
- Save Path: `models/phase2_finetuned_best.pth`

**Process**:
1. Load Phase 1 best weights as initialization
2. Unfreeze all layers
3. Train entire network with very low learning rate
4. Save best model based on validation accuracy
5. Validate on same 20% split as Phase 1

## Data Processing Pipeline

### Training Data Processing
Each audio sample goes through the following pipeline (when `is_training=True`):

1. **Load**: Read WAV file at 16kHz sample rate
2. **Standardize Length**: Pad or truncate to 53,248 samples
3. **Data Augmentation** (applied with 50% probability):
   - **Pitch Shifting**: Random pitch shift of -2, -1, 0, 1, or +2 semitones
   - **White Noise**: Add white noise at level 0.005-0.015
   - **Environmental Noise**: Overlay random real environmental noise (10-50% volume)
   - **Time Slicing**: Roll audio by 10-50% of max_length
4. **Mel-Spectrogram**: Convert to 64-bin mel-spectrogram
5. **dB Conversion**: Convert to decibels (power, top_db=80)
6. **Normalization**: Apply Z-score normalization (per-sample basis)

### Validation Data Processing
Same as training, but **without** augmentations:
1. Load and standardize length
2. Convert to Mel-Spectrogram
3. dB conversion
4. Z-score normalization

## Class Imbalance Handling

### Weighted Random Sampling
To prevent the model from overfitting to common classes:

**Weight Formula**:
```
weight[class] = total_training_samples / num_samples_in_class
```

- Rare classes get **larger weights** (oversampled more frequently)
- Common classes get **smaller weights** (sampled less frequently)
- Ensures balanced class representation during training

**Implementation**: `WeightedRandomSampler` in `prepare_dataloaders_with_google_speech()`

## Model Architecture

**Model**: MobileNetV2 (pretrained on ImageNet)
- **Base**: Frozen ImageNet-pretrained MobileNetV2
- **Custom Head**: 
  - Removes original 1000-class classifier
  - Adds custom linear layer: 1280 features → 14 classes
  - ReLU activation between layers
  - Input: Gray-scale spectrogram (1 channel) duplicated to 3 channels for ImageNet compatibility

**Phase 1 Flow**:
```
Mel-Spectrogram [1, 64, 334] 
  → Duplicate to 3 channels [3, 64, 334]
  → MobileNetV2 Base (frozen) 
  → 1280 features
  → Custom Head (trainable)
  → 14 logits
```

## Training Hyperparameters Summary

| Parameter | Phase 1 | Phase 2 |
|-----------|---------|---------|
| Model | MobileNetV2 (frozen base) | MobileNetV2 (unfrozen) |
| Learning Rate | 1e-3 | 1e-5 |
| Optimizer | Adam | Adam |
| Weight Decay | 1e-4 | 1e-4 |
| Epochs | 10 | 60 |
| Batch Size | 32 | 32 |
| Val Split | 20% | 20% (same split as Phase 1) |
| LR Scheduler | StepLR (step=5, γ=0.5) | StepLR (step=35, γ=0.5) |
| Loss Function | CrossEntropyLoss | CrossEntropyLoss |
| Save Metric | Best Val Accuracy | Best Val Accuracy |

## Dataset Statistics

### Distribution After Merge
- **Local Classes**: Variable counts (depends on recorded data)
- **Google Speech**: 3,000 samples (class 13)
- **Total Training**: 80% of merged dataset
- **Total Validation**: 20% of merged dataset
- **Stratification**: Train/val split maintains class proportion

## Output Artifacts

### Saved Models
1. **Phase 1 Model**: `models/best_keyword_model.pth`
   - Best weights from classifier head training
   - Used as initialization for Phase 2

2. **Phase 2 Model**: `models/phase2_finetuned_best.pth`
   - Final production model
   - Used in inference scripts

### Logs and Visualizations
- **Class Mapping**: `logs/class_mapping.txt` - Maps class names to IDs
- **Training Curves**: `plots/training_curves.png` - Loss and accuracy curves
- **Training Logs**: `logs/` directory - Detailed epoch-by-epoch metrics

## Inference Configuration

When using the trained model for inference (see `inference_script_with_cnn.py`):
- Use model: `phase2_finetuned_best.pth`
- Apply **same preprocessing** as validation:
  - Load audio at 16kHz
  - Pad/truncate to 53,248 samples
  - Convert to 64-bin mel-spectrogram
  - Apply dB conversion and Z-score normalization
  - No augmentations
- Output: 14-class softmax predictions with confidence scores

## Key Design Decisions

1. **Two-Phase Training**: 
   - First phase adapts the classifier on frozen features (fast, stable)
   - Second phase fine-tunes entire network with microscopic LR (preserves ImageNet knowledge)

2. **Max Length = 53,248 samples**:
   - Chosen experimentally to capture typical speech keyword duration (~3.3s)
   - Balances GPU memory usage with adequate temporal context

3. **Weighted Sampling**:
   - Prevents model collapse to predicting only common classes
   - Ensures all classes contribute meaningfully to training

4. **Data Augmentation**:
   - Pitch shifting: Handles speaker variability
   - White noise: Robustness to environmental noise
   - Environmental noise overlay: Real-world augmentation
   - Time shifting: Positional invariance (without wraparound)

5. **Z-score Normalization**:
   - Applied per-sample independently
   - Normalizes each spectrogram to mean=0, std=1
   - Improves training stability

## Running the Pipeline

```bash
cd /home/st1/Documents/training_cnn_for_waveform/src
python finetuning_pretrained_pipeline_with_google_speech.py
```

The pipeline will:
1. Load local classes and Google Speech data
2. Create merged dataset with class mapping
3. Prepare weighted dataloaders
4. Execute Phase 1 training (10 epochs)
5. Execute Phase 2 fine-tuning (60 epochs)
6. Save best model and training visualizations 
