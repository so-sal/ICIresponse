# 🔧 Unified Parameter Guide

Complete parameter reference for all scripts in the STAD framework.

## 📖 Table of Contents
- [Feature Extraction](#feature-extraction)
- [MIL Training](#mil-training)
- [Visualization](#visualization)
- [Tissue Analysis](#tissue-analysis)
- [Configuration Files](#configuration-files)

---

## Feature Extraction

### Script: `src/feature_extraction.py`

#### Required Parameters
```bash
--model_type {DINO,UNI,GigaPath,Lunit}    # Foundation model for feature extraction
--dataset {JNUH,AJOU,TCGA,STFD}           # Dataset to process
```

#### Model Configuration
```bash
# Foundation model selection
--model_type DINO          # DINO ViT-S/16 (384-dim features)
--model_type UNI           # UNI ViT-L/16 (1024-dim features)  
--model_type GigaPath      # GigaPath model (1536-dim features)
--model_type Lunit         # Lunit SSL model (512-dim features)
```

#### Dataset Configuration
```bash
# Dataset selection (each has different default settings)
--dataset JNUH            # Jeonbuk National University Hospital
--dataset AJOU            # Ajou University Hospital  
--dataset TCGA            # The Cancer Genome Atlas
--dataset STFD            # Stanford Dataset
```

#### Processing Parameters
```bash
--device cuda:0           # Computation device (default: cuda:0)
--batch_size 32           # Batch size for inference (default: 32)
--num_workers 4           # Data loading workers (default: 4)
```

#### Patch Extraction Parameters
```bash
--patch_size 448          # Patch size in pixels (default: 448)
--stride 448              # Stride between patches (default: 448)
```

#### Processing Control
```bash
--slide_ids slide1 slide2  # Process specific slides only
--start_idx 0              # Starting index for batch processing
--end_idx 100              # Ending index for batch processing  
--overwrite                # Overwrite existing features
```

#### Dataset-Specific Defaults

**JNUH (Default)**
- `patch_size`: 448, `stride`: 448, `downsample`: 2
- `sat_thresh`: 9, `area_thresh`: 0.4
- Slide directory: `/mnt/e/JNUH/STAD_WSI/Gastric_Ca_ImmunoTx/`

**AJOU**
- `patch_size`: 448, `stride`: 448, `downsample`: 2  
- `sat_thresh`: 7, `area_thresh`: 0.5
- Slide directory: `/mnt/f/AJOU/STAD_WSI/`

**TCGA**  
- `patch_size`: 448, `stride`: 448, `downsample`: 1
- `sat_thresh`: 5, `area_thresh`: 0.3
- Slide directory: `/mnt/e/TCGA/STAD/svs/`

**STFD**
- `patch_size`: 448, `stride`: 448, `downsample`: 2
- `sat_thresh`: 7, `area_thresh`: 0.4
- Slide directory: `/mnt/g/Stanford/STAD/`

### Usage Examples
```bash
# Basic feature extraction
python src/feature_extraction.py --model_type UNI --dataset AJOU

# High-throughput processing
python src/feature_extraction.py \
    --model_type DINO \
    --dataset TCGA \
    --batch_size 64 \
    --num_workers 8 \
    --start_idx 0 \
    --end_idx 50

# Process specific slides
python src/feature_extraction.py \
    --model_type UNI \
    --dataset AJOU \
    --slide_ids GA001 GA002 GA003 \
    --overwrite
```

---

## MIL Training

### Script: `src/mil_training.py`

#### Required Parameters
```bash
--model {CLAM_SB,CLAM_MB,TransMIL,ACMIL,DSMIL,ABMIL,GABMIL,MeanMIL,MaxMIL}
--dataset {JNUH,AJOU,TCGA,STFD}
```

#### Model Selection
```bash
--model CLAM_SB           # CLAM Single Branch
--model CLAM_MB           # CLAM Multi Branch  
--model TransMIL          # Transformer MIL
--model ACMIL             # Attention-Constrained MIL
--model DSMIL             # Dual-Stream MIL
--model ABMIL             # Attention-Based MIL
--model GABMIL            # Gated Attention-Based MIL
--model MeanMIL           # Mean Pooling MIL
--model MaxMIL            # Max Pooling MIL
```

#### Training Parameters
```bash
--device cuda:0          # Training device (default: cuda:0)
--epochs 50               # Number of training epochs (default: 50)
--batch_size 1            # Batch size (default: 1)
--learning_rate 1e-4      # Learning rate (default: 1e-4)
--weight_decay 1e-5       # Weight decay (default: 1e-5)
```

#### Cross-Validation Parameters
```bash
--fold 0                  # Single fold index (0-4)
--n_folds 5               # Number of CV folds (default: 5)
--run_all_folds           # Run all folds sequentially
```

#### Model-Specific Parameters
```bash
--dropout 0.25            # Dropout rate (default: 0.25)
--n_classes 2             # Number of classes (default: 2)
```

#### Feature & Dataset Parameters
```bash
--backbone_model UNI      # Feature extraction backbone
                          # Options: {DINO,UNI,GigaPath,Lunit}
```

#### Output Parameters
```bash
--save_dir checkpoints    # Model save directory (default: checkpoints)
--config path/config.yml  # Custom configuration file
```

#### Model-Specific Configurations

**CLAM (SB/MB)**
- Default feature dimension: 1024
- Attention branches (B): 8  
- Bag loss: CrossEntropy, Instance loss: CrossEntropy

**TransMIL**
- Masked patch modeling with transformer architecture
- Number of tokens: 5, Masked patches: 10
- Mask drop rate: 0.6

**DSMIL**
- Dual-stream architecture with instance classifier
- Separate bag and instance classification heads

### Usage Examples
```bash
# Train single model, single fold
python src/mil_training.py \
    --model CLAM_SB \
    --dataset AJOU \
    --backbone_model UNI \
    --fold 0 \
    --epochs 50

# Cross-validation training
python src/mil_training.py \
    --model TransMIL \
    --dataset TCGA \
    --backbone_model DINO \
    --run_all_folds \
    --epochs 30 \
    --learning_rate 5e-5

# Custom configuration
python src/mil_training.py \
    --model CLAM_MB \
    --dataset STFD \
    --config configs/dataset_config.yaml \
    --dropout 0.3 \
    --weight_decay 1e-4 \
    --save_dir custom_models/
```

---

## Visualization

### Script: `src/visualization.py`

#### Required Parameters
```bash
--model {CLAM_SB,CLAM_MB,TransMIL,ACMIL,DSMIL,ABMIL,GABMIL,MeanMIL,MaxMIL}
--model_path path/to/checkpoint.pth
--dataset {JNUH,AJOU,TCGA,STFD}
```

#### Processing Parameters
```bash
--device cuda:0          # Computation device
--backbone_model UNI     # Feature extraction backbone
```

#### Visualization Parameters  
```bash
--scaling_method minmax   # Attention scaling method
                          # Options: {minmax,zscore,none}
--alpha 0.4               # Attention overlay transparency (0.0-1.0)
--figsize 15              # Figure size for heatmaps
--patch_size 448          # Original patch size used
--downsample 28           # Downsampling for visualization
```

#### Output Parameters
```bash
--output_dir results/     # Output directory for heatmaps
--create_multiscale       # Generate multi-scale comparison heatmaps
```

#### Processing Control
```bash
--slide_ids slide1 slide2 # Process specific slides only
```

#### Scaling Methods

**minmax**: Normalize attention to [0,1] range
```bash
--scaling_method minmax
```

**zscore**: Z-score normalization then scale to [0,1]  
```bash
--scaling_method zscore
```

**none**: Use raw attention weights
```bash
--scaling_method none
```

### Usage Examples
```bash
# Basic attention heatmap
python src/visualization.py \
    --model CLAM_SB \
    --model_path checkpoints/CLAM_SB_fold0_best.pth \
    --dataset AJOU \
    --backbone_model UNI

# Multi-scale comparison heatmaps
python src/visualization.py \
    --model TransMIL \
    --model_path models/TransMIL_best.pth \
    --dataset TCGA \
    --backbone_model DINO \
    --create_multiscale \
    --scaling_method zscore \
    --alpha 0.5

# Specific slides with custom settings
python src/visualization.py \
    --model CLAM_MB \
    --model_path checkpoints/best_model.pth \
    --dataset STFD \
    --slide_ids slide001 slide002 slide003 \
    --output_dir custom_heatmaps/ \
    --figsize 20 \
    --alpha 0.3
```

---

## Tissue Analysis

### Script: `src/tissue_analysis.py`

#### Required Parameters
```bash
--dataset {JNUH,AJOU,TCGA,STFD}
--tissue_patch_dir path/to/tissue/patches
```

#### Processing Parameters
```bash
--device cuda:0          # Computation device
--backbone_model UNI     # Feature extraction backbone
```

#### Analysis Parameters
```bash
--create_correlation_analysis  # Generate correlation matrix analysis
--patch_size 448              # Patch size for spatial mapping
--downsample 28               # Visualization downsampling
--skip_umap                   # Skip UMAP embedding computation
```

#### Output Parameters
```bash
--output_dir tissue_results/  # Output directory
```

#### Processing Control
```bash
--slide_ids slide1 slide2     # Process specific slides only
```

#### Tissue Pattern Classes
The framework analyzes 4 predefined tissue patterns:
- **NR_Fibrosis**: Non-responder fibrotic patterns
- **NR_Signet_Ring_Cell**: Non-responder signet ring cells  
- **R_Hyperchromasia**: Responder nuclear hyperchromasia
- **R_Enlarged_Nuclei**: Responder enlarged nuclei patterns

#### Tissue Patch Directory Structure
```
tissue_patches/
├── NR_Fibrosis/
│   ├── patch001.png
│   └── patch002.png
├── NR_Signet_Ring_Cell/
│   ├── patch001.png
│   └── patch002.png
├── R_Hyperchromasia/
│   ├── patch001.png
│   └── patch002.png
└── R_Enlarged_Nuclei/
    ├── patch001.png
    └── patch002.png
```

### Usage Examples
```bash
# Complete tissue analysis
python src/tissue_analysis.py \
    --dataset AJOU \
    --backbone_model UNI \
    --tissue_patch_dir ../Patch_Noh \
    --create_correlation_analysis \
    --device cuda:0

# Specific slides, no correlation analysis
python src/tissue_analysis.py \
    --dataset TCGA \
    --backbone_model DINO \
    --tissue_patch_dir tissue_examples/ \
    --slide_ids slide001 slide002 slide003 \
    --output_dir tissue_results/ \
    --skip_umap

# Custom settings
python src/tissue_analysis.py \
    --dataset STFD \
    --backbone_model UNI \
    --tissue_patch_dir custom_tissues/ \
    --create_correlation_analysis \
    --patch_size 512 \
    --downsample 32
```

---

## Configuration Files

### Main Config: `configs/dataset_config.yaml`

#### Dataset Sections
Each dataset has its own configuration section:

```yaml
datasets:
  AJOU:
    slide_dir: "/path/to/slides/"
    clinical_file: "clinical_data.csv"
    patch_extraction:
      patch_size: 448
      stride: 448
      sat_thresh: 7
    clinical:
      response_column: "Best_of_Response"
      positive_responses: ["CR", "PR"]
```

#### Model Configurations
```yaml
models:
  feature_extraction:
    UNI:
      feature_dim: 1024
      transforms:
        mean: [0.485, 0.456, 0.406]
        std: [0.229, 0.224, 0.225]
  
  mil_architectures:
    CLAM_SB:
      default_params:
        feat_d: 1024
        dropout: 0.25
        B: 8
```

#### Training Configurations
```yaml
training:
  default:
    epochs: 50
    learning_rate: 1e-4
    weight_decay: 1e-5
  cross_validation:
    n_splits: 5
    group_based: true
```

### Loading Custom Configurations
```bash
# Use custom config file
python src/mil_training.py \
    --model CLAM_SB \
    --dataset AJOU \
    --config path/to/custom_config.yaml

# Config file overrides command line parameters for:
# - Model architecture parameters  
# - Dataset paths and settings
# - Training hyperparameters
```

---

## 🎛️ Advanced Parameter Combinations

### Multi-GPU Processing
```bash
# Feature extraction on specific GPU
python src/feature_extraction.py \
    --model_type UNI \
    --dataset TCGA \
    --device cuda:1 \
    --batch_size 64

# Training on different GPU
python src/mil_training.py \
    --model CLAM_SB \
    --dataset AJOU \
    --device cuda:2
```

### Batch Processing Workflows
```bash
# Extract features for slides 0-100
python src/feature_extraction.py \
    --model_type DINO \
    --dataset TCGA \
    --start_idx 0 \
    --end_idx 100

# Extract features for slides 100-200
python src/feature_extraction.py \
    --model_type DINO \
    --dataset TCGA \
    --start_idx 100 \
    --end_idx 200
```

### Cross-Dataset Validation
```bash
# Train on AJOU
python src/mil_training.py \
    --model CLAM_SB \
    --dataset AJOU \
    --backbone_model UNI \
    --save_dir models_ajou/

# Test on TCGA  
python src/visualization.py \
    --model CLAM_SB \
    --model_path models_ajou/best_model.pth \
    --dataset TCGA \
    --backbone_model UNI
```

---

## ⚠️ Important Notes

1. **GPU Memory**: Larger models (UNI, GigaPath) require more GPU memory. Reduce `batch_size` if you encounter OOM errors.

2. **Dataset Paths**: Update dataset paths in `configs/dataset_config.yaml` to match your local setup.

3. **Feature Compatibility**: Ensure the same `backbone_model` is used for feature extraction and subsequent training/visualization.

4. **Cross-Validation**: Use `--run_all_folds` for complete evaluation, or specify individual `--fold` indices for parallel processing.

5. **File Naming**: Output files include model name, dataset, and parameters for easy identification:
   - `CLAM_SB_fold0_epoch25_loss0.234_auroc0.856.pth`
   - `slide001_CLAM_SB_minmax_attention.png`

6. **Visualization Requirements**: Attention visualization requires a trained model checkpoint. Ensure model training is completed before running visualization scripts.