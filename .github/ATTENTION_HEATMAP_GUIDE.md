# 🎨 Attention Heatmap Generation Guide

This guide explains how to generate and interpret attention heatmaps for tissue-specific analysis in the STAD pathological image analysis framework.

## 🎯 Overview

The attention heatmap generation process creates interpretable visualizations that highlight tissue regions important for treatment response prediction. The framework supports multiple heatmap types and scaling approaches for clinical interpretation.

## 🔄 Attention Map Generation Workflow

### 1. **Feature Extraction Phase**
```python
# Extract features using DINO or UNI
python 01.DINO_PatchFeatureExtractor.py --dataset AJOU
python 01.UNI_PatchFeatureExtractor_TCGA.py --dataset TCGA
```

### 2. **MIL Training Phase**
```python
# Train MIL model with attention mechanism
python 03.MIL_Response.py --model CLAM --attention_branches 8
```

### 3. **Visualization Generation**
```python
# Generate standard attention heatmaps
python 04.MIL_Visualization_AJOU.py --fold 0 --model_path checkpoints/

# Generate tissue-specific analysis
cd Patch_Noh/
python Visualization.py --dataset AJOU
```

## 🏗️ Heatmap Types and Structure

### Standard Attention Heatmaps (`AttentionHeatmap/`)

**File Naming Convention:**
```
Fold{X}_Label{Y}_{confidence}+{sample_id}.png
```
- `X`: Cross-validation fold (0-4)
- `Y`: Ground truth label (0=Non-Responder, 1=Responder) 
- `confidence`: Model prediction confidence (0.000-1.000)
- `sample_id`: Patient/sample identifier

**Examples:**
- `Fold0_Label1_0.986+GI075_11584_37_HE.png` - High confidence responder prediction
- `Fold0_Label0_0.026+GI014_10064_20_HE.png` - Low confidence non-responder prediction

### AJOU Dataset Specific (`AttentionHeatmap_AJOU/`)

**Enhanced Organization:**
- **Fold-wise separation** for cross-validation analysis
- **Confidence-based sorting** for model reliability assessment
- **Patient ID mapping** for clinical correlation

### Advanced Tissue Analysis (`Patch_Noh/`)

#### A. Tissue Pattern Classes
The framework identifies 4 key histological patterns:

| Pattern | Description | Clinical Significance |
|---------|-------------|---------------------|
| **NR_Fibrosis** | Fibrotic tissue patterns in non-responders | Associated with treatment resistance |
| **NR_Signet_Ring_Cell** | Signet ring cells in non-responders | Aggressive cancer phenotype |
| **R_Hyperchromasia** | Nuclear hyperchromasia in responders | Active cellular response |  
| **R_Enlarged_Nuclei** | Enlarged nuclei in responders | Proliferative activity |

#### B. Correlation-Based Analysis
```python
# Generate best representative vectors for each tissue type
BestVector = {}
for tissue_type in ['NR_Fibrosis', 'NR_Signet_Ring_Cell', 'R_Hyperchromasia', 'R_Enlarged_Nuclei']:
    correlation_matrix = compute_correlation(patch_features[tissue_type])
    best_idx = correlation_matrix.sum(axis=1).argmax()
    BestVector[tissue_type] = patch_features[tissue_type][best_idx]
```

#### C. Visualization Output Types

1. **Standard Heatmaps (`Heatmap/`)**
   - Direct attention weights from MIL models
   - Standard color mapping (jet colormap)

2. **MinMax Scaled (`Heatmap_minMax/`)**
   - Normalized to [0,1] range for consistent interpretation
   - Better for cross-slide comparisons

3. **Scaled Heatmaps (`Heatmap_scale/`)**  
   - Z-score normalization for statistical consistency
   - Removes bias from different attention magnitudes

4. **Vertical Layout (`Heatmap_scale_vertical/`)**
   - Optimized for portrait-oriented slides
   - Clinical presentation format

## 🎨 Visualization Parameters

### Color Mapping
```python
# Jet colormap with transparency
cmap_jet = plt.cm.jet
cmap_jet[:1, -1] = 0  # First color transparent
cmap_jet = ListedColormap(cmap_jet)

# Patch overlay mapping  
cmap_patch = np.array([[0, 0, 0, 1], [0, 0, 0, 0]])  # Black & Transparent
cmap_patch = ListedColormap(cmap_patch)
```

### Scaling Functions
```python
def minMax(x): 
    return (x - x.min()) / (x.max() - x.min())

def z_score_normalize(attention_weights):
    return (attention_weights - attention_weights.mean()) / attention_weights.std()
```

## 🔧 Generation Process Details

### 1. Patch-Level Correlation
```python
# For each slide, compute correlations with tissue patterns
correlations = []
for tissue_type in tissue_patterns:
    correlation = []
    for patch_idx in range(n_patches):
        cor, _ = pearsonr(slide_features[patch_idx], BestVector[tissue_type])
        correlation.append(cor)
    
    # Normalize by maximum correlation for this tissue type
    correlations.append(np.array(correlation) / max_correlation[tissue_type])
```

### 2. Spatial Mapping
```python
# Map patch correlations to slide coordinates
patch_size = 448
downsample = 28

for idx, (x_pos, y_pos) in enumerate(patch_coordinates):
    x_grid = int(x_pos / patch_size)
    y_grid = int(y_pos / patch_size) 
    
    # Assign tissue-specific correlations
    spatial_map[y_grid, x_grid, :] = [
        correlations[0][idx],  # NR_Fibrosis
        correlations[1][idx],  # NR_Signet_Ring_Cell  
        correlations[2][idx],  # R_Hyperchromasia
        correlations[3][idx]   # R_Enlarged_Nuclei
    ]
```

### 3. Multi-Panel Visualization
```python
# Create 6-panel visualization
fig, axes = plt.subplots(nrows=1, ncols=6, figsize=(x_pix*6, y_pix*1))

panels = [
    'Original WSI',
    'UMAP Embedding', 
    'NR_Fibrosis',
    'NR_Signet_Ring_Cell',
    'R_Hyperchromasia', 
    'R_Enlarged_Nuclei'
]

for i, panel_name in enumerate(panels):
    axes[i].imshow(wsi_thumbnail)
    if i > 0:  # Overlay correlations
        axes[i].imshow(correlation_maps[:,:,i-1], alpha=0.3, vmax=0.5, vmin=0, cmap=cmap_jet)
```

## 📊 Interpretation Guidelines

### Attention Weight Interpretation
- **High attention (red)**: Regions strongly associated with prediction
- **Medium attention (yellow/green)**: Moderately important regions
- **Low attention (blue/transparent)**: Less relevant tissue areas

### Clinical Relevance
1. **Treatment Response**: Red regions indicate tissue patterns predictive of response
2. **Biomarker Discovery**: Consistent attention patterns across patients suggest novel biomarkers
3. **Quality Control**: Attention outside tissue boundaries indicates model issues

### Statistical Validation
- Compare attention patterns between responders vs non-responders
- Validate patterns across different datasets (AJOU, TCGA, STFD)
- Correlate with known biomarkers and clinical outcomes

## 🚀 Usage Examples

### Generate Basic Heatmaps
```bash
# For AJOU dataset
python 04.MIL_Visualization_AJOU.py \
    --model_path checkpoints/CLAM_fold0_best.pth \
    --output_dir AttentionHeatmap_AJOU/ \
    --confidence_threshold 0.8

# For TCGA dataset  
python 04.MIL_Visualization_TCGA.py \
    --model_path checkpoints/CLAM_TCGA.pth \
    --batch_size 1 \
    --alpha 0.4
```

### Generate Tissue-Specific Analysis
```bash
cd Patch_Noh/

# Standard scaling
python Visualization.py \
    --dataset AJOU \
    --output_type scale \
    --device 0

# MinMax scaling  
python Visualization.py \
    --dataset AJOU \
    --output_type minmax \
    --alpha 0.3
```

### Batch Processing
```bash
# Process all folds
for fold in {0..4}; do
    python 04.MIL_Visualization_AJOU.py \
        --fold $fold \
        --model_path checkpoints/fold_${fold}_best.pth
done
```

## 🔍 Quality Control

### Validation Checks
1. **Attention Coverage**: Ensure attention focuses on tissue regions, not background
2. **Consistency**: Similar patterns should emerge across cross-validation folds  
3. **Biological Plausibility**: Attention should align with known histological features
4. **Calibration**: High confidence predictions should show clear attention patterns

### Common Issues and Solutions

| Issue | Cause | Solution |
|-------|-------|----------|
| Attention on background | Poor tissue segmentation | Improve patch filtering |
| Inconsistent patterns | Model overfitting | Increase regularization |
| Low attention contrast | Insufficient training | More epochs or data |
| Missing visualizations | Path/memory issues | Check file paths and RAM |

## 📈 Advanced Analysis

### Cross-Dataset Comparison
```python
# Compare attention patterns across datasets
datasets = ['AJOU', 'TCGA', 'STFD']
attention_similarity = compute_cross_dataset_similarity(datasets)
```

### Survival Correlation
```python
# Correlate attention patterns with survival outcomes
survival_data = load_survival_data()
attention_survival_correlation = correlate_attention_survival(attention_maps, survival_data)
```

### Biomarker Discovery
```python
# Identify novel attention-based biomarkers
biomarker_candidates = discover_attention_biomarkers(
    attention_maps, 
    clinical_outcomes, 
    statistical_threshold=0.05
)
```

---

For additional support and examples, see the main [README.md](README.md) or open an issue on GitHub.