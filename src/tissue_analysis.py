#!/usr/bin/env python3
"""
Unified Tissue-Specific Analysis Script (Patch_Noh functionality)

Performs correlation-based tissue pattern analysis and generates specialized visualizations
"""

import os
import sys
import argparse
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
import cv2
import openslide
from PIL import Image
import pickle
import umap
from scipy.stats import pearsonr
from sklearn.feature_selection import mutual_info_classif
from scipy import stats
from glob import glob
from typing import List, Dict, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import load_data, get_dataset_paths, Struct, minMax, load_model_by_type, get_model_transforms


# Setup color maps
cmap = plt.cm.jet
cmap_jet = cmap(np.arange(cmap.N))
cmap_jet[:1, -1] = 0  # First color transparent
cmap_jet = ListedColormap(cmap_jet)

cmap_patch = np.array([[0, 0, 0, 1], [0, 0, 0, 0]])  # Black & Transparent
cmap_patch = ListedColormap(cmap_patch)


class TissuePatternAnalyzer:
    """
    Tissue-specific pattern analysis for treatment response prediction
    """
    
    def __init__(self, dataset: str, backbone_model: str, device: torch.device):
        """
        Initialize tissue pattern analyzer
        
        Args:
            dataset: Dataset name
            backbone_model: Feature extraction backbone
            device: Device for computation
        """
        self.dataset = dataset
        self.backbone_model = backbone_model
        self.device = device
        
        # Dataset configuration
        self.dataset_config = get_dataset_paths(dataset)
        
        # Load feature extraction model
        print(f"Loading {backbone_model} model...")
        self.model = load_model_by_type(backbone_model, device)
        self.transform = get_model_transforms(backbone_model)
        
        # Tissue pattern classes
        self.tissue_classes = [
            'NR_Fibrosis',           # Non-responder fibrotic patterns
            'NR_Signet_Ring_Cell',   # Non-responder signet ring cells
            'R_Hyperchromasia',      # Responder nuclear hyperchromasia
            'R_Enlarged_Nuclei'      # Responder enlarged nuclei
        ]
        
        print(f"Tissue Pattern Analyzer initialized:")
        print(f"  Dataset: {dataset}")
        print(f"  Backbone: {backbone_model}")
        print(f"  Tissue classes: {len(self.tissue_classes)}")
    
    def extract_tissue_features(self, tissue_patch_dir: str) -> Dict[str, np.ndarray]:
        """
        Extract features from tissue-specific patch examples
        
        Args:
            tissue_patch_dir: Directory containing tissue pattern patches
            
        Returns:
            Dictionary mapping tissue class to features
        """
        patch_features = {}
        
        for tissue_class in self.tissue_classes:
            print(f"Processing tissue class: {tissue_class}")
            
            class_dir = os.path.join(tissue_patch_dir, tissue_class)
            if not os.path.exists(class_dir):
                print(f"Warning: Directory not found: {class_dir}")
                continue
            
            # Get patch files
            patch_files = glob(os.path.join(class_dir, "*.png"))
            if not patch_files:
                print(f"Warning: No patch files found in {class_dir}")
                continue
            
            print(f"Found {len(patch_files)} patches")
            
            # Extract features
            features = []
            self.model.eval()
            
            with torch.no_grad():
                for patch_file in tqdm(patch_files, desc=f"Extracting {tissue_class}"):
                    try:
                        # Load and preprocess patch
                        img = Image.open(patch_file).convert('RGB')
                        img_tensor = self.transform(img).unsqueeze(0).to(self.device)
                        
                        # Extract feature
                        feature = self.model(img_tensor)
                        
                        # Handle different output formats
                        if isinstance(feature, tuple):
                            feature = feature[0]
                        
                        # Ensure 2D output
                        if len(feature.shape) > 2:
                            feature = feature.mean(dim=1)
                        
                        features.append(feature.cpu().numpy()[0])
                        
                    except Exception as e:
                        print(f"Error processing {patch_file}: {e}")
                        continue
            
            if features:
                patch_features[tissue_class] = np.stack(features)
                print(f"Extracted features shape: {patch_features[tissue_class].shape}")
            else:
                print(f"No features extracted for {tissue_class}")
        
        return patch_features
    
    def compute_best_representatives(self, patch_features: Dict[str, np.ndarray]) -> Tuple[Dict, Dict]:
        """
        Compute best representative vectors for each tissue type
        
        Args:
            patch_features: Dictionary of tissue features
            
        Returns:
            Tuple of (best_vectors, max_correlations)
        """
        best_vectors = {}
        max_correlations = {}
        
        for tissue_class, features in patch_features.items():
            if len(features) == 0:
                continue
            
            print(f"Computing best representative for {tissue_class}...")
            
            # Compute correlation matrix
            corr_matrix = pd.DataFrame(features).T.corr()
            
            # Find maximum correlation (excluding self-correlation)
            mask = ~np.eye(corr_matrix.shape[0], dtype=bool)
            max_correlations[tissue_class] = corr_matrix.values[mask].max()
            
            # Find patch with highest sum of correlations (most representative)
            correlation_sums = corr_matrix.sum(axis=1)
            best_idx = correlation_sums.argmax()
            
            best_vectors[tissue_class] = features[best_idx]
            
            print(f"  Best representative index: {best_idx}")
            print(f"  Max correlation: {max_correlations[tissue_class]:.3f}")
        
        return best_vectors, max_correlations
    
    def compute_slide_correlations(self, slide_features: np.ndarray,
                                 best_vectors: Dict, max_correlations: Dict) -> np.ndarray:
        """
        Compute correlations between slide patches and tissue patterns
        
        Args:
            slide_features: Features for all patches in a slide
            best_vectors: Best representative vectors for each tissue type
            max_correlations: Maximum correlations for normalization
            
        Returns:
            Correlation matrix (n_patches x n_tissue_types)
        """
        correlations = []
        
        for tissue_class in self.tissue_classes:
            if tissue_class not in best_vectors:
                # Use zeros if tissue class not available
                correlation = np.zeros(slide_features.shape[0])
            else:
                correlation = []
                best_vector = best_vectors[tissue_class]
                max_corr = max_correlations[tissue_class]
                
                for i in range(slide_features.shape[0]):
                    try:
                        corr, _ = pearsonr(slide_features[i], best_vector)
                        correlation.append(corr / max_corr if max_corr > 0 else 0)
                    except:
                        correlation.append(0)
                
                correlation = np.array(correlation)
            
            correlations.append(correlation)
        
        return np.stack(correlations).transpose()  # Shape: (n_patches, n_tissue_types)
    
    def create_tissue_correlation_heatmap(self, slide_id: str, 
                                        best_vectors: Dict, max_correlations: Dict,
                                        output_path: str = None,
                                        patch_size: int = 448, downsample: int = 28,
                                        create_umap: bool = True) -> str:
        """
        Create tissue correlation heatmap for a slide
        
        Args:
            slide_id: Slide identifier
            best_vectors: Best representative vectors for each tissue type
            max_correlations: Maximum correlations for normalization
            output_path: Output file path
            patch_size: Patch size used for extraction
            downsample: Downsampling factor for visualization
            create_umap: Whether to include UMAP visualization
            
        Returns:
            Path to saved heatmap
        """
        print(f"Creating tissue correlation heatmap for {slide_id}...")
        
        try:
            # Load slide data
            base_path = self.dataset_config["output_dir"]
            coord_path = os.path.join(base_path, f"Coord_{self.backbone_model}")
            feature_path = os.path.join(base_path, f"Feature_{self.backbone_model}")
            slide_dir = self.dataset_config["slide_dir"]
            
            # Load features and coordinates
            feature_file = os.path.join(feature_path, f"{slide_id}.pickle")
            coord_file = os.path.join(coord_path, f"{slide_id}.pickle")
            
            with open(feature_file, 'rb') as f:
                features = pickle.load(f)
            
            with open(coord_file, 'rb') as f:
                coordinates = pickle.load(f)
            
            # Find slide file
            slide_extensions = ['.svs', '.tif', '.tiff', '.ndpi', '.vms', '.vmu', '.scn']
            slide_path = None
            
            for ext in slide_extensions:
                potential_path = os.path.join(slide_dir, f"{slide_id}{ext}")
                if os.path.exists(potential_path):
                    slide_path = potential_path
                    break
            
            if slide_path is None:
                raise FileNotFoundError(f"Slide file not found for {slide_id}")
            
            # Open slide
            slide = openslide.open_slide(slide_path)
            slide_thumbnail = np.array(
                slide.get_thumbnail(np.array(slide.dimensions) // downsample)
            )
            
            print(f"Slide dimensions: {slide.dimensions}")
            print(f"Thumbnail shape: {slide_thumbnail.shape}")
            
            # Compute correlations
            correlations = self.compute_slide_correlations(
                features, best_vectors, max_correlations
            )
            
            # Create UMAP embedding if requested
            umap_embedding = None
            if create_umap:
                print("Computing UMAP embedding...")
                umap_model = umap.UMAP(n_components=1, random_state=42)
                umap_embedding = umap_model.fit_transform(features)
                umap_embedding = minMax(umap_embedding.flatten())
            
            # Create spatial maps
            n_panels = len(self.tissue_classes) + (2 if create_umap else 1)  # +1 for original, +1 for UMAP
            
            # Setup figure
            y_pix = 1
            x_pix = slide_thumbnail.shape[1] / slide_thumbnail.shape[0]
            
            fig, axes = plt.subplots(1, n_panels, figsize=(x_pix * n_panels * 3, y_pix * 3))
            if n_panels == 1:
                axes = [axes]
            
            panel_idx = 0
            
            # Original slide
            axes[panel_idx].imshow(slide_thumbnail)
            axes[panel_idx].set_title('Original WSI')
            axes[panel_idx].axis('off')
            panel_idx += 1
            
            # UMAP embedding
            if create_umap:
                axes[panel_idx].imshow(slide_thumbnail)
                
                # Create UMAP spatial map
                umap_spatial_map = np.zeros((*slide_thumbnail.shape[:2], 1))
                for i, (x, y) in enumerate(coordinates):
                    x_scaled = int(x // downsample)
                    y_scaled = int(y // downsample)
                    
                    if (0 <= x_scaled < umap_spatial_map.shape[1] and 
                        0 <= y_scaled < umap_spatial_map.shape[0]):
                        umap_spatial_map[y_scaled, x_scaled, 0] = umap_embedding[i]
                
                # Resize and smooth
                umap_spatial_map = cv2.resize(
                    umap_spatial_map, slide_thumbnail.shape[:2][::-1], 
                    interpolation=cv2.INTER_CUBIC
                )
                umap_spatial_map = cv2.GaussianBlur(umap_spatial_map, (5, 5), 0)
                
                axes[panel_idx].imshow(umap_spatial_map, alpha=0.4, vmax=1, vmin=0, cmap=cmap_jet)
                axes[panel_idx].set_title('UMAP Embedding')
                axes[panel_idx].axis('off')
                panel_idx += 1
            
            # Tissue correlation maps
            for tissue_idx, tissue_class in enumerate(self.tissue_classes):
                axes[panel_idx].imshow(slide_thumbnail)
                
                # Create spatial correlation map
                corr_spatial_map = np.zeros((*slide_thumbnail.shape[:2], 1))
                
                for i, (x, y) in enumerate(coordinates):
                    x_scaled = int(x // downsample)
                    y_scaled = int(y // downsample)
                    
                    if (0 <= x_scaled < corr_spatial_map.shape[1] and 
                        0 <= y_scaled < corr_spatial_map.shape[0]):
                        corr_spatial_map[y_scaled, x_scaled, 0] = correlations[i, tissue_idx]
                
                # Resize and smooth
                corr_spatial_map = cv2.resize(
                    corr_spatial_map, slide_thumbnail.shape[:2][::-1], 
                    interpolation=cv2.INTER_CUBIC
                )
                corr_spatial_map = cv2.GaussianBlur(corr_spatial_map, (5, 5), 0)
                
                # Overlay correlation map
                im = axes[panel_idx].imshow(
                    corr_spatial_map, alpha=0.4, vmax=0.5, vmin=0, cmap=cmap_jet
                )
                
                # Clean up tissue class name for display
                display_name = tissue_class.replace('_', ' ')
                axes[panel_idx].set_title(display_name)
                axes[panel_idx].axis('off')
                
                panel_idx += 1
            
            # Add white overlay for patch boundaries
            cmap_white = LinearSegmentedColormap.from_list(
                'custom_cmap', [(0, (1, 1, 1, 0)), (1, (1, 1, 1, 1))]
            )
            
            patch_map_white = np.zeros((*slide_thumbnail.shape[:2], 1))
            patch_map_white += 1
            
            for x, y in coordinates:
                x_scaled = int(x // downsample)
                y_scaled = int(y // downsample)
                
                if (0 <= x_scaled < patch_map_white.shape[1] and 
                    0 <= y_scaled < patch_map_white.shape[0]):
                    patch_map_white[y_scaled, x_scaled, 0] = 0
            
            patch_map_white = cv2.resize(
                patch_map_white, slide_thumbnail.shape[:2][::-1]
            ) > 0.5
            
            # Apply white overlay to all panels except the first
            for i in range(1, len(axes)):
                axes[i].imshow(patch_map_white, cmap=cmap_white, vmin=0, vmax=1, alpha=0.3)
            
            plt.suptitle(f'{slide_id} - Tissue Pattern Analysis', fontsize=16)
            plt.tight_layout()
            
            # Save figure
            if output_path is None:
                output_dir = f"tissue_analysis_{self.dataset}_{self.backbone_model}"
                os.makedirs(output_dir, exist_ok=True)
                output_path = os.path.join(output_dir, f"{slide_id}_tissue_analysis.png")
            
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"Tissue analysis heatmap saved: {output_path}")
            return output_path
            
        except Exception as e:
            print(f"Error creating tissue heatmap for {slide_id}: {e}")
            return None
    
    def analyze_correlation_matrix(self, patch_features: Dict[str, np.ndarray],
                                 output_dir: str = "tissue_correlation_analysis"):
        """
        Analyze and visualize inter-patch correlations within tissue types
        
        Args:
            patch_features: Dictionary of tissue features
            output_dir: Output directory for analysis results
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Combine all features for mutual information analysis
        all_features = []
        all_labels = []
        
        for tissue_class, features in patch_features.items():
            if len(features) > 0:
                all_features.append(features)
                all_labels.extend([tissue_class] * len(features))
        
        if not all_features:
            print("No features available for correlation analysis")
            return
        
        all_features = np.vstack(all_features)
        
        # Compute mutual information
        print("Computing mutual information...")
        mi_scores = mutual_info_classif(all_features, all_labels)
        
        # Select top features
        top_n = min(128, len(mi_scores))
        top_feature_indices = np.argsort(mi_scores)[-top_n:]
        
        # Create correlation matrix with top features
        selected_features = all_features[:, top_feature_indices]
        correlation_matrix = pd.DataFrame(selected_features).T.corr()
        
        # Add labels
        correlation_matrix.columns = all_labels
        correlation_matrix.index = all_labels
        
        # Save correlation matrix
        correlation_matrix.to_csv(
            os.path.join(output_dir, f"inter_correlation_{top_n}_features.csv")
        )
        
        # Create correlation heatmap
        plt.figure(figsize=(12, 10))
        
        # Create custom colormap
        colors = ['blue', 'white', 'red']
        n_bins = 100
        cmap_corr = LinearSegmentedColormap.from_list('correlation', colors, N=n_bins)
        
        # Plot heatmap
        im = plt.imshow(correlation_matrix, cmap=cmap_corr, vmin=-1, vmax=1, aspect='auto')
        
        # Add colorbar
        cbar = plt.colorbar(im, fraction=0.046, pad=0.04)
        cbar.set_label('Correlation', rotation=270, labelpad=15)
        
        plt.title(f'Inter-Patch Correlation Matrix (Top {top_n} Features)')
        plt.tight_layout()
        
        # Save heatmap
        plt.savefig(
            os.path.join(output_dir, f"correlation_heatmap_{top_n}.png"), 
            dpi=300, bbox_inches='tight'
        )
        plt.close()
        
        print(f"Correlation analysis saved to: {output_dir}")
    
    def batch_tissue_analysis(self, tissue_patch_dir: str, 
                            slide_ids: List[str] = None,
                            output_dir: str = None,
                            create_correlation_analysis: bool = True):
        """
        Perform tissue analysis for multiple slides
        
        Args:
            tissue_patch_dir: Directory containing tissue pattern examples
            slide_ids: List of slide IDs to process (None for all)
            output_dir: Output directory
            create_correlation_analysis: Whether to create correlation analysis
        """
        print("Starting batch tissue analysis...")
        
        # Extract tissue features from examples
        print("Extracting tissue pattern features...")
        patch_features = self.extract_tissue_features(tissue_patch_dir)
        
        if not patch_features:
            print("No tissue features extracted. Exiting.")
            return
        
        # Compute best representatives
        best_vectors, max_correlations = self.compute_best_representatives(patch_features)
        
        # Create correlation analysis if requested
        if create_correlation_analysis:
            self.analyze_correlation_matrix(patch_features)
        
        # Get slide IDs if not provided
        if slide_ids is None:
            base_path = self.dataset_config["output_dir"]
            coord_path = os.path.join(base_path, f"Coord_{self.backbone_model}")
            slide_ids = [
                f.split('.pickle')[0] 
                for f in os.listdir(coord_path)
                if f.endswith('.pickle')
            ]
        
        print(f"Processing {len(slide_ids)} slides...")
        
        # Create output directory
        if output_dir is None:
            output_dir = f"tissue_analysis_{self.dataset}_{self.backbone_model}"
        os.makedirs(output_dir, exist_ok=True)
        
        successful = 0
        failed = []
        
        # Process each slide
        for slide_id in tqdm(slide_ids, desc="Creating tissue analyses"):
            try:
                output_path = os.path.join(output_dir, f"{slide_id}_tissue_analysis.png")
                
                result = self.create_tissue_correlation_heatmap(
                    slide_id=slide_id,
                    best_vectors=best_vectors,
                    max_correlations=max_correlations,
                    output_path=output_path
                )
                
                if result:
                    successful += 1
                else:
                    failed.append(slide_id)
                    
            except Exception as e:
                print(f"Failed to process {slide_id}: {e}")
                failed.append(slide_id)
        
        print(f"\nTissue analysis completed:")
        print(f"  Successful: {successful}")
        print(f"  Failed: {len(failed)}")
        
        if failed:
            print(f"  Failed slides: {failed[:5]}{'...' if len(failed) > 5 else ''}")


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Tissue-Specific Pattern Analysis for STAD")
    
    # Dataset arguments
    parser.add_argument("--dataset", type=str, required=True,
                       choices=["JNUH", "AJOU", "TCGA", "STFD"],
                       help="Dataset name")
    
    parser.add_argument("--backbone_model", type=str, default="UNI",
                       choices=["DINO", "UNI", "GigaPath", "Lunit"],
                       help="Feature extraction backbone")
    
    # Processing arguments
    parser.add_argument("--device", type=str, default="cuda:0",
                       help="Device for computation")
    
    parser.add_argument("--tissue_patch_dir", type=str, required=True,
                       help="Directory containing tissue pattern examples")
    
    parser.add_argument("--slide_ids", nargs="+", default=None,
                       help="Specific slide IDs to process")
    
    parser.add_argument("--output_dir", type=str, default=None,
                       help="Output directory for analysis results")
    
    # Analysis arguments
    parser.add_argument("--create_correlation_analysis", action="store_true",
                       help="Create inter-patch correlation analysis")
    
    parser.add_argument("--patch_size", type=int, default=448,
                       help="Patch size used for feature extraction")
    
    parser.add_argument("--downsample", type=int, default=28,
                       help="Downsampling factor for visualization")
    
    parser.add_argument("--skip_umap", action="store_true",
                       help="Skip UMAP embedding computation")
    
    return parser.parse_args()


def main():
    """Main execution function"""
    args = parse_arguments()
    
    # Setup device
    if torch.cuda.is_available() and "cuda" in args.device:
        device = torch.device(args.device)
        print(f"Using GPU: {device}")
    else:
        device = torch.device("cpu")
        print("Using CPU")
    
    # Initialize analyzer
    try:
        analyzer = TissuePatternAnalyzer(
            dataset=args.dataset,
            backbone_model=args.backbone_model,
            device=device
        )
        
        # Perform tissue analysis
        analyzer.batch_tissue_analysis(
            tissue_patch_dir=args.tissue_patch_dir,
            slide_ids=args.slide_ids,
            output_dir=args.output_dir,
            create_correlation_analysis=args.create_correlation_analysis
        )
        
        print("\n✓ Tissue analysis completed!")
        
    except Exception as e:
        print(f"✗ Error during tissue analysis: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())