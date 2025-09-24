#!/usr/bin/env python3
"""
Unified Visualization Script for STAD Pathological Image Analysis

Generates attention heatmaps and visualizations for all datasets and models
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
import matplotlib.patches as patches
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
import cv2
import openslide
from PIL import Image
import pickle
from typing import List, Dict, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import load_data, get_dataset_paths, Struct, minMax
from mil_training import load_mil_model, PatchFeatureGenerator


# Color maps for visualization
def setup_colormaps():
    """Setup color maps for visualization"""
    # Jet colormap with transparency
    cmap = plt.cm.jet
    cmap_jet = cmap(np.arange(cmap.N))
    cmap_jet[:1, -1] = 0  # First color transparent
    cmap_jet = ListedColormap(cmap_jet)
    
    # Patch overlay colormap (Black & Transparent)
    cmap_patch = np.array([[0, 0, 0, 1], [0, 0, 0, 0]])
    cmap_patch = ListedColormap(cmap_patch)
    
    # White transparent colormap
    cmap_white = LinearSegmentedColormap.from_list(
        'custom_cmap', [(0, (1, 1, 1, 0)), (1, (1, 1, 1, 1))]
    )
    
    return cmap_jet, cmap_patch, cmap_white


class AttentionVisualizer:
    """
    Unified attention visualization class supporting multiple datasets and models
    """
    
    def __init__(self, model_name: str, dataset: str, backbone_model: str, 
                 device: torch.device, model_path: str = None):
        """
        Initialize attention visualizer
        
        Args:
            model_name: MIL architecture name
            dataset: Dataset name
            backbone_model: Feature extraction backbone
            device: Device for computation
            model_path: Path to trained model checkpoint
        """
        self.model_name = model_name
        self.dataset = dataset
        self.backbone_model = backbone_model
        self.device = device
        self.model_path = model_path
        
        # Setup color maps
        self.cmap_jet, self.cmap_patch, self.cmap_white = setup_colormaps()
        
        # Dataset configuration
        self.dataset_config = get_dataset_paths(dataset)
        
        # Load model if path provided
        self.model = None
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
        
        print(f"Visualizer initialized:")
        print(f"  Model: {model_name}")
        print(f"  Dataset: {dataset}")
        print(f"  Backbone: {backbone_model}")
    
    def load_model(self, model_path: str):
        """Load trained model from checkpoint"""
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Create config from checkpoint
        config_dict = checkpoint.get('config', {})
        config_dict.update({
            'n_class': 2,
            'dropout': 0.25,
            'feat_d': 1024,
            'D_feat': 1024
        })
        config = Struct(**config_dict)
        
        # Load model
        self.model = load_mil_model(self.model_name, config)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        print(f"Model loaded from: {model_path}")
    
    def get_attention_weights(self, features: torch.Tensor) -> np.ndarray:
        """
        Extract attention weights from model
        
        Args:
            features: Input features tensor
            
        Returns:
            Attention weights array
        """
        if self.model is None:
            raise ValueError("Model not loaded. Provide model_path or load model first.")
        
        with torch.no_grad():
            features = features.to(self.device)
            output = self.model(features)
            
            # Extract attention weights based on model type
            if isinstance(output, dict):
                if 'attention' in output:
                    attention = output['attention']
                elif 'A' in output:  # CLAM
                    attention = output['A']
                else:
                    # Use uniform attention if not available
                    attention = torch.ones(features.shape[1], 1) / features.shape[1]
            else:
                # For models without explicit attention
                attention = torch.ones(features.shape[1], 1) / features.shape[1]
            
            return attention.cpu().numpy().flatten()
    
    def load_slide_data(self, slide_id: str) -> Tuple[np.ndarray, List, str, np.ndarray]:
        """
        Load features, coordinates and slide for a specific slide
        
        Args:
            slide_id: Slide identifier
            
        Returns:
            Tuple of (features, coordinates, slide_path, attention_weights)
        """
        # Paths
        base_path = self.dataset_config["output_dir"]
        coord_path = os.path.join(base_path, f"Coord_{self.backbone_model}")
        feature_path = os.path.join(base_path, f"Feature_{self.backbone_model}")
        slide_dir = self.dataset_config["slide_dir"]
        
        # Load features and coordinates
        feature_file = os.path.join(feature_path, f"{slide_id}.pickle")
        coord_file = os.path.join(coord_path, f"{slide_id}.pickle")
        
        if not os.path.exists(feature_file) or not os.path.exists(coord_file):
            raise FileNotFoundError(f"Features or coordinates not found for {slide_id}")
        
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
        
        # Get attention weights
        features_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0)
        attention_weights = self.get_attention_weights(features_tensor)
        
        return features, coordinates, slide_path, attention_weights
    
    def create_attention_heatmap(self, slide_id: str, output_path: str = None,
                               patch_size: int = 448, downsample: int = 28,
                               alpha: float = 0.4, figsize: int = 15,
                               scaling_method: str = "minmax") -> str:
        """
        Create attention heatmap for a slide
        
        Args:
            slide_id: Slide identifier
            output_path: Output file path
            patch_size: Patch size used for extraction
            downsample: Downsampling factor for visualization
            alpha: Transparency of attention overlay
            figsize: Figure size
            scaling_method: Attention scaling method ("minmax", "zscore", "none")
            
        Returns:
            Path to saved heatmap
        """
        print(f"Creating attention heatmap for {slide_id}...")
        
        try:
            # Load slide data
            features, coordinates, slide_path, attention_weights = self.load_slide_data(slide_id)
            
            # Open slide
            slide = openslide.open_slide(slide_path)
            slide_thumbnail = np.array(
                slide.get_thumbnail(np.array(slide.dimensions) // downsample)
            )
            
            print(f"Slide dimensions: {slide.dimensions}")
            print(f"Thumbnail shape: {slide_thumbnail.shape}")
            print(f"Number of patches: {len(coordinates)}")
            
            # Scale attention weights
            if scaling_method == "minmax":
                attention_weights = minMax(attention_weights)
            elif scaling_method == "zscore":
                attention_weights = (attention_weights - attention_weights.mean()) / \
                                  (attention_weights.std() + 1e-8)
                attention_weights = minMax(attention_weights)  # Normalize to [0,1]
            # "none" keeps original weights
            
            # Create spatial attention map
            spatial_map = np.zeros(
                (*slide_thumbnail.shape[:2], 1),
                dtype=np.float32
            )
            
            patch_size_scaled = patch_size // downsample
            
            for i, (x, y) in enumerate(coordinates):
                x_scaled = int(x // downsample)
                y_scaled = int(y // downsample)
                
                # Ensure coordinates are within bounds
                if (0 <= x_scaled < spatial_map.shape[1] and 
                    0 <= y_scaled < spatial_map.shape[0]):
                    spatial_map[y_scaled, x_scaled, 0] = attention_weights[i]
            
            # Apply Gaussian blur for smoother visualization
            spatial_map[:, :, 0] = cv2.GaussianBlur(
                spatial_map[:, :, 0], (5, 5), 0
            )
            
            # Create visualization
            fig, ax = plt.subplots(figsize=(figsize, figsize * slide_thumbnail.shape[0] / slide_thumbnail.shape[1]))
            
            # Show original slide
            ax.imshow(slide_thumbnail)
            
            # Overlay attention heatmap
            im = ax.imshow(
                spatial_map[:, :, 0], 
                alpha=alpha,
                vmin=0, 
                vmax=1,
                cmap=self.cmap_jet
            )
            
            # Add colorbar
            cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label('Attention Weight', rotation=270, labelpad=20)
            
            # Remove axes
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(f'{slide_id} - {self.model_name} Attention', fontsize=16)
            
            # Save figure
            if output_path is None:
                output_dir = f"heatmaps_{self.dataset}_{self.backbone_model}"
                os.makedirs(output_dir, exist_ok=True)
                output_path = os.path.join(
                    output_dir, 
                    f"{slide_id}_{self.model_name}_attention.png"
                )
            
            plt.tight_layout()
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"Heatmap saved: {output_path}")
            return output_path
            
        except Exception as e:
            print(f"Error creating heatmap for {slide_id}: {e}")
            return None
    
    def create_multi_scale_heatmap(self, slide_id: str, output_path: str = None) -> str:
        """
        Create multi-scale attention heatmap with different views
        
        Args:
            slide_id: Slide identifier
            output_path: Output file path
            
        Returns:
            Path to saved heatmap
        """
        try:
            features, coordinates, slide_path, attention_weights = self.load_slide_data(slide_id)
            
            # Open slide
            slide = openslide.open_slide(slide_path)
            downsample = 28
            slide_thumbnail = np.array(
                slide.get_thumbnail(np.array(slide.dimensions) // downsample)
            )
            
            # Different scaling methods
            attention_minmax = minMax(attention_weights)
            attention_zscore = (attention_weights - attention_weights.mean()) / \
                             (attention_weights.std() + 1e-8)
            attention_zscore = minMax(attention_zscore)
            
            # Create figure with multiple subplots
            fig, axes = plt.subplots(1, 3, figsize=(20, 7))
            
            scaling_methods = [
                ("Original", attention_weights),
                ("Min-Max Scaled", attention_minmax),
                ("Z-Score Scaled", attention_zscore)
            ]
            
            for idx, (title, attention) in enumerate(scaling_methods):
                # Create spatial map
                spatial_map = np.zeros((*slide_thumbnail.shape[:2], 1))
                
                for i, (x, y) in enumerate(coordinates):
                    x_scaled = int(x // downsample)
                    y_scaled = int(y // downsample)
                    
                    if (0 <= x_scaled < spatial_map.shape[1] and 
                        0 <= y_scaled < spatial_map.shape[0]):
                        spatial_map[y_scaled, x_scaled, 0] = attention[i]
                
                # Smooth
                spatial_map[:, :, 0] = cv2.GaussianBlur(
                    spatial_map[:, :, 0], (5, 5), 0
                )
                
                # Plot
                axes[idx].imshow(slide_thumbnail)
                im = axes[idx].imshow(
                    spatial_map[:, :, 0],
                    alpha=0.4,
                    vmin=0,
                    vmax=1,
                    cmap=self.cmap_jet
                )
                
                axes[idx].set_title(f'{title}', fontsize=14)
                axes[idx].set_xticks([])
                axes[idx].set_yticks([])
                
                # Add colorbar
                cbar = plt.colorbar(im, ax=axes[idx], fraction=0.046, pad=0.04)
                cbar.set_label('Attention', rotation=270, labelpad=15)
            
            plt.suptitle(f'{slide_id} - {self.model_name} Multi-Scale Attention', fontsize=16)
            
            # Save
            if output_path is None:
                output_dir = f"heatmaps_multiscale_{self.dataset}_{self.backbone_model}"
                os.makedirs(output_dir, exist_ok=True)
                output_path = os.path.join(
                    output_dir, 
                    f"{slide_id}_{self.model_name}_multiscale.png"
                )
            
            plt.tight_layout()
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"Multi-scale heatmap saved: {output_path}")
            return output_path
            
        except Exception as e:
            print(f"Error creating multi-scale heatmap for {slide_id}: {e}")
            return None
    
    def batch_visualize(self, slide_ids: List[str] = None, 
                       output_dir: str = None,
                       scaling_method: str = "minmax",
                       create_multiscale: bool = False):
        """
        Create attention heatmaps for multiple slides
        
        Args:
            slide_ids: List of slide IDs to process (None for all)
            output_dir: Output directory
            scaling_method: Attention scaling method
            create_multiscale: Whether to create multi-scale heatmaps
        """
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
            output_dir = f"heatmaps_{self.dataset}_{self.backbone_model}_{self.model_name}"
        os.makedirs(output_dir, exist_ok=True)
        
        successful = 0
        failed = []
        
        # Process each slide
        for slide_id in tqdm(slide_ids, desc="Creating heatmaps"):
            try:
                # Standard heatmap
                output_path = os.path.join(
                    output_dir,
                    f"{slide_id}_{scaling_method}_attention.png"
                )
                
                result = self.create_attention_heatmap(
                    slide_id=slide_id,
                    output_path=output_path,
                    scaling_method=scaling_method
                )
                
                if result:
                    successful += 1
                    
                    # Multi-scale heatmap if requested
                    if create_multiscale:
                        multiscale_path = os.path.join(
                            output_dir,
                            f"{slide_id}_multiscale.png"
                        )
                        self.create_multi_scale_heatmap(slide_id, multiscale_path)
                else:
                    failed.append(slide_id)
                    
            except Exception as e:
                print(f"Failed to process {slide_id}: {e}")
                failed.append(slide_id)
        
        print(f"\nVisualization completed:")
        print(f"  Successful: {successful}")
        print(f"  Failed: {len(failed)}")
        
        if failed:
            print(f"  Failed slides: {failed[:5]}{'...' if len(failed) > 5 else ''}")


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Unified Visualization for STAD Analysis")
    
    # Model arguments
    parser.add_argument("--model", type=str, required=True,
                       choices=["CLAM_SB", "CLAM_MB", "TransMIL", "ACMIL", "DSMIL", 
                               "ABMIL", "GABMIL", "MeanMIL", "MaxMIL"],
                       help="MIL architecture name")
    
    parser.add_argument("--model_path", type=str, required=True,
                       help="Path to trained model checkpoint")
    
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
    
    parser.add_argument("--slide_ids", nargs="+", default=None,
                       help="Specific slide IDs to process")
    
    # Visualization arguments
    parser.add_argument("--output_dir", type=str, default=None,
                       help="Output directory for heatmaps")
    
    parser.add_argument("--scaling_method", type=str, default="minmax",
                       choices=["minmax", "zscore", "none"],
                       help="Attention scaling method")
    
    parser.add_argument("--alpha", type=float, default=0.4,
                       help="Transparency of attention overlay")
    
    parser.add_argument("--figsize", type=int, default=15,
                       help="Figure size for heatmaps")
    
    parser.add_argument("--create_multiscale", action="store_true",
                       help="Create multi-scale heatmaps")
    
    parser.add_argument("--patch_size", type=int, default=448,
                       help="Patch size used for feature extraction")
    
    parser.add_argument("--downsample", type=int, default=28,
                       help="Downsampling factor for visualization")
    
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
    
    # Initialize visualizer
    try:
        visualizer = AttentionVisualizer(
            model_name=args.model,
            dataset=args.dataset,
            backbone_model=args.backbone_model,
            device=device,
            model_path=args.model_path
        )
        
        # Create visualizations
        visualizer.batch_visualize(
            slide_ids=args.slide_ids,
            output_dir=args.output_dir,
            scaling_method=args.scaling_method,
            create_multiscale=args.create_multiscale
        )
        
        print("\n✓ Visualization completed!")
        
    except Exception as e:
        print(f"✗ Error during visualization: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())