#!/usr/bin/env python3
"""
Unified Feature Extraction Script for STAD Pathological Image Analysis

Supports multiple foundation models (DINO, UNI, GigaPath, Lunit) and datasets (JNUH, AJOU, TCGA, STFD)
"""

import os
import sys
import argparse
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image
import gc
import multiprocessing as mp
from typing import List, Dict, Tuple
import warnings
warnings.filterwarnings('ignore')

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import (
    PatchDataset, load_model_by_type, get_model_transforms, 
    get_feature_dim, save_features_and_coords, get_dataset_paths
)


class FeatureExtractor:
    """
    Unified feature extractor supporting multiple models and datasets
    """
    
    def __init__(self, model_type: str, dataset: str, device: torch.device,
                 patch_size: int = 448, stride: int = 448, 
                 batch_size: int = 32, num_workers: int = 4):
        """
        Initialize feature extractor
        
        Args:
            model_type: Foundation model type (DINO, UNI, GigaPath, Lunit)
            dataset: Dataset name (JNUH, AJOU, TCGA, STFD)
            device: Device for computation
            patch_size: Size of patches to extract
            stride: Stride between patches
            batch_size: Batch size for inference
            num_workers: Number of workers for data loading
        """
        self.model_type = model_type
        self.dataset = dataset
        self.device = device
        self.patch_size = patch_size
        self.stride = stride
        self.batch_size = batch_size
        self.num_workers = num_workers
        
        # Load model and transforms
        print(f"Loading {model_type} model...")
        self.model = load_model_by_type(model_type, device)
        self.transform = get_model_transforms(model_type)
        self.feature_dim = get_feature_dim(model_type)
        
        # Get dataset-specific paths
        self.dataset_config = get_dataset_paths(dataset)
        
        print(f"Model: {model_type}")
        print(f"Dataset: {dataset}")
        print(f"Feature dimension: {self.feature_dim}")
        print(f"Patch size: {patch_size}")
        print(f"Batch size: {batch_size}")
    
    def extract_slide_features(self, slide_path: str, slide_id: str) -> Tuple[np.ndarray, List]:
        """
        Extract features from a single slide
        
        Args:
            slide_path: Path to slide file
            slide_id: Slide identifier
            
        Returns:
            Tuple of (features, coordinates)
        """
        print(f"\nProcessing slide: {slide_id}")
        
        # Get dataset-specific downsample factor
        downsample_factors = {
            "JNUH": 2,
            "AJOU": 2, 
            "TCGA": 1,
            "STFD": 2
        }
        downsample = downsample_factors.get(self.dataset, 2)
        
        # Extract patches
        try:
            patch_dataset = PatchDataset(
                slide_path=slide_path,
                dataset=self.dataset,
                patch_size=self.patch_size,
                stride=self.stride,
                downsample=downsample
            )
            
            patches, coords = patch_dataset.patchify()
            print(f"Extracted {len(patches)} patches")
            
            if len(patches) == 0:
                print("No valid patches found!")
                return np.array([]), []
            
        except Exception as e:
            print(f"Error extracting patches: {e}")
            return np.array([]), []
        
        # Extract features in batches
        features = []
        self.model.eval()
        
        with torch.no_grad():
            for i in tqdm(range(0, len(patches), self.batch_size), desc="Extracting features"):
                batch_patches = patches[i:i + self.batch_size]
                
                # Prepare batch
                batch_tensors = []
                for patch in batch_patches:
                    patch_pil = Image.fromarray(patch)
                    patch_tensor = self.transform(patch_pil)
                    batch_tensors.append(patch_tensor)
                
                if batch_tensors:
                    batch = torch.stack(batch_tensors).to(self.device)
                    
                    # Extract features
                    try:
                        batch_features = self.model(batch)
                        
                        # Handle different output formats
                        if isinstance(batch_features, tuple):
                            batch_features = batch_features[0]
                        
                        # Ensure 2D output
                        if len(batch_features.shape) > 2:
                            batch_features = batch_features.mean(dim=1)
                        
                        features.append(batch_features.cpu().numpy())
                        
                    except Exception as e:
                        print(f"Error processing batch {i}: {e}")
                        continue
                
                # Clean up GPU memory
                if i % (self.batch_size * 10) == 0:
                    torch.cuda.empty_cache()
                    gc.collect()
        
        if features:
            features = np.vstack(features)
            print(f"Final feature shape: {features.shape}")
        else:
            features = np.array([])
        
        return features, coords
    
    def process_dataset(self, slide_ids: List[str] = None, start_idx: int = 0, end_idx: int = None):
        """
        Process entire dataset or subset
        
        Args:
            slide_ids: List of specific slide IDs to process
            start_idx: Starting index for processing
            end_idx: Ending index for processing
        """
        slide_dir = self.dataset_config["slide_dir"]
        output_dir = self.dataset_config["output_dir"]
        
        # Get all slide files
        slide_extensions = ['.svs', '.tif', '.tiff', '.ndpi', '.vms', '.vmu', '.scn']
        all_slides = []
        
        for ext in slide_extensions:
            pattern = os.path.join(slide_dir, f"*{ext}")
            slides = [f for f in os.listdir(slide_dir) if f.lower().endswith(ext.lower())]
            all_slides.extend([(os.path.join(slide_dir, f), f.split('.')[0]) for f in slides])
        
        if slide_ids:
            # Filter by specified slide IDs
            all_slides = [(path, sid) for path, sid in all_slides if sid in slide_ids]
        
        # Apply start/end indices
        if start_idx > 0 or end_idx is not None:
            end_idx = end_idx or len(all_slides)
            all_slides = all_slides[start_idx:end_idx]
        
        print(f"Found {len(all_slides)} slides to process")
        
        # Process each slide
        for i, (slide_path, slide_id) in enumerate(all_slides):
            try:
                # Check if already processed
                feature_dir = os.path.join(output_dir, f"Feature_{self.model_type}")
                feature_file = os.path.join(feature_dir, f"{slide_id}.pickle")
                
                if os.path.exists(feature_file) and not args.overwrite:
                    print(f"Skipping {slide_id} (already processed)")
                    continue
                
                # Extract features
                features, coords = self.extract_slide_features(slide_path, slide_id)
                
                if len(features) > 0:
                    # Save results
                    save_features_and_coords(features, coords, slide_id, output_dir, self.model_type)
                    print(f"✓ Completed {slide_id} ({i+1}/{len(all_slides)})")
                else:
                    print(f"✗ Failed {slide_id} - No features extracted")
                
            except Exception as e:
                print(f"✗ Error processing {slide_id}: {e}")
                continue
            
            # Memory cleanup
            if i % 5 == 0:
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Unified Feature Extraction for STAD Analysis")
    
    # Model arguments
    parser.add_argument("--model_type", type=str, required=True,
                       choices=["DINO", "UNI", "GigaPath", "Lunit"],
                       help="Foundation model type")
    
    # Dataset arguments  
    parser.add_argument("--dataset", type=str, required=True,
                       choices=["JNUH", "AJOU", "TCGA", "STFD"],
                       help="Dataset name")
    
    # Processing arguments
    parser.add_argument("--device", type=str, default="cuda:0",
                       help="Device for computation (e.g., cuda:0, cpu)")
    parser.add_argument("--batch_size", type=int, default=32,
                       help="Batch size for feature extraction")
    parser.add_argument("--num_workers", type=int, default=4,
                       help="Number of data loading workers")
    
    # Patch arguments
    parser.add_argument("--patch_size", type=int, default=448,
                       help="Size of patches to extract")
    parser.add_argument("--stride", type=int, default=448,
                       help="Stride between patches")
    
    # Processing control
    parser.add_argument("--slide_ids", nargs="+", default=None,
                       help="Specific slide IDs to process")
    parser.add_argument("--start_idx", type=int, default=0,
                       help="Starting index for batch processing")
    parser.add_argument("--end_idx", type=int, default=None,
                       help="Ending index for batch processing")
    parser.add_argument("--overwrite", action="store_true",
                       help="Overwrite existing features")
    
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
    
    # Initialize feature extractor
    try:
        extractor = FeatureExtractor(
            model_type=args.model_type,
            dataset=args.dataset,
            device=device,
            patch_size=args.patch_size,
            stride=args.stride,
            batch_size=args.batch_size,
            num_workers=args.num_workers
        )
        
        # Process dataset
        extractor.process_dataset(
            slide_ids=args.slide_ids,
            start_idx=args.start_idx,
            end_idx=args.end_idx
        )
        
        print("\n✓ Feature extraction completed!")
        
    except Exception as e:
        print(f"✗ Error during feature extraction: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())