"""
Utility functions for STAD pathological image analysis framework
"""
import os
import cv2
import numpy as np
import pandas as pd
import pickle
import torch
import torch.nn as nn
from PIL import Image
from tqdm import tqdm
import openslide
from typing import List, Tuple, Dict, Optional, Union


class PatchDataset:
    """
    Universal patch extraction class supporting different datasets and resolutions
    """
    
    def __init__(self, slide_path: str, dataset: str = "JNUH", 
                 patch_size: int = 448, stride: int = 448, 
                 downsample: int = 2, transform=None):
        """
        Initialize patch dataset with dataset-specific configurations
        
        Args:
            slide_path: Path to the slide file
            dataset: Dataset name (JNUH, AJOU, STFD, TCGA)
            patch_size: Size of patches to extract
            stride: Stride between patches
            downsample: Downsampling factor
            transform: Image transforms to apply
        """
        self.slide = openslide.open_slide(slide_path)
        self.dataset = dataset
        self.patch_size = patch_size
        self.stride = stride
        self.downsample = downsample
        self.transform = transform
        
        # Dataset-specific configurations
        self.dataset_configs = self._get_dataset_configs()
        self.config = self.dataset_configs.get(dataset, self.dataset_configs["JNUH"])
        
        self.Xmax = self.slide.dimensions[0]
        self.Ymax = self.slide.dimensions[1]
        self.magnification = self._get_magnification()
        
        print(f"Dataset: {dataset}")
        print(f"Slide dimensions: {self.slide.dimensions}")
        print(f"Magnification: {self.magnification}")
        print(f"Patch size: {patch_size}, Stride: {stride}, Downsample: {downsample}")
        
    def _get_dataset_configs(self) -> Dict:
        """Get dataset-specific configurations"""
        return {
            "JNUH": {
                "sat_thresh": 9,
                "area_thresh": 0.4,
                "magnification_key": "aperio.AppMag",
                "default_magnification": 40
            },
            "AJOU": {
                "sat_thresh": 7,
                "area_thresh": 0.5,
                "magnification_key": "aperio.AppMag",
                "default_magnification": 40
            },
            "TCGA": {
                "sat_thresh": 5,
                "area_thresh": 0.3,
                "magnification_key": "openslide.objective-power",
                "default_magnification": 40
            },
            "STFD": {
                "sat_thresh": 7,
                "area_thresh": 0.4,
                "magnification_key": "openslide.objective-power", 
                "default_magnification": 20
            }
        }
    
    def _get_magnification(self) -> float:
        """Extract magnification from slide properties"""
        mag_key = self.config["magnification_key"]
        default_mag = self.config["default_magnification"]
        
        try:
            return float(self.slide.properties.get(mag_key, default_mag))
        except (ValueError, TypeError):
            return default_mag
    
    def patchify(self) -> Tuple[List[np.ndarray], List[List[int]]]:
        """Extract patches from the slide"""
        patches, coords = [], []
        
        for ypos in tqdm(range(0, self.Ymax, self.stride), desc="Extracting patches"):
            # Read full width strip
            try:
                image = self.slide.read_region(
                    location=(0, ypos), 
                    level=0, 
                    size=(self.Xmax, self.patch_size)
                )
                
                # Resize according to downsample factor
                new_size = (
                    int(image.size[0] / self.downsample), 
                    int(self.patch_size / self.downsample)
                )
                image = image.resize(new_size)
                image = np.array(image)[:, :, :3]  # Remove alpha channel
                
                # Extract patches from the strip
                stride_downsampled = int(self.stride / self.downsample)
                patch_size_downsampled = int(self.patch_size / self.downsample)
                
                for xpos in range(0, image.shape[1], stride_downsampled):
                    patch = image[:, xpos:xpos + patch_size_downsampled, :]
                    
                    if self._is_valid_patch(patch):
                        patches.append(patch)
                        coords.append([xpos * self.downsample, ypos])
                        
            except Exception as e:
                print(f"Error processing position ({0}, {ypos}): {e}")
                continue
                
        return patches, coords
    
    def _is_valid_patch(self, patch: np.ndarray) -> bool:
        """Check if patch is valid (not white, black, or wrong size)"""
        if self._is_wrong_size_patch(patch):
            return False
        if self._is_white_patch(patch):
            return False
        if self._is_black_patch(patch):
            return False
        return True
    
    def _is_white_patch(self, patch: np.ndarray) -> bool:
        """Check if patch is predominantly white"""
        sat_thresh = self.config["sat_thresh"]
        patch_hsv = cv2.cvtColor(patch, cv2.COLOR_RGB2HSV)
        return np.mean(patch_hsv[:, :, 1]) < sat_thresh
    
    def _is_black_patch(self, patch: np.ndarray) -> bool:
        """Check if patch is predominantly black"""
        area_thresh = self.config["area_thresh"]
        return np.mean(patch) < area_thresh * 255
    
    def _is_wrong_size_patch(self, patch: np.ndarray) -> bool:
        """Check if patch has wrong dimensions"""
        expected_size = int(self.patch_size / self.downsample)
        return patch.shape[0] != expected_size or patch.shape[1] != expected_size


def load_model_by_type(model_type: str, device: torch.device) -> torch.nn.Module:
    """
    Load different foundation models for feature extraction
    
    Args:
        model_type: One of ["DINO", "UNI", "GigaPath", "Lunit"]
        device: Device to load model on
        
    Returns:
        Loaded model
    """
    if model_type == "DINO":
        return load_dino_model(device)
    elif model_type == "UNI":
        return load_uni_model(device)
    elif model_type == "GigaPath":
        return load_gigapath_model(device)
    elif model_type == "Lunit":
        return load_lunit_model(device)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def load_dino_model(device: torch.device) -> torch.nn.Module:
    """Load DINO model"""
    import timm
    from timm.models.vision_transformer import VisionTransformer
    
    def get_pretrained_url(key):
        URL_PREFIX = "https://github.com/lunit-io/benchmark-ssl-pathology/releases/download/pretrained-weights"
        model_zoo_registry = {
            "DINO_p16": "dino_vit_small_patch16_ep200.torch",
            "DINO_p8": "dino_vit_small_patch8_ep200.torch",
        }
        return f"{URL_PREFIX}/{model_zoo_registry.get(key)}"
    
    model = VisionTransformer(
        img_size=224, patch_size=16, embed_dim=384, 
        num_heads=6, num_classes=0
    )
    
    pretrained_url = get_pretrained_url("DINO_p16")
    state_dict = torch.hub.load_state_dict_from_url(pretrained_url, progress=True)
    model.load_state_dict(state_dict)
    
    return model.to(device).eval()


def load_uni_model(device: torch.device) -> torch.nn.Module:
    """Load UNI model"""
    import timm
    
    model = timm.create_model(
        "vit_large_patch16_224", 
        img_size=224, 
        patch_size=16, 
        init_values=1e-5, 
        num_classes=0, 
        dynamic_img_size=True
    )
    
    # Load pre-trained weights
    # Note: You'll need to adjust the path to your UNI weights
    weights_path = "../../../../../home/sosal/.cache/huggingface/hub/models--MahmoodLab--UNI/blobs/56ef09b44a25dc5c7eedc55551b3d47bcd17659a7a33837cf9abc9ec4e2ffb40"
    if os.path.exists(weights_path):
        model.load_state_dict(torch.load(weights_path, map_location="cpu"), strict=True)
    
    return model.to(device).eval()


def load_gigapath_model(device: torch.device) -> torch.nn.Module:
    """Load GigaPath model"""
    # Placeholder for GigaPath model loading
    # You'll need to implement this based on GigaPath's API
    raise NotImplementedError("GigaPath model loading not implemented yet")


def load_lunit_model(device: torch.device) -> torch.nn.Module:
    """Load Lunit model"""
    # Placeholder for Lunit model loading
    # You'll need to implement this based on Lunit's API
    raise NotImplementedError("Lunit model loading not implemented yet")


def get_model_transforms(model_type: str):
    """Get appropriate transforms for different models"""
    from torchvision import transforms
    
    if model_type in ["DINO", "UNI"]:
        return transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        ])
    elif model_type == "GigaPath":
        # GigaPath specific transforms
        return transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        ])
    elif model_type == "Lunit":
        # Lunit specific transforms
        return transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        ])
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def get_feature_dim(model_type: str) -> int:
    """Get feature dimension for different models"""
    feature_dims = {
        "DINO": 384,
        "UNI": 1024,
        "GigaPath": 1536,  # Placeholder
        "Lunit": 512       # Placeholder
    }
    return feature_dims.get(model_type, 512)


def save_features_and_coords(features: np.ndarray, coords: List[List[int]], 
                           slide_id: str, output_dir: str, model_type: str):
    """Save extracted features and coordinates"""
    os.makedirs(output_dir, exist_ok=True)
    
    feature_dir = os.path.join(output_dir, f"Feature_{model_type}")
    coord_dir = os.path.join(output_dir, f"Coord_{model_type}")
    
    os.makedirs(feature_dir, exist_ok=True)
    os.makedirs(coord_dir, exist_ok=True)
    
    # Save features
    feature_path = os.path.join(feature_dir, f"{slide_id}.pickle")
    with open(feature_path, 'wb') as f:
        pickle.dump(features, f)
    
    # Save coordinates
    coord_path = os.path.join(coord_dir, f"{slide_id}.pickle")
    with open(coord_path, 'wb') as f:
        pickle.dump(coords, f)
    
    print(f"Saved features: {feature_path}")
    print(f"Saved coordinates: {coord_path}")


def load_data(indices: List[int], filenames: List[str], labels: List[int],
              feature_path: str, coord_path: str) -> Tuple[List, List, List, List]:
    """Load features, coordinates, and metadata"""
    features, loaded_labels, coords, barcodes = [], [], [], []
    
    for idx in indices:
        filename = filenames[idx]
        label = labels[idx]
        
        # Load features
        feature_file = os.path.join(feature_path, f"{filename}.pickle")
        if os.path.exists(feature_file):
            with open(feature_file, 'rb') as f:
                feature = pickle.load(f)
                features.append(feature)
        
        # Load coordinates
        coord_file = os.path.join(coord_path, f"{filename}.pickle")
        if os.path.exists(coord_file):
            with open(coord_file, 'rb') as f:
                coord = pickle.load(f)
                coords.append(coord)
        
        loaded_labels.append(label)
        barcodes.append(filename)
    
    return features, loaded_labels, coords, barcodes


def get_dataset_paths(dataset: str) -> Dict[str, str]:
    """Get dataset-specific file paths"""
    dataset_paths = {
        "JNUH": {
            "slide_dir": "/mnt/e/JNUH/STAD_WSI/Gastric_Ca_ImmunoTx/",
            "clinical_file": "STAD.tsv",
            "output_dir": "Features_STAD_JNUH"
        },
        "AJOU": {
            "slide_dir": "/mnt/f/AJOU/STAD_WSI/",
            "clinical_file": "AJOU.csv", 
            "output_dir": "Features_STAD_AJOU"
        },
        "TCGA": {
            "slide_dir": "/mnt/e/TCGA/STAD/svs/",
            "clinical_file": "STAD.tsv",
            "output_dir": "Features_TCGA"
        },
        "STFD": {
            "slide_dir": "/mnt/g/Stanford/STAD/",
            "clinical_file": "SFTD.xlsx",
            "output_dir": "Features_STFD"
        }
    }
    
    return dataset_paths.get(dataset, dataset_paths["JNUH"])


class Struct:
    """Convert dictionary to object with attribute access"""
    def __init__(self, **entries):
        self.__dict__.update(entries)


def minMax(x: np.ndarray) -> np.ndarray:
    """Min-max normalization"""
    return (x - x.min()) / (x.max() - x.min() + 1e-8)


def z_score_normalize(x: np.ndarray) -> np.ndarray:
    """Z-score normalization"""
    return (x - x.mean()) / (x.std() + 1e-8)