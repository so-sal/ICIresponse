#!/usr/bin/env python3
"""
Unified MIL Training Script for STAD Pathological Image Analysis

Supports multiple MIL architectures and datasets with flexible configuration
"""

import os
import sys
import argparse
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.optim import lr_scheduler
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import GroupKFold, StratifiedKFold
from sklearn.metrics import roc_auc_score, accuracy_score
import warnings
warnings.filterwarnings('ignore')

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import load_data, get_dataset_paths, Struct


class PatchFeatureGenerator(Dataset):
    """Dataset class for MIL training"""
    
    def __init__(self, features, labels, filenames=None, randomPatchSelection=False, 
                 min_ratio=0.8, max_ratio=1.0):
        self.features = features
        self.labels = labels
        self.filenames = filenames if filenames is not None else list(range(len(features)))
        self.randomPatchSelection = randomPatchSelection
        self.min_ratio = min_ratio
        self.max_ratio = max_ratio
    
    def __len__(self):
        return len(self.features)
    
    def select_random_patches(self, features, min_ratio=0.8, max_ratio=1.0):
        """Randomly select a subset of patches"""
        num_patches = features.shape[0]
        random_ratio = np.random.uniform(min_ratio, max_ratio)
        selected_patches = np.random.choice(
            num_patches, int(num_patches * random_ratio), replace=False
        )
        return features[selected_patches]
    
    def __getitem__(self, idx):
        if self.randomPatchSelection:
            feature = self.select_random_patches(self.features[idx])
            feature = torch.tensor(feature, dtype=torch.float32)
        else:
            feature = torch.tensor(self.features[idx], dtype=torch.float32)
        
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return feature, label


def load_mil_model(model_name: str, config: Struct) -> nn.Module:
    """
    Load MIL model by name
    
    Args:
        model_name: Name of the MIL model
        config: Configuration object
        
    Returns:
        Initialized MIL model
    """
    sys.path.append('..')
    
    if model_name == "CLAM_SB":
        from architecture.clam import CLAM_SB
        return CLAM_SB(config)
    
    elif model_name == "CLAM_MB":
        from architecture.clam import CLAM_MB
        return CLAM_MB(config)
    
    elif model_name == "TransMIL":
        from architecture.transMIL import TransMIL
        return TransMIL(config)
    
    elif model_name == "ACMIL":
        from architecture.acmil import AttnMIL6
        return AttnMIL6(config)
    
    elif model_name == "DSMIL":
        from architecture.dsmil import MILNet, FCLayer, BClassifier
        i_classifier = FCLayer(config.D_feat, config.n_class)
        b_classifier = BClassifier(config, nonlinear=False)
        return MILNet(i_classifier, b_classifier)
    
    elif model_name == "ABMIL":
        from architecture.abmil import ABMIL
        return ABMIL(config)
    
    elif model_name == "GABMIL":
        from architecture.abmil import GatedABMIL
        return GatedABMIL(config)
    
    elif model_name == "MeanMIL":
        from modules.mean_max import MeanMIL
        return MeanMIL(config)
    
    elif model_name == "MaxMIL":
        from modules.mean_max import MaxMIL
        return MaxMIL(config)
    
    else:
        raise ValueError(f"Unknown model: {model_name}")


def loss_acmil(pred, target, config, criterion, device):
    """ACMIL specific loss function"""
    bag_loss = criterion(pred['bag_logits'], target)
    
    # Instance loss if available
    inst_loss = 0
    if 'inst_logits' in pred:
        inst_labels = torch.zeros_like(pred['inst_logits']).to(device)
        # Set positive instances for positive bags
        for i, label in enumerate(target):
            if label == 1:
                # Use attention weights to determine positive instances
                if 'attention' in pred:
                    attention = pred['attention'][i]
                    top_k = int(0.1 * attention.shape[0])  # Top 10% instances
                    _, top_indices = torch.topk(attention.squeeze(), top_k)
                    inst_labels[i][top_indices] = 1
        
        inst_loss = criterion(pred['inst_logits'].view(-1, config.n_class), 
                            inst_labels.view(-1).long())
    
    total_loss = bag_loss + 0.3 * inst_loss
    return total_loss


def evaluate_model(model, data_loader, criterion, device, model_name=""):
    """Evaluate model on given dataset"""
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for x, y in data_loader:
            x, y = x.to(device), y.to(device)
            
            pred = model(x)
            
            # Handle different output formats
            if isinstance(pred, dict):
                logits = pred.get('bag_logits', pred.get('logits', pred))
            else:
                logits = pred
            
            # Calculate loss
            if model_name == 'ACMIL' and isinstance(pred, dict):
                loss = loss_acmil(pred, y, None, criterion, device)
            else:
                loss = criterion(logits, y)
            
            total_loss += loss.item()
            
            # Store predictions and labels
            if len(logits.shape) > 1:
                probs = torch.softmax(logits, dim=1)
                preds = probs[:, 1] if probs.shape[1] == 2 else probs.max(1)[0]
            else:
                preds = torch.sigmoid(logits)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())
    
    avg_loss = total_loss / len(data_loader)
    
    # Calculate AUROC
    try:
        auroc = roc_auc_score(all_labels, all_preds)
    except:
        auroc = 0.5
    
    return avg_loss, auroc


def nested_cross_validation(filenames, labels, filenames_id, feature_path, coord_path, 
                          fold_idx, n_splits=5, get_filename=False):
    """Perform nested cross-validation split"""
    # Group-based split to ensure same patient's slides stay together
    group_kfold = GroupKFold(n_splits=n_splits)
    splits = list(group_kfold.split(filenames, labels, groups=filenames_id))
    
    train_idx, test_idx = splits[fold_idx]
    
    # Further split training set for validation
    train_filenames = [filenames[i] for i in train_idx]
    train_labels = [labels[i] for i in train_idx]
    train_ids = [filenames_id[i] for i in train_idx]
    
    # Validation split
    val_group_kfold = GroupKFold(n_splits=4)  # 75% train, 25% validation
    val_splits = list(val_group_kfold.split(train_filenames, train_labels, groups=train_ids))
    train_train_idx, val_idx = val_splits[0]
    
    # Final indices
    final_train_idx = [train_idx[i] for i in train_train_idx]
    final_val_idx = [train_idx[i] for i in val_idx]
    
    # Load data
    train_features, train_labels, _, train_files = load_data(
        final_train_idx, filenames, labels, feature_path, coord_path
    )
    val_features, val_labels, _, val_files = load_data(
        final_val_idx, filenames, labels, feature_path, coord_path
    )
    test_features, test_labels, _, test_files = load_data(
        test_idx, filenames, labels, feature_path, coord_path
    )
    
    # Create datasets
    train_dataset = PatchFeatureGenerator(
        train_features, train_labels, 
        train_files if get_filename else None, 
        randomPatchSelection=True
    )
    val_dataset = PatchFeatureGenerator(val_features, val_labels)
    test_dataset = PatchFeatureGenerator(test_features, test_labels)
    
    return train_dataset, val_dataset, test_dataset, {}


class MILTrainer:
    """
    Unified MIL trainer supporting multiple architectures and datasets
    """
    
    def __init__(self, model_name: str, dataset: str, backbone_model: str, 
                 config: Struct, device: torch.device):
        """
        Initialize MIL trainer
        
        Args:
            model_name: MIL architecture name
            dataset: Dataset name
            backbone_model: Feature extraction backbone
            config: Training configuration
            device: Device for training
        """
        self.model_name = model_name
        self.dataset = dataset
        self.backbone_model = backbone_model
        self.config = config
        self.device = device
        
        # Load model
        self.model = load_mil_model(model_name, config)
        
        # Dataset configuration
        self.dataset_config = get_dataset_paths(dataset)
        
        print(f"Model: {model_name}")
        print(f"Dataset: {dataset}")
        print(f"Backbone: {backbone_model}")
    
    def load_dataset(self, fold_idx: int = 0):
        """Load and prepare dataset"""
        # Paths
        base_feature_path = self.dataset_config["output_dir"]
        coord_path = os.path.join(base_feature_path, f"Coord_{self.backbone_model}")
        feature_path = os.path.join(base_feature_path, f"Feature_{self.backbone_model}")
        
        # Clinical data
        clinical_file = self.dataset_config["clinical_file"]
        
        if self.dataset == "AJOU":
            clinical_data = pd.read_csv(clinical_file, sep="\t")
            clinical_data.set_index('연구ID', inplace=True)
            response_column = 'Best_of_Response'
            positive_responses = ["CR", "PR"]
        elif self.dataset == "TCGA":
            clinical_data = pd.read_csv(clinical_file, sep="\t", index_col='ID')
            response_column = 'Response'
            positive_responses = ["R"]
        elif self.dataset == "STFD":
            clinical_data = pd.read_excel(clinical_file, index_col='ID')
            response_column = 'Response'
            positive_responses = ["R", "CR", "PR"]
        else:  # JNUH
            clinical_data = pd.read_csv(clinical_file, sep="\t", index_col='ID')
            response_column = 'Response'
            positive_responses = ["R"]
        
        # Get filenames and labels
        filenames = [f.split('.pickle')[0] for f in sorted(os.listdir(coord_path))]
        
        if self.dataset == "AJOU":
            filenames, filenames_id = zip(*[
                (f, f.split('.svs')[0]) for f in filenames 
                if f.split('.svs')[0] in clinical_data.index.tolist()
            ])
        else:
            filenames, filenames_id = zip(*[
                (f, f.split('_')[0]) for f in filenames 
                if f.split('_')[0] in clinical_data.index
            ])
        
        # Labels
        labels = [
            int(clinical_data.loc[fid, response_column] in positive_responses) 
            for fid in filenames_id
        ]
        
        print(f"Loaded {len(filenames)} slides")
        print(f"Positive cases: {sum(labels)}, Negative cases: {len(labels) - sum(labels)}")
        
        # Cross-validation split
        train_dataset, val_dataset, test_dataset, _ = nested_cross_validation(
            list(filenames), labels, list(filenames_id), 
            feature_path, coord_path, fold_idx
        )
        
        return train_dataset, val_dataset, test_dataset
    
    def train_fold(self, fold_idx: int, epochs: int = 50, batch_size: int = 1, 
                   learning_rate: float = 1e-4, save_dir: str = "checkpoints"):
        """Train model for one fold"""
        print(f"\n=== Training Fold {fold_idx} ===")
        
        # Load data
        train_dataset, val_dataset, test_dataset = self.load_dataset(fold_idx)
        
        # Data loaders
        train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=0)
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)
        
        # Model setup
        self.model.to(self.device)
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=self.config.wd)
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-6)
        criterion = nn.CrossEntropyLoss()
        
        # Training variables
        best_val_loss = float('inf')
        best_model_path = None
        
        # Create save directory
        os.makedirs(save_dir, exist_ok=True)
        
        # Training loop
        for epoch in range(epochs):
            # Training phase
            self.model.train()
            train_loss = 0
            batch_counter = 0
            loss_batch = 0
            
            for x, y in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
                x, y = x.to(self.device), y.to(self.device)
                
                pred = self.model(x)
                
                # Calculate loss
                if self.model_name == 'ACMIL':
                    loss = loss_acmil(pred, y, self.config, criterion, self.device)
                else:
                    if isinstance(pred, dict):
                        pred = pred.get('bag_logits', pred.get('logits', pred))
                    loss = criterion(pred, y)
                
                loss_batch += loss
                train_loss += loss.item()
                batch_counter += 1
                
                # Batch optimization
                if batch_counter == batch_size:
                    loss_batch.backward()
                    optimizer.step()
                    optimizer.zero_grad()
                    loss_batch = 0
                    batch_counter = 0
            
            # Handle remaining batch
            if batch_counter != 0:
                loss_batch.backward()
                optimizer.step()
                optimizer.zero_grad()
            
            train_loss /= len(train_loader)
            
            # Validation phase
            self.model.eval()
            val_loss, val_auroc = evaluate_model(self.model, val_loader, criterion, self.device, self.model_name)
            test_loss, test_auroc = evaluate_model(self.model, test_loader, criterion, self.device, self.model_name)
            
            # Learning rate scheduling
            scheduler.step(val_loss)
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                
                # Remove previous best model
                if best_model_path and os.path.exists(best_model_path):
                    os.remove(best_model_path)
                
                # Save new best model
                best_model_path = os.path.join(
                    save_dir, 
                    f"{self.model_name}_fold{fold_idx}_epoch{epoch+1:02d}_"
                    f"loss{val_loss:.3f}_auroc{val_auroc:.3f}.pth"
                )
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': val_loss,
                    'val_auroc': val_auroc,
                    'test_auroc': test_auroc,
                    'config': self.config.__dict__
                }, best_model_path)
            
            # Print progress
            print(f"Epoch {epoch+1:02d}: Train Loss: {train_loss:.4f}, "
                  f"Val Loss: {val_loss:.4f}, Val AUROC: {val_auroc:.4f}, "
                  f"Test AUROC: {test_auroc:.4f}")
        
        print(f"Best model saved: {best_model_path}")
        return best_model_path, best_val_loss, val_auroc


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Unified MIL Training for STAD Analysis")
    
    # Model arguments
    parser.add_argument("--model", type=str, required=True,
                       choices=["CLAM_SB", "CLAM_MB", "TransMIL", "ACMIL", "DSMIL", 
                               "ABMIL", "GABMIL", "MeanMIL", "MaxMIL"],
                       help="MIL architecture name")
    
    # Dataset arguments
    parser.add_argument("--dataset", type=str, required=True,
                       choices=["JNUH", "AJOU", "TCGA", "STFD"],
                       help="Dataset name")
    
    parser.add_argument("--backbone_model", type=str, default="UNI",
                       choices=["DINO", "UNI", "GigaPath", "Lunit"],
                       help="Feature extraction backbone")
    
    # Training arguments
    parser.add_argument("--device", type=str, default="cuda:0",
                       help="Device for training")
    parser.add_argument("--epochs", type=int, default=50,
                       help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=1,
                       help="Batch size for training")
    parser.add_argument("--learning_rate", type=float, default=1e-4,
                       help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-5,
                       help="Weight decay")
    
    # Cross-validation arguments
    parser.add_argument("--fold", type=int, default=0,
                       help="Cross-validation fold index")
    parser.add_argument("--n_folds", type=int, default=5,
                       help="Number of cross-validation folds")
    parser.add_argument("--run_all_folds", action="store_true",
                       help="Run all folds sequentially")
    
    # Model-specific arguments
    parser.add_argument("--dropout", type=float, default=0.25,
                       help="Dropout rate")
    parser.add_argument("--n_classes", type=int, default=2,
                       help="Number of classes")
    
    # Output arguments
    parser.add_argument("--save_dir", type=str, default="checkpoints",
                       help="Directory to save models")
    parser.add_argument("--config", type=str, default=None,
                       help="Path to configuration file")
    
    return parser.parse_args()


def create_config(args):
    """Create configuration object"""
    config_dict = {
        'n_class': args.n_classes,
        'dropout': args.dropout,
        'wd': args.weight_decay,
        'feat_d': 1024,  # Default feature dimension
        'D_feat': 1024,  # For TransMIL
        'bag_loss': 'ce',
        'inst_loss': 'ce',
        'B': 8,  # CLAM attention branches
        'n_token': 5,
        'n_masked_patch': 10,
        'mask_drop': 0.6
    }
    
    # Load from config file if provided
    if args.config and os.path.exists(args.config):
        with open(args.config, 'r') as f:
            file_config = yaml.load(f, Loader=yaml.FullLoader)
            config_dict.update(file_config)
    
    return Struct(**config_dict)


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
    
    # Create configuration
    config = create_config(args)
    
    # Initialize trainer
    trainer = MILTrainer(
        model_name=args.model,
        dataset=args.dataset,
        backbone_model=args.backbone_model,
        config=config,
        device=device
    )
    
    try:
        if args.run_all_folds:
            # Train all folds
            results = []
            for fold in range(args.n_folds):
                print(f"\n{'='*50}")
                print(f"Training Fold {fold}/{args.n_folds}")
                print(f"{'='*50}")
                
                model_path, val_loss, val_auroc = trainer.train_fold(
                    fold_idx=fold,
                    epochs=args.epochs,
                    batch_size=args.batch_size,
                    learning_rate=args.learning_rate,
                    save_dir=args.save_dir
                )
                
                results.append({
                    'fold': fold,
                    'model_path': model_path,
                    'val_loss': val_loss,
                    'val_auroc': val_auroc
                })
            
            # Summary
            mean_auroc = np.mean([r['val_auroc'] for r in results])
            std_auroc = np.std([r['val_auroc'] for r in results])
            
            print(f"\n{'='*50}")
            print(f"Cross-Validation Results")
            print(f"{'='*50}")
            print(f"Mean AUROC: {mean_auroc:.4f} ± {std_auroc:.4f}")
            
            for result in results:
                print(f"Fold {result['fold']}: AUROC = {result['val_auroc']:.4f}")
        
        else:
            # Train single fold
            model_path, val_loss, val_auroc = trainer.train_fold(
                fold_idx=args.fold,
                epochs=args.epochs,
                batch_size=args.batch_size,
                learning_rate=args.learning_rate,
                save_dir=args.save_dir
            )
            
            print(f"\nTraining completed!")
            print(f"Best model: {model_path}")
            print(f"Validation AUROC: {val_auroc:.4f}")
    
    except Exception as e:
        print(f"Error during training: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())