"""
Enhanced Training Script for Multi-Dataset Document Detection
Supports multiple datasets, advanced augmentation, and multi-stage training
"""

import os
import json
import yaml
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
import wandb
from tqdm import tqdm
import pandas as pd
from sklearn.model_selection import train_test_split
import random
from ultralytics import YOLO
import timm
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.cuda.amp import autocast, GradScaler


class MultiDatasetConfig:
    """Configuration for multiple datasets"""
    
    DATASET_CONFIGS = {
        'publaynet': {
            'path': 'data/publaynet',
            'annotations': 'annotations.json',
            'classes': ['text', 'title', 'list', 'table', 'figure'],
            'weight': 1.0
        },
        'docbank': {
            'path': 'data/docbank',
            'annotations': 'annotations.json',
            'classes': ['abstract', 'author', 'caption', 'date', 'equation', 
                       'figure', 'footer', 'list', 'paragraph', 'reference', 
                       'section', 'table', 'title'],
            'weight': 1.2
        },
        'pubtables': {
            'path': 'data/pubtables',
            'annotations': 'annotations.json',
            'classes': ['table', 'table_column', 'table_row', 'table_cell'],
            'weight': 0.8
        },
        'musescore': {
            'path': 'data/musescore',
            'annotations': 'annotations.json',
            'classes': ['staff', 'measure', 'note', 'clef', 'time_signature', 'lyrics'],
            'weight': 1.5
        },
        'funsd': {
            'path': 'data/funsd',
            'annotations': 'annotations.json',
            'classes': ['header', 'question', 'answer', 'other'],
            'weight': 1.0
        }
    }


class DocumentAugmentation:
    """Advanced augmentation pipeline for document images"""
    
    @staticmethod
    def get_train_transforms(image_size: int = 640):
        return A.Compose([
            # Geometric transforms
            A.RandomRotate90(p=0.5),
            A.ShiftScaleRotate(
                shift_limit=0.05, 
                scale_limit=0.1, 
                rotate_limit=5, 
                p=0.5
            ),
            
            # Perspective and affine
            A.OneOf([
                A.Perspective(scale=(0.02, 0.05), p=0.5),
                A.Affine(
                    scale=(0.9, 1.1),
                    translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)},
                    rotate=(-5, 5),
                    shear=(-5, 5),
                    p=0.5
                ),
            ], p=0.5),
            
            # Document-specific augmentations
            A.OneOf([
                # Simulate scanning artifacts
                A.RandomBrightnessContrast(
                    brightness_limit=0.2, 
                    contrast_limit=0.2, 
                    p=0.5
                ),
                # Simulate photocopy
                A.RandomGamma(gamma_limit=(80, 120), p=0.5),
                # Simulate old document
                A.HueSaturationValue(
                    hue_shift_limit=10, 
                    sat_shift_limit=20, 
                    val_shift_limit=20, 
                    p=0.5
                ),
            ], p=0.7),
            
            # Noise and blur
            A.OneOf([
                A.GaussNoise(var_limit=(10.0, 50.0), p=0.5),
                A.ISONoise(p=0.5),
                A.MultiplicativeNoise(multiplier=(0.9, 1.1), p=0.5),
            ], p=0.3),
            
            A.OneOf([
                A.MotionBlur(blur_limit=3, p=0.5),
                A.MedianBlur(blur_limit=3, p=0.5),
                A.GaussianBlur(blur_limit=3, p=0.5),
            ], p=0.3),
            
            # Compression artifacts
            A.ImageCompression(quality_lower=70, quality_upper=100, p=0.3),
            
            # Grid distortion for document warping
            A.GridDistortion(p=0.3),
            
            # Cutout/CoarseDropout for robustness
            A.CoarseDropout(
                max_holes=5, 
                max_height=50, 
                max_width=50, 
                p=0.3
            ),
            
            # Resize
            A.Resize(image_size, image_size),
            
            # Normalize
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
            ToTensorV2()
        ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels']))
    
    @staticmethod
    def get_val_transforms(image_size: int = 640):
        return A.Compose([
            A.Resize(image_size, image_size),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
            ToTensorV2()
        ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels']))


class UnifiedDocumentDataset(Dataset):
    """Dataset that can load from multiple document datasets"""
    
    def __init__(self, 
                 dataset_name: str,
                 dataset_config: Dict,
                 split: str = 'train',
                 transform=None,
                 class_mapping: Optional[Dict] = None):
        
        self.dataset_name = dataset_name
        self.config = dataset_config
        self.split = split
        self.transform = transform
        self.class_mapping = class_mapping or {}
        
        # Load annotations
        self.annotations = self._load_annotations()
        
        # Dataset weight for balanced sampling
        self.weight = dataset_config.get('weight', 1.0)
        
    def _load_annotations(self) -> List[Dict]:
        """Load and parse annotations based on dataset format"""
        ann_path = Path(self.config['path']) / self.config['annotations']
        
        with open(ann_path, 'r') as f:
            data = json.load(f)
        
        # Handle different annotation formats
        if self.dataset_name == 'publaynet':
            return self._parse_coco_annotations(data)
        elif self.dataset_name == 'docbank':
            return self._parse_docbank_annotations(data)
        elif self.dataset_name == 'musescore':
            return self._parse_music_annotations(data)
        # Add more parsers as needed
        
        return data
    
    def _parse_coco_annotations(self, coco_data: Dict) -> List[Dict]:
        """Parse COCO format annotations"""
        images = {img['id']: img for img in coco_data['images']}
        categories = {cat['id']: cat['name'] for cat in coco_data['categories']}
        
        # Group annotations by image
        img_annotations = {}
        for ann in coco_data['annotations']:
            img_id = ann['image_id']
            if img_id not in img_annotations:
                img_annotations[img_id] = []
            
            img_annotations[img_id].append({
                'bbox': ann['bbox'],  # COCO format: [x, y, w, h]
                'category': categories[ann['category_id']]
            })
        
        # Create dataset entries
        annotations = []
        for img_id, anns in img_annotations.items():
            img_info = images[img_id]
            annotations.append({
                'image_path': Path(self.config['path']) / 'images' / img_info['file_name'],
                'width': img_info['width'],
                'height': img_info['height'],
                'annotations': anns
            })
        
        return annotations
    
    def _parse_docbank_annotations(self, data: Dict) -> List[Dict]:
        """Parse DocBank format annotations"""
        # Implementation specific to DocBank format
        pass
    
    def _parse_music_annotations(self, data: Dict) -> List[Dict]:
        """Parse music score annotations"""
        # Implementation specific to music score format
        pass
    
    def __len__(self):
        return len(self.annotations)
    
    def __getitem__(self, idx):
        ann = self.annotations[idx]
        
        # Load image
        image = cv2.imread(str(ann['image_path']))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Prepare bboxes and labels
        bboxes = []
        class_labels = []
        
        for obj in ann['annotations']:
            bbox = obj['bbox']
            
            # Convert COCO to Pascal VOC format if needed
            if len(bbox) == 4 and bbox[2] > 1 and bbox[3] > 1:  # Likely COCO format
                x, y, w, h = bbox
                bbox = [x, y, x + w, y + h]
            
            bboxes.append(bbox)
            
            # Map class name to unified class ID
            class_name = obj['category']
            class_id = self.class_mapping.get(class_name, -1)
            class_labels.append(class_id)
        
        # Apply transforms
        if self.transform:
            transformed = self.transform(
                image=image,
                bboxes=bboxes,
                class_labels=class_labels
            )
            
            image = transformed['image']
            bboxes = transformed['bboxes']
            class_labels = transformed['class_labels']
        
        # Convert to tensors
        if bboxes:
            bboxes = torch.tensor(bboxes, dtype=torch.float32)
            class_labels = torch.tensor(class_labels, dtype=torch.long)
        else:
            bboxes = torch.zeros((0, 4), dtype=torch.float32)
            class_labels = torch.zeros((0,), dtype=torch.long)
        
        return {
            'image': image,
            'bboxes': bboxes,
            'labels': class_labels,
            'image_id': idx,
            'dataset': self.dataset_name
        }


class MultiStageTrainer:
    """Trainer for multi-stage training strategy"""
    
    def __init__(self, config_path: str):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.scaler = GradScaler()
        
        # Initialize wandb
        if self.config.get('use_wandb', True):
            wandb.init(
                project=self.config.get('wandb_project', 'document-detection'),
                config=self.config
            )
        
        # Create unified class mapping
        self.class_mapping = self._create_class_mapping()
        
        # Initialize datasets
        self.train_datasets = self._init_datasets('train')
        self.val_datasets = self._init_datasets('val')
        
        # Create data loaders
        self.train_loader = self._create_dataloader(self.train_datasets, 'train')
        self.val_loader = self._create_dataloader(self.val_datasets, 'val')
        
        # Initialize model
        self.model = self._init_model()
        
        # Initialize optimizer and scheduler
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=self.config['training']['learning_rate'],
            weight_decay=self.config['training']['weight_decay']
        )
        
        self.scheduler = CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=self.config['training']['scheduler_T0'],
            T_mult=2,
            eta_min=1e-6
        )
        
        # Loss functions
        self.criterion = self._init_loss_functions()
        
    def _create_class_mapping(self) -> Dict[str, int]:
        """Create unified class mapping across all datasets"""
        all_classes = set()
        
        for dataset_config in MultiDatasetConfig.DATASET_CONFIGS.values():
            all_classes.update(dataset_config['classes'])
        
        # Add any additional classes from EnhancedClasses
        from enhanced_content_detector import EnhancedClasses
        all_classes.update(EnhancedClasses.CLASSES.keys())
        
        # Create mapping
        class_mapping = {cls: idx for idx, cls in enumerate(sorted(all_classes))}
        
        return class_mapping
    
    def _init_datasets(self, split: str) -> List[UnifiedDocumentDataset]:
        """Initialize all datasets for given split"""
        datasets = []
        
        transform = (DocumentAugmentation.get_train_transforms(self.config['image_size']) 
                    if split == 'train' 
                    else DocumentAugmentation.get_val_transforms(self.config['image_size']))
        
        for dataset_name, dataset_config in MultiDatasetConfig.DATASET_CONFIGS.items():
            if self.config.get('datasets', {}).get(dataset_name, {}).get('use', True):
                try:
                    dataset = UnifiedDocumentDataset(
                        dataset_name=dataset_name,
                        dataset_config=dataset_config,
                        split=split,
                        transform=transform,
                        class_mapping=self.class_mapping
                    )
                    datasets.append(dataset)
                    print(f"Loaded {dataset_name} dataset with {len(dataset)} samples")
                except Exception as e:
                    print(f"Failed to load {dataset_name}: {e}")
        
        return datasets
    
    def _create_dataloader(self, datasets: List[UnifiedDocumentDataset], 
                          split: str) -> DataLoader:
        """Create dataloader with weighted sampling"""
        if not datasets:
            raise ValueError(f"No datasets loaded for {split}")
        
        # Combine datasets
        combined_dataset = ConcatDataset(datasets)
        
        # Create weighted sampler if training
        if split == 'train':
            # Calculate sample weights based on dataset weights
            sample_weights = []
            for dataset in datasets:
                dataset_weight = dataset.weight
                sample_weights.extend([dataset_weight] * len(dataset))
            
            sampler = torch.utils.data.WeightedRandomSampler(
                weights=sample_weights,
                num_samples=len(sample_weights),
                replacement=True
            )
            
            return DataLoader(
                combined_dataset,
                batch_size=self.config['training']['batch_size'],
                sampler=sampler,
                num_workers=self.config['training']['num_workers'],
                collate_fn=self.collate_fn,
                pin_memory=True
            )
        else:
            return DataLoader(
                combined_dataset,
                batch_size=self.config['training']['batch_size'],
                shuffle=False,
                num_workers=self.config['training']['num_workers'],
                collate_fn=self.collate_fn,
                pin_memory=True
            )
    
    def collate_fn(self, batch):
        """Custom collate function for variable number of boxes"""
        images = torch.stack([item['image'] for item in batch])
        
        # Pad bboxes and labels
        max_boxes = max(len(item['bboxes']) for item in batch)
        
        padded_bboxes = []
        padded_labels = []
        box_masks = []
        
        for item in batch:
            num_boxes = len(item['bboxes'])
            
            if num_boxes > 0:
                # Pad boxes
                pad_boxes = torch.zeros((max_boxes - num_boxes, 4))
                bboxes = torch.cat([item['bboxes'], pad_boxes], dim=0)
                
                # Pad labels
                pad_labels = torch.full((max_boxes - num_boxes,), -1, dtype=torch.long)
                labels = torch.cat([item['labels'], pad_labels], dim=0)
                
                # Create mask
                mask = torch.zeros(max_boxes, dtype=torch.bool)
                mask[:num_boxes] = True
            else:
                bboxes = torch.zeros((max_boxes, 4))
                labels = torch.full((max_boxes,), -1, dtype=torch.long)
                mask = torch.zeros(max_boxes, dtype=torch.bool)
            
            padded_bboxes.append(bboxes)
            padded_labels.append(labels)
            box_masks.append(mask)
        
        return {
            'images': images,
            'bboxes': torch.stack(padded_bboxes),
            'labels': torch.stack(padded_labels),
            'masks': torch.stack(box_masks),
            'image_ids': [item['image_id'] for item in batch],
            'datasets': [item['dataset'] for item in batch]
        }
    
    def _init_model(self):
        """Initialize the model based on configuration"""
        model_type = self.config['model']['type']
        
        if model_type == 'yolo':
            # For YOLO, we'll use the ultralytics implementation
            model = YOLO(self.config['model']['backbone'])
            return model
        
        elif model_type == 'hybrid':
            # Use our custom hybrid model
            from enhanced_content_detector import HybridDetectionHead
            
            # Create backbone
            backbone = timm.create_model(
                self.config['model']['backbone'],
                pretrained=True,
                features_only=True
            )
            
            # Get feature info
            feature_info = backbone.feature_info.channels()
            
            # Create detection head
            detection_head = HybridDetectionHead(
                in_channels=feature_info[-1],
                num_classes=len(self.class_mapping)
            )
            
            # Combine into full model
            class HybridModel(nn.Module):
                def __init__(self, backbone, head):
                    super().__init__()
                    self.backbone = backbone
                    self.head = head
                
                def forward(self, x):
                    features = self.backbone(x)
                    return self.head(features)
            
            model = HybridModel(backbone, detection_head)
            return model.to(self.device)
        
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    def _init_loss_functions(self):
        """Initialize loss functions"""
        return {
            'classification': nn.CrossEntropyLoss(ignore_index=-1),
            'bbox_regression': nn.SmoothL1Loss(),
            'objectness': nn.BCEWithLogitsLoss()
        }
    
    def train_epoch(self, epoch: int):
        """Train for one epoch"""
        self.model.train()
        
        epoch_losses = {
            'total': 0,
            'classification': 0,
            'bbox_regression': 0,
            'objectness': 0
        }
        
        progress_bar = tqdm(self.train_loader, desc=f'Epoch {epoch}')
        
        for batch_idx, batch in enumerate(progress_bar):
            # Move to device
            images = batch['images'].to(self.device)
            bboxes = batch['bboxes'].to(self.device)
            labels = batch['labels'].to(self.device)
            masks = batch['masks'].to(self.device)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass with mixed precision
            with autocast():
                outputs = self.model(images)
                
                # Calculate losses
                loss_dict = self.calculate_losses(outputs, bboxes, labels, masks)
                total_loss = sum(loss_dict.values())
            
            # Backward pass
            self.scaler.scale(total_loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            # Update metrics
            for key, value in loss_dict.items():
                epoch_losses[key] += value.item()
            epoch_losses['total'] += total_loss.item()
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f"{total_loss.item():.4f}",
                'lr': f"{self.optimizer.param_groups[0]['lr']:.6f}"
            })
            
            # Log to wandb
            if batch_idx % self.config['training']['log_interval'] == 0:
                if self.config.get('use_wandb', True):
                    wandb.log({
                        'train/loss': total_loss.item(),
                        'train/lr': self.optimizer.param_groups[0]['lr'],
                        'step': epoch * len(self.train_loader) + batch_idx
                    })
        
        # Average losses
        for key in epoch_losses:
            epoch_losses[key] /= len(self.train_loader)
        
        return epoch_losses
    
    def calculate_losses(self, outputs, bboxes, labels, masks):
        """Calculate all losses"""
        losses = {}
        
        # Implementation depends on model output format
        # This is a simplified version
        
        # For YOLO-style outputs
        if isinstance(outputs, dict) and 'level_0' in outputs:
            for level_key, level_outputs in outputs.items():
                if level_key.startswith('level_'):
                    # Classification loss
                    cls_pred = level_outputs['class']
                    # ... implement loss calculation
                    
                    # Regression loss
                    bbox_pred = level_outputs['bbox']
                    # ... implement loss calculation
                    
                    # Objectness loss
                    obj_pred = level_outputs['objectness']
                    # ... implement loss calculation
        
        return losses
    
    def validate(self, epoch: int):
        """Validate the model"""
        self.model.eval()
        
        val_losses = {
            'total': 0,
            'classification': 0,
            'bbox_regression': 0,
            'objectness': 0
        }
        
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc='Validation'):
                images = batch['images'].to(self.device)
                bboxes = batch['bboxes'].to(self.device)
                labels = batch['labels'].to(self.device)
                masks = batch['masks'].to(self.device)
                
                # Forward pass
                outputs = self.model(images)
                
                # Calculate losses
                loss_dict = self.calculate_losses(outputs, bboxes, labels, masks)
                
                for key, value in loss_dict.items():
                    val_losses[key] += value.item()
                val_losses['total'] += sum(loss_dict.values()).item()
                
                # Store predictions for mAP calculation
                # ... implement prediction extraction
        
        # Average losses
        for key in val_losses:
            val_losses[key] /= len(self.val_loader)
        
        # Calculate mAP
        # mAP = calculate_map(all_predictions, all_targets)
        
        # Log to wandb
        if self.config.get('use_wandb', True):
            wandb.log({
                'val/loss': val_losses['total'],
                # 'val/mAP': mAP,
                'epoch': epoch
            })
        
        return val_losses
    
    def train(self):
        """Main training loop"""
        best_val_loss = float('inf')
        
        for epoch in range(self.config['training']['epochs']):
            print(f"\nEpoch {epoch + 1}/{self.config['training']['epochs']}")
            
            # Training
            train_losses = self.train_epoch(epoch)
            print(f"Train losses: {train_losses}")
            
            # Validation
            val_losses = self.validate(epoch)
            print(f"Val losses: {val_losses}")
            
            # Update learning rate
            self.scheduler.step()
            
            # Save checkpoint
            if val_losses['total'] < best_val_loss:
                best_val_loss = val_losses['total']
                self.save_checkpoint(epoch, val_losses['total'], is_best=True)
            
            # Regular checkpoint
            if (epoch + 1) % self.config['training']['save_interval'] == 0:
                self.save_checkpoint(epoch, val_losses['total'])
    
    def save_checkpoint(self, epoch: int, val_loss: float, is_best: bool = False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'val_loss': val_loss,
            'config': self.config,
            'class_mapping': self.class_mapping
        }
        
        save_dir = Path(self.config['training']['checkpoint_dir'])
        save_dir.mkdir(parents=True, exist_ok=True)
        
        if is_best:
            torch.save(checkpoint, save_dir / 'best_model.pth')
        
        torch.save(checkpoint, save_dir / f'checkpoint_epoch_{epoch}.pth')


def main():
    parser = argparse.ArgumentParser(description='Train enhanced document detector')
    parser.add_argument('--config', type=str, required=True, 
                       help='Path to training configuration file')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from')
    args = parser.parse_args()
    
    # Create trainer
    trainer = MultiStageTrainer(args.config)
    
    # Resume if specified
    if args.resume:
        checkpoint = torch.load(args.resume)
        trainer.model.load_state_dict(checkpoint['model_state_dict'])
        trainer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        trainer.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
    else:
        start_epoch = 0
    
    # Train
    trainer.train()


if __name__ == '__main__':
    main()