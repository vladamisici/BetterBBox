"""
Dataset Preparation Script for Enhanced Document Detection
Handles multiple dataset formats and creates unified annotations
"""

import os
import json
import argparse
import shutil
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
import cv2
from tqdm import tqdm
import pandas as pd
import xml.etree.ElementTree as ET
from collections import defaultdict
import requests
import zipfile
import tarfile
import gdown


class DatasetDownloader:
    """Handles downloading of various public datasets"""
    
    DATASET_URLS = {
        'publaynet': {
            'train': 'https://dax-cdn.cdn.appdomain.cloud/dax-publaynet/1.0.0/train-{}.tar.gz',
            'val': 'https://dax-cdn.cdn.appdomain.cloud/dax-publaynet/1.0.0/val.tar.gz',
            'labels': 'https://dax-cdn.cdn.appdomain.cloud/dax-publaynet/1.0.0/labels.tar.gz'
        },
        'docbank': {
            'url': 'https://github.com/doc-analysis/DocBank',  # Note: Requires manual download
            'info': 'Please download DocBank from the official repository'
        },
        'pubtables': {
            'url': 'https://github.com/microsoft/table-transformer',
            'info': 'Download PubTables-1M from Microsoft'
        },
        'funsd': {
            'url': 'https://guillaumejaume.github.io/FUNSD/dataset.zip'
        }
    }
    
    @staticmethod
    def download_publaynet(output_dir: Path):
        """Download PubLayNet dataset"""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Download train splits
        for i in range(7):  # train-0 to train-6
            url = DatasetDownloader.DATASET_URLS['publaynet']['train'].format(i)
            filename = f'train-{i}.tar.gz'
            filepath = output_dir / filename
            
            if not filepath.exists():
                print(f"Downloading {filename}...")
                DatasetDownloader._download_file(url, filepath)
                DatasetDownloader._extract_tar(filepath, output_dir)
        
        # Download validation
        val_url = DatasetDownloader.DATASET_URLS['publaynet']['val']
        val_path = output_dir / 'val.tar.gz'
        if not val_path.exists():
            print("Downloading validation set...")
            DatasetDownloader._download_file(val_url, val_path)
            DatasetDownloader._extract_tar(val_path, output_dir)
        
        # Download labels
        labels_url = DatasetDownloader.DATASET_URLS['publaynet']['labels']
        labels_path = output_dir / 'labels.tar.gz'
        if not labels_path.exists():
            print("Downloading labels...")
            DatasetDownloader._download_file(labels_url, labels_path)
            DatasetDownloader._extract_tar(labels_path, output_dir)
    
    @staticmethod
    def download_funsd(output_dir: Path):
        """Download FUNSD dataset"""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        url = DatasetDownloader.DATASET_URLS['funsd']['url']
        zip_path = output_dir / 'funsd.zip'
        
        if not zip_path.exists():
            print("Downloading FUNSD...")
            DatasetDownloader._download_file(url, zip_path)
            DatasetDownloader._extract_zip(zip_path, output_dir)
    
    @staticmethod
    def _download_file(url: str, filepath: Path):
        """Download file with progress bar"""
        response = requests.get(url, stream=True)
        total_size = int(response.headers.get('content-length', 0))
        
        with open(filepath, 'wb') as f:
            with tqdm(total=total_size, unit='B', unit_scale=True, desc=filepath.name) as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
                    pbar.update(len(chunk))
    
    @staticmethod
    def _extract_tar(tar_path: Path, extract_to: Path):
        """Extract tar.gz file"""
        print(f"Extracting {tar_path.name}...")
        with tarfile.open(tar_path, 'r:gz') as tar:
            tar.extractall(path=extract_to)
        tar_path.unlink()  # Remove archive after extraction
    
    @staticmethod
    def _extract_zip(zip_path: Path, extract_to: Path):
        """Extract zip file"""
        print(f"Extracting {zip_path.name}...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
        zip_path.unlink()  # Remove archive after extraction


class DatasetProcessor:
    """Base class for dataset processors"""
    
    def __init__(self, dataset_path: Path, output_path: Path):
        self.dataset_path = dataset_path
        self.output_path = output_path
        self.output_path.mkdir(parents=True, exist_ok=True)
        
    def process(self):
        """Process the dataset - to be implemented by subclasses"""
        raise NotImplementedError


class PubLayNetProcessor(DatasetProcessor):
    """Process PubLayNet dataset"""
    
    def process(self):
        """Convert PubLayNet to unified format"""
        print("Processing PubLayNet dataset...")
        
        # Load COCO annotations
        train_ann_path = self.dataset_path / 'labels' / 'publaynet' / 'train.json'
        val_ann_path = self.dataset_path / 'labels' / 'publaynet' / 'val.json'
        
        # Process train and val splits
        for split, ann_path in [('train', train_ann_path), ('val', val_ann_path)]:
            print(f"Processing {split} split...")
            
            with open(ann_path, 'r') as f:
                coco_data = json.load(f)
            
            # Convert to unified format
            unified_annotations = self._convert_coco_to_unified(coco_data, split)
            
            # Save unified annotations
            output_file = self.output_path / f'{split}_annotations.json'
            with open(output_file, 'w') as f:
                json.dump(unified_annotations, f, indent=2)
            
            print(f"Saved {len(unified_annotations)} annotations to {output_file}")
    
    def _convert_coco_to_unified(self, coco_data: Dict, split: str) -> List[Dict]:
        """Convert COCO format to unified format"""
        images = {img['id']: img for img in coco_data['images']}
        categories = {cat['id']: cat['name'] for cat in coco_data['categories']}
        
        # Group annotations by image
        img_annotations = defaultdict(list)
        for ann in tqdm(coco_data['annotations'], desc='Processing annotations'):
            img_id = ann['image_id']
            
            # Convert bbox from COCO [x,y,w,h] to [x1,y1,x2,y2]
            x, y, w, h = ann['bbox']
            bbox = [x, y, x + w, y + h]
            
            img_annotations[img_id].append({
                'bbox': bbox,
                'category': categories[ann['category_id']],
                'confidence': 1.0  # Ground truth
            })
        
        # Create unified format
        unified_data = []
        for img_id, annotations in img_annotations.items():
            img_info = images[img_id]
            
            # Handle different image paths for train splits
            if split == 'train':
                # PubLayNet train images are in train-0 to train-6 folders
                # You might need to search for the actual file
                image_path = self._find_train_image(img_info['file_name'])
            else:
                image_path = self.dataset_path / split / img_info['file_name']
            
            unified_data.append({
                'image_id': img_info['file_name'],
                'image_path': str(image_path),
                'width': img_info['width'],
                'height': img_info['height'],
                'annotations': annotations
            })
        
        return unified_data
    
    def _find_train_image(self, filename: str) -> Path:
        """Find train image in train-0 to train-6 folders"""
        for i in range(7):
            potential_path = self.dataset_path / f'train-{i}' / filename
            if potential_path.exists():
                return potential_path
        
        # If not found in numbered folders, try main train folder
        return self.dataset_path / 'train' / filename


class DocBankProcessor(DatasetProcessor):
    """Process DocBank dataset"""
    
    def process(self):
        """Convert DocBank to unified format"""
        print("Processing DocBank dataset...")
        
        # DocBank structure: DocBank_500K_txt_img/
        # Process each split
        for split in ['train', 'val', 'test']:
            split_dir = self.dataset_path / f'DocBank_500K_txt_img' / split
            if not split_dir.exists():
                print(f"Split {split} not found, skipping...")
                continue
            
            print(f"Processing {split} split...")
            unified_annotations = []
            
            # Get all txt files (contain annotations)
            txt_files = list(split_dir.glob('*.txt'))
            
            for txt_file in tqdm(txt_files, desc=f'Processing {split}'):
                # Corresponding image file
                img_file = txt_file.with_suffix('.jpg')
                if not img_file.exists():
                    img_file = txt_file.with_suffix('.png')
                
                if not img_file.exists():
                    continue
                
                # Read image to get dimensions
                img = cv2.imread(str(img_file))
                if img is None:
                    continue
                
                height, width = img.shape[:2]
                
                # Parse annotations
                annotations = self._parse_docbank_txt(txt_file)
                
                unified_annotations.append({
                    'image_id': txt_file.stem,
                    'image_path': str(img_file),
                    'width': width,
                    'height': height,
                    'annotations': annotations
                })
            
            # Save unified annotations
            output_file = self.output_path / f'{split}_annotations.json'
            with open(output_file, 'w') as f:
                json.dump(unified_annotations, f, indent=2)
            
            print(f"Saved {len(unified_annotations)} annotations to {output_file}")
    
    def _parse_docbank_txt(self, txt_file: Path) -> List[Dict]:
        """Parse DocBank txt annotation file"""
        annotations = []
        
        with open(txt_file, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) < 10:  # DocBank format has 10 columns
                    continue
                
                # Extract bbox coordinates
                x0, y0, x1, y1 = map(float, parts[1:5])
                
                # Extract category (column 9)
                category = parts[9]
                
                annotations.append({
                    'bbox': [x0, y0, x1, y1],
                    'category': category.lower(),
                    'confidence': 1.0
                })
        
        return annotations


class MusicScoreProcessor(DatasetProcessor):
    """Process music score datasets (e.g., DeepScores, MUSCIMA++)"""
    
    def process(self):
        """Convert music score dataset to unified format"""
        print("Processing Music Score dataset...")
        
        # This is a template - actual implementation depends on the specific dataset
        # For example, DeepScores uses XML annotations
        
        annotation_dir = self.dataset_path / 'annotations'
        image_dir = self.dataset_path / 'images'
        
        if not annotation_dir.exists() or not image_dir.exists():
            print("Expected directory structure not found!")
            return
        
        unified_annotations = []
        
        # Process XML annotations (DeepScores style)
        xml_files = list(annotation_dir.glob('*.xml'))
        
        for xml_file in tqdm(xml_files, desc='Processing music scores'):
            # Parse XML
            tree = ET.parse(xml_file)
            root = tree.getroot()
            
            # Get image info
            filename = root.find('filename').text
            img_path = image_dir / filename
            
            if not img_path.exists():
                continue
            
            # Get image dimensions
            size = root.find('size')
            width = int(size.find('width').text)
            height = int(size.find('height').text)
            
            # Parse objects
            annotations = []
            for obj in root.findall('object'):
                name = obj.find('name').text
                bbox = obj.find('bndbox')
                
                xmin = float(bbox.find('xmin').text)
                ymin = float(bbox.find('ymin').text)
                xmax = float(bbox.find('xmax').text)
                ymax = float(bbox.find('ymax').text)
                
                # Map music-specific classes
                category = self._map_music_class(name)
                
                annotations.append({
                    'bbox': [xmin, ymin, xmax, ymax],
                    'category': category,
                    'confidence': 1.0
                })
            
            unified_annotations.append({
                'image_id': xml_file.stem,
                'image_path': str(img_path),
                'width': width,
                'height': height,
                'annotations': annotations
            })
        
        # Save unified annotations
        output_file = self.output_path / 'music_annotations.json'
        with open(output_file, 'w') as f:
            json.dump(unified_annotations, f, indent=2)
        
        print(f"Saved {len(unified_annotations)} annotations to {output_file}")
    
    def _map_music_class(self, music_class: str) -> str:
        """Map music-specific classes to unified classes"""
        mapping = {
            'notehead': 'note',
            'stem': 'note',
            'beam': 'note',
            'rest': 'note',
            'accidental': 'note',
            'barline': 'measure',
            'clef': 'clef',
            'keySignature': 'clef',
            'timeSignature': 'time_signature',
            'staff': 'staff',
            'slur': 'note',
            'tie': 'note',
            'dynamic': 'note',
            'lyrics': 'lyrics'
        }
        
        return mapping.get(music_class, music_class.lower())


class FUNSDProcessor(DatasetProcessor):
    """Process FUNSD dataset for form understanding"""
    
    def process(self):
        """Convert FUNSD to unified format"""
        print("Processing FUNSD dataset...")
        
        for split in ['train', 'test']:
            split_dir = self.dataset_path / 'dataset' / split
            if not split_dir.exists():
                print(f"Split {split} not found, skipping...")
                continue
            
            print(f"Processing {split} split...")
            unified_annotations = []
            
            # FUNSD has JSON annotations for each image
            json_files = list(split_dir.glob('*.json'))
            
            for json_file in tqdm(json_files, desc=f'Processing {split}'):
                with open(json_file, 'r') as f:
                    data = json.load(f)
                
                # Get image path
                img_path = json_file.with_suffix('.png')
                if not img_path.exists():
                    continue
                
                # Get image dimensions
                img = cv2.imread(str(img_path))
                if img is None:
                    continue
                
                height, width = img.shape[:2]
                
                # Parse form annotations
                annotations = []
                for form_item in data['form']:
                    bbox = form_item['box']  # [x1, y1, x2, y2]
                    category = form_item['label'].lower()  # header, question, answer, other
                    
                    # Map to unified categories
                    if category == 'question':
                        category = 'input_field'
                    elif category == 'answer':
                        category = 'text'
                    
                    annotations.append({
                        'bbox': bbox,
                        'category': category,
                        'confidence': 1.0
                    })
                
                unified_annotations.append({
                    'image_id': json_file.stem,
                    'image_path': str(img_path),
                    'width': width,
                    'height': height,
                    'annotations': annotations
                })
            
            # Save unified annotations
            output_file = self.output_path / f'{split}_annotations.json'
            with open(output_file, 'w') as f:
                json.dump(unified_annotations, f, indent=2)
            
            print(f"Saved {len(unified_annotations)} annotations to {output_file}")


class DatasetAugmenter:
    """Generate synthetic data for underrepresented classes"""
    
    @staticmethod
    def generate_synthetic_music_scores(output_dir: Path, num_samples: int = 1000):
        """Generate synthetic music score data using music21 or similar"""
        print(f"Generating {num_samples} synthetic music scores...")
        
        output_dir.mkdir(parents=True, exist_ok=True)
        annotations = []
        
        # This is a placeholder - actual implementation would use
        # music generation libraries like music21, LilyPond, etc.
        
        for i in tqdm(range(num_samples), desc='Generating music scores'):
            # Generate synthetic music score
            # ... implementation ...
            pass
        
        return annotations
    
    @staticmethod
    def generate_synthetic_forms(output_dir: Path, num_samples: int = 1000):
        """Generate synthetic form data"""
        print(f"Generating {num_samples} synthetic forms...")
        
        output_dir.mkdir(parents=True, exist_ok=True)
        annotations = []
        
        # Generate forms with various layouts
        # This could use HTML/CSS rendering or programmatic generation
        
        for i in tqdm(range(num_samples), desc='Generating forms'):
            # Generate synthetic form
            # ... implementation ...
            pass
        
        return annotations


class DatasetMerger:
    """Merge multiple processed datasets into final training set"""
    
    def __init__(self, dataset_dirs: List[Path], output_dir: Path):
        self.dataset_dirs = dataset_dirs
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def merge(self):
        """Merge all datasets into unified training and validation sets"""
        print("Merging datasets...")
        
        all_train_annotations = []
        all_val_annotations = []
        
        # Collect all annotations
        for dataset_dir in self.dataset_dirs:
            # Train annotations
            train_file = dataset_dir / 'train_annotations.json'
            if train_file.exists():
                with open(train_file, 'r') as f:
                    annotations = json.load(f)
                    # Add dataset source
                    for ann in annotations:
                        ann['dataset'] = dataset_dir.name
                    all_train_annotations.extend(annotations)
            
            # Val annotations
            val_file = dataset_dir / 'val_annotations.json'
            if val_file.exists():
                with open(val_file, 'r') as f:
                    annotations = json.load(f)
                    # Add dataset source
                    for ann in annotations:
                        ann['dataset'] = dataset_dir.name
                    all_val_annotations.extend(annotations)
        
        # Create stratified split if needed
        if not all_val_annotations and all_train_annotations:
            # Split train into train/val
            from sklearn.model_selection import train_test_split
            all_train_annotations, all_val_annotations = train_test_split(
                all_train_annotations, 
                test_size=0.1, 
                random_state=42
            )
        
        # Save merged annotations
        train_output = self.output_dir / 'merged_train_annotations.json'
        val_output = self.output_dir / 'merged_val_annotations.json'
        
        with open(train_output, 'w') as f:
            json.dump(all_train_annotations, f, indent=2)
        
        with open(val_output, 'w') as f:
            json.dump(all_val_annotations, f, indent=2)
        
        print(f"Merged {len(all_train_annotations)} train annotations")
        print(f"Merged {len(all_val_annotations)} val annotations")
        
        # Generate statistics
        self._generate_statistics(all_train_annotations, all_val_annotations)
    
    def _generate_statistics(self, train_annotations: List[Dict], 
                           val_annotations: List[Dict]):
        """Generate dataset statistics"""
        stats = {
            'train': self._calculate_stats(train_annotations),
            'val': self._calculate_stats(val_annotations)
        }
        
        # Save statistics
        stats_file = self.output_dir / 'dataset_statistics.json'
        with open(stats_file, 'w') as f:
            json.dump(stats, f, indent=2)
        
        # Print summary
        print("\nDataset Statistics:")
        print(f"{'Split':<10} {'Images':<10} {'Annotations':<15} {'Avg/Image':<10}")
        print("-" * 50)
        
        for split, split_stats in stats.items():
            print(f"{split:<10} {split_stats['num_images']:<10} "
                  f"{split_stats['num_annotations']:<15} "
                  f"{split_stats['avg_annotations_per_image']:<10.2f}")
        
        print("\nClass Distribution:")
        all_classes = set()
        for split_stats in stats.values():
            all_classes.update(split_stats['class_distribution'].keys())
        
        for class_name in sorted(all_classes):
            train_count = stats['train']['class_distribution'].get(class_name, 0)
            val_count = stats['val']['class_distribution'].get(class_name, 0)
            print(f"{class_name:<20} Train: {train_count:<10} Val: {val_count:<10}")
    
    def _calculate_stats(self, annotations: List[Dict]) -> Dict:
        """Calculate statistics for annotations"""
        stats = {
            'num_images': len(annotations),
            'num_annotations': 0,
            'class_distribution': defaultdict(int),
            'dataset_distribution': defaultdict(int)
        }
        
        for img_ann in annotations:
            num_boxes = len(img_ann['annotations'])
            stats['num_annotations'] += num_boxes
            
            # Count classes
            for ann in img_ann['annotations']:
                stats['class_distribution'][ann['category']] += 1
            
            # Count datasets
            if 'dataset' in img_ann:
                stats['dataset_distribution'][img_ann['dataset']] += 1
        
        stats['avg_annotations_per_image'] = (
            stats['num_annotations'] / stats['num_images'] 
            if stats['num_images'] > 0 else 0
        )
        
        # Convert defaultdicts to regular dicts for JSON serialization
        stats['class_distribution'] = dict(stats['class_distribution'])
        stats['dataset_distribution'] = dict(stats['dataset_distribution'])
        
        return stats


def main():
    parser = argparse.ArgumentParser(description='Prepare datasets for enhanced document detection')
    parser.add_argument('--download', action='store_true',
                       help='Download datasets')
    parser.add_argument('--process', action='store_true',
                       help='Process datasets to unified format')
    parser.add_argument('--merge', action='store_true',
                       help='Merge all processed datasets')
    parser.add_argument('--datasets', nargs='+', 
                       default=['publaynet', 'docbank', 'funsd'],
                       help='Datasets to process')
    parser.add_argument('--data-dir', type=str, default='data/raw',
                       help='Root directory for raw datasets')
    parser.add_argument('--output-dir', type=str, default='data/processed',
                       help='Output directory for processed datasets')
    
    args = parser.parse_args()
    
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    
    # Download datasets
    if args.download:
        for dataset in args.datasets:
            dataset_dir = data_dir / dataset
            
            if dataset == 'publaynet':
                DatasetDownloader.download_publaynet(dataset_dir)
            elif dataset == 'funsd':
                DatasetDownloader.download_funsd(dataset_dir)
            else:
                print(f"Automatic download not available for {dataset}")
                print(f"Please download manually from: {DatasetDownloader.DATASET_URLS.get(dataset, {}).get('url', 'N/A')}")
    
    # Process datasets
    if args.process:
        processed_dirs = []
        
        for dataset in args.datasets:
            dataset_dir = data_dir / dataset
            dataset_output = output_dir / dataset
            
            if not dataset_dir.exists():
                print(f"Dataset {dataset} not found at {dataset_dir}")
                continue
            
            if dataset == 'publaynet':
                processor = PubLayNetProcessor(dataset_dir, dataset_output)
            elif dataset == 'docbank':
                processor = DocBankProcessor(dataset_dir, dataset_output)
            elif dataset == 'musescore':
                processor = MusicScoreProcessor(dataset_dir, dataset_output)
            elif dataset == 'funsd':
                processor = FUNSDProcessor(dataset_dir, dataset_output)
            else:
                print(f"No processor available for {dataset}")
                continue
            
            processor.process()
            processed_dirs.append(dataset_output)
    
    # Merge datasets
    if args.merge:
        processed_dirs = [output_dir / dataset for dataset in args.datasets]
        merger = DatasetMerger(processed_dirs, output_dir / 'merged')
        merger.merge()


if __name__ == '__main__':
    main()