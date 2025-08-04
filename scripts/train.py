"""
Simplified Training Script - Works with YOLO and synthetic/real data
"""

import argparse
import yaml
import os
from ultralytics import YOLO
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description='Train YOLO model for document detection')
    parser.add_argument('--config', default='configs/config.yaml',
                        help='Path to YAML config file')
    parser.add_argument('--weights', default='models/yolov8m.pt',
                        help='Pretrained weights')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of training epochs')
    parser.add_argument('--imgsz', type=int, default=640,
                        help='Image size')
    parser.add_argument('--batch', type=int, default=16,
                        help='Batch size')
    parser.add_argument('--device', default='0',
                        help='Device to train on')
    parser.add_argument('--project', default='models/checkpoints',
                        help='Project directory')
    parser.add_argument('--name', default='document_detection',
                        help='Experiment name')
    
    args = parser.parse_args()
    
    # Load config
    print(f"Loading config from {args.config}")
    if os.path.exists(args.config):
        with open(args.config, 'r') as f:
            cfg = yaml.safe_load(f)
    else:
        print(f"Config file not found: {args.config}")
        print("Using default configuration...")
        cfg = {
            'path': 'data',
            'train': 'images/train',
            'val': 'images/val',
            'nc': 26,  # number of classes
            'names': [
                'text', 'title', 'list', 'table', 'figure', 'caption',
                'header', 'footer', 'page_number', 'staff', 'measure',
                'note', 'clef', 'time_signature', 'lyrics', 'checkbox',
                'input_field', 'signature_field', 'dropdown', 'flowchart',
                'graph', 'equation', 'barcode', 'qr_code', 'logo', 'stamp'
            ]
        }
    
    # Check if data exists
    data_path = Path(cfg.get('path', 'data'))
    train_path = data_path / cfg.get('train', 'images/train')
    val_path = data_path / cfg.get('val', 'images/val')
    
    if not train_path.exists():
        print(f"\n⚠️  Training data not found at: {train_path}")
        print("\nTo get started:")
        print("1. Generate synthetic data:")
        print("   python generate_synthetic_data.py --output data/synthetic --samples 1000")
        print("\n2. Or download real datasets:")
        print("   python prepare_datasets.py --download --datasets publaynet")
        print("\n3. Then prepare the data:")
        print("   python prepare_datasets.py --process --merge")
        return
    
    # Initialize model
    print(f"\nLoading base model from {args.weights}")
    model = YOLO(args.weights)
    
    # Create temporary data.yaml for YOLO
    data_yaml = {
        'path': str(data_path.absolute()),
        'train': cfg.get('train', 'images/train'),
        'val': cfg.get('val', 'images/val'),
        'nc': cfg.get('nc', 26),
        'names': cfg.get('names', ['class_' + str(i) for i in range(26)])
    }
    
    # Save temporary data.yaml
    temp_yaml_path = 'temp_data.yaml'
    with open(temp_yaml_path, 'w') as f:
        yaml.dump(data_yaml, f)
    
    print("\nStarting training...")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch}")
    print(f"Image size: {args.imgsz}")
    print(f"Device: {args.device}")
    
    # Train the model
    try:
        results = model.train(
            data=temp_yaml_path,
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            device=args.device,
            project=args.project,
            name=args.name,
            exist_ok=True,
            patience=10,
            save=True,
            plots=True,
            verbose=True
        )
        
        print("\n✅ Training completed successfully!")
        print(f"Results saved to: {args.project}/{args.name}")
        
        # Clean up temporary file
        if os.path.exists(temp_yaml_path):
            os.remove(temp_yaml_path)
            
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        
        # Clean up temporary file
        if os.path.exists(temp_yaml_path):
            os.remove(temp_yaml_path)


if __name__ == '__main__':
    main()