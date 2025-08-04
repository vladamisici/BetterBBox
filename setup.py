#!/usr/bin/env python3
"""
Automated Setup Script for Enhanced Document Detection System
Downloads models, prepares datasets, and configures the environment
"""

import os
import sys
import subprocess
import urllib.request
import zipfile
import tarfile
from pathlib import Path
import shutil
import json
from tqdm import tqdm

class ProjectSetup:
    def __init__(self):
        self.project_root = Path(__file__).parent
        self.models_dir = self.project_root / "models"
        self.data_dir = self.project_root / "data"
        self.checkpoints_dir = self.models_dir / "checkpoints"
        
        # URLs for pre-trained models
        self.model_urls = {
            'yolov8m.pt': 'https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8m.pt',
            'yolov8x.pt': 'https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8x.pt',
        }
        
        # Sample data URL (you can replace with actual URLs)
        self.sample_data_url = None  # Will create synthetic samples instead
        
    def setup_directories(self):
        """Create all necessary directories"""
        print("📁 Creating directory structure...")
        
        directories = [
            self.models_dir,
            self.checkpoints_dir,
            self.models_dir / "optimized",
            self.data_dir / "raw",
            self.data_dir / "processed", 
            self.data_dir / "synthetic",
            self.project_root / "logs",
            self.project_root / "results",
            self.project_root / "cache"
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
            
        print("✅ Directory structure created")
    
    def download_file(self, url: str, destination: Path):
        """Download file with progress bar"""
        try:
            with urllib.request.urlopen(url) as response:
                total_size = int(response.headers.get('Content-Length', 0))
                
                with open(destination, 'wb') as f:
                    with tqdm(total=total_size, unit='B', unit_scale=True, desc=destination.name) as pbar:
                        while True:
                            chunk = response.read(8192)
                            if not chunk:
                                break
                            f.write(chunk)
                            pbar.update(len(chunk))
            return True
        except Exception as e:
            print(f"❌ Error downloading {url}: {e}")
            return False
    
    def download_models(self):
        """Download pre-trained models"""
        print("\n🤖 Downloading pre-trained models...")
        
        for model_name, url in self.model_urls.items():
            model_path = self.models_dir / model_name
            
            if model_path.exists():
                print(f"✅ {model_name} already exists, skipping...")
                continue
                
            print(f"📥 Downloading {model_name}...")
            if self.download_file(url, model_path):
                print(f"✅ {model_name} downloaded successfully")
            else:
                print(f"⚠️  Failed to download {model_name}, you'll need to download it manually")
    
    def create_sample_data(self):
        """Create sample images for testing"""
        print("\n🎨 Creating sample data...")
        
        samples_dir = self.project_root / "samples"
        samples_dir.mkdir(exist_ok=True)
        
        # Create sample images using numpy/opencv
        try:
            import numpy as np
            import cv2
            
            # Academic paper sample
            img = np.ones((1200, 900, 3), dtype=np.uint8) * 255
            cv2.rectangle(img, (200, 50), (700, 120), (0, 0, 0), 2)
            cv2.putText(img, "Sample Academic Paper", (220, 90), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 2)
            # Add some text blocks
            for i in range(5):
                y = 200 + i * 80
                cv2.rectangle(img, (100, y), (800, y + 60), (100, 100, 100), 1)
            cv2.imwrite(str(samples_dir / "academic_paper.jpg"), img)
            
            # Form sample
            img = np.ones((1200, 900, 3), dtype=np.uint8) * 255
            cv2.putText(img, "Application Form", (300, 80), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 2)
            # Add form fields
            fields = ["Name:", "Email:", "Phone:", "Address:"]
            for i, field in enumerate(fields):
                y = 200 + i * 100
                cv2.putText(img, field, (100, y), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 1)
                cv2.rectangle(img, (250, y - 30), (700, y + 10), (0, 0, 0), 1)
            cv2.imwrite(str(samples_dir / "form.jpg"), img)
            
            # Mixed document
            img = np.ones((1200, 900, 3), dtype=np.uint8) * 255
            # Title
            cv2.putText(img, "Mixed Document", (300, 80), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 2)
            # Table
            for i in range(4):
                for j in range(3):
                    x = 100 + j * 200
                    y = 200 + i * 50
                    cv2.rectangle(img, (x, y), (x + 190, y + 45), (0, 0, 0), 1)
            # Figure
            cv2.rectangle(img, (100, 500), (400, 800), (0, 0, 0), 2)
            cv2.circle(img, (250, 650), 80, (150, 150, 150), -1)
            cv2.imwrite(str(samples_dir / "mixed.jpg"), img)
            
            print("✅ Sample images created")
            
        except ImportError:
            print("⚠️  OpenCV not installed, skipping sample creation")
            print("    Install with: pip install opencv-python")
    
    def setup_config_files(self):
        """Create configuration files"""
        print("\n⚙️  Setting up configuration files...")
        
        # Copy .env.example to .env if not exists
        env_example = self.project_root / ".env.example"
        env_file = self.project_root / ".env"
        
        if not env_file.exists() and env_example.exists():
            shutil.copy(env_example, env_file)
            print("✅ Created .env from .env.example")
            print("⚠️  Please edit .env with your configuration!")
        
        # Create empty model checkpoint for initial testing
        checkpoint_path = self.checkpoints_dir / "best_model.pth"
        if not checkpoint_path.exists():
            # Create a minimal checkpoint
            checkpoint = {
                'epoch': 0,
                'model_state_dict': {},
                'optimizer_state_dict': {},
                'loss': 0.0,
                'config': {
                    'num_classes': 26,
                    'model_type': 'yolo'
                }
            }
            import torch
            torch.save(checkpoint, checkpoint_path)
            print("✅ Created placeholder checkpoint")
    
    def install_dependencies(self):
        """Install Python dependencies"""
        print("\n📦 Installing Python dependencies...")
        
        try:
            subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], 
                          check=True)
            print("✅ Dependencies installed successfully")
        except subprocess.CalledProcessError:
            print("❌ Failed to install dependencies")
            print("   Please run manually: pip install -r requirements.txt")
    
    def check_gpu(self):
        """Check GPU availability"""
        print("\n🎮 Checking GPU availability...")
        
        try:
            import torch
            if torch.cuda.is_available():
                gpu_count = torch.cuda.device_count()
                gpu_name = torch.cuda.get_device_name(0)
                print(f"✅ Found {gpu_count} GPU(s)")
                print(f"   Primary GPU: {gpu_name}")
                print(f"   CUDA Version: {torch.version.cuda}")
            else:
                print("⚠️  No GPU detected. Training will be slow on CPU!")
        except ImportError:
            print("❌ PyTorch not installed yet")
    
    def create_initial_dataset_config(self):
        """Create initial dataset configuration"""
        print("\n📊 Creating dataset configuration...")
        
        dataset_config = {
            "datasets": {
                "synthetic": {
                    "path": str(self.data_dir / "synthetic"),
                    "train_annotations": "annotations/train_annotations.json",
                    "val_annotations": "annotations/val_annotations.json",
                    "enabled": True
                }
            },
            "num_classes": 26,
            "class_names": [
                "text", "title", "list", "table", "figure", "caption",
                "header", "footer", "page_number", "staff", "measure",
                "note", "clef", "time_signature", "lyrics", "checkbox",
                "input_field", "signature_field", "dropdown", "flowchart",
                "graph", "equation", "barcode", "qr_code", "logo", "stamp"
            ]
        }
        
        config_path = self.project_root / "configs" / "dataset_config.json"
        with open(config_path, 'w') as f:
            json.dump(dataset_config, f, indent=2)
        
        print("✅ Dataset configuration created")
    
    def print_next_steps(self):
        """Print next steps for the user"""
        print("\n" + "="*60)
        print("🎉 SETUP COMPLETE!")
        print("="*60)
        
        print("\n📝 Next Steps:")
        print("\n1. IMMEDIATE TEST (no training needed):")
        print("   python infer.py --weights models/yolov8m.pt --source samples/academic_paper.jpg")
        
        print("\n2. GENERATE SYNTHETIC DATA:")
        print("   python generate_synthetic_data.py --output data/synthetic --samples 1000")
        
        print("\n3. TRAIN ON SYNTHETIC DATA:")
        print("   python train.py --config configs/config.yaml --weights models/yolov8m.pt")
        
        print("\n4. START WEB DEMO:")
        print("   streamlit run demo_app.py")
        
        print("\n5. START API SERVER:")
        print("   python api_server.py")
        
        print("\n6. OR USE DOCKER:")
        print("   docker-compose up -d")
        
        print("\n⚠️  IMPORTANT:")
        print("   - Edit .env file with your configuration")
        print("   - For real training, download actual datasets:")
        print("     python prepare_datasets.py --download --datasets publaynet")
        
        print("\n📚 Documentation: README.md")
        print("🚀 Quick Start: QUICKSTART.md")
        print("\n" + "="*60)
    
    def run(self):
        """Run complete setup"""
        print("🚀 Enhanced Document Detection System Setup")
        print("="*60)
        
        # Create directories
        self.setup_directories()
        
        # Install dependencies
        self.install_dependencies()
        
        # Download models
        self.download_models()
        
        # Create sample data
        self.create_sample_data()
        
        # Setup config files
        self.setup_config_files()
        
        # Create dataset config
        self.create_initial_dataset_config()
        
        # Check GPU
        self.check_gpu()
        
        # Print next steps
        self.print_next_steps()


if __name__ == "__main__":
    setup = ProjectSetup()
    setup.run()