"""
Comprehensive Test Suite for Enhanced Document Detection System
Includes unit tests, integration tests, and performance benchmarks
"""

import os
import sys
import json
import time
import tempfile
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import unittest
from unittest.mock import Mock, patch, MagicMock
import numpy as np
import cv2
import torch
import pytest
import requests
from PIL import Image
import asyncio
import aiohttp
from concurrent.futures import ThreadPoolExecutor
import pandas as pd

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from enhanced_content_detector import (
    EnhancedContentDetector, 
    BoundingBox, 
    DocumentType,
    EnhancedClasses,
    FusionModule
)


class TestEnhancedContentDetector(unittest.TestCase):
    """Unit tests for EnhancedContentDetector"""
    
    @classmethod
    def setUpClass(cls):
        """Set up test fixtures"""
        cls.detector = EnhancedContentDetector()
        cls.test_image = cls._create_test_image()
        cls.test_boxes = cls._create_test_boxes()
    
    @staticmethod
    def _create_test_image(width: int = 640, height: int = 480) -> np.ndarray:
        """Create a test image"""
        # Create white background
        img = np.ones((height, width, 3), dtype=np.uint8) * 255
        
        # Add some text-like rectangles
        cv2.rectangle(img, (50, 50), (200, 100), (0, 0, 0), 2)
        cv2.rectangle(img, (50, 150), (400, 200), (0, 0, 0), 2)
        cv2.rectangle(img, (100, 250), (300, 350), (0, 0, 0), 2)
        
        return img
    
    @staticmethod
    def _create_test_boxes() -> List[BoundingBox]:
        """Create test bounding boxes"""
        return [
            BoundingBox(x1=50, y1=50, x2=200, y2=100, 
                       confidence=0.9, class_id=0, class_name='text'),
            BoundingBox(x1=50, y1=150, x2=400, y2=200, 
                       confidence=0.85, class_id=1, class_name='title'),
            BoundingBox(x1=100, y1=250, x2=300, y2=350, 
                       confidence=0.7, class_id=3, class_name='table')
        ]
    
    def test_initialization(self):
        """Test detector initialization"""
        self.assertIsNotNone(self.detector)
        self.assertIsNotNone(self.detector.config)
        self.assertIn('thresholds', self.detector.config)
    
    def test_bounding_box_creation(self):
        """Test BoundingBox dataclass"""
        box = BoundingBox(
            x1=10, y1=20, x2=100, y2=200,
            confidence=0.95, class_id=0, class_name='text'
        )
        
        self.assertEqual(box.x1, 10)
        self.assertEqual(box.y1, 20)
        self.assertEqual(box.x2, 100)
        self.assertEqual(box.y2, 200)
        self.assertEqual(box.confidence, 0.95)
        self.assertEqual(box.class_id, 0)
        self.assertEqual(box.class_name, 'text')
    
    def test_class_mapping(self):
        """Test enhanced class definitions"""
        self.assertIn('text', EnhancedClasses.CLASSES)
        self.assertIn('staff', EnhancedClasses.CLASSES)
        self.assertIn('checkbox', EnhancedClasses.CLASSES)
        
        # Test inverse mapping
        self.assertEqual(
            EnhancedClasses.get_class_name(0),
            'text'
        )
    
    @patch('enhanced_content_detector.YOLO')
    def test_detection_mock(self, mock_yolo):
        """Test detection with mocked YOLO"""
        # Mock YOLO response
        mock_model = MagicMock()
        mock_result = MagicMock()
        mock_result.boxes.xyxy = torch.tensor([[50, 50, 200, 100]])
        mock_result.boxes.cls = torch.tensor([0])
        mock_result.boxes.conf = torch.tensor([0.9])
        mock_model.return_value = [mock_result]
        mock_yolo.return_value = mock_model
        
        # Create detector with mocked model
        detector = EnhancedContentDetector()
        detector.model = mock_model
        
        # Test detection
        boxes = detector._yolo_detect(self.test_image)
        
        self.assertIsInstance(boxes, list)
        self.assertTrue(len(boxes) > 0)
    
    def test_nms(self):
        """Test Non-Maximum Suppression"""
        # Create overlapping boxes
        boxes = [
            BoundingBox(x1=50, y1=50, x2=200, y2=200, 
                       confidence=0.9, class_id=0, class_name='text'),
            BoundingBox(x1=60, y1=60, x2=210, y2=210, 
                       confidence=0.8, class_id=0, class_name='text'),
            BoundingBox(x1=300, y1=300, x2=400, y2=400, 
                       confidence=0.95, class_id=1, class_name='title')
        ]
        
        # Apply NMS
        nms_boxes = self.detector._nms(boxes, threshold=0.5)
        
        # Should keep highest confidence box and non-overlapping box
        self.assertEqual(len(nms_boxes), 2)
        self.assertEqual(nms_boxes[0].confidence, 0.9)
        self.assertEqual(nms_boxes[1].confidence, 0.95)
    
    def test_iou_calculation(self):
        """Test Intersection over Union calculation"""
        box1 = BoundingBox(x1=0, y1=0, x2=100, y2=100,
                          confidence=1.0, class_id=0, class_name='text')
        box2 = BoundingBox(x1=50, y1=50, x2=150, y2=150,
                          confidence=1.0, class_id=0, class_name='text')
        
        fusion = FusionModule()
        iou = fusion._iou(box1, box2)
        
        # Expected IoU = 2500 / 17500 ≈ 0.143
        self.assertAlmostEqual(iou, 0.143, places=2)
    
    def test_fusion_module(self):
        """Test prediction fusion"""
        fusion = FusionModule(strategy='weighted_voting')
        
        # Create overlapping predictions from different models
        all_boxes = [
            # Model 1 predictions
            BoundingBox(x1=50, y1=50, x2=200, y2=200, 
                       confidence=0.9, class_id=0, class_name='text'),
            # Model 2 predictions (overlapping)
            BoundingBox(x1=55, y1=55, x2=205, y2=205, 
                       confidence=0.85, class_id=0, class_name='text'),
            # Different detection
            BoundingBox(x1=300, y1=300, x2=400, y2=400, 
                       confidence=0.7, class_id=1, class_name='title')
        ]
        
        fused_boxes = fusion.fuse(all_boxes)
        
        # Should fuse overlapping boxes
        self.assertEqual(len(fused_boxes), 2)
        
        # Check fused box coordinates (weighted average)
        fused_text_box = [b for b in fused_boxes if b.class_name == 'text'][0]
        self.assertTrue(50 <= fused_text_box.x1 <= 55)
        self.assertTrue(200 <= fused_text_box.x2 <= 205)


class TestDataPreparation(unittest.TestCase):
    """Test data preparation and augmentation"""
    
    def test_document_augmentation(self):
        """Test augmentation pipeline"""
        from prepare_datasets import DocumentAugmentation
        
        # Get transforms
        train_transform = DocumentAugmentation.get_train_transforms(640)
        val_transform = DocumentAugmentation.get_val_transforms(640)
        
        # Create test image and annotations
        image = np.ones((800, 600, 3), dtype=np.uint8) * 255
        bboxes = [[50, 50, 200, 200], [300, 300, 500, 500]]
        class_labels = [0, 1]
        
        # Apply train transform
        augmented = train_transform(
            image=image,
            bboxes=bboxes,
            class_labels=class_labels
        )
        
        # Check output
        self.assertEqual(augmented['image'].shape, (3, 640, 640))
        self.assertEqual(len(augmented['bboxes']), len(bboxes))
        self.assertEqual(augmented['class_labels'], class_labels)
    
    def test_synthetic_data_generation(self):
        """Test synthetic data generator"""
        from generate_synthetic_data import AcademicDocumentGenerator
        
        with tempfile.TemporaryDirectory() as tmpdir:
            generator = AcademicDocumentGenerator(tmpdir, num_samples=2)
            
            # Generate single document
            doc_image, doc_annotations = generator._generate_document(0)
            
            # Check output
            self.assertIsInstance(doc_image, np.ndarray)
            self.assertEqual(len(doc_image.shape), 3)  # H, W, C
            self.assertIsInstance(doc_annotations, list)
            self.assertTrue(len(doc_annotations) > 0)
            
            # Check annotation format
            for ann in doc_annotations:
                self.assertIn('bbox', ann)
                self.assertIn('category', ann)
                self.assertIn('confidence', ann)
                self.assertEqual(len(ann['bbox']), 4)


class TestAPI(unittest.TestCase):
    """Test REST API endpoints"""
    
    @classmethod
    def setUpClass(cls):
        """Set up API test client"""
        cls.api_url = os.getenv("TEST_API_URL", "http://localhost:8000")
        cls.test_image_path = cls._create_test_image_file()
    
    @classmethod
    def _create_test_image_file(cls) -> str:
        """Create and save test image"""
        img = np.ones((640, 480, 3), dtype=np.uint8) * 255
        cv2.rectangle(img, (50, 50), (200, 150), (0, 0, 0), -1)
        
        temp_file = tempfile.NamedTemporaryFile(suffix='.jpg', delete=False)
        cv2.imwrite(temp_file.name, img)
        return temp_file.name
    
    @classmethod
    def tearDownClass(cls):
        """Clean up test files"""
        if os.path.exists(cls.test_image_path):
            os.unlink(cls.test_image_path)
    
    def test_health_endpoint(self):
        """Test health check endpoint"""
        response = requests.get(f"{self.api_url}/health")
        
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data['status'], 'healthy')
        self.assertIn('model_loaded', data)
    
    def test_authentication(self):
        """Test authentication flow"""
        # Get token
        response = requests.post(
            f"{self.api_url}/auth/token",
            data={"username": "demo", "password": "demo123"}
        )
        
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn('access_token', data)
        self.assertEqual(data['token_type'], 'bearer')
    
    @pytest.mark.skipif(not os.getenv("RUN_INTEGRATION_TESTS"), 
                       reason="Integration tests disabled")
    def test_detection_endpoint(self):
        """Test detection endpoint"""
        # Get auth token
        auth_response = requests.post(
            f"{self.api_url}/auth/token",
            data={"username": "demo", "password": "demo123"}
        )
        token = auth_response.json()['access_token']
        
        # Test detection
        with open(self.test_image_path, 'rb') as f:
            response = requests.post(
                f"{self.api_url}/api/v1/detect",
                files={'file': f},
                headers={'Authorization': f'Bearer {token}'},
                params={'confidence_threshold': 0.5}
            )
        
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertTrue(data['success'])
        self.assertIn('detections', data)
        self.assertIn('processing_time', data)
    
    def test_supported_classes_endpoint(self):
        """Test supported classes endpoint"""
        response = requests.get(f"{self.api_url}/api/v1/classes")
        
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn('total_classes', data)
        self.assertIn('classes', data)
        self.assertTrue(len(data['classes']) > 0)


class TestPerformance(unittest.TestCase):
    """Performance and load tests"""
    
    @classmethod
    def setUpClass(cls):
        """Set up performance test environment"""
        cls.detector = EnhancedContentDetector()
        cls.test_images = cls._generate_test_images()
    
    @classmethod
    def _generate_test_images(cls, num_images: int = 10) -> List[np.ndarray]:
        """Generate multiple test images"""
        images = []
        for i in range(num_images):
            size = (640 + i * 10, 480 + i * 10)
            img = np.random.randint(0, 255, (*size, 3), dtype=np.uint8)
            images.append(img)
        return images
    
    def test_inference_speed(self):
        """Test inference speed"""
        times = []
        
        for img in self.test_images:
            start_time = time.time()
            _ = self.detector.detect(img, use_ensemble=False)
            inference_time = time.time() - start_time
            times.append(inference_time)
        
        avg_time = np.mean(times)
        fps = 1.0 / avg_time
        
        print(f"\nInference Performance:")
        print(f"  Average time: {avg_time:.3f}s")
        print(f"  FPS: {fps:.1f}")
        print(f"  Min time: {np.min(times):.3f}s")
        print(f"  Max time: {np.max(times):.3f}s")
        
        # Assert reasonable performance
        self.assertLess(avg_time, 1.0)  # Should be faster than 1 second
    
    def test_memory_usage(self):
        """Test memory usage"""
        import psutil
        process = psutil.Process()
        
        # Initial memory
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Process multiple images
        for _ in range(10):
            img = np.random.randint(0, 255, (1024, 1024, 3), dtype=np.uint8)
            _ = self.detector.detect(img)
        
        # Final memory
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory
        
        print(f"\nMemory Usage:")
        print(f"  Initial: {initial_memory:.1f} MB")
        print(f"  Final: {final_memory:.1f} MB")
        print(f"  Increase: {memory_increase:.1f} MB")
        
        # Assert no major memory leak
        self.assertLess(memory_increase, 500)  # Less than 500MB increase
    
    @pytest.mark.asyncio
    async def test_concurrent_requests(self):
        """Test concurrent API requests"""
        api_url = os.getenv("TEST_API_URL", "http://localhost:8000")
        num_concurrent = 10
        
        async def make_request(session, image_data):
            async with session.post(
                f"{api_url}/api/v1/detect",
                data={'file': image_data}
            ) as response:
                return await response.json()
        
        # Create test image data
        img = np.ones((640, 480, 3), dtype=np.uint8) * 255
        _, buffer = cv2.imencode('.jpg', img)
        image_data = buffer.tobytes()
        
        # Make concurrent requests
        async with aiohttp.ClientSession() as session:
            tasks = [make_request(session, image_data) 
                    for _ in range(num_concurrent)]
            
            start_time = time.time()
            results = await asyncio.gather(*tasks)
            total_time = time.time() - start_time
        
        print(f"\nConcurrent Request Performance:")
        print(f"  Requests: {num_concurrent}")
        print(f"  Total time: {total_time:.2f}s")
        print(f"  Requests/second: {num_concurrent / total_time:.1f}")
        
        # Check all requests succeeded
        for result in results:
            self.assertTrue(result.get('success', False))


class TestOptimization(unittest.TestCase):
    """Test model optimization techniques"""
    
    def test_quantization(self):
        """Test INT8 quantization"""
        from optimize_model import ModelOptimizer
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create dummy model
            model = torch.nn.Sequential(
                torch.nn.Conv2d(3, 64, 3),
                torch.nn.ReLU(),
                torch.nn.Conv2d(64, 128, 3)
            )
            
            # Save model
            model_path = os.path.join(tmpdir, 'model.pth')
            torch.save(model.state_dict(), model_path)
            
            # Test quantization
            optimizer = ModelOptimizer(model_path)
            optimizer.detector.model = model
            
            # Benchmark original
            original_results = optimizer.benchmark_model(model, (3, 224, 224))
            
            # Quantize
            quantized_model = optimizer.quantize_model()
            
            # Benchmark quantized
            quantized_results = optimizer.benchmark_model(
                quantized_model, (3, 224, 224)
            )
            
            # Check size reduction
            self.assertLess(
                quantized_results['model_size_mb'],
                original_results['model_size_mb']
            )
    
    def test_onnx_export(self):
        """Test ONNX export"""
        import onnx
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create simple model
            model = torch.nn.Conv2d(3, 64, 3)
            
            # Export to ONNX
            dummy_input = torch.randn(1, 3, 224, 224)
            onnx_path = os.path.join(tmpdir, 'model.onnx')
            
            torch.onnx.export(
                model,
                dummy_input,
                onnx_path,
                opset_version=13
            )
            
            # Verify ONNX model
            onnx_model = onnx.load(onnx_path)
            onnx.checker.check_model(onnx_model)
            
            self.assertTrue(os.path.exists(onnx_path))


class TestIntegration(unittest.TestCase):
    """End-to-end integration tests"""
    
    def test_full_pipeline(self):
        """Test complete detection pipeline"""
        # Create test document
        doc_image = self._create_complex_document()
        
        # Initialize detector
        detector = EnhancedContentDetector()
        
        # Detect elements
        boxes = detector.detect(doc_image, use_ensemble=False)
        
        # Verify results
        self.assertIsInstance(boxes, list)
        
        # Check different element types detected
        detected_classes = set(box.class_name for box in boxes)
        
        print(f"\nDetected classes: {detected_classes}")
        print(f"Total detections: {len(boxes)}")
    
    def _create_complex_document(self) -> np.ndarray:
        """Create a complex test document"""
        # Create white background
        img = np.ones((1200, 900, 3), dtype=np.uint8) * 255
        
        # Add title
        cv2.rectangle(img, (200, 50), (700, 100), (0, 0, 0), 2)
        cv2.putText(img, "Document Title", (250, 80), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        
        # Add text blocks
        for i in range(3):
            y = 150 + i * 100
            cv2.rectangle(img, (50, y), (850, y + 60), (0, 0, 0), 1)
        
        # Add table
        table_x, table_y = 100, 500
        for i in range(4):
            for j in range(3):
                x = table_x + j * 150
                y = table_y + i * 40
                cv2.rectangle(img, (x, y), (x + 140, y + 35), (0, 0, 0), 1)
        
        # Add figure
        cv2.rectangle(img, (500, 700), (800, 1000), (0, 0, 0), 2)
        cv2.circle(img, (650, 850), 50, (0, 0, 0), -1)
        
        return img


def run_all_tests():
    """Run all test suites"""
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test classes
    test_classes = [
        TestEnhancedContentDetector,
        TestDataPreparation,
        TestAPI,
        TestPerformance,
        TestOptimization,
        TestIntegration
    ]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # Print summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success rate: {(result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100:.1f}%")
    
    return result.wasSuccessful()


if __name__ == '__main__':
    # Run all tests
    success = run_all_tests()
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)