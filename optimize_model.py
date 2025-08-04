"""
Model Optimization Script for Production Deployment
Includes quantization, ONNX export, TensorRT optimization, and pruning
"""

import os
import argparse
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.quantization as quantization
from torch.utils.data import DataLoader
import onnx
import onnxruntime as ort
import tensorrt as trt
import cv2
from tqdm import tqdm
import json

from enhanced_content_detector import EnhancedContentDetector, BoundingBox


class ModelOptimizer:
    """Comprehensive model optimization for deployment"""
    
    def __init__(self, model_path: str, config_path: Optional[str] = None):
        self.model_path = model_path
        self.config_path = config_path
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load model
        self.detector = EnhancedContentDetector()
        self._load_model()
        
        # Optimization results
        self.optimization_results = {
            'original': {},
            'quantized': {},
            'onnx': {},
            'tensorrt': {},
            'pruned': {}
        }
    
    def _load_model(self):
        """Load the original model"""
        checkpoint = torch.load(self.model_path, map_location=self.device)
        
        if 'model_state_dict' in checkpoint:
            self.detector.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.detector.model.load_state_dict(checkpoint)
        
        self.detector.model.eval()
        print(f"Model loaded from {self.model_path}")
    
    def benchmark_model(self, model, input_shape: Tuple[int, ...], 
                       num_runs: int = 100, warmup_runs: int = 10) -> Dict:
        """Benchmark model performance"""
        # Create dummy input
        dummy_input = torch.randn(1, *input_shape).to(self.device)
        
        # Warmup
        for _ in range(warmup_runs):
            with torch.no_grad():
                _ = model(dummy_input)
        
        # Benchmark
        torch.cuda.synchronize()
        start_time = time.time()
        
        for _ in range(num_runs):
            with torch.no_grad():
                _ = model(dummy_input)
        
        torch.cuda.synchronize()
        end_time = time.time()
        
        # Calculate metrics
        avg_time = (end_time - start_time) / num_runs
        fps = 1.0 / avg_time
        
        # Get model size
        model_size = self._get_model_size(model)
        
        return {
            'avg_inference_time': avg_time,
            'fps': fps,
            'model_size_mb': model_size,
            'memory_usage_mb': torch.cuda.max_memory_allocated() / 1024 / 1024
        }
    
    def _get_model_size(self, model) -> float:
        """Get model size in MB"""
        param_size = 0
        buffer_size = 0
        
        for param in model.parameters():
            param_size += param.nelement() * param.element_size()
        
        for buffer in model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()
        
        size_mb = (param_size + buffer_size) / 1024 / 1024
        return size_mb
    
    def quantize_model(self, calibration_data: Optional[DataLoader] = None) -> nn.Module:
        """Apply INT8 quantization to the model"""
        print("\nApplying INT8 Quantization...")
        
        # Prepare model for quantization
        model_fp32 = self.detector.model.cpu()
        model_fp32.eval()
        
        # Configure quantization
        model_fp32.qconfig = quantization.get_default_qconfig('qnnpack')
        
        # Prepare model
        model_prepared = quantization.prepare(model_fp32)
        
        # Calibrate with representative data if provided
        if calibration_data:
            print("Calibrating with representative data...")
            with torch.no_grad():
                for batch in tqdm(calibration_data, desc='Calibration'):
                    model_prepared(batch['images'].cpu())
        
        # Convert to quantized model
        model_int8 = quantization.convert(model_prepared)
        
        # Save quantized model
        output_path = Path(self.model_path).parent / 'model_quantized.pth'
        torch.save({
            'model_state_dict': model_int8.state_dict(),
            'model_config': self.detector.config
        }, output_path)
        
        print(f"Quantized model saved to {output_path}")
        
        # Benchmark quantized model
        self.optimization_results['quantized'] = self.benchmark_model(
            model_int8, (3, 640, 640)
        )
        
        return model_int8
    
    def export_onnx(self, input_shape: Tuple[int, ...] = (3, 640, 640),
                    opset_version: int = 13,
                    dynamic_axes: Optional[Dict] = None) -> str:
        """Export model to ONNX format"""
        print("\nExporting to ONNX...")
        
        # Prepare model
        model = self.detector.model.eval()
        
        # Create dummy input
        dummy_input = torch.randn(1, *input_shape).to(self.device)
        
        # Output path
        output_path = Path(self.model_path).parent / 'model.onnx'
        
        # Dynamic axes for variable batch size and image dimensions
        if dynamic_axes is None:
            dynamic_axes = {
                'input': {0: 'batch_size', 2: 'height', 3: 'width'},
                'output': {0: 'batch_size'}
            }
        
        # Export
        torch.onnx.export(
            model,
            dummy_input,
            output_path,
            export_params=True,
            opset_version=opset_version,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes=dynamic_axes,
            verbose=False
        )
        
        # Verify ONNX model
        onnx_model = onnx.load(str(output_path))
        onnx.checker.check_model(onnx_model)
        
        print(f"ONNX model exported to {output_path}")
        
        # Benchmark ONNX model
        self._benchmark_onnx(str(output_path), input_shape)
        
        return str(output_path)
    
    def _benchmark_onnx(self, onnx_path: str, input_shape: Tuple[int, ...]):
        """Benchmark ONNX model with ONNXRuntime"""
        # Create ONNX Runtime session
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        session = ort.InferenceSession(onnx_path, providers=providers)
        
        # Prepare input
        input_name = session.get_inputs()[0].name
        dummy_input = np.random.randn(1, *input_shape).astype(np.float32)
        
        # Warmup
        for _ in range(10):
            _ = session.run(None, {input_name: dummy_input})
        
        # Benchmark
        start_time = time.time()
        num_runs = 100
        
        for _ in range(num_runs):
            _ = session.run(None, {input_name: dummy_input})
        
        end_time = time.time()
        
        avg_time = (end_time - start_time) / num_runs
        fps = 1.0 / avg_time
        
        # Get model size
        model_size = os.path.getsize(onnx_path) / 1024 / 1024
        
        self.optimization_results['onnx'] = {
            'avg_inference_time': avg_time,
            'fps': fps,
            'model_size_mb': model_size
        }
    
    def optimize_tensorrt(self, onnx_path: str,
                         precision: str = 'FP16',
                         max_batch_size: int = 1,
                         workspace_size: int = 1 << 30) -> str:
        """Optimize model with TensorRT"""
        print(f"\nOptimizing with TensorRT ({precision} precision)...")
        
        # Create logger and builder
        logger = trt.Logger(trt.Logger.WARNING)
        builder = trt.Builder(logger)
        
        # Create network
        network = builder.create_network(
            1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        )
        
        # Parse ONNX
        parser = trt.OnnxParser(network, logger)
        
        with open(onnx_path, 'rb') as f:
            if not parser.parse(f.read()):
                print("Failed to parse ONNX file")
                for error in range(parser.num_errors):
                    print(parser.get_error(error))
                return None
        
        # Configure builder
        config = builder.create_builder_config()
        config.max_workspace_size = workspace_size
        
        # Set precision
        if precision == 'FP16':
            config.set_flag(trt.BuilderFlag.FP16)
        elif precision == 'INT8':
            config.set_flag(trt.BuilderFlag.INT8)
            # Would need calibration data for INT8
        
        # Set optimization profile for dynamic shapes
        profile = builder.create_optimization_profile()
        
        # Get input shape from network
        input_tensor = network.get_input(0)
        input_name = input_tensor.name
        
        # Set dynamic shape ranges
        min_shape = (1, 3, 320, 320)
        opt_shape = (1, 3, 640, 640)
        max_shape = (max_batch_size, 3, 1280, 1280)
        
        profile.set_shape(input_name, min_shape, opt_shape, max_shape)
        config.add_optimization_profile(profile)
        
        # Build engine
        print("Building TensorRT engine... (this may take a while)")
        engine = builder.build_engine(network, config)
        
        if engine is None:
            print("Failed to build TensorRT engine")
            return None
        
        # Save engine
        output_path = Path(self.model_path).parent / f'model_{precision.lower()}.trt'
        
        with open(output_path, 'wb') as f:
            f.write(engine.serialize())
        
        print(f"TensorRT engine saved to {output_path}")
        
        # Benchmark TensorRT engine
        self._benchmark_tensorrt(engine, opt_shape)
        
        return str(output_path)
    
    def _benchmark_tensorrt(self, engine, input_shape: Tuple[int, ...]):
        """Benchmark TensorRT engine"""
        # Create execution context
        context = engine.create_execution_context()
        
        # Allocate buffers
        inputs, outputs, bindings, stream = self._allocate_buffers(engine)
        
        # Set input shape for dynamic engines
        context.set_binding_shape(0, (1, *input_shape))
        
        # Prepare input data
        input_data = np.random.randn(1, *input_shape).astype(np.float32)
        np.copyto(inputs[0].host, input_data.ravel())
        
        # Warmup
        for _ in range(10):
            self._do_inference(context, bindings, inputs, outputs, stream)
        
        # Benchmark
        start_time = time.time()
        num_runs = 100
        
        for _ in range(num_runs):
            self._do_inference(context, bindings, inputs, outputs, stream)
        
        end_time = time.time()
        
        avg_time = (end_time - start_time) / num_runs
        fps = 1.0 / avg_time
        
        self.optimization_results['tensorrt'] = {
            'avg_inference_time': avg_time,
            'fps': fps,
            'precision': 'FP16'  # or INT8
        }
    
    def _allocate_buffers(self, engine):
        """Allocate buffers for TensorRT inference"""
        class HostDeviceMem:
            def __init__(self, host_mem, device_mem):
                self.host = host_mem
                self.device = device_mem
        
        inputs = []
        outputs = []
        bindings = []
        stream = cuda.Stream()
        
        for binding in engine:
            binding_idx = engine.get_binding_index(binding)
            size = trt.volume(engine.get_binding_shape(binding_idx))
            dtype = trt.nptype(engine.get_binding_dtype(binding))
            
            # Allocate host and device buffers
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            
            # Append the device buffer to device bindings
            bindings.append(int(device_mem))
            
            # Append to the appropriate list
            if engine.binding_is_input(binding_idx):
                inputs.append(HostDeviceMem(host_mem, device_mem))
            else:
                outputs.append(HostDeviceMem(host_mem, device_mem))
        
        return inputs, outputs, bindings, stream
    
    def _do_inference(self, context, bindings, inputs, outputs, stream):
        """Run inference on TensorRT engine"""
        # Transfer input data to the GPU
        [cuda.memcpy_htod_async(inp.device, inp.host, stream) for inp in inputs]
        
        # Run inference
        context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
        
        # Transfer predictions back from the GPU
        [cuda.memcpy_dtoh_async(out.host, out.device, stream) for out in outputs]
        
        # Synchronize the stream
        stream.synchronize()
        
        return [out.host for out in outputs]
    
    def prune_model(self, pruning_rate: float = 0.3) -> nn.Module:
        """Apply structured pruning to reduce model size"""
        print(f"\nApplying Structured Pruning (rate={pruning_rate})...")
        
        import torch.nn.utils.prune as prune
        
        model = self.detector.model
        
        # Get all Conv2d and Linear layers
        parameters_to_prune = []
        for name, module in model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                parameters_to_prune.append((module, 'weight'))
        
        # Apply structured pruning
        for module, param_name in parameters_to_prune:
            prune.ln_structured(
                module, 
                name=param_name, 
                amount=pruning_rate, 
                n=2, 
                dim=0
            )
            
            # Make pruning permanent
            prune.remove(module, param_name)
        
        # Count remaining parameters
        total_params = sum(p.numel() for p in model.parameters())
        nonzero_params = sum((p != 0).sum().item() for p in model.parameters())
        
        print(f"Pruning complete. Remaining parameters: {nonzero_params}/{total_params} "
              f"({100 * nonzero_params / total_params:.1f}%)")
        
        # Save pruned model
        output_path = Path(self.model_path).parent / 'model_pruned.pth'
        torch.save({
            'model_state_dict': model.state_dict(),
            'model_config': self.detector.config,
            'pruning_rate': pruning_rate
        }, output_path)
        
        print(f"Pruned model saved to {output_path}")
        
        # Benchmark pruned model
        self.optimization_results['pruned'] = self.benchmark_model(
            model, (3, 640, 640)
        )
        
        return model
    
    def optimize_for_mobile(self) -> str:
        """Optimize model for mobile deployment"""
        print("\nOptimizing for Mobile Deployment...")
        
        # Convert to TorchScript
        model = self.detector.model.eval()
        
        # Create example input
        example_input = torch.randn(1, 3, 640, 640).to(self.device)
        
        # Trace the model
        traced_model = torch.jit.trace(model, example_input)
        
        # Optimize for mobile
        from torch.utils.mobile_optimizer import optimize_for_mobile
        
        optimized_model = optimize_for_mobile(traced_model)
        
        # Save the model
        output_path = Path(self.model_path).parent / 'model_mobile.pt'
        optimized_model._save_for_lite_interpreter(str(output_path))
        
        print(f"Mobile-optimized model saved to {output_path}")
        
        # Get model size
        model_size = os.path.getsize(output_path) / 1024 / 1024
        
        self.optimization_results['mobile'] = {
            'model_size_mb': model_size,
            'format': 'TorchScript Mobile'
        }
        
        return str(output_path)
    
    def compare_optimizations(self, test_image_path: Optional[str] = None):
        """Compare all optimization methods"""
        print("\n" + "=" * 80)
        print("OPTIMIZATION COMPARISON")
        print("=" * 80)
        
        # Headers
        headers = ['Method', 'Model Size (MB)', 'Inference Time (ms)', 'FPS', 'Speedup']
        row_format = "{:<15} {:<18} {:<20} {:<10} {:<10}"
        
        print(row_format.format(*headers))
        print("-" * 80)
        
        # Original model baseline
        if 'original' not in self.optimization_results or not self.optimization_results['original']:
            self.optimization_results['original'] = self.benchmark_model(
                self.detector.model, (3, 640, 640)
            )
        
        baseline_time = self.optimization_results['original']['avg_inference_time']
        
        # Print results for each optimization
        for method, results in self.optimization_results.items():
            if results:
                size = f"{results.get('model_size_mb', 0):.2f}"
                time_ms = f"{results.get('avg_inference_time', 0) * 1000:.2f}"
                fps = f"{results.get('fps', 0):.1f}"
                
                if method == 'original':
                    speedup = "1.0x"
                else:
                    speedup = f"{baseline_time / results.get('avg_inference_time', 1):.2f}x"
                
                print(row_format.format(method.title(), size, time_ms, fps, speedup))
        
        print("=" * 80)
        
        # Test on actual image if provided
        if test_image_path and os.path.exists(test_image_path):
            self._test_on_image(test_image_path)
    
    def _test_on_image(self, image_path: str):
        """Test optimized models on a real image"""
        print(f"\nTesting on image: {image_path}")
        
        # Load image
        image = cv2.imread(image_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Test original model
        start_time = time.time()
        original_results = self.detector.detect(image_rgb)
        original_time = time.time() - start_time
        
        print(f"Original model: {len(original_results)} detections in {original_time:.3f}s")
        
        # Compare detection counts to ensure optimization didn't break the model
        # (Additional testing code would go here for each optimized version)
    
    def generate_optimization_report(self, output_path: str):
        """Generate detailed optimization report"""
        report = {
            'model_path': self.model_path,
            'optimization_results': self.optimization_results,
            'recommendations': self._get_recommendations()
        }
        
        # Save JSON report
        output_file = Path(output_path) / 'optimization_report.json'
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"\nOptimization report saved to {output_file}")
    
    def _get_recommendations(self) -> List[str]:
        """Get optimization recommendations based on results"""
        recommendations = []
        
        if self.optimization_results.get('quantized'):
            quantized_speedup = (
                self.optimization_results['original']['avg_inference_time'] /
                self.optimization_results['quantized']['avg_inference_time']
            )
            if quantized_speedup > 1.5:
                recommendations.append(
                    f"INT8 quantization provides {quantized_speedup:.1f}x speedup. "
                    "Recommended for CPU deployment."
                )
        
        if self.optimization_results.get('tensorrt'):
            trt_speedup = (
                self.optimization_results['original']['avg_inference_time'] /
                self.optimization_results['tensorrt']['avg_inference_time']
            )
            if trt_speedup > 2.0:
                recommendations.append(
                    f"TensorRT provides {trt_speedup:.1f}x speedup. "
                    "Highly recommended for NVIDIA GPU deployment."
                )
        
        if self.optimization_results.get('pruned'):
            size_reduction = (
                1 - self.optimization_results['pruned']['model_size_mb'] /
                self.optimization_results['original']['model_size_mb']
            ) * 100
            if size_reduction > 30:
                recommendations.append(
                    f"Pruning reduces model size by {size_reduction:.1f}%. "
                    "Consider fine-tuning after pruning to recover accuracy."
                )
        
        return recommendations


def main():
    parser = argparse.ArgumentParser(description='Optimize document detection model')
    parser.add_argument('--model', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--output', type=str, default='optimized_models',
                       help='Output directory for optimized models')
    
    # Optimization options
    parser.add_argument('--quantize', action='store_true',
                       help='Apply INT8 quantization')
    parser.add_argument('--onnx', action='store_true',
                       help='Export to ONNX format')
    parser.add_argument('--tensorrt', action='store_true',
                       help='Optimize with TensorRT')
    parser.add_argument('--prune', action='store_true',
                       help='Apply structured pruning')
    parser.add_argument('--mobile', action='store_true',
                       help='Optimize for mobile deployment')
    parser.add_argument('--all', action='store_true',
                       help='Apply all optimizations')
    
    # Additional options
    parser.add_argument('--pruning-rate', type=float, default=0.3,
                       help='Pruning rate (0-1)')
    parser.add_argument('--trt-precision', type=str, default='FP16',
                       choices=['FP32', 'FP16', 'INT8'],
                       help='TensorRT precision')
    parser.add_argument('--test-image', type=str, default=None,
                       help='Test image for comparison')
    parser.add_argument('--calibration-data', type=str, default=None,
                       help='Calibration dataset for INT8 quantization')
    
    args = parser.parse_args()
    
    # Initialize optimizer
    optimizer = ModelOptimizer(args.model)
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Apply optimizations
    if args.all:
        args.quantize = args.onnx = args.tensorrt = args.prune = args.mobile = True
    
    if args.quantize:
        # Load calibration data if provided
        calibration_loader = None
        if args.calibration_data:
            # Implement calibration data loading
            pass
        
        optimizer.quantize_model(calibration_loader)
    
    if args.prune:
        optimizer.prune_model(args.pruning_rate)
    
    if args.onnx:
        onnx_path = optimizer.export_onnx()
        
        if args.tensorrt and onnx_path:
            # Import pycuda here to avoid issues if not installed
            try:
                import pycuda.driver as cuda
                import pycuda.autoinit
                
                optimizer.optimize_tensorrt(
                    onnx_path,
                    precision=args.trt_precision
                )
            except ImportError:
                print("PyCUDA not installed. Skipping TensorRT optimization.")
    
    if args.mobile:
        optimizer.optimize_for_mobile()
    
    # Compare all optimizations
    optimizer.compare_optimizations(args.test_image)
    
    # Generate report
    optimizer.generate_optimization_report(args.output)


if __name__ == '__main__':
    main()