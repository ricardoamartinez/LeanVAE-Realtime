#!/usr/bin/env python3
"""
Comprehensive test runner for Active Inference Agent

Features:
- Runs all unit tests
- Generates coverage reports
- Performance profiling
- Memory leak detection
- Creates test report summary
"""

import sys
import os
import unittest
import time
import psutil
import torch
import numpy as np
from datetime import datetime
import json

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


class TestSuiteRunner:
    """Comprehensive test suite runner with reporting"""
    
    def __init__(self):
        self.results = {
            'timestamp': datetime.now().isoformat(),
            'system_info': self.get_system_info(),
            'test_results': {},
            'performance_metrics': {},
            'memory_metrics': {},
            'issues_found': []
        }
        
    def get_system_info(self):
        """Gather system information"""
        return {
            'python_version': sys.version,
            'torch_version': torch.__version__,
            'cuda_available': torch.cuda.is_available(),
            'cuda_version': torch.version.cuda if torch.cuda.is_available() else None,
            'cpu_count': psutil.cpu_count(),
            'memory_gb': psutil.virtual_memory().total / (1024**3)
        }
        
    def run_unit_tests(self):
        """Run all unit tests"""
        print("=" * 80)
        print("RUNNING UNIT TESTS")
        print("=" * 80)
        
        # Discover all test modules
        loader = unittest.TestLoader()
        start_dir = os.path.dirname(__file__)
        suite = loader.discover(start_dir, pattern='test_*.py')
        
        # Run tests with custom result collector
        runner = unittest.TextTestRunner(verbosity=2)
        result = runner.run(suite)
        
        # Store results
        self.results['test_results'] = {
            'total_tests': result.testsRun,
            'failures': len(result.failures),
            'errors': len(result.errors),
            'skipped': len(result.skipped),
            'success_rate': (result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun if result.testsRun > 0 else 0
        }
        
        # Log failures and errors
        for test, traceback in result.failures + result.errors:
            self.results['issues_found'].append({
                'test': str(test),
                'type': 'failure' if (test, traceback) in result.failures else 'error',
                'traceback': traceback
            })
            
        return result.wasSuccessful()
        
    def run_performance_tests(self):
        """Run performance benchmarks"""
        print("\n" + "=" * 80)
        print("RUNNING PERFORMANCE TESTS")
        print("=" * 80)
        
        from active_inference_agent import (
            PhysicsInformedWaveletTransform,
            TemporalKoopmanOperator,
            MicroLeanVAE
        )
        
        metrics = {}
        
        # Test wavelet transform performance
        print("\n1. Testing Wavelet Transform Performance...")
        wavelet = PhysicsInformedWaveletTransform(input_size=128)
        x = torch.randn(8, 3, 128, 128)
        
        # Warm up
        _ = wavelet.forward_2d_spatial_wavelet(x)
        
        # Time multiple runs
        times = []
        for _ in range(10):
            start = time.time()
            _ = wavelet.forward_2d_spatial_wavelet(x)
            times.append(time.time() - start)
            
        metrics['wavelet_transform'] = {
            'mean_time_ms': np.mean(times) * 1000,
            'std_time_ms': np.std(times) * 1000,
            'min_time_ms': np.min(times) * 1000,
            'max_time_ms': np.max(times) * 1000
        }
        print(f"   Average time: {metrics['wavelet_transform']['mean_time_ms']:.2f}ms")
        
        # Test VAE inference performance
        print("\n2. Testing VAE Inference Performance...")
        vae = MicroLeanVAE(input_size=64, latent_dim=32)
        vae.eval()
        x_vae = torch.randn(4, 3, 64, 64)
        
        # Warm up
        with torch.no_grad():
            _ = vae(x_vae)
            
        # Time inference
        times = []
        for _ in range(20):
            start = time.time()
            with torch.no_grad():
                _ = vae(x_vae)
            times.append(time.time() - start)
            
        metrics['vae_inference'] = {
            'mean_time_ms': np.mean(times) * 1000,
            'std_time_ms': np.std(times) * 1000,
            'fps_potential': 1000 / (np.mean(times) * 1000)
        }
        print(f"   Average time: {metrics['vae_inference']['mean_time_ms']:.2f}ms")
        print(f"   Potential FPS: {metrics['vae_inference']['fps_potential']:.1f}")
        
        self.results['performance_metrics'] = metrics
        
    def run_memory_tests(self):
        """Test for memory leaks"""
        print("\n" + "=" * 80)
        print("RUNNING MEMORY TESTS")
        print("=" * 80)
        
        from active_inference_agent import MicroLeanVAE
        
        process = psutil.Process(os.getpid())
        
        # Initial memory
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Create and destroy multiple models
        print("\n1. Testing model creation/destruction...")
        for i in range(5):
            vae = MicroLeanVAE(input_size=64, latent_dim=32)
            x = torch.randn(4, 3, 64, 64)
            _ = vae(x)
            del vae
            
        # Force garbage collection
        import gc
        gc.collect()
        
        mid_memory = process.memory_info().rss / 1024 / 1024
        
        # Run intensive operations
        print("\n2. Testing intensive operations...")
        vae = MicroLeanVAE(input_size=128, latent_dim=64)
        for i in range(10):
            x = torch.randn(8, 3, 128, 128)
            _ = vae(x)
            
        final_memory = process.memory_info().rss / 1024 / 1024
        
        self.results['memory_metrics'] = {
            'initial_mb': initial_memory,
            'after_creation_mb': mid_memory,
            'after_intensive_mb': final_memory,
            'total_increase_mb': final_memory - initial_memory,
            'potential_leak': (final_memory - initial_memory) > 100  # Flag if > 100MB increase
        }
        
        print(f"\n   Memory increase: {final_memory - initial_memory:.1f}MB")
        if self.results['memory_metrics']['potential_leak']:
            print("   ⚠️  WARNING: Potential memory leak detected!")
            self.results['issues_found'].append({
                'type': 'memory_leak',
                'description': f'Memory increased by {final_memory - initial_memory:.1f}MB'
            })
            
    def run_integration_tests(self):
        """Run basic integration tests"""
        print("\n" + "=" * 80)
        print("RUNNING INTEGRATION TESTS")
        print("=" * 80)
        
        from active_inference_agent import UltraFast60FpsLeanVAE
        
        print("\n1. Testing full inference pipeline...")
        
        # Create processor
        processor = UltraFast60FpsLeanVAE(device='cpu', input_resolution=(256, 256))
        
        # Simulate frame processing
        frame = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        
        try:
            reconstruction, inference_time, motion_score = processor.process_frame_ultra_fast(frame)
            
            if reconstruction is not None:
                print("   ✓ Inference pipeline successful")
                print(f"   Inference time: {inference_time*1000:.2f}ms")
            else:
                print("   ✗ Inference pipeline failed")
                self.results['issues_found'].append({
                    'type': 'integration_failure',
                    'description': 'Inference pipeline returned None'
                })
                
        except Exception as e:
            print(f"   ✗ Integration test failed: {e}")
            self.results['issues_found'].append({
                'type': 'integration_error',
                'description': str(e)
            })
        finally:
            processor.cleanup()
            
    def generate_report(self):
        """Generate comprehensive test report"""
        print("\n" + "=" * 80)
        print("TEST REPORT SUMMARY")
        print("=" * 80)
        
        # Save detailed JSON report
        report_path = os.path.join(os.path.dirname(__file__), '..', 'test_report.json')
        with open(report_path, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        print(f"\nDetailed report saved to: {report_path}")
        
        # Print summary
        print("\n📊 Test Results:")
        if 'test_results' in self.results:
            tr = self.results['test_results']
            print(f"   Total tests: {tr['total_tests']}")
            print(f"   Success rate: {tr['success_rate']*100:.1f}%")
            print(f"   Failures: {tr['failures']}")
            print(f"   Errors: {tr['errors']}")
            
        print("\n⚡ Performance Metrics:")
        if 'performance_metrics' in self.results:
            pm = self.results['performance_metrics']
            if 'wavelet_transform' in pm:
                print(f"   Wavelet transform: {pm['wavelet_transform']['mean_time_ms']:.2f}ms")
            if 'vae_inference' in pm:
                print(f"   VAE inference: {pm['vae_inference']['mean_time_ms']:.2f}ms")
                print(f"   Potential FPS: {pm['vae_inference']['fps_potential']:.1f}")
                
        print("\n💾 Memory Usage:")
        if 'memory_metrics' in self.results:
            mm = self.results['memory_metrics']
            print(f"   Total increase: {mm['total_increase_mb']:.1f}MB")
            if mm['potential_leak']:
                print("   ⚠️  Potential memory leak detected!")
                
        print(f"\n🐛 Issues Found: {len(self.results['issues_found'])}")
        for issue in self.results['issues_found'][:5]:  # Show first 5 issues
            print(f"   - {issue.get('type', 'unknown')}: {issue.get('description', issue.get('test', 'N/A'))}")
            
        if len(self.results['issues_found']) > 5:
            print(f"   ... and {len(self.results['issues_found']) - 5} more")
            
        # Overall status
        print("\n" + "=" * 80)
        if len(self.results['issues_found']) == 0:
            print("✅ ALL TESTS PASSED!")
        else:
            print(f"❌ TESTS COMPLETED WITH {len(self.results['issues_found'])} ISSUES")
        print("=" * 80)
        
        return len(self.results['issues_found']) == 0


def main():
    """Main test runner"""
    runner = TestSuiteRunner()
    
    # Run all test suites
    unit_success = runner.run_unit_tests()
    runner.run_performance_tests()
    runner.run_memory_tests()
    runner.run_integration_tests()
    
    # Generate report
    overall_success = runner.generate_report()
    
    # Exit with appropriate code
    sys.exit(0 if overall_success else 1)


if __name__ == '__main__':
    main()