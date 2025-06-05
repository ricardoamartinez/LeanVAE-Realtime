import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from realtime_leanvae_ultra_fast_60fps import ImmediateFrequencyAdaptiveWavelet, MicroLeanVAE

def debug_wavelet_pipeline():
    """Debug the wavelet reconstruction pipeline step by step"""
    print("🔍 DEBUGGING WAVELET RECONSTRUCTION PIPELINE")
    
    # Create a simple test image
    test_image = np.random.rand(240, 240, 3) * 255
    test_image = test_image.astype(np.uint8)
    print(f"✅ Test image created: {test_image.shape}, range: {test_image.min()}-{test_image.max()}")
    
    # Convert to tensor
    tensor_input = torch.tensor(test_image).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    print(f"✅ Tensor input: {tensor_input.shape}, range: {tensor_input.min():.3f}-{tensor_input.max():.3f}")
    
    # Test wavelet transform
    wavelet_transform = ImmediateFrequencyAdaptiveWavelet(input_size=240, n_channels=3)
    
    print("\n🌊 Testing forward wavelet transform...")
    try:
        wavelet_coeffs = wavelet_transform.forward_2d_wavelet(tensor_input)
        print(f"✅ Wavelet coefficients: {wavelet_coeffs.shape}")
        print(f"   Range: {wavelet_coeffs.min():.3f} to {wavelet_coeffs.max():.3f}")
        print(f"   Mean: {wavelet_coeffs.mean():.3f}, Std: {wavelet_coeffs.std():.3f}")
        
        # Check for NaN or inf
        if torch.isnan(wavelet_coeffs).any():
            print("❌ NaN detected in wavelet coefficients!")
        if torch.isinf(wavelet_coeffs).any():
            print("❌ Inf detected in wavelet coefficients!")
            
    except Exception as e:
        print(f"❌ Forward wavelet failed: {e}")
        return
    
    print("\n🔄 Testing inverse wavelet transform...")
    try:
        reconstructed = wavelet_transform.inverse_2d_wavelet(wavelet_coeffs, 240, 240)
        print(f"✅ Reconstructed: {reconstructed.shape}")
        print(f"   Range: {reconstructed.min():.3f} to {reconstructed.max():.3f}")
        print(f"   Mean: {reconstructed.mean():.3f}, Std: {reconstructed.std():.3f}")
        
        # Check for NaN or inf
        if torch.isnan(reconstructed).any():
            print("❌ NaN detected in reconstruction!")
        if torch.isinf(reconstructed).any():
            print("❌ Inf detected in reconstruction!")
            
    except Exception as e:
        print(f"❌ Inverse wavelet failed: {e}")
        return
    
    print("\n🎯 Testing sigmoid activation...")
    try:
        sigmoid_output = torch.sigmoid(reconstructed)
        print(f"✅ Sigmoid output: {sigmoid_output.shape}")
        print(f"   Range: {sigmoid_output.min():.3f} to {sigmoid_output.max():.3f}")
        print(f"   Mean: {sigmoid_output.mean():.3f}, Std: {sigmoid_output.std():.3f}")
        
        # Check for NaN or inf
        if torch.isnan(sigmoid_output).any():
            print("❌ NaN detected in sigmoid output!")
        if torch.isinf(sigmoid_output).any():
            print("❌ Inf detected in sigmoid output!")
            
    except Exception as e:
        print(f"❌ Sigmoid failed: {e}")
        return
    
    print("\n🖼️ Testing image conversion...")
    try:
        # Convert back to image
        recon_np = sigmoid_output.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
        print(f"✅ Numpy conversion: {recon_np.shape}")
        print(f"   Range: {recon_np.min():.3f} to {recon_np.max():.3f}")
        
        # Apply nan_to_num
        recon_np = np.nan_to_num(recon_np, nan=0.0, posinf=1.0, neginf=0.0)
        print(f"✅ After nan_to_num: range {recon_np.min():.3f} to {recon_np.max():.3f}")
        
        # Scale to 0-255
        recon_scaled = (recon_np * 255).clip(0, 255).astype(np.uint8)
        print(f"✅ Final image: {recon_scaled.shape}, range: {recon_scaled.min()}-{recon_scaled.max()}")
        
        # Check if image is all black
        if recon_scaled.max() == 0:
            print("❌ PROBLEM FOUND: Output image is all black!")
        else:
            print("✅ Image has non-zero values - looks good!")
            
    except Exception as e:
        print(f"❌ Image conversion failed: {e}")
        return
    
    print("\n🧠 Testing full model...")
    try:
        model = MicroLeanVAE(input_size=240, latent_dim=32)
        model.eval()
        
        with torch.no_grad():
            reconstructed, mean, logvar = model(tensor_input)
            
        print(f"✅ Model output: {reconstructed.shape}")
        print(f"   Range: {reconstructed.min():.3f} to {reconstructed.max():.3f}")
        print(f"   Mean: {reconstructed.mean():.3f}, Std: {reconstructed.std():.3f}")
        
        # Check for issues
        if torch.isnan(reconstructed).any():
            print("❌ NaN detected in model output!")
        if torch.isinf(reconstructed).any():
            print("❌ Inf detected in model output!")
        if reconstructed.max() < 0.01:
            print("❌ PROBLEM: Model output is essentially zero (all black)!")
            print("   This suggests the model is not properly initialized or trained")
        else:
            print("✅ Model output looks reasonable!")
            
    except Exception as e:
        print(f"❌ Model test failed: {e}")
        return

if __name__ == "__main__":
    debug_wavelet_pipeline()
