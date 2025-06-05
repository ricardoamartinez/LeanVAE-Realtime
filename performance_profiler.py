import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
from realtime_leanvae_ultra_fast_60fps import MicroLeanVAE, ImmediateFrequencyAdaptiveWavelet

def profile_inference_pipeline():
    """Profile each component of the inference pipeline to find bottlenecks"""
    print("🔍 PERFORMANCE PROFILER - Finding the slowest components")
    print("=" * 60)
    
    # Create test input (similar to real camera frame)
    test_frame = np.random.randint(0, 255, (240, 240, 3), dtype=np.uint8)
    
    # Initialize model
    model = MicroLeanVAE(input_size=240, latent_dim=32)
    model.eval()
    
    # Test multiple iterations for accurate timing
    iterations = 10
    print(f"Running {iterations} iterations for accurate timing...\n")
    
    total_times = {
        'preprocessing': [],
        'wavelet_forward': [],
        'encoder': [],
        'koopman': [],
        'decoder': [],
        'wavelet_inverse': [],
        'postprocessing': []
    }
    
    for i in range(iterations):
        print(f"Iteration {i+1}/{iterations}")
        
        # 1. PREPROCESSING
        start_time = time.time()
        frame_rgb = cv2.cvtColor(test_frame, cv2.COLOR_BGR2RGB)
        frame_tensor = torch.tensor(frame_rgb).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        preprocessing_time = time.time() - start_time
        total_times['preprocessing'].append(preprocessing_time)
        
        with torch.no_grad():
            # 2. WAVELET FORWARD TRANSFORM
            start_time = time.time()
            wavelet_coeffs = model.wavelet_transform.forward_2d_wavelet(frame_tensor)
            wavelet_forward_time = time.time() - start_time
            total_times['wavelet_forward'].append(wavelet_forward_time)
            
            # 3. ENCODER
            start_time = time.time()
            batch_size = wavelet_coeffs.shape[0]
            flattened = wavelet_coeffs.reshape(batch_size, -1)
            encoded = model.encoder(flattened)
            mean, logvar = torch.chunk(encoded, 2, dim=1)
            z = model.reparameterize(mean, logvar)
            encoder_time = time.time() - start_time
            total_times['encoder'].append(encoder_time)
            
            # 4. KOOPMAN OPERATOR (if previous latent exists)
            start_time = time.time()
            if model.prev_latent is not None:
                predicted_z = model.koopman_operator(z)
                temporal_prediction = model.temporal_predictor(torch.cat([z, model.prev_latent], dim=-1))
                z = 0.7 * z + 0.2 * predicted_z + 0.1 * temporal_prediction
            model.prev_latent = z.detach()
            koopman_time = time.time() - start_time
            total_times['koopman'].append(koopman_time)
            
            # 5. DECODER
            start_time = time.time()
            decoded = model.decoder(z)
            wavelet_coeffs_decoded = decoded.reshape(batch_size, 3, model.wavelet_features)
            decoder_time = time.time() - start_time
            total_times['decoder'].append(decoder_time)
            
            # 6. WAVELET INVERSE TRANSFORM
            start_time = time.time()
            reconstruction = model.wavelet_transform.inverse_2d_wavelet(
                wavelet_coeffs_decoded, 240, 240
            )
            reconstruction = torch.sigmoid(reconstruction)
            wavelet_inverse_time = time.time() - start_time
            total_times['wavelet_inverse'].append(wavelet_inverse_time)
        
        # 7. POSTPROCESSING
        start_time = time.time()
        recon_np = reconstruction.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
        recon_np = np.nan_to_num(recon_np, nan=0.0, posinf=1.0, neginf=0.0)
        recon_np = (recon_np * 255).clip(0, 255).astype(np.uint8)
        recon_upscaled = cv2.resize(recon_np, (256, 256), interpolation=cv2.INTER_LINEAR)
        recon_bgr = cv2.cvtColor(recon_upscaled, cv2.COLOR_RGB2BGR)
        postprocessing_time = time.time() - start_time
        total_times['postprocessing'].append(postprocessing_time)
    
    # Calculate averages and display results
    print("\n" + "=" * 60)
    print("📊 PERFORMANCE RESULTS (Average over {} iterations)".format(iterations))
    print("=" * 60)
    
    total_avg_time = 0
    for component, times in total_times.items():
        avg_time = np.mean(times) * 1000  # Convert to milliseconds
        std_time = np.std(times) * 1000
        total_avg_time += np.mean(times)
        
        # Add visual indicator for slow components
        if avg_time > 50:  # >50ms is definitely a bottleneck
            status = "🔴 MAJOR BOTTLENECK"
        elif avg_time > 10:  # >10ms is concerning
            status = "🟡 BOTTLENECK"
        else:
            status = "🟢 OK"
        
        print(f"{component.upper():<20}: {avg_time:6.1f}ms ± {std_time:4.1f}ms {status}")
    
    total_avg_time_ms = total_avg_time * 1000
    print("-" * 60)
    print(f"{'TOTAL INFERENCE':<20}: {total_avg_time_ms:6.1f}ms")
    print(f"{'TARGET (60 FPS)':<20}: {16:6.1f}ms")
    print(f"{'SPEEDUP NEEDED':<20}: {total_avg_time_ms/16:6.1f}x")
    
    # Show breakdown percentages
    print("\n📈 TIME BREAKDOWN:")
    for component, times in total_times.items():
        percentage = (np.mean(times) / total_avg_time) * 100
        bar_length = int(percentage / 2)  # Scale down for display
        bar = "█" * bar_length + "░" * (50 - bar_length)
        print(f"{component.upper():<20}: {percentage:5.1f}% {bar}")
    
    print("\n🎯 RECOMMENDATIONS:")
    # Find the biggest bottleneck
    avg_times_ms = {k: np.mean(v) * 1000 for k, v in total_times.items()}
    slowest_component = max(avg_times_ms, key=avg_times_ms.get)
    slowest_time = avg_times_ms[slowest_component]
    
    print(f"1. Focus on optimizing: {slowest_component.upper()} ({slowest_time:.1f}ms)")
    
    # Show top 3 bottlenecks
    sorted_components = sorted(avg_times_ms.items(), key=lambda x: x[1], reverse=True)
    print("2. Top 3 bottlenecks:")
    for i, (comp, time_ms) in enumerate(sorted_components[:3]):
        print(f"   {i+1}. {comp.upper()}: {time_ms:.1f}ms")

if __name__ == "__main__":
    profile_inference_pipeline()
