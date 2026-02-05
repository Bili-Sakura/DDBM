#!/usr/bin/env python
"""
Example: Using DDBM with Hugging Face diffusers API

This example demonstrates how to use the DDBM pipeline with the diffusers-compatible
API for image-to-image translation tasks.
"""

import torch
from PIL import Image
import numpy as np

# Import DDBM diffusers components
from ddbm import DDBMScheduler, DDBMPipeline

# For this example, we'll use a simple mock model
# In practice, you would use the actual DDBM UNet model
class MockDDBMUNet(torch.nn.Module):
    """Mock UNet for demonstration purposes."""
    def __init__(self, channels=3):
        super().__init__()
        self.linear = torch.nn.Linear(1, 1)  # Dummy param
        
    def forward(self, x, timesteps, xT=None):
        # Mock denoising: gradually blend towards xT
        alpha = 0.1
        if xT is not None:
            return x * (1 - alpha) + xT * alpha
        return x * (1 - alpha)


def main():
    # 1. Create the scheduler
    print("Creating DDBM scheduler...")
    scheduler = DDBMScheduler(
        sigma_max=1.0,          # Maximum sigma (VP mode)
        sigma_min=0.0001,       # Minimum sigma
        pred_mode='vp',         # Variance-preserving schedule
        beta_d=2.0,             # Beta_d parameter
        beta_min=0.1,           # Beta_min parameter
        num_train_timesteps=40, # Default training steps
    )
    
    # 2. Create the model (use your trained DDBM model here)
    print("Loading model...")
    model = MockDDBMUNet()
    
    # 3. Create the pipeline
    pipeline = DDBMPipeline(unet=model, scheduler=scheduler)
    
    # 4. Prepare input image
    print("Preparing input...")
    # Create a random input image (replace with your actual source image)
    source_image = torch.randn(1, 3, 64, 64)
    
    # 5. Run inference
    print("Running DDBM inference...")
    result = pipeline(
        source_image=source_image,
        num_inference_steps=40,    # Number of diffusion steps
        guidance=1.0,              # Guidance weight
        churn_step_ratio=0.33,     # Stochastic sampling ratio
        output_type='pil',         # Output as PIL images
    )
    
    # 6. Get results
    images = result["images"]
    nfe = result["nfe"]
    
    print(f"Generated {len(images)} images")
    print(f"Number of function evaluations: {nfe}")
    
    # Save the output image
    images[0].save("output.png")
    print("Saved output to output.png")
    
    return images


def example_with_pil_input():
    """Example using PIL images as input."""
    print("\n--- Example with PIL input ---")
    
    scheduler = DDBMScheduler(pred_mode='vp', sigma_max=1.0)
    model = MockDDBMUNet()
    pipeline = DDBMPipeline(unet=model, scheduler=scheduler)
    
    # Load source image
    # source_img = Image.open("source.png")
    # For demo, create random image
    source_img = Image.fromarray(
        np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
    )
    
    # Run inference
    result = pipeline(
        source_image=source_img,
        num_inference_steps=20,
        output_type='pil',
    )
    
    print(f"Output image size: {result['images'][0].size}")


def example_with_numpy_output():
    """Example with numpy array output."""
    print("\n--- Example with numpy output ---")
    
    scheduler = DDBMScheduler(pred_mode='ve', sigma_max=80.0)
    model = MockDDBMUNet()
    pipeline = DDBMPipeline(unet=model, scheduler=scheduler)
    
    source_image = torch.randn(2, 3, 32, 32)  # Batch of 2
    
    result = pipeline(
        source_image=source_image,
        num_inference_steps=10,
        output_type='np',
    )
    
    print(f"Output shape: {result['images'].shape}")
    print(f"Output range: [{result['images'].min():.3f}, {result['images'].max():.3f}]")


def example_scheduler_standalone():
    """Example using scheduler standalone for custom sampling loops."""
    print("\n--- Standalone scheduler example ---")
    
    scheduler = DDBMScheduler(
        sigma_max=1.0,
        sigma_min=0.0001,
        pred_mode='vp',
    )
    
    # Set timesteps
    scheduler.set_timesteps(40)
    
    print(f"Number of sigmas: {len(scheduler.sigmas)}")
    print(f"First sigma: {scheduler.sigmas[0]:.6f}")
    print(f"Last non-zero sigma: {scheduler.sigmas[-2]:.6f}")
    
    # Add noise to samples
    batch_size = 2
    original_samples = torch.randn(batch_size, 3, 64, 64)
    noise = torch.randn_like(original_samples)
    x_T = torch.randn_like(original_samples)
    timesteps = torch.tensor([0.5, 0.8])  # Sigma values
    
    noisy = scheduler.add_noise(original_samples, noise, timesteps, x_T)
    print(f"Noisy samples shape: {noisy.shape}")


if __name__ == "__main__":
    main()
    example_with_pil_input()
    example_with_numpy_output()
    example_scheduler_standalone()
