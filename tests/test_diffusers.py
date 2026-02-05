# Copyright 2024 The DDBM Authors.
# Licensed under the Apache License, Version 2.0 (the "License");
"""
Tests for the diffusers-compatible DDBM components.
"""

import pytest
import torch
import numpy as np


class TestDDBMScheduler:
    """Tests for the DDBMScheduler class."""

    def test_scheduler_import(self):
        """Test that the scheduler can be imported."""
        from ddbm.schedulers import DDBMScheduler
        assert DDBMScheduler is not None

    def test_scheduler_init_default(self):
        """Test scheduler initialization with default parameters."""
        from ddbm.schedulers import DDBMScheduler
        
        scheduler = DDBMScheduler()
        
        assert scheduler.sigma_min == 0.002
        assert scheduler.sigma_max == 80.0
        assert scheduler.sigma_data == 0.5
        assert scheduler.pred_mode == "vp"
        assert scheduler.num_train_timesteps == 40

    def test_scheduler_init_custom(self):
        """Test scheduler initialization with custom parameters."""
        from ddbm.schedulers import DDBMScheduler
        
        scheduler = DDBMScheduler(
            sigma_min=0.001,
            sigma_max=1.0,
            pred_mode="ve",
            num_train_timesteps=100,
        )
        
        assert scheduler.sigma_min == 0.001
        assert scheduler.sigma_max == 1.0
        assert scheduler.pred_mode == "ve"

    def test_set_timesteps(self):
        """Test setting timesteps."""
        from ddbm.schedulers import DDBMScheduler
        
        scheduler = DDBMScheduler()
        scheduler.set_timesteps(40)
        
        assert scheduler.sigmas is not None
        assert len(scheduler.sigmas) == 41  # 40 steps + 1 zero at end
        assert scheduler.sigmas[-1] == 0  # Last sigma should be 0
        assert scheduler.num_inference_steps == 40

    def test_sigmas_are_decreasing(self):
        """Test that sigmas are monotonically decreasing."""
        from ddbm.schedulers import DDBMScheduler
        
        scheduler = DDBMScheduler()
        scheduler.set_timesteps(40)
        
        sigmas = scheduler.sigmas[:-1]  # Exclude final zero
        for i in range(len(sigmas) - 1):
            assert sigmas[i] > sigmas[i + 1], f"Sigmas not decreasing at index {i}"

    def test_scale_model_input(self):
        """Test scale_model_input returns input unchanged."""
        from ddbm.schedulers import DDBMScheduler
        
        scheduler = DDBMScheduler()
        sample = torch.randn(2, 3, 64, 64)
        
        scaled = scheduler.scale_model_input(sample, timestep=0)
        
        assert torch.allclose(sample, scaled)


class TestDDBMPipeline:
    """Tests for the DDBMPipeline class."""

    def test_pipeline_import(self):
        """Test that the pipeline can be imported."""
        from ddbm.pipelines import DDBMPipeline
        assert DDBMPipeline is not None

    def test_pipeline_with_mock_model(self):
        """Test pipeline initialization with a mock model."""
        from ddbm.pipelines import DDBMPipeline
        from ddbm.schedulers import DDBMScheduler
        
        # Create a simple mock model
        class MockUNet(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(1, 1)  # Dummy parameter
            
            def forward(self, x, timesteps, xT=None):
                return x  # Return input as denoised output
        
        model = MockUNet()
        scheduler = DDBMScheduler()
        
        pipeline = DDBMPipeline(unet=model, scheduler=scheduler)
        
        assert pipeline.unet is model
        assert pipeline.scheduler is scheduler


class TestDDBMIntegration:
    """Integration tests for DDBM diffusers components."""

    def test_main_package_exports(self):
        """Test that main package exports diffusers components."""
        from ddbm import DDBMScheduler, DDBMPipeline
        
        assert DDBMScheduler is not None
        assert DDBMPipeline is not None

    def test_scheduler_config_mixin(self):
        """Test that scheduler properly inherits from ConfigMixin."""
        from ddbm.schedulers import DDBMScheduler
        
        scheduler = DDBMScheduler(sigma_max=20.0, pred_mode="ve")
        
        # ConfigMixin should provide config attribute
        assert hasattr(scheduler, "config")
        assert scheduler.config.sigma_max == 20.0
        assert scheduler.config.pred_mode == "ve"


class TestAddNoise:
    """Tests for the add_noise method."""

    def test_add_noise_ve_mode(self):
        """Test add_noise with VE mode."""
        from ddbm.schedulers import DDBMScheduler
        
        scheduler = DDBMScheduler(pred_mode="ve", sigma_max=80.0)
        
        batch_size = 2
        channels = 3
        height = width = 32
        
        original_samples = torch.randn(batch_size, channels, height, width)
        noise = torch.randn_like(original_samples)
        x_T = torch.randn_like(original_samples)
        timesteps = torch.tensor([0.5, 0.8])  # Sigma values
        
        noisy = scheduler.add_noise(original_samples, noise, timesteps, x_T)
        
        assert noisy.shape == original_samples.shape

    def test_add_noise_vp_mode(self):
        """Test add_noise with VP mode."""
        from ddbm.schedulers import DDBMScheduler
        
        scheduler = DDBMScheduler(pred_mode="vp", sigma_max=1.0)
        
        batch_size = 2
        channels = 3
        height = width = 32
        
        original_samples = torch.randn(batch_size, channels, height, width)
        noise = torch.randn_like(original_samples)
        x_T = torch.randn_like(original_samples)
        timesteps = torch.tensor([0.5, 0.8])
        
        noisy = scheduler.add_noise(original_samples, noise, timesteps, x_T)
        
        assert noisy.shape == original_samples.shape


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
