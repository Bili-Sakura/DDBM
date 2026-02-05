"""
Denoising Diffusion Bridge Models (DDBM)

This package provides a Hugging Face diffusers-compatible implementation of DDBM
for image-to-image translation using diffusion bridges.

Original paper: https://arxiv.org/abs/2309.16948

Quick Start:
    ```python
    from ddbm import DDBMScheduler, DDBMPipeline

    # Create scheduler
    scheduler = DDBMScheduler(pred_mode='vp', sigma_max=1.0)

    # Load your trained model
    model = ...  # Your trained DDBM UNet

    # Create pipeline
    pipeline = DDBMPipeline(unet=model, scheduler=scheduler)

    # Run inference
    result = pipeline(source_image=your_image, num_inference_steps=40)
    images = result.images
    ```
"""

# Diffusers-compatible components (primary API)
from .schedulers import DDBMScheduler, DDBMSchedulerOutput
from .pipelines import DDBMPipeline, DDBMPipelineOutput

__version__ = "0.3.0"

__all__ = [
    # Diffusers-compatible (primary API)
    "DDBMScheduler",
    "DDBMSchedulerOutput",
    "DDBMPipeline",
    "DDBMPipelineOutput",
    # Original components (lazy loaded for backward compatibility)
    "KarrasDenoiser",
    "karras_sample",
    "UNetModel",
    "SongUNet",
    "create_model",
    "create_model_and_diffusion",
    "model_and_diffusion_defaults",
]


def __getattr__(name):
    """Lazy imports for original components that have heavier dependencies."""
    if name == "KarrasDenoiser":
        from .karras_diffusion import KarrasDenoiser
        return KarrasDenoiser
    elif name == "karras_sample":
        from .karras_diffusion import karras_sample
        return karras_sample
    elif name == "UNetModel":
        from .unet import UNetModel
        return UNetModel
    elif name == "SongUNet":
        from .edm_unet import SongUNet
        return SongUNet
    elif name == "create_model":
        from .script_util import create_model
        return create_model
    elif name == "create_model_and_diffusion":
        from .script_util import create_model_and_diffusion
        return create_model_and_diffusion
    elif name == "model_and_diffusion_defaults":
        from .script_util import model_and_diffusion_defaults
        return model_and_diffusion_defaults
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
