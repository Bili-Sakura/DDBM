"""
Denoising Diffusion Bridge Models (DDBM)

This package provides implementation of DDBM with both original API 
and Hugging Face diffusers-compatible components.

Original paper: https://arxiv.org/abs/2309.16948
"""

# Diffusers-compatible components (always available)
from .schedulers import DDBMScheduler
from .pipelines import DDBMPipeline

__all__ = [
    # Diffusers-compatible
    "DDBMScheduler",
    "DDBMPipeline",
    # Original (imported lazily to avoid dependency issues)
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
