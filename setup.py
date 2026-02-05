from setuptools import setup, find_packages

setup(
    name="ddbm",
    version="0.2.0",
    packages=find_packages(),
    install_requires=[
        "blobfile>=1.0.5",
        "packaging",
        "tqdm",
        "numpy",
        "scipy",
        "pandas",
        "Cython",
        "piq==0.7.0",
        "joblib==0.14.0",
        "albumentations==0.4.3",
        "lmdb",
        "clip @ git+https://github.com/openai/CLIP.git",
        "mpi4py",
        "flash-attn==2.0.4",
        "pillow",
        "wandb",
        'omegaconf',
        'torchmetrics[image]',
        'prdc',
        'clean-fid==0.1.35',
        # Diffusers integration (with secure versions)
        'diffusers>=0.25.0',
        'transformers>=4.48.0',  # Security fix for deserialization vulnerability
        'accelerate>=0.20.0',
    ],
    extras_require={
        "dev": [
            "pytest",
            "pytest-cov",
        ],
    },
    python_requires=">=3.8",
    author="DDBM Authors",
    description="Denoising Diffusion Bridge Models with Hugging Face diffusers integration",
    url="https://github.com/alexzhou907/DDBM",
)
