# Copyright 2024 The DDBM Authors and The Hugging Face Team.
# Licensed under the Apache License, Version 2.0 (the "License");
"""
DDBM Training and Sampling Scripts.

This module provides command-line scripts for training and sampling with DDBM:

- `train_ddbm_diffusers.py`: Diffusers-style training using accelerate
- `ddbm_train.py`: Legacy training script using mpi4py
- `image_sample.py`: Legacy sampling script for evaluation

Usage:
    # Diffusers-style training (recommended)
    python scripts/train_ddbm_diffusers.py --data_dir /path/to/data
    
    # Or with accelerate for multi-GPU
    accelerate launch scripts/train_ddbm_diffusers.py --data_dir /path/to/data
"""
