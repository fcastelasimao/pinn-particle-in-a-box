"""Shared device selection and seeding — the one deduplication justified by the plan."""

import random
import numpy as np
import torch


def get_device():
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using Apple GPU (MPS)")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using NVIDIA GPU")
    else:
        device = torch.device("cpu")
        print("Using CPU")
    return device


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
