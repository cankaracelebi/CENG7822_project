"""
World Model Package for 2D Shooter RL
=====================================
VAE-based world model with latent dynamics and planning.
"""

from .vae import VAE, VAEEncoder, VAEDecoder
from .dynamics import DynamicsModel
from .world_agent import WorldModelAgent, LatentPolicy
from .data_collector import FrameDataCollector

__all__ = [
    "VAE",
    "VAEEncoder", 
    "VAEDecoder",
    "DynamicsModel",
    "WorldModelAgent",
    "LatentPolicy",
    "FrameDataCollector",
]
