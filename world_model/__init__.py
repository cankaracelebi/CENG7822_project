"""
World Model Package - VAE + MDN-RNN for latent-space learning and planning.
Based on Ha & Schmidhuber 2018 "World Models" paper.
"""

from .vae import VAE, VAEEncoder, VAEDecoder
from .mdn_rnn import MDNRNN
from .dynamics import DynamicsModel  # Legacy MLP-based dynamics (kept for backward compatibility)
from .world_agent import WorldModelAgent, LatentPolicy

__all__ = [
    'VAE', 'VAEEncoder', 'VAEDecoder',
    'MDNRNN', 'DynamicsModel',
    'WorldModelAgent', 'LatentPolicy',
]
