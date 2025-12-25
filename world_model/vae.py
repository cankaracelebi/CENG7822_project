"""
Variational Autoencoder (VAE) for encoding game frames to latent space.
Architecture: Conv encoder → latent (μ, σ) → Conv decoder
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class VAEEncoder(nn.Module):
    """CNN encoder that maps images to latent distribution parameters (μ, σ)"""
    
    def __init__(self, in_channels: int = 3, latent_dim: int = 32):
        super().__init__()
        self.latent_dim = latent_dim
        
        # Conv layers: (3, 64, 64) -> (32, 32, 32) -> (64, 16, 16) -> (128, 8, 8) -> (256, 4, 4)
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1)
        
        # Flatten: 256 * 4 * 4 = 4096
        self.fc_mu = nn.Linear(256 * 4 * 4, latent_dim)
        self.fc_logvar = nn.Linear(256 * 4 * 4, latent_dim)
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Input image tensor (B, C, H, W), expected (B, 3, 64, 64)
        Returns:
            mu: Mean of latent distribution (B, latent_dim)
            logvar: Log variance of latent distribution (B, latent_dim)
        """
        # Conv layers with ReLU
        h = F.relu(self.conv1(x))
        h = F.relu(self.conv2(h))
        h = F.relu(self.conv3(h))
        h = F.relu(self.conv4(h))
        
        # Flatten and project to latent params
        h = h.contiguous().view(h.size(0), -1)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        
        return mu, logvar


class VAEDecoder(nn.Module):
    """CNN decoder that reconstructs images from latent vectors"""
    
    def __init__(self, out_channels: int = 3, latent_dim: int = 32):
        super().__init__()
        self.latent_dim = latent_dim
        
        # Project latent to conv feature map
        self.fc = nn.Linear(latent_dim, 256 * 4 * 4)
        
        # Deconv layers: (256, 4, 4) -> (128, 8, 8) -> (64, 16, 16) -> (32, 32, 32) -> (3, 64, 64)
        self.deconv1 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1)
        self.deconv2 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
        self.deconv3 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)
        self.deconv4 = nn.ConvTranspose2d(32, out_channels, kernel_size=4, stride=2, padding=1)
        
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: Latent vector (B, latent_dim)
        Returns:
            recon: Reconstructed image (B, C, 64, 64)
        """
        h = F.relu(self.fc(z))
        h = h.view(h.size(0), 256, 4, 4)
        
        h = F.relu(self.deconv1(h))
        h = F.relu(self.deconv2(h))
        h = F.relu(self.deconv3(h))
        recon = torch.sigmoid(self.deconv4(h))  # Sigmoid for [0, 1] pixel values
        
        return recon


class VAE(nn.Module):
    """
    Variational Autoencoder combining encoder and decoder.
    Used to compress game frames into a compact latent representation.
    """
    
    def __init__(self, in_channels: int = 3, latent_dim: int = 32):
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder = VAEEncoder(in_channels, latent_dim)
        self.decoder = VAEDecoder(in_channels, latent_dim)
        
    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick: z = mu + sigma * epsilon"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Input image (B, C, H, W)
        Returns:
            recon: Reconstructed image
            mu: Latent mean
            logvar: Latent log variance
        """
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decoder(z)
        return recon, mu, logvar
    
    def encode(self, x: torch.Tensor, deterministic: bool = True) -> torch.Tensor:
        """Encode image to latent vector"""
        mu, logvar = self.encoder(x)
        if deterministic:
            return mu
        return self.reparameterize(mu, logvar)
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent vector to image"""
        return self.decoder(z)
    
    @staticmethod
    def loss_function(recon: torch.Tensor, target: torch.Tensor, 
                      mu: torch.Tensor, logvar: torch.Tensor,
                      beta: float = 1.0) -> Tuple[torch.Tensor, dict]:
        """
        VAE loss = Reconstruction loss + β * KL divergence
        
        Args:
            recon: Reconstructed image
            target: Original image
            mu: Latent mean
            logvar: Latent log variance
            beta: Weight for KL term (β-VAE)
        Returns:
            loss: Total loss
            metrics: Dict with individual loss components
        """
        # Reconstruction loss (MSE)
        recon_loss = F.mse_loss(recon, target, reduction='sum') / target.size(0)
        
        # KL divergence: -0.5 * sum(1 + log(σ²) - μ² - σ²)
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / target.size(0)
        
        total_loss = recon_loss + beta * kl_loss
        
        return total_loss, {
            'loss': total_loss.item(),
            'recon_loss': recon_loss.item(),
            'kl_loss': kl_loss.item(),
        }


def test_vae():
    """Quick test of VAE architecture"""
    vae = VAE(in_channels=3, latent_dim=32)
    
    # Test with random input (batch=4, channels=3, height=64, width=64)
    x = torch.randn(4, 3, 64, 64)
    
    # Forward pass
    recon, mu, logvar = vae(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Reconstruction shape: {recon.shape}")
    print(f"Latent mu shape: {mu.shape}")
    print(f"Latent logvar shape: {logvar.shape}")
    
    # Compute loss
    loss, metrics = VAE.loss_function(recon, x, mu, logvar)
    print(f"Loss: {loss.item():.4f}")
    print(f"Metrics: {metrics}")
    
    # Test encoding/decoding
    z = vae.encode(x)
    print(f"Encoded z shape: {z.shape}")
    
    recon2 = vae.decode(z)
    print(f"Decoded shape: {recon2.shape}")
    
    print("VAE test passed!")


if __name__ == "__main__":
    test_vae()
