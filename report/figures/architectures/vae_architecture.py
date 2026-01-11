#!/usr/bin/env python3
"""
VAE Architecture Diagram for World Models
Using PlotNeuralNet

Architecture:
  Encoder: Input(64x64x3) -> Conv1(32) -> Conv2(64) -> Conv3(128) -> Conv4(256) -> FC(mu,sigma) -> z(32)
  Decoder: z(32) -> FC(4096) -> DeConv1(128) -> DeConv2(64) -> DeConv3(32) -> DeConv4(3) -> Output(64x64x3)
"""

import sys
import os

# Add PlotNeuralNet to path
sys.path.insert(0, '/home/guava/Desktop/CENG7822_project/PlotNeuralNet')
from pycore.tikzeng import *


def to_LatentBox(name, n_units, offset="(0,0,0)", to="(0,0,0)", width=2, height=3, depth=10, caption=" "):
    """Latent space representation - small highlighted box"""
    return r"""
\pic[shift={""" + offset + r"""}] at """ + to + r""" 
    {Box={
        name=""" + name + r""",
        caption=""" + caption + r""",
        xlabel={{""" + str(n_units) + r""", }},
        fill={rgb:green,3;blue,2;white,3},
        height=""" + str(height) + r""",
        width=""" + str(width) + r""",
        depth=""" + str(depth) + r"""
        }
    };
"""


def to_DeConv(name, s_filer=256, n_filer=64, offset="(0,0,0)", to="(0,0,0)", width=1, height=40, depth=40, caption=" "):
    """Deconvolution/transpose convolution layer"""
    return r"""
\pic[shift={""" + offset + r"""}] at """ + to + r""" 
    {Box={
        name=""" + name + r""",
        caption=""" + caption + r""",
        xlabel={{""" + str(n_filer) + r""", }},
        zlabel=""" + str(s_filer) + r""",
        fill=\UnpoolColor,
        height=""" + str(height) + r""",
        width=""" + str(width) + r""",
        depth=""" + str(depth) + r"""
        }
    };
"""


def to_InputImage(name, offset="(0,0,0)", to="(0,0,0)", width=2, height=32, depth=32, caption="Input"):
    """Input image representation"""
    return r"""
\pic[shift={""" + offset + r"""}] at """ + to + r""" 
    {Box={
        name=""" + name + r""",
        caption=""" + caption + r""",
        xlabel={{3, }},
        zlabel=64,
        fill={rgb:white,5;black,1},
        height=""" + str(height) + r""",
        width=""" + str(width) + r""",
        depth=""" + str(depth) + r"""
        }
    };
"""


def to_OutputImage(name, offset="(0,0,0)", to="(0,0,0)", width=2, height=32, depth=32, caption="Output"):
    """Output image representation"""
    return r"""
\pic[shift={""" + offset + r"""}] at """ + to + r""" 
    {Box={
        name=""" + name + r""",
        caption=""" + caption + r""",
        xlabel={{3, }},
        zlabel=64,
        fill={rgb:white,5;black,1},
        height=""" + str(height) + r""",
        width=""" + str(width) + r""",
        depth=""" + str(depth) + r"""
        }
    };
"""


def to_MuSigma(name, offset="(0,0,0)", to="(0,0,0)"):
    """mu and sigma representation for VAE"""
    return r"""
\pic[shift={""" + offset + r"""}] at """ + to + r""" 
    {Box={
        name=""" + name + r"""_mu,
        caption=$\mu$,
        xlabel={{ ,32}},
        fill={rgb:red,4;white,3},
        height=2,
        width=1,
        depth=8
        }
    };
\pic[shift={(0,-1.5,0)}] at (""" + name + r"""_mu-south) 
    {Box={
        name=""" + name + r"""_sigma,
        caption=$\sigma$,
        xlabel={{ ,32}},
        fill={rgb:blue,4;white,3},
        height=2,
        width=1,
        depth=8
        }
    };
"""


def to_FC(name, n_units, offset="(0,0,0)", to="(0,0,0)", width=1.5, height=2, depth=10, caption=" "):
    """Fully connected layer"""
    return r"""
\pic[shift={""" + offset + r"""}] at """ + to + r""" 
    {Box={
        name=""" + name + r""",
        caption=""" + caption + r""",
        xlabel={{""" + str(n_units) + r""", }},
        fill=\FcColor,
        height=""" + str(height) + r""",
        width=""" + str(width) + r""",
        depth=""" + str(depth) + r"""
        }
    };
"""


# Architecture definition
arch = [
    to_head('/home/guava/Desktop/CENG7822_project/PlotNeuralNet'),
    to_cor(),
    to_begin(),
    
    # === ENCODER ===
    # Input image 64x64x3
    to_InputImage("input", offset="(0,0,0)", to="(0,0,0)", width=2, height=32, depth=32, caption="64x64x3"),
    
    # Conv1: 64x64x3 -> 32x32x32
    to_Conv("conv1", s_filer=32, n_filer=32, offset="(2,0,0)", to="(input-east)", 
            width=2, height=28, depth=28, caption="Conv1"),
    to_connection("input", "conv1"),
    
    # Conv2: 32x32x32 -> 16x16x64
    to_Conv("conv2", s_filer=16, n_filer=64, offset="(1.5,0,0)", to="(conv1-east)", 
            width=3, height=22, depth=22, caption="Conv2"),
    to_connection("conv1", "conv2"),
    
    # Conv3: 16x16x64 -> 8x8x128
    to_Conv("conv3", s_filer=8, n_filer=128, offset="(1.5,0,0)", to="(conv2-east)", 
            width=4, height=16, depth=16, caption="Conv3"),
    to_connection("conv2", "conv3"),
    
    # Conv4: 8x8x128 -> 4x4x256
    to_Conv("conv4", s_filer=4, n_filer=256, offset="(1.5,0,0)", to="(conv3-east)", 
            width=5, height=10, depth=10, caption="Conv4"),
    to_connection("conv3", "conv4"),
    
    # Flatten + FC -> mu, sigma (both receive same flattened input)
    to_MuSigma("latent_params", offset="(2,0,0)", to="(conv4-east)"),
    to_connection("conv4", "latent_params_mu"),
    to_connection("conv4", "latent_params_sigma"),
    
    # Latent z (32-dim) - central point with reparameterization (z = mu + sigma*eps)
    to_LatentBox("z", n_units=32, offset="(2.5,0,0)", to="(latent_params_mu-east)", 
                 width=1.5, height=3, depth=10, caption="z"),
    to_connection("latent_params_mu", "z"),
    to_connection("latent_params_sigma", "z"),
    
    # === DECODER ===
    # FC: z -> 4x4x256
    to_FC("fc_dec", n_units="4096", offset="(2,0,0)", to="(z-east)", 
          width=1.5, height=3, depth=10, caption="FC"),
    to_connection("z", "fc_dec"),
    
    # Reshape placeholder (conceptual)
    to_Conv("reshape", s_filer=4, n_filer=256, offset="(1.5,0,0)", to="(fc_dec-east)", 
            width=5, height=10, depth=10, caption="Reshape"),
    to_connection("fc_dec", "reshape"),
    
    # DeConv1: 4x4x256 -> 8x8x128
    to_DeConv("deconv1", s_filer=8, n_filer=128, offset="(1.5,0,0)", to="(reshape-east)", 
              width=4, height=16, depth=16, caption="DeConv1"),
    to_connection("reshape", "deconv1"),
    
    # DeConv2: 8x8x128 -> 16x16x64
    to_DeConv("deconv2", s_filer=16, n_filer=64, offset="(1.5,0,0)", to="(deconv1-east)", 
              width=3, height=22, depth=22, caption="DeConv2"),
    to_connection("deconv1", "deconv2"),
    
    # DeConv3: 16x16x64 -> 32x32x32
    to_DeConv("deconv3", s_filer=32, n_filer=32, offset="(1.5,0,0)", to="(deconv2-east)", 
              width=2, height=28, depth=28, caption="DeConv3"),
    to_connection("deconv2", "deconv3"),
    
    # DeConv4: 32x32x32 -> 64x64x3
    to_OutputImage("output", offset="(2,0,0)", to="(deconv3-east)", 
                   width=2, height=32, depth=32, caption="64x64x3"),
    to_connection("deconv3", "output"),
    
    to_end()
]


def main():
    output_dir = '/home/guava/Desktop/CENG7822_project/report/figures/architectures'
    os.makedirs(output_dir, exist_ok=True)
    to_generate(arch, os.path.join(output_dir, 'vae_architecture.tex'))
    print(f"Generated vae_architecture.tex in {output_dir}")


if __name__ == '__main__':
    main()
