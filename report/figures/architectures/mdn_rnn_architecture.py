#!/usr/bin/env python
"""
MDN-RNN Architecture Diagram for World Models
Using PlotNeuralNet

Architecture:
  Input: [z_t (32), a_t (80)] concatenated → 112 dim
  LSTM: 112 → 256 hidden
  MDN Head: 256 → (π, μ, σ) for 5 Gaussians × 32 latent dims
  Reward Head: 256 → 1
  Done Head: 256 → 1
"""

import sys
import os

sys.path.insert(0, '/home/guava/Desktop/CENG7822_project/PlotNeuralNet')
from pycore.tikzeng import *


def to_RNN_Cell(name, hidden_dim=256, offset="(0,0,0)", to="(0,0,0)", width=3, height=8, depth=8, caption="LSTM"):
    """LSTM/RNN cell representation"""
    return r"""
\pic[shift={""" + offset + """}] at """ + to + """ 
    {Box={
        name=""" + name + """,
        caption=""" + caption + r""",
        xlabel={{""" + str(hidden_dim) + """, }},
        fill={rgb:orange,5;red,2;white,3},
        height=""" + str(height) + """,
        width=""" + str(width) + """,
        depth=""" + str(depth) + """
        }
    };
"""


def to_ConcatBox(name, dim, offset="(0,0,0)", to="(0,0,0)", width=1.5, height=4, depth=12, caption=" "):
    """Concatenation/input box"""
    return r"""
\pic[shift={""" + offset + """}] at """ + to + """ 
    {Box={
        name=""" + name + """,
        caption=""" + caption + r""",
        xlabel={{""" + str(dim) + """, }},
        fill={rgb:green,3;blue,2;white,4},
        height=""" + str(height) + """,
        width=""" + str(width) + """,
        depth=""" + str(depth) + """
        }
    };
"""


def to_InputVector(name, dim, offset="(0,0,0)", to="(0,0,0)", width=1, height=3, depth=8, caption=" "):
    """Input vector representation"""
    return r"""
\pic[shift={""" + offset + """}] at """ + to + """ 
    {Box={
        name=""" + name + """,
        caption=""" + caption + r""",
        xlabel={{""" + str(dim) + """, }},
        fill={rgb:blue,4;white,4},
        height=""" + str(height) + """,
        width=""" + str(width) + """,
        depth=""" + str(depth) + """
        }
    };
"""


def to_OutputHead(name, dim, offset="(0,0,0)", to="(0,0,0)", width=1, height=2, depth=6, caption=" ", color="magenta"):
    """Output head (reward, done, MDN components)"""
    if color == "magenta":
        fill = r"{rgb:magenta,5;black,3}"
    elif color == "red":
        fill = r"{rgb:red,5;white,3}"
    elif color == "blue":
        fill = r"{rgb:blue,5;white,3}"
    elif color == "green":
        fill = r"{rgb:green,5;white,3}"
    else:
        fill = r"\SoftmaxColor"
    
    return r"""
\pic[shift={""" + offset + """}] at """ + to + """ 
    {Box={
        name=""" + name + """,
        caption=""" + caption + r""",
        xlabel={{""" + str(dim) + """, }},
        fill=""" + fill + """,
        height=""" + str(height) + """,
        width=""" + str(width) + """,
        depth=""" + str(depth) + """
        }
    };
"""


def to_MDN_Output(name, offset="(0,0,0)", to="(0,0,0)"):
    """MDN output representation: π (weights), μ (means), σ (stds)"""
    return r"""
% MDN mixture weights π
\pic[shift={""" + offset + """}] at """ + to + """ 
    {Box={
        name=""" + name + """_pi,
        caption=$\pi$,
        xlabel={{5, }},
        fill={rgb:yellow,5;red,2;white,3},
        height=2,
        width=1,
        depth=6
        }
    };
% MDN means μ  
\pic[shift={(0,-2.2,0)}] at (""" + name + """_pi-south) 
    {Box={
        name=""" + name + """_mu,
        caption=$\mu$,
        xlabel={{160, }},
        fill={rgb:red,4;white,3},
        height=2,
        width=1,
        depth=10
        }
    };
% MDN stds σ
\pic[shift={(0,-2.2,0)}] at (""" + name + """_mu-south) 
    {Box={
        name=""" + name + """_sigma,
        caption=$\sigma$,
        xlabel={{160, }},
        fill={rgb:blue,4;white,3},
        height=2,
        width=1,
        depth=10
        }
    };
"""


def to_hidden_state(name, offset="(0,0,0)", to="(0,0,0)"):
    """Hidden state h and cell state c for LSTM"""
    return r"""
\pic[shift={""" + offset + """}] at """ + to + """ 
    {Box={
        name=""" + name + """_h,
        caption=$h_t$,
        xlabel={{256, }},
        fill={rgb:purple,4;white,3},
        height=1.5,
        width=0.8,
        depth=5
        }
    };
\pic[shift={(0,-1.5,0)}] at (""" + name + """_h-south) 
    {Box={
        name=""" + name + """_c,
        caption=$c_t$,
        xlabel={{256, }},
        fill={rgb:purple,3;white,4},
        height=1.5,
        width=0.8,
        depth=5
        }
    };
"""


# Architecture definition
arch = [
    to_head('/home/guava/Desktop/CENG7822_project/PlotNeuralNet'),
    to_cor(),
    to_begin(),
    
    # === INPUTS ===
    # Latent z_t (from VAE)
    to_InputVector("z_in", dim=32, offset="(0,0,0)", to="(0,0,0)", 
                   width=1, height=3, depth=8, caption="$z_t$"),
    
    # Action a_t
    to_InputVector("a_in", dim=80, offset="(0,-3,0)", to="(z_in-south)", 
                   width=1, height=3, depth=12, caption="$a_t$"),
    
    # Concatenation
    to_ConcatBox("concat", dim=112, offset="(2.5,1.5,0)", to="(a_in-east)", 
                 width=1.5, height=4, depth=14, caption="Concat"),
    to_connection("z_in", "concat"),
    to_connection("a_in", "concat"),
    
    # === LSTM ===
    to_RNN_Cell("lstm", hidden_dim=256, offset="(3,0,0)", to="(concat-east)", 
                width=5, height=10, depth=10, caption="LSTM"),
    to_connection("concat", "lstm"),
    
    # Hidden states (recurrent connection shown conceptually)
    to_hidden_state("hidden", offset="(0,7,0)", to="(lstm-north)"),
    
    # === OUTPUT HEADS ===
    # MDN Head (π, μ, σ)
    to_MDN_Output("mdn", offset="(3,2,0)", to="(lstm-east)"),
    to_connection("lstm", "mdn_pi"),
    
    # Reward prediction head
    to_OutputHead("reward", dim=1, offset="(3,-1,0)", to="(lstm-east)", 
                  width=1, height=2, depth=4, caption="$\\hat{r}$", color="green"),
    to_connection("lstm", "reward"),
    
    # Done prediction head
    to_OutputHead("done", dim=1, offset="(3,-3,0)", to="(lstm-east)", 
                  width=1, height=2, depth=4, caption="$\\hat{d}$", color="red"),
    to_connection("lstm", "done"),
    
    # === SAMPLED OUTPUT ===
    # z_next (sampled from mixture)
    to_InputVector("z_next", dim=32, offset="(3,0,0)", to="(mdn_mu-east)", 
                   width=1, height=3, depth=8, caption="$z_{t+1}$"),
    to_connection("mdn_mu", "z_next"),
    
    to_end()
]


def main():
    output_dir = '/home/guava/Desktop/CENG7822_project/report/figures/architectures'
    os.makedirs(output_dir, exist_ok=True)
    to_generate(arch, os.path.join(output_dir, 'mdn_rnn_architecture.tex'))
    print(f"Generated mdn_rnn_architecture.tex in {output_dir}")


if __name__ == '__main__':
    main()
