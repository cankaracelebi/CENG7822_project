# Applied Reinforcement Learning for 2D Shooter Game

This project implements and compares reinforcement learning approaches for training autonomous agents in a 2D top-down shooter game environment.

## Quick Start

### Installation

```bash
# Create conda environment
conda create -n rlan python=3.10
conda activate rlan

# Install dependencies
pip install -r requirements.txt
```

### Run the Game (Human Play)

```bash
cd game && python game.py
```

## Training Agents

### Model-Free RL (DQN, PPO, SAC) 

```bash
# Train PPO with aggressive rewards 
python -m rl.train --algo ppo --reward-config aggressive --timesteps 1500000

# Train DQN
python -m rl.train --algo dqn --reward-config aggressive --timesteps 1000000

# Evaluate a trained model
python -m rl.evaluate --model-path trained_models/ppo/ppo_aggressive.zip --n-episodes 10 --record-video
```

### Pre-trained Models
We provide the trained models in the `trained_models/` directory:
- `trained_models/ppo/ppo_aggressive.zip`: **Best Agent (+38.74)** (PPO with aggressive rewards)
- `trained_models/world_model/dyna_agent.zip`: Best World Model agent (-4.87)
- `trained_models/world_model/vae.pt`: Pre-trained VAE
- `trained_models/world_model/mdn_rnn.pt`: Pre-trained MDN-RNN

### World Model Approaches

**1. Pure Dream Training (Failed)**
```bash
# Train entirely in imagination (requires pre-trained VAE/MDN)
python -m world_model.train_world_model_ppo \
    --vae-path trained_models/world_model/vae.pt \
    --mdn-path trained_models/world_model/mdn_rnn.pt \
    --skip-collect --skip-vae --skip-mdn \
    --save-dir ./world_model_dream \
    --ppo-timesteps 500000

# Evaluate
python -m rl.evaluate --model-path world_model_dream/ppo_dream.zip --n-episodes 10 --record-video
```

**2. Hybrid Training (Real Rewards + VAE Obs)**
```bash
# Train with real rewards but latent observations
python -m world_model.train_world_model_ppo \
    --vae-path trained_models/world_model/vae.pt \
    --mdn-path trained_models/world_model/mdn_rnn.pt \
    --skip-collect --skip-vae --skip-mdn \
    --save-dir ./world_model_hybrid \
    --use-real-env --ppo-timesteps 500000

# Evaluate (requires latent wrapper, handled by evaluate.py if model type detected, or use training script eval)
python -m world_model.train_world_model_ppo --eval-only --model-path world_model_hybrid/ppo_hybrid.zip
```

**3. Dyna-Style Training **
```bash
# Train interleaving real and dream experiences
python -m world_model.train_dyna \
    --vae-path trained_models/world_model/vae.pt \
    --mdn-path trained_models/world_model/mdn_rnn.pt \
    --save-dir ./world_model_dyna \
    --timesteps 1500000

# Evaluate (this script handles the specific dyna wrapper)
python -m world_model.train_dyna --eval-only --model-path trained_models/world_model/dyna_agent.zip
```

**World Model Options:**
| Flag | Description |
|------|-------------|
| `--use-real-env` | Use real environment rewards instead of MDN-RNN predictions |
| `--normalize-rewards` | Normalize rewards before MDN-RNN training |
| `--reward-weight N` | Weight for reward loss in MDN-RNN (default: 1.0) |
| `--vae-path PATH` | Load pre-trained VAE from specified path |
| `--mdn-path PATH` | Load pre-trained MDN-RNN from specified path |
| `--skip-collect/--skip-vae/--skip-mdn` | Skip data collection/VAE/MDN-RNN training |

**Verify World Model Quality:**
```bash
python -m world_model.verify_world_model --model-dir ./trained_models/world_model
```

## Project Structure

```
├── game/                    # Game environment
│   └── game.py             # ShooterEnv (Gymnasium-compatible)
├── rl/                      # Model-free RL implementations
│   ├── train.py            # Training script
│   └── evaluate.py         # Evaluation with video recording
├── world_model/             # World Models implementation
│   ├── vae.py              # Variational Autoencoder (BCE loss)
│   ├── mdn_rnn.py          # Mixture Density Network RNN
│   ├── train_world_model_ppo.py  # Full training pipeline
│   ├── train_dyna.py       # Dyna-style training (recommended)
│   └── verify_world_model.py
├── report/                  # LaTeX report
├── presentation/            # Beamer presentation
└── report_figures/          # Generated figures
```



## References

- Ha & Schmidhuber, "World Models", 2018
- Tomilin et al., "COOM: A Game Benchmark for Continual RL", NeurIPS 2023
- Schulman et al., "Proximal Policy Optimization", 2017
- Sutton, "Dyna: An Integrated Architecture for Learning, Planning, and Reacting", 1991

## License

MIT License
