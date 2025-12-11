# Quick Start Guide

## ✅ Setup Complete!

You've successfully:
- ✅ Installed all dependencies in the `rlan` conda environment
- ✅ Migrated from pygame to Arcade
- ✅ Organized project structure (game/2D/, game/3D/, rl/)

## 🚀 Test the Environment (Headless - No Graphics Needed)

Since you don't have OpenGL drivers for rendering, **use headless mode** (which is what you need for RL training anyway!):

```bash
# Make sure you're in the rlan environment
conda activate rlan

# Test the environment without rendering
python -c "
import importlib
game_2d = importlib.import_module('game.2D')
env = game_2d.ShooterEnv(render_mode=None)  # No rendering!
obs, info = env.reset(seed=42)
print(f'✓ Environment works! Obs shape: {obs.shape}')

# Run 10 steps
for i in range(10):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    print(f'Step {i+1}: reward={reward:.3f}, health={info[\"health\"]:.2f}')
    if terminated or truncated:
        break

env.close()
print('✓ Test complete!')
"
```

## 🎯 Start Training

```bash
# Train with PPO (recommended)
python -m rl.train --algo ppo --timesteps 100000

# Or train with DQN
python -m rl.train --algo dqn --timesteps 100000

# Monitor with TensorBoard (in another terminal)
tensorboard --logdir ./tensorboard_logs
```

## 📊 Evaluate Trained Model

```bash
# Evaluate (without rendering)
python -m rl.evaluate models/ppo/best_model --algo ppo --n-episodes 5 --no-render
```

##  Common Commands

```bash
# Check your environment
conda env list

# Activate rlan environment
conda activate rlan

# Check installed packages
pip list | grep -E "gymnasium|arcade|stable"
```

## ⚠️ About Rendering

- **For RL Training**: You don't need rendering! Use `render_mode=None`
- **For Visualization**: Requires OpenGL drivers (you currently don't have them, but training works fine without!)

## 📁 Project Structure

```
CENG7822_project/
├── game/2D/          # Your Arcade-based shooter environment
├── game/3D/          # Placeholder for future 3D
├── rl/               # Training & evaluation scripts
│   ├── train.py
│   ├── evaluate.py
│   └── configs/
└── requirements.txt  # All dependencies
```

## 🎮 Environment Details

- **Action Space**: MultiDiscrete([5, 2, 8])
  - Move: 5 directions (stay, up, down, left, right)
  - Shoot: 2 options (don't shoot, shoot)
  - Aim: 8 directions
  
- **Observation Space**: Box(32,) - Vector observations
  - Agent: position, velocity, health, cooldown
  - Top-5 nearest enemies: relative positions and velocities
  - Top-3 nearest prizes: relative positions

## 🔥 Next Steps

1. Test the environment ☝️ (run the test command above)
2. Start training with PPO
3. Monitor results with TensorBoard
4. Evaluate your trained agent!
