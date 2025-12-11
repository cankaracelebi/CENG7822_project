#!/bin/bash
# Test script that ensures we're in the rlan conda environment

echo "============================================================"
echo "Testing ShooterEnv in Headless Mode (No Graphics Needed)"
echo "============================================================"
echo ""
echo "Checking conda environment..."

if [ -z "$CONDA_DEFAULT_ENV" ]; then
    echo "ERROR: No conda environment activated!"
    echo "Please run: conda activate rlan"
    exit 1
fi

if [ "$CONDA_DEFAULT_ENV" != "rlan" ]; then
    echo "WARNING: You're in '$CONDA_DEFAULT_ENV' environment"
    echo "Expected: rlan"
    echo "Consider running: conda activate rlan"
fi

echo "✓ Conda environment: $CONDA_DEFAULT_ENV"
echo ""

# Run the Python test
python -c "
import importlib

print('Importing ShooterEnv...')
game_2d = importlib.import_module('game.2D')
ShooterEnv = game_2d.ShooterEnv

print('✓ Import successful!')
print('')

# Create environment
print('Creating environment (headless mode)...')
env = ShooterEnv(render_mode=None)
print('✓ Environment created')
print(f'  - Observation space: {env.observation_space.shape}')
print(f'  - Action space: {env.action_space}')
print('')

# Reset
print('Resetting environment...')
obs, info = env.reset(seed=42)
print('✓ Reset successful')
print(f'  - Observation shape: {obs.shape}')
print(f'  - Info: {info}')
print('')

# Run some steps
print('Running 50 steps with random actions...')
total_reward = 0
for i in range(50):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    total_reward += reward
    if terminated or truncated:
        print(f'  Episode ended at step {i+1}')
        break

print(f'✓ Completed {i+1} steps')
print(f'  - Total reward: {total_reward:.3f}')
print(f'  - Final health: {info[\"health\"]:.2f}')
print(f'  - Enemies: {info[\"num_enemies\"]}')
print(f'  - Prizes: {info[\"num_prizes\"]}')
print('')

env.close()
print('✓ Environment closed')

print('')
print('='*60)
print('SUCCESS! Your environment is working perfectly!')
print('='*60)
print('')
print('You can now:')
print('  1. Train RL agent: python -m rl.train --algo ppo')
print('  2. See training config: rl/configs/shooter_config.py')
print('')
print('Note: Rendering requires OpenGL drivers (not needed for training)')
print('='*60)
"
