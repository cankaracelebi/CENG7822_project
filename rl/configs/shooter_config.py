"""
Training configuration for shooter environment
"""

# Environment parameters
ENV_CONFIG = {
    # "render_mode": None,  # Don't render during training - it's too slow with parallel envs
    "width": 800,
    "height": 600,
    "dt": 1/30,
    "max_steps": 1800,  # 60 seconds at 30 FPS
    "k_enemies": 5,
    "m_prizes": 3,
    "max_enemies": 10,
    "enemy_spawn_interval": 1.5,
    "prize_spawn_interval": 1.0,
    "agent_speed": 140.0,
    "bullet_speed": 260.0,
    "shoot_cooldown_steps": 6,
    "damage_on_contact": 0.2,
}

# PPO hyperparameters
PPO_CONFIG = {
    "policy": "MlpPolicy",
    "learning_rate": 3e-4,
    "n_steps": 1024,
    "batch_size": 256,
    "n_epochs": 10,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip_range": 0.2,
    "ent_coef": 0.01,
    "vf_coef": 0.5,
    "max_grad_norm": 0.5,
    "verbose": 1,
}

# DQN hyperparameters (alternative)
DQN_CONFIG = {
    "policy": "MlpPolicy",
    "learning_rate": 1e-4,
    "buffer_size": 100_000,
    "learning_starts": 1000,
    "batch_size": 128,
    "tau": 1.0,
    "gamma": 0.99,
    "train_freq": 4,
    "gradient_steps": 1,
    "target_update_interval": 1000,
    "exploration_fraction": 0.1,
    "exploration_initial_eps": 1.0,
    "exploration_final_eps": 0.05,
    "verbose": 1,
}

# Training settings
TRAINING_CONFIG = {
    "total_timesteps": 500_000,
    "save_freq": 10_000,
    "log_dir": "./logs",
    "model_dir": "./models",
    "tensorboard_log": "./tensorboard_logs",
}
