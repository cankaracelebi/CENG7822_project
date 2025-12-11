"""
ShooterEnv - a compact 2D top-down shooter RL environment
---------------------------------------------------------
- Arcade for simulation/rendering (migrated from Pygame)
- Gymnasium API
- 1 RL agent that moves + shoots (with cooldown)
- Random prizes for reward
- Enemies that chase and damage the agent on contact
- Vector observation: agent state + top-K nearest enemies + top-M nearest prizes
- Discrete MultiDiscrete action space: [move(5), shoot(2), aim(8)]

This is designed to be:
- Easy to train with PPO/DQN-style algorithms
- Easy to explain in a term project report
- Easy to extend (enemy types, obstacles, enemy projectiles, image obs)

Install:
    pip install gymnasium arcade numpy

Quick test:
    python -m game.2D.shooter_env
"""

from __future__ import annotations

import math
import random
import warnings
from typing import List, Optional, Tuple, Dict, Any

import numpy as np
import gymnasium as gym
from gymnasium import spaces
import arcade

from .utils import clamp, normalize, circle_collide, seed_everything
from .entities import Agent, Enemy, Prize, Bullet

# Suppress the harmless warning about numeric folder names in module paths
warnings.filterwarnings('ignore', message='.*found in sys.modules.*', category=RuntimeWarning)


class ShooterEnv(gym.Env):
    """2D top-down shooter environment using Arcade"""
    
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}

    def __init__(
        self,
        render_mode: Optional[str] = None,
        obs_mode: str = "vector",
        width: int = 800,
        height: int = 600,
        dt: float = 1 / 30,
        max_steps: int = 1800,  # 60s at 30 FPS
        k_enemies: int = 5,
        m_prizes: int = 3,
        max_enemies: int = 10,
        enemy_spawn_interval: float = 1.5,  # seconds
        prize_spawn_interval: float = 1.0,  # seconds
        agent_speed: float = 140.0,
        bullet_speed: float = 260.0,
        shoot_cooldown_steps: int = 6,
        damage_on_contact: float = 0.2,  # fraction of health per hit
        prize_ttl_range: Tuple[float, float] = (4.0, 8.0),
        enable_prize_ttl: bool = True,
    ):
        super().__init__()

        assert obs_mode in ("vector",), "Only 'vector' is implemented in this compact version."
        self.render_mode = render_mode
        self.obs_mode = obs_mode

        # Arena
        self.width = width
        self.height = height
        self.dt = dt
        self.max_steps = max_steps

        # Observation config
        self.k_enemies = k_enemies
        self.m_prizes = m_prizes

        # Gameplay config
        self.max_enemies = max_enemies
        self.enemy_spawn_interval = enemy_spawn_interval
        self.prize_spawn_interval = prize_spawn_interval
        self.agent_speed = agent_speed
        self.bullet_speed = bullet_speed
        self.shoot_cooldown_steps = shoot_cooldown_steps
        self.damage_on_contact = damage_on_contact
        self.prize_ttl_range = prize_ttl_range
        self.enable_prize_ttl = enable_prize_ttl

        # Internal timers
        self._enemy_spawn_timer = 0.0
        self._prize_spawn_timer = 0.0

        # Action space:
        # move: 0 stay, 1 up, 2 down, 3 left, 4 right
        # shoot: 0/1
        # aim: 0..7 (8 directions)
        self.action_space = spaces.MultiDiscrete([5, 2, 8])

        # Observation space (vector)
        # Agent: pos(2) vel(2) health(1) cooldown(1)
        # Each enemy: rel pos(2) rel vel(2)
        # Each prize: rel pos(2)
        obs_dim = 2 + 2 + 1 + 1 + (self.k_enemies * 4) + (self.m_prizes * 2)
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(obs_dim,), dtype=np.float32
        )

        # Arcade rendering state
        self._window = None

        # World state
        self.agent: Agent = None  # type: ignore
        self.enemies: List[Enemy] = []
        self.prizes: List[Prize] = []
        self.bullets: List[Bullet] = []

        # Step state
        self._step_count = 0

        # Event flags for reward computation
        self._events: Dict[str, float] = {}

        # Precompute aim directions (8-way)
        self._aim_dirs = []
        for i in range(8):
            ang = (math.pi * 2) * (i / 8.0)
            self._aim_dirs.append((math.cos(ang), math.sin(ang)))

    # ----------------------------
    # Gym API
    # ----------------------------

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        seed_everything(seed)

        self._step_count = 0
        self._enemy_spawn_timer = 0.0
        self._prize_spawn_timer = 0.0

        # Initialize agent in center
        self.agent = Agent(x=self.width * 0.5, y=self.height * 0.5)
        self.enemies = []
        self.prizes = []
        self.bullets = []

        # Spawn a couple of initial prizes/enemies (light start)
        self._spawn_prize()
        self._spawn_prize()
        self._spawn_enemy()

        obs = self._get_obs()
        info = self._get_info()
        return obs, info

    def step(self, action):
        self._events = {"hit": 0.0, "kill": 0.0, "prize": 0.0, "damage": 0.0, "shot": 0.0}

        move, shoot, aim = int(action[0]), int(action[1]), int(action[2])

        # Apply agent action
        self._apply_move(move)
        self._apply_shoot(shoot, aim)

        # Update world
        self._update_agent()
        self._update_enemies()
        self._update_bullets()
        self._update_prizes()

        # Handle collisions
        self._handle_collisions()

        # Spawn logic
        self._spawn_logic()

        # Decrease cooldown
        if self.agent.cooldown > 0:
            self.agent.cooldown -= 1

        # Compute reward
        reward = self._compute_reward()

        # Termination
        terminated = self.agent.health <= 0.0
        self._step_count += 1
        truncated = self._step_count >= self.max_steps

        obs = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self.render()

        return obs, reward, terminated, truncated, info

    # ----------------------------
    # Core mechanics
    # ----------------------------

    def _apply_move(self, move: int):
        # 0 stay, 1 up, 2 down, 3 left, 4 right
        vx, vy = 0.0, 0.0
        if move == 1:
            vy = -1.0
        elif move == 2:
            vy = 1.0
        elif move == 3:
            vx = -1.0
        elif move == 4:
            vx = 1.0

        # normalize diagonal
        vx, vy = normalize(vx, vy)
        self.agent.vx = vx * self.agent_speed
        self.agent.vy = vy * self.agent_speed

    def _apply_shoot(self, shoot: int, aim: int):
        if shoot == 0:
            return
        if self.agent.cooldown > 0:
            return

        dx, dy = self._aim_dirs[aim % 8]

        # Spawn bullet slightly in front of agent
        bx = self.agent.x + dx * (self.agent.radius + 6)
        by = self.agent.y + dy * (self.agent.radius + 6)
        bvx = dx * self.bullet_speed
        bvy = dy * self.bullet_speed

        self.bullets.append(Bullet(x=bx, y=by, vx=bvx, vy=bvy))
        self.agent.cooldown = self.shoot_cooldown_steps

        self._events["shot"] += 1.0

    def _update_agent(self):
        self.agent.x += self.agent.vx * self.dt
        self.agent.y += self.agent.vy * self.dt

        # Keep in bounds
        r = self.agent.radius
        self.agent.x = clamp(self.agent.x, r, self.width - r)
        self.agent.y = clamp(self.agent.y, r, self.height - r)

    def _update_enemies(self):
        ax, ay = self.agent.x, self.agent.y

        for e in self.enemies:
            if not e.alive:
                continue

            # Chase agent with mild noise
            to_ax = ax - e.x
            to_ay = ay - e.y
            nx, ny = normalize(to_ax, to_ay)

            # Add a little stochasticity for richer behavior
            jitter = 0.15
            nx += random.uniform(-jitter, jitter)
            ny += random.uniform(-jitter, jitter)
            nx, ny = normalize(nx, ny)

            e.vx = nx * e.speed
            e.vy = ny * e.speed

            e.x += e.vx * self.dt
            e.y += e.vy * self.dt

            # Keep enemies in bounds too
            r = e.radius
            e.x = clamp(e.x, r, self.width - r)
            e.y = clamp(e.y, r, self.height - r)

        # Remove dead enemies
        self.enemies = [e for e in self.enemies if e.alive]

    def _update_bullets(self):
        for b in self.bullets:
            if not b.alive:
                continue

            b.x += b.vx * self.dt
            b.y += b.vy * self.dt
            b.ttl -= self.dt

            if b.ttl <= 0:
                b.alive = False
                continue

            # Out of bounds -> kill
            if b.x < -10 or b.x > self.width + 10 or b.y < -10 or b.y > self.height + 10:
                b.alive = False

        self.bullets = [b for b in self.bullets if b.alive]

    def _update_prizes(self):
        if not self.enable_prize_ttl:
            return

        for p in self.prizes:
            if p.ttl is None:
                continue
            p.ttl -= self.dt

        self.prizes = [p for p in self.prizes if (p.ttl is None or p.ttl > 0)]

    def _handle_collisions(self):
        # Agent vs prizes
        remaining_prizes = []
        for p in self.prizes:
            if circle_collide(self.agent.x, self.agent.y, self.agent.radius, p.x, p.y, p.radius):
                self._events["prize"] += p.value
            else:
                remaining_prizes.append(p)
        self.prizes = remaining_prizes

        # Agent vs enemies (contact damage)
        for e in self.enemies:
            if circle_collide(self.agent.x, self.agent.y, self.agent.radius, e.x, e.y, e.radius):
                dmg = self.damage_on_contact
                self.agent.health = clamp(self.agent.health - dmg, 0.0, 1.0)
                self._events["damage"] += dmg

                # Small knockback to avoid "sticky" overlaps
                dx = self.agent.x - e.x
                dy = self.agent.y - e.y
                nx, ny = normalize(dx, dy)
                self.agent.x = clamp(self.agent.x + nx * 6, self.agent.radius, self.width - self.agent.radius)
                self.agent.y = clamp(self.agent.y + ny * 6, self.agent.radius, self.height - self.agent.radius)

        # Bullets vs enemies
        for b in self.bullets:
            if not b.alive:
                continue
            for e in self.enemies:
                if not e.alive:
                    continue
                if circle_collide(b.x, b.y, b.radius, e.x, e.y, e.radius):
                    # One-hit kill for simplicity
                    e.alive = False
                    b.alive = False
                    self._events["hit"] += 1.0
                    self._events["kill"] += 1.0
                    break

        # Cleanup after bullet collisions
        self.enemies = [e for e in self.enemies if e.alive]
        self.bullets = [b for b in self.bullets if b.alive]

    def _spawn_logic(self):
        # Enemy spawn timer
        self._enemy_spawn_timer += self.dt
        if self._enemy_spawn_timer >= self.enemy_spawn_interval:
            self._enemy_spawn_timer = 0.0
            if len(self.enemies) < self.max_enemies:
                self._spawn_enemy()

        # Prize spawn timer
        self._prize_spawn_timer += self.dt
        if self._prize_spawn_timer >= self.prize_spawn_interval:
            self._prize_spawn_timer = 0.0
            self._spawn_prize()

    def _spawn_enemy(self):
        # Spawn on a random edge to encourage pursuit dynamics
        side = random.choice(["top", "bottom", "left", "right"])
        margin = 20

        if side == "top":
            x = random.uniform(margin, self.width - margin)
            y = margin
        elif side == "bottom":
            x = random.uniform(margin, self.width - margin)
            y = self.height - margin
        elif side == "left":
            x = margin
            y = random.uniform(margin, self.height - margin)
        else:
            x = self.width - margin
            y = random.uniform(margin, self.height - margin)

        # Slightly randomized speed
        speed = random.uniform(70.0, 110.0)

        self.enemies.append(Enemy(x=x, y=y, vx=0.0, vy=0.0, speed=speed))

    def _spawn_prize(self):
        margin = 30
        x = random.uniform(margin, self.width - margin)
        y = random.uniform(margin, self.height - margin)

        ttl = None
        if self.enable_prize_ttl:
            ttl = random.uniform(*self.prize_ttl_range)

        value = 1.0
        self.prizes.append(Prize(x=x, y=y, value=value, ttl=ttl))

    # ----------------------------
    # Observation / reward / info
    # ----------------------------

    def _get_obs(self) -> np.ndarray:
        # Agent state
        ax = self.agent.x / self.width
        ay = self.agent.y / self.height

        # Normalize velocity by max speed
        avx = self.agent.vx / max(1e-6, self.agent_speed)
        avy = self.agent.vy / max(1e-6, self.agent_speed)

        health = self.agent.health
        cooldown = self.agent.cooldown / max(1, self.shoot_cooldown_steps)

        obs_parts = [ax * 2 - 1, ay * 2 - 1,  # map to [-1,1]
                     clamp(avx, -1, 1), clamp(avy, -1, 1),
                     health * 2 - 1,
                     clamp(cooldown * 2 - 1, -1, 1)]

        # Enemies: top-K nearest
        enemies_sorted = sorted(
            self.enemies,
            key=lambda e: (e.x - self.agent.x) ** 2 + (e.y - self.agent.y) ** 2
        )
        for i in range(self.k_enemies):
            if i < len(enemies_sorted):
                e = enemies_sorted[i]
                dx = (e.x - self.agent.x) / self.width
                dy = (e.y - self.agent.y) / self.height
                dvx = (e.vx - self.agent.vx) / max(1e-6, self.agent_speed)
                dvy = (e.vy - self.agent.vy) / max(1e-6, self.agent_speed)

                obs_parts += [
                    clamp(dx, -1, 1),
                    clamp(dy, -1, 1),
                    clamp(dvx, -1, 1),
                    clamp(dvy, -1, 1),
                ]
            else:
                obs_parts += [0.0, 0.0, 0.0, 0.0]

        # Prizes: top-M nearest
        prizes_sorted = sorted(
            self.prizes,
            key=lambda p: (p.x - self.agent.x) ** 2 + (p.y - self.agent.y) ** 2
        )
        for i in range(self.m_prizes):
            if i < len(prizes_sorted):
                p = prizes_sorted[i]
                dx = (p.x - self.agent.x) / self.width
                dy = (p.y - self.agent.y) / self.height
                obs_parts += [clamp(dx, -1, 1), clamp(dy, -1, 1)]
            else:
                obs_parts += [0.0, 0.0]

        obs = np.array(obs_parts, dtype=np.float32)
        return obs

    def _compute_reward(self) -> float:
        # lets work on some initial reward shaping then we wil lfocus on something wiht more detail
        R_PRIZE = 1.0
        R_HIT = 0.3
        R_KILL = 1.0
        R_DAMAGE = 1.0  # multiplied by damage fraction
        R_SHOT = 0.02
        R_TIME = 0.001
        R_DEATH = 5.0

        reward = 0.0

        reward += R_PRIZE * self._events.get("prize", 0.0) #obtaining prizes is important 
        reward += R_HIT * self._events.get("hit", 0.0) #
        reward += R_KILL * self._events.get("kill", 0.0)

        reward -= R_DAMAGE * self._events.get("damage", 0.0)
        reward -= R_SHOT * self._events.get("shot", 0.0)
        reward -= R_TIME

        if self.agent.health <= 0.0:
            reward -= R_DEATH

        return float(reward)

    def _get_info(self) -> Dict[str, Any]:
        return {
            "health": self.agent.health,
            "cooldown": self.agent.cooldown,
            "num_enemies": len(self.enemies),
            "num_prizes": len(self.prizes),
            "num_bullets": len(self.bullets),
            "step": self._step_count,
        }

    # ----------------------------
    # Rendering with Arcade
    # ----------------------------

    def render(self):
        if self.render_mode is None:
            return None

        if self._window is None and self.render_mode == "human":
            # Create Arcade window for human rendering
            self._window = ShooterWindow(self, self.width, self.height)
            
        if self.render_mode == "human" and self._window:
            # Arcade window handles its own rendering loop
            # We just need to update it
            self._window.on_draw()
            return None
        elif self.render_mode == "rgb_array":
            # For rgb_array mode, we'll use arcade's offscreen rendering
            return self._render_rgb_array()

    def _render_rgb_array(self):
        """Render to RGB array using Arcade's offscreen rendering"""
        # Create an offscreen buffer
        if not hasattr(self, '_offscreen_buffer'):
            self._offscreen_buffer = arcade.create_offscreen()
        
        # TODO: Implement offscreen rendering for rgb_array mode
        # For now, return a placeholder
        return np.zeros((self.height, self.width, 3), dtype=np.uint8)

    def close(self):
        if self._window is not None:
            self._window.close()
            self._window = None


class ShooterWindow(arcade.Window):
    """Arcade window for rendering the shooter environment"""
    
    def __init__(self, env: ShooterEnv, width: int, height: int):
        super().__init__(width, height, "ShooterEnv - Arcade")
        self.env = env
        
        # Colors
        self.BG = arcade.color.BLACK
        self.AGENT_C = (80, 200, 120)
        self.ENEMY_C = (220, 80, 80)
        self.PRIZE_C = (240, 210, 80)
        self.BULLET_C = (180, 180, 220)
        self.HUD_C = (220, 220, 220)
        
    def on_draw(self):
        """Draw the current game state"""
        self.clear()
        arcade.set_background_color((18, 18, 22))
        
        # Draw prizes
        for p in self.env.prizes:
            arcade.draw_circle_filled(p.x, p.y, p.radius, self.PRIZE_C)
        
        # Draw enemies
        for e in self.env.enemies:
            arcade.draw_circle_filled(e.x, e.y, e.radius, self.ENEMY_C)
        
        # Draw bullets
        for b in self.env.bullets:
            arcade.draw_circle_filled(b.x, b.y, b.radius, self.BULLET_C)
        
        # Draw agent
        arcade.draw_circle_filled(
            self.env.agent.x, self.env.agent.y, 
            self.env.agent.radius, self.AGENT_C
        )
        
        # HUD - Health bar
        bar_w, bar_h = 180, 10
        x0, y0 = 12, self.height - 22
        # Draw background bar using lrbt (left, right, bottom, top) format
        arcade.draw_lrbt_rectangle_filled(
            x0, x0 + bar_w, y0, y0 + bar_h, (60, 60, 60)
        )
        # Draw health fill
        fill = bar_w * clamp(self.env.agent.health, 0, 1)
        if fill > 0:
            arcade.draw_lrbt_rectangle_filled(
                x0, x0 + fill, y0, y0 + bar_h, self.AGENT_C
            )
        
        # Text HUD
        txt = (f"HP: {self.env.agent.health:.2f}  "
               f"Enemies: {len(self.env.enemies)}  "
               f"Prizes: {len(self.env.prizes)}  "
               f"Step: {self.env._step_count}")
        arcade.draw_text(txt, 12, self.height - 40, self.HUD_C, 14)


# ----------------------------
# Quick sanity test
# ----------------------------

def run_random_episode(render: bool = True):
    """Run a random episode for testing"""
    env = ShooterEnv(render_mode="human" if render else None)
    obs, info = env.reset(seed=42)

    terminated = False
    truncated = False
    total = 0.0

    print("Running episode... Press ESC or close window to exit early.")
    
    import time
    while not (terminated or truncated):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total += reward
        
        # For rendering, we need to:
        # 1. Process window events (so it doesn't freeze)
        # 2. Actually show the frame
        # 3. Control the frame rate
        if render and env._window:
            # Dispatch window events (needed to keep window responsive)
            env._window.dispatch_events()
            # Force redraw
            env._window.on_draw()
            env._window.flip()
            # Control frame rate - INCREASE THIS to slow down more
            time.sleep(0.03)  # Change to 0.1 for slower, 0.2 for very slow

    print(f"Random episode return: {total}")
    
    if render:
        print("Episode finished! Window will close in 10 seconds...")
        # Keep window open so you can see the final state
        import time
        try:
            # Show final state for 10 seconds (change this number as needed)
            for i in range(1):
                if env._window:
                    env._window.on_draw()
                    env._window.flip()
                time.sleep(1)
                print(f"Closing in {10-i} seconds...")
        except (KeyboardInterrupt, Exception):
            pass
    
    env.close()


if __name__ == "__main__":
    # Run a random episode with rendering
    # Use: python -m game.2D.shooter_env
    # Or with wrapper: ./run_with_rendering.sh python -m game.2D.shooter_env
    run_random_episode(render=True)
