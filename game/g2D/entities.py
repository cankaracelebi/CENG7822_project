"""
Game entity dataclasses
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class Agent:
    """Player agent entity"""
    x: float
    y: float
    vx: float = 0.0
    vy: float = 0.0
    radius: float = 14.0
    health: float = 1.0  # normalized [0,1]
    cooldown: int = 0


@dataclass
class Enemy:
    """Enemy entity that chases the player"""
    x: float
    y: float
    vx: float
    vy: float
    radius: float = 12.0
    speed: float = 90.0  # px/s
    alive: bool = True


@dataclass
class Prize:
    """Collectible prize entity"""
    x: float
    y: float
    radius: float = 10.0
    value: float = 1.0
    ttl: Optional[float] = None  # seconds left; None for infinite


@dataclass
class Bullet:
    """Bullet projectile entity"""
    x: float
    y: float
    vx: float
    vy: float
    radius: float = 4.0
    ttl: float = 1.2  # seconds
    alive: bool = True
