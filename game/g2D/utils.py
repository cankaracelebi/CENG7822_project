"""
Utility functions for game mechanics
"""

from __future__ import annotations
import math
import random
from typing import Tuple, Optional
import numpy as np


def clamp(x: float, lo: float, hi: float) -> float:
    """Clamp value between low and high bounds"""
    return lo if x < lo else hi if x > hi else x


def vec_len(x: float, y: float) -> float:
    """Calculate vector length (magnitude)"""
    return math.hypot(x, y)


def normalize(x: float, y: float, eps: float = 1e-8) -> Tuple[float, float]:
    """Normalize a vector to unit length"""
    l = math.hypot(x, y)
    if l < eps:
        return 0.0, 0.0
    return x / l, y / l


def circle_collide(x1, y1, r1, x2, y2, r2) -> bool:
    """Check if two circles collide"""
    dx = x1 - x2
    dy = y1 - y2
    rr = r1 + r2
    return (dx * dx + dy * dy) <= (rr * rr)


def seed_everything(py_seed: Optional[int]):
    """Seed all random number generators"""
    if py_seed is None:
        return
    random.seed(py_seed)
    np.random.seed(py_seed)
