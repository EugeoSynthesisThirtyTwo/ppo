import torch
from typing import Callable

DerivateFunction = Callable[[float, torch.Tensor], torch.Tensor]
"""
(t, y) -> y'
"""

def runge_kutta(t: float, y: torch.Tensor, dt: float, derivate: DerivateFunction) -> torch.Tensor:
    half_dt = dt / 2
    k1 = derivate(t, y)
    k2 = derivate(t + half_dt, y + half_dt * k1)
    k3 = derivate(t + half_dt, y + half_dt * k2)
    k4 = derivate(t + dt, y + dt * k3)
    return y + dt / 6 * (k1 + 2 * (k2 + k3) + k4)

def map(x, xmin, xmax, ymin, ymax):
    return (x - xmin) / (xmax - xmin) * (ymax - ymin) + ymin