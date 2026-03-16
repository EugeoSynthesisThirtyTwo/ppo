import time

import pygame
from pygame.event import Event
from pygame.surface import Surface
from pygame.time import Clock
import torch
from math import sqrt

from src.window.game import Game
from src.pendulum.pendulum_on_rail import PendulumOnRail, PendulumOnRailWithFriction
from src.util import map

class UiPendulumOnRail(Game):
    def __init__(self):
        super().__init__()
        self._real_time = True
        self._clock = Clock()
    
    def on_event(self, event: Event):
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_SPACE:
                self._real_time = not self._real_time

    def update(self, dt: float):
        pass
    
    def _world_to_screen_vector(self, v: torch.Tensor) -> tuple:
        v = map(v, self._world_xlim[0], self._world_xlim[1], 0, self._width)
        return tuple(torch.tensor((v[0], (self._height + self._width) / 2 - v[1])).tolist())
    
    def _world_to_screen_ratio(self) -> float:
        return (self._width / (self._world_xlim[1] - self._world_xlim[0])).item()
    
    def render(self, screen: Surface, pendulum: PendulumOnRail, dt: float, index_in_batch: int = 0):
        self._world_xlim = pendulum.get_xlim()
        world_to_screen_ratio = self._world_to_screen_ratio()

        with torch.no_grad():
            state = pendulum.get_state()
            x_pos = state[index_in_batch, 0]
            x_spd = state[index_in_batch, 1]
            theta_pos = state[index_in_batch, 2]
            theta_spd = state[index_in_batch, 3]

            v0_world = torch.tensor((x_pos, 0))
            v1_world = v0_world + pendulum.get_radius() * torch.tensor((torch.sin(theta_pos), -torch.cos(theta_pos)))
            v0_screen = self._world_to_screen_vector(v0_world)
            v1_screen = self._world_to_screen_vector(v1_world)

        screen.fill((0, 0, 0))
        pygame.draw.line(screen, (255, 255, 255), v0_screen, v1_screen, 3)
        wagon_size = int(world_to_screen_ratio * sqrt(pendulum.get_surface_wagon()))
        wagon_rect = pygame.Rect(0, 0, wagon_size, wagon_size)
        wagon_rect.center = v0_screen
        pygame.draw.rect(screen, (30, 100, 255), wagon_rect)
        pendulum_radius = int(world_to_screen_ratio * sqrt(pendulum.get_surface_pendulum()) / 2)
        pygame.draw.circle(screen, (255, 0, 0), v1_screen, pendulum_radius)

        desired_framerate = 1 / dt if self._real_time else 0
        self._clock.tick(desired_framerate)