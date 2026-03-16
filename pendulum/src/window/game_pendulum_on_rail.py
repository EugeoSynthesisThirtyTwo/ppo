import pygame
from pygame.event import Event
from pygame.surface import Surface
import torch
from math import sqrt

from src.window.game import Game
from src.pendulum.pendulum_on_rail import PendulumOnRail, PendulumOnRailWithFriction
from src.util import map

class GamePendulumOnRail(Game):
    def __init__(self):
        super().__init__()
        self._pendulum = PendulumOnRailWithFriction(1, 50, 0.5, randomized_state=False, batch_size=1, device="cpu")
        # self._pendulum = PendulumOnRail(1, 50, 0.5, randomized_state=False, batch_size=1, device="cpu")
        # self._pendulum = PendulumOnRailWithFriction(1, 50, 0.5, randomized_state=False, batch_size=1024 * 1024, device="cuda:0")
        # self._pendulum = PendulumOnRail(1, 50, 0.5, randomized_state=False, batch_size=1024 * 1024, device="cuda:0")
        self._world_xlim = self._pendulum.get_xlim()
        self._push_force = 0
    
    def on_event(self, event: Event):
        push_force = 350

        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_LEFT:
                self._push_force -= push_force
            elif event.key == pygame.K_RIGHT:
                self._push_force += push_force 
        if event.type == pygame.KEYUP:
            if event.key == pygame.K_LEFT:
                self._push_force += push_force
            elif event.key == pygame.K_RIGHT:
                self._push_force -= push_force

    def update(self, dt: float):
        steps = 8

        for _ in range(steps):
            self._pendulum.next_state(dt / steps, torch.tensor(self._push_force))
    
    def _world_to_screen_vector(self, v: torch.Tensor) -> list:
        v = map(v, self._world_xlim[0], self._world_xlim[1], 0, self._width)
        return torch.tensor((v[0], (self._height + self._width) / 2 - v[1])).tolist()
    
    def _world_to_screen_ratio(self) -> float:
        return self._width / (self._world_xlim[1] - self._world_xlim[0])
    
    def render(self, screen: Surface, index_in_batch=0):
        world_to_screen_ratio = self._world_to_screen_ratio()

        with torch.no_grad():
            state = self._pendulum.get_state()
            x_pos = state[index_in_batch, 0]
            x_spd = state[index_in_batch, 1]
            theta_pos = state[index_in_batch, 2]
            theta_spd = state[index_in_batch, 3]

            v0_world = torch.tensor((x_pos, 0))
            v1_world = v0_world + self._pendulum._radius * torch.tensor((torch.sin(theta_pos), -torch.cos(theta_pos)))
            v0_screen = self._world_to_screen_vector(v0_world)
            v1_screen = self._world_to_screen_vector(v1_world)

        screen.fill((0, 0, 0))
        pygame.draw.line(screen, (255, 255, 255), v0_screen, v1_screen, 3)
        wagon_size = int(world_to_screen_ratio * sqrt(self._pendulum.get_surface_wagon()))
        wagon_rect = pygame.Rect(0, 0, wagon_size, wagon_size)
        wagon_rect.center = v0_screen
        pygame.draw.rect(screen, (30, 100, 255), wagon_rect)
        pendulum_radius = int(world_to_screen_ratio * sqrt(self._pendulum.get_surface_pendulum()) / 2)
        pygame.draw.circle(screen, (255, 0, 0), v1_screen, pendulum_radius)