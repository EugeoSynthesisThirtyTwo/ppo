import pygame
from pygame.event import Event
from pygame.surface import Surface

import torch
import torch.nn.functional as F
import numpy as np

from src.window.game import Game


class JeuDeLaVie(Game):
    def __init__(self):
        torch.set_grad_enabled(False)
        self._winsize = (3200, 1800)
        super().__init__(self._winsize)
        self._device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self._grid = torch.randint(0, 2, (1, 1, self._winsize[0], self._winsize[1]), dtype=torch.float16).to(self._device)
        self._kernel = torch.ones((1, 1, 3, 3), dtype=torch.float16).to(self._device)
        self._kernel[0, 0, 1, 1] = 0
        
    def on_event(self, event: Event):
        pass
    
    def update(self, dt: float):
        for i in range(32):
            neighbors = F.conv2d(self._grid, self._kernel, padding=1, stride=1)
            self._grid = ((self._grid == 1) & (neighbors == 2)) | (neighbors == 3)
            self._grid = self._grid.to(torch.float16)

    def render(self, screen: Surface, *args, **kw_args):
        arr = self._grid.to(torch.uint8).squeeze().cpu().numpy() * 255
        surface = pygame.surfarray.make_surface(arr)
        surface = pygame.transform.scale(surface, self._winsize)
        screen.blit(surface, (0, 0))
