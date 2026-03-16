import os
import abc
import time

import pygame
from pygame.event import Event
from pygame.surface import Surface

class Game:
    def __init__(self):
        self._last_time = time.perf_counter()
        self._last_second = self._last_time
        self._frame_count = 0
        os.environ['SDL_VIDEO_CENTERED'] = '1'

        pygame.init()

        screen_info = pygame.display.Info()
        height = int(screen_info.current_h * 0.9)
        width = int(height * (16 / 9))

        self._width = width
        self._height = height
        self._title = "PPO"

        self._window = pygame.display.set_mode((self._width, self._height))
        pygame.display.set_caption(self._title)
        self._running = True
    
    def is_running(self) -> bool:
        return self._running
    
    def next_frame_no_render(self, *args, **kw_args):
        if not self._running:
            return
        
        current_time = time.perf_counter()
        dt = current_time - self._last_time
        self._frame_count += 1
        fps_dt = current_time - self._last_second

        if fps_dt >= 1:
            self._last_second = current_time
            pygame.display.set_caption(f"{self._title} {int(self._frame_count / fps_dt)} fps")
            self._last_second = current_time
            self._frame_count = 0
        
        self._handle_events()
        self.update(dt)
        self._last_time = current_time

    def next_frame(self, *args, **kw_args):
        if not self._running:
            return
        
        current_time = time.perf_counter()
        dt = current_time - self._last_time
        self._frame_count += 1
        fps_dt = current_time - self._last_second

        if fps_dt >= 1:
            self._last_second = current_time
            pygame.display.set_caption(f"{self._title} {int(self._frame_count / fps_dt)} fps")
            self._last_second = current_time
            self._frame_count = 0
        
        self._handle_events()
        self.update(dt)
        self.render(self._window, *args, **kw_args)
        pygame.display.flip()
        self._last_time = current_time

    def loop_forever(self):
        while self._running:
            self.next_frame()

        self.quit()
    
    def quit(self):
        self._running = False
        pygame.quit()

    def _handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self._running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self._running = False
            self.on_event(event)
    
    @abc.abstractmethod
    def on_event(self, event: Event):
        pass
    
    @abc.abstractmethod
    def update(self, dt: float):
        pass

    @abc.abstractmethod
    def render(self, screen: Surface, *args, **kw_args):
        pass
