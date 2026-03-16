import time
import matplotlib.pyplot as plt
import tempfile
import threading
from src.debug.logger import logger

def profile_decorator(
    profiler: dict[str, list[float]] | None = None, print_each_time: bool = False, error_when_greater_than: float = -1
):
    def decorateur_wrapper(fonction):
        def fonction_wrapper(*args, **kw_args):
            start = time.perf_counter()
            result = fonction(*args, **kw_args)
            end = time.perf_counter()
            dt = end - start
            key = fonction.__name__

            if profiler is not None:
                if key in profiler.keys():
                    profiler[key].append(dt)
                else:
                    profiler[key] = [dt]

            if profiler is None or print_each_time:
                logger.debug(f"{key} : {round(1000 * dt, 2)} ms")

            if dt >= error_when_greater_than >= 0:
                raise TimeoutError()

            return result

        return fonction_wrapper

    return decorateur_wrapper


class profile_context:
    def __init__(
        self,
        key: str,
        profiler: dict[str, list[float]] | None = None,
        print_each_time: bool = False,
        error_when_greater_than: float = -1,
    ):
        self.key = key
        self.profiler = profiler
        self.print_each_time = print_each_time
        self.error_when_greater_than = error_when_greater_than

    def __enter__(self):
        self.start = time.perf_counter()

    def __exit__(self, exc_type, exc_value, traceback):
        self.end = time.perf_counter()
        dt = self.end - self.start

        if self.profiler is not None:
            if self.key in self.profiler.keys():
                self.profiler[self.key].append(dt)
            else:
                self.profiler[self.key] = [dt]

        if self.profiler is None or self.print_each_time:
            logger.debug(f"{self.key} : {round(1000 * dt, 2)} ms")

        if dt >= self.error_when_greater_than >= 0:
            raise TimeoutError()


def filter(profiler, total_rate_area_removed=0.01):
    for key in profiler.keys():
        profiler[key] = (sum(profiler[key]), len(profiler[key]))

    profiler = dict(sorted(profiler.items(), key=lambda item: item[1][0], reverse=True))
    result = dict(profiler)
    remaining = sum([val[0] for val in profiler.values()]) * (1 - total_rate_area_removed)

    for key, value in profiler.items():
        if remaining <= 0:
            del result[key]

        remaining -= value[0]

    return result


def plot_with_pygame(scale: float = 1):
    import pygame

    # Init pygame
    pygame.init()

    # Get screen resolution
    screen_info = pygame.display.Info()

    # Get the temporary directory
    temp_dir = tempfile.gettempdir()

    # Save the Matplotlib figure to an image file
    img_path = f"{temp_dir}/pygame_plot.png"
    plt.savefig(img_path, dpi=int(80 * scale * screen_info.current_h / 1080))

    # Clears the plot
    plt.clf()
    plt.cla()

    # Load the image into pygame
    img: pygame.Surface = pygame.image.load(img_path)
    # img = pygame.transform.scale(img, (img.get_width() // 2, img.get_height() // 2))

    fps = 60
    fpsClock = pygame.time.Clock()

    screen = pygame.display.set_mode((img.get_width(), img.get_height()))
    opened = True

    # Game loop
    while opened:
        screen.fill((0, 0, 0))
        screen.blit(img, (0, 0))

        pygame.display.flip()
        fpsClock.tick(fps)

        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                opened = False

    # Quit pygame
    pygame.quit()


def show_histogram(
    profiler: dict,
    in_pygame: bool = False,
    pygame_scale: float = 1,
    in_separate_thread=False,
    total_rate_area_removed=0.01,
) -> bool:
    if len(profiler) == 0:
        return False

    def job():
        profiler_filtered = filter(profiler, total_rate_area_removed)
        keys = list(profiler_filtered.keys())
        total = [value[0] for value in profiler_filtered.values()]
        average = [1000 * value[0] / value[1] for value in profiler_filtered.values()]

        fig, axs = plt.subplots(2, 1, figsize=(16, 12))
        width = 0.3
        bins = list(map(lambda x: x - width / 2, range(1, len(total) + 1)))

        axs[0].bar(bins, total, width=width)
        axs[1].bar(bins, average, width=width)

        axs[0].set_xticks(list(map(lambda x: x, range(1, len(total) + 1))))
        axs[0].set_xticklabels(keys, rotation=20, rotation_mode="anchor", ha="right")
        axs[0].set_ylabel("durée totale d'exécution (s)")

        axs[1].set_xticks(list(map(lambda x: x, range(1, len(total) + 1))))
        axs[1].set_xticklabels(keys, rotation=20, rotation_mode="anchor", ha="right")
        axs[1].set_ylabel("durée moyenne d'exécution (ms)")

        if in_pygame:
            plot_with_pygame(pygame_scale)
        else:
            plt.show()

    if in_separate_thread:
        threading.Thread(target=job).start()
    else:
        job()

    return True
