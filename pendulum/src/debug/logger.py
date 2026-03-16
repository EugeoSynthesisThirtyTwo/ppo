import contextlib
import os
import sys

import logging
from rich.logging import RichHandler

logging.basicConfig(
    level=logging.DEBUG,
    # format="%(filename)s:%(lineno)d - %(message)s",
    format="%(message)s",
    datefmt="[%H:%M:%S.%f]",
    handlers=[RichHandler(rich_tracebacks=True)]
)

logger = logging.getLogger("rich")

for noisy_module in [
    'matplotlib', 'asyncio', 'grpc', 'PIL', 'urllib3', 'tensorflow',
    'matplotlib.font_manager'
]:
    logging.getLogger(noisy_module).setLevel(logging.WARNING)

@contextlib.contextmanager
def suppress_stdout_stderr():
    with open(os.devnull, 'w') as fnull:
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        sys.stdout = fnull
        sys.stderr = fnull
        try:
            yield
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr
