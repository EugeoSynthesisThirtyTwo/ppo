import time


class FpsCounter:
    def __init__(self):
        self._times: list[float] = []

    def update(self) -> float:
        current_time = time.perf_counter()
        self._times.append(current_time)

        while current_time - self._times[0] > 1:
            del self._times[0]

        return self.get()

    def get(self) -> float:
        if not self._times:
            return 0

        delta_time = self._times[-1] - self._times[0]

        if delta_time <= 0:
            return 0

        return (len(self._times) - 1) / delta_time
