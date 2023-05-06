import time


class Timer:
    def __init__(self):
        self._total_time = {}
        self._count_time = {}
        self._last_time = {}
        self._keys = []

    def register_timer(self, key):
        self._total_time[key] = 0.0
        self._count_time[key] = 0
        self._last_time[key] = None
        self._keys.append(key)

    def tic(self, key):
        if key not in self._keys:
            self.register_timer(key)
        self._last_time[key] = time.time()

    def toc(self, key):
        assert key in self._keys
        assert self._last_time[key] is not None, "'tic' must be called before 'toc'."
        duration = time.time() - self._last_time[key]
        self._total_time[key] += duration
        self._count_time[key] += 1
        self._last_time[key] = None

    def get_time(self, key):
        assert key in self._keys
        return self._total_time[key] / (self._count_time[key] + 1e-12)

    def tostring(self, keys=None, verbose=True):
        if keys is None:
            keys = self._keys
        if verbose:
            log_strings = [f"{key}: {self.get_time(key):.3f}s" for key in keys if key in self._keys]
            format_string = ", ".join(log_strings)
        else:
            log_strings = [f"{self.get_time(key):.3f}s" for key in keys if key in self._keys]
            format_string = "time: " + "/".join(log_strings)
        return format_string

    def summary(self, keys=None):
        if keys is None:
            keys = self._keys
        summary_dict = {key: self.get_time(key) for key in keys}
        return summary_dict
