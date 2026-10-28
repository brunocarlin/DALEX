"""Benchmark measurement helpers independent of the DALEX implementation."""
import threading

import numpy as np
import psutil


class RSS:
    def __enter__(self):
        self.process = psutil.Process()
        self.baseline = self.peak = self.process.memory_info().rss
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self.sample, daemon=True)
        self.thread.start()
        return self

    def sample(self):
        while not self.stop.wait(.002):
            self.peak = max(self.peak, self.process.memory_info().rss)

    def __exit__(self, *args):
        self.peak = max(self.peak, self.process.memory_info().rss)
        self.stop.set()
        self.thread.join()


def summary(values):
    return {'median': float(np.median(values)), 'q25': float(np.quantile(values, .25)),
            'q75': float(np.quantile(values, .75)), 'min': min(values), 'max': max(values)}
