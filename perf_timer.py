import time


class PerformanceTimer:
    
    def __init__(self):
        self._start = None
        self._stop = None
    
    def start(self):
        self._start = time.perf_counter()
        
    def stop(self):
        self._stop = time.perf_counter()
        print(f"Elapsed time: {self._stop - self._start} s")