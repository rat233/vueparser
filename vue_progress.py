"""Optional, throttled stage reporting shared by the command-line exporters."""
import time


def console_progress(message):
    print(message, flush=True)


class ExportProgress:
    def __init__(self, callback=None, name=''):
        self.callback = callback
        self.name = name
        self.started = self.last_update = time.perf_counter()

    def __call__(self, message):
        if self.callback is not None:
            now = time.perf_counter()
            self.callback(f'[{self.name} +{now-self.started:.1f}s] {message}')
            self.last_update = now

    def count(self, stage, completed, total):
        if self.callback is not None and (
                completed in (0, total) or time.perf_counter()-self.last_update >= 2):
            percent = completed*100//total if total else 100
            self(f'{stage}: {completed}/{total} ({percent:.0f}%)')
