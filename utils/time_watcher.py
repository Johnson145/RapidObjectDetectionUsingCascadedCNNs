from time import time
from datetime import timedelta

from utils import log


class TimeWatcher:
    """A TimeWatcher object can be used to evaluate the runtime of individual code parts."""

    def __init__(self, name: str):
        """Create a new TimeWatcher.

        :param name: The name of the time watcher. Will be included in logs to identify it.
        """
        self._name = name
        self.start()

    def start(self):
        """Starts counting the time."""
        self._start_time = time()
        self._end_time = None
        self._elapsed_seconds = None

        log.log("TimeWatcher Start: {}".format(
            self._name
        ))

    def stop(self):
        """Stop the previously-started runtime evaluation."""
        self._end_time = time()
        self._elapsed_seconds = self._end_time - self._start_time

        log.log("TimeWatcher Stop {}: {}".format(
            self._name,
            self.seconds_to_str(self._elapsed_seconds)
        ))

    @property
    def elapsed_seconds(self):
        """Get the elapsed time in seconds.

        If called before self.stop() was called the first time, None will be returned.
        """
        return self._elapsed_seconds

    @staticmethod
    def seconds_to_str(time_in_seconds: int) -> str:
        """Get a readable time representation of the given duration in seconds."""
        return str(timedelta(seconds=time_in_seconds))
