import abc
import os

import sys

from utils import log
from utils.time_watcher import TimeWatcher
import config as cf
from subprocess import call


class BaseApp(metaclass=abc.ABCMeta):
    """This class is the base class of all "apps" created in this project."""

    def __init__(self, run_now=True):
        """Create a new BaseApp.

        :param run_now: Whether this app should run right now.
        """
        self._time_watcher = None

        # overclock GPU
        if cf.get("overclock_gpu_shell") is not None:
            log.log("automatically overclocking the GPU by using the following shell script: {}".format(
                cf.get("overclock_gpu_shell")
            ))
            call(cf.get("overclock_gpu_shell"), shell=True)

        if run_now:
            self.run()

    def run(self):
        """Run this app.

        This method is wrapping the main method to introduce some additional events.
        :return:
        """
        self._time_watcher = TimeWatcher(os.path.basename(sys.argv[0]).replace(".py", ""))
        try:
            self._main()
        except KeyboardInterrupt:
            log.log("WARNING: User interrupted progress.")
            self._on_cancel()

        self._on_finished()

        self._time_watcher.stop()

    @abc.abstractmethod
    def _main(self):
        """This method will be called on object initialization to run the actual programme."""
        return

    def _on_cancel(self):
        """This method will be called when the user interrupted the main method."""
        return

    def _on_finished(self):
        """This method will be called when the main method is done (either finished regularly or cancelled)."""
        # save log files
        log.log_set_name(self.__class__.__name__)

        # we don't flush the log here, because other apps include each other
        log.log_save(cf.get("log_dir"), flush=False)
        return
