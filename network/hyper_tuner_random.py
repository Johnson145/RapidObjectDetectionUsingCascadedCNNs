import random

from network import criteria
from network.hyper_tuner import HyperTuner
from utils import log
import config as cf


class HyperTunerRandom(HyperTuner):
    """This Hyper Tuner always tunes the complete list of requested parameters at once.

    Each round, a new random set of parameter values will be used.
    """

    def __init__(self, param_keys=None):
        """Create new HyperTunerRandom."""
        HyperTuner.__init__(self, param_keys)

        # get initial changes
        self._get_next_changes()

        # all results will be saved in the following list
        # (no further grouping as in the HyperTunerSuccessive)
        self._results = []

    def _apply_current_settings(self):
        for param_key, param_value in self._next_changes.items():
            # apply
            self._override_configuration_entry(param_key, param_value)
            # log
            log.log(' - {} = {}'.format(
                param_key,
                param_value
            ))

    def _get_next_changes(self):
        self._next_changes = self._choose_random_settings()

    def _choose_random_settings(self):
        """Randomly select one value for each requested parameter.

        The values are still taken from the predefined value sets.
        """
        random_changes = dict()
        for param_key in self._param_keys:
            random_changes[param_key] = random.choice(self._value_sets_per_param[param_key])

        return random_changes

    def receive_results(self, latest_results):
        """Receive the latest training results achieved with self._next_changes."""
        # add information about the used values
        new = latest_results.copy()
        new["config_changes"] = self._next_changes

        # save
        self._results.append(new)

    def log_best_values(self):
        if len(self._results) > 0:
            # get the best result
            best_main_criteria_value = 0
            best_config = None
            for run in self._results:
                if run[cf.get("tuning_main_criteria")] > best_main_criteria_value:
                    best_main_criteria_value = run[cf.get("tuning_main_criteria")]
                    best_config = run["config_changes"]

            # print it
            log.log("The following configuration changes achieved the best results so far ({}):".format(
                criteria.get(cf.get("tuning_main_criteria")).format(best_main_criteria_value)
            ))
            for param_key, param_value in best_config.items():
                log.log(' - {} = {}'.format(
                    param_key,
                    param_value
                ))
        else:
            log.log("Warning: The Hyper Tuner didn't receive any results yet. So it can't print the best config.")
