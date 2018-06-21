import os

import config as cf
from network.hyper_tuner import HyperTuner
from utils import log


class HyperTunerSuccessive(HyperTuner):
    """This Hyper Tuner tunes only one parameter at once.

    As soon as all values of one parameter have been tested, the next parameter will be tuned.
    Later tuning iterations always use the best value of the previously tested parameters.
    """

    def __init__(self, param_keys=None):
        """Create new HyperTunerSuccessive."""
        HyperTuner.__init__(self, param_keys)

        # index of current parameter's value
        self._curr_value_index = 0

        # empty list as a second argument is important here!
        self._current_parameter = ["", []]

        # keys are the names of parameters
        # value is a list with one entry for each run of this parameter. The entry equals the result dictionary merged
        # with the "param_value".
        self._results_per_parameter = dict()

    def _get_next_changes(self):
        # choose the next value for the current parameter
        self._curr_value_index += 1

        # if the last parameter config has been finished, go to the next one
        if self._curr_value_index >= self.current_parameter_number_of_values():
            self._select_next_parameter()

    def _select_next_parameter(self):
        """Select the next parameter that should be tuned (and its predefined values).

        If no more parameters are available, hyper tuning will be disabled.
        """
        # reset all possible changes caused by the last parameter
        self._curr_value_index = 0
        cf.reset()

        if len(self._parameter_selection) > 0:
            self._current_parameter = self._parameter_selection.pop(0)
        else:
            self.finalize()

    def finalize(self):
        HyperTuner.finalize(self)
        self._parameter_selection = []
        self._current_parameter = ["", [0]]

    def _apply_current_settings(self):

        # continue logger output
        log.log(' - parameter: {}'.format(self.current_parameter_name()))
        log.log(' - current value: {}'.format(self.current_parameter_value()))
        log.log(' - value list to be checked: {}'.format(self._current_parameter[1]))
        log.log(' - iteration {}/{} for this parameter'.format(
            self._curr_value_index + 1,
            self.current_parameter_number_of_values()
        ))

        # reset all previously tested parameters to the best values found so far
        self.restore_best_values()

        # save evaluations of a specific parameter in its own subfolder
        cf.set("log_dir", os.path.join(cf.get("log_dir_init"), self.current_parameter_name()))

        # actually change the current parameter to the current value
        self._override_configuration_entry(self.current_parameter_name(), self.current_parameter_value())

    def current_parameter_name(self):
        return self._current_parameter[0]

    def current_parameter_value(self):
        return self._current_parameter[1][self._curr_value_index]

    def current_parameter_number_of_values(self):
        return len(self._current_parameter[1])

    def receive_results(self, latest_results):
        """Receive the results for the currently tuned parameter and its used value."""
        if self.current_parameter_name() not in self._results_per_parameter:
            self._results_per_parameter[self.current_parameter_name()] = []

        # merge the result dictionary with the used parameter value and save everything
        new = latest_results.copy()
        new["param_value"] = self.current_parameter_value()
        self._results_per_parameter[self.current_parameter_name()].append(new)

    def restore_best_values(self):
        """Sets all parameters to the previously-tested value that has the best results."""

        # if the best validation accuracy of a parameter is still worse than this minimum, we will not restore it
        # as it's too close at 0.50 guessing accuracy. Usually, the already chosen parameter is at least better for the
        # training data set.
        min_val_acc_to_restore = 0.53

        for param_name, param_runs in self._results_per_parameter.items():

            # do not reset the currently tuned parameter
            if param_name != self.current_parameter_name():
                best_tuning_main_criteria = 0  # this holds the actually-searched optimum
                accuracy = 0  # this is the accuracy value belonging to best_tuning_main_criteria
                best_value = None  # the value used for the current parameter in order to achieve the collected result
                for run in param_runs:
                    if run[cf.get("tuning_main_criteria")] > best_tuning_main_criteria:
                        accuracy = run["accuracy"]
                        best_tuning_main_criteria = run[cf.get("tuning_main_criteria")]
                        best_value = run["param_value"]

                if best_value is not None:
                    if accuracy >= min_val_acc_to_restore:
                        log.log("Restoring {} to {} ({}).".format(param_name, best_value, best_tuning_main_criteria))
                        cf.set(param_name, best_value)
                    else:
                        log.log("NOT Restoring {}, cause all validation accuracies have been below {:.3f}%.".format(
                            param_name,
                            min_val_acc_to_restore * 100
                        ))

    def log_best_values(self):
        """Log only information about finished parameters. The current one will be excluded."""
        self.restore_best_values()

    # TODO unit test assert:
    # self.current_parameter_name() == "" or self._curr_value_index >= self.current_parameter_number_of_values()
    # <=> self.finished()
