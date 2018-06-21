import abc

import config as cf
from utils import log


class HyperTuner(metaclass=abc.ABCMeta):
    """A HyperTuner can be used to modify the current configuration to explore (better) hyper parameter values."""

    def __init__(self, param_keys=None):
        """Create a new HyperTuner.

        :param param_keys: List with keys of all parameters that should be tuned. If None, nothing will be tuned at all.
        """
        self._init_value_sets_per_param()

        # save a list with keys of all parameters that should be tuned.
        self._param_keys = self._filter_param_keys(param_keys)

        self._required_iterations = 0
        self._iter_total = 0  # total number of done iterations (not counting repetitions)

        # Load predefined parameter sets for all requested parameters
        # TODO don't save _parameter_selection for the randomized hyper tuner
        self._parameter_selection = []
        for key in self._param_keys:
            self._parameter_selection.append([key, self._value_sets_per_param[key]])
            self._required_iterations += len(self._value_sets_per_param[key])

    @abc.abstractmethod
    def _get_next_changes(self):
        """Do what ever is necessary to determine a new set of changes that is going to be applied."""
        return

    @abc.abstractmethod
    def _apply_current_settings(self):
        """Internal help method that will be called inside of tune to allow different tuning methods."""
        return

    @abc.abstractmethod
    def receive_results(self, latest_results):
        """Receive a dictionary containing the latest training results."""
        return

    @abc.abstractmethod
    def log_best_values(self):
        """Log all parameter values that have been used to achieve the best results (so far)."""
        return

    def _filter_param_keys(self, param_keys_original):
        """ Filter param_keys_original such that only keys with existing preconfigurations are kept.

        :param param_keys_original:
        :return: Return always a list. Even if param_keys_original is None.
        """
        param_keys_filtered = []
        if param_keys_original is not None:
            for key in param_keys_original:
                if key in self._value_sets_per_param:
                    param_keys_filtered.append(key)
                else:
                    log.log("Error: Can't tune parameter {}, because of missing preconfiguration.".format(key))
        return param_keys_filtered

    def _init_value_sets_per_param(self):
        """Define possible values for supported parameters.

        This does not only cover parameters, that are included in self._param_keys, but all supported ones.
        """
        self._value_sets_per_param = dict()

        # configurations to automatically evaluate hyperparameters:
        self._value_sets_per_param["learning_rate_init"] = [0.00001, 0.0001, 0.001, 0.01, 0.1, 0.5, 0.05, 0.005]

        self._value_sets_per_param["batch_size"] = [128, 256, 400, 500, 600, 1000, 2000, 5000]

        self._value_sets_per_param["learning_rate_decay"] = [0.5, 0.7, 0.9, 0.95, 0.99, 1]

        self._value_sets_per_param["momentum"] = [0, 0.25, 0.5, 0.72, 1]

        self._value_sets_per_param["dropout_rate"] = [0.25, 0.75, 0.5, 1.0]

        self._value_sets_per_param["optimizer"] = [1, 0, 2]

        self._value_sets_per_param["standardization"] = [True, False]

        self._value_sets_per_param["fc1_size"] = [16, 32, 64, 128, 256, 512]

        self._value_sets_per_param["L2_regularization_strength"] = [0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1]

        self._value_sets_per_param["L1_regularization_strength"] = [0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1]

        self._value_sets_per_param["cascade_n_nets"] = [3, 4, 5, 6, 7, 10, 15]

        self._value_sets_per_param["f_beta_cascade_loss_very_last"] = [True, False]

        # when changing these values, keep in mind that min_beta must not be greater than max_beta
        self._value_sets_per_param["min_beta"] = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        self._value_sets_per_param["max_beta"] = [16, 20, 24, 28, 32, 36, 48]

        # max pooling layer
        self._value_sets_per_param["pooling_size"] = [2, 3]
        self._value_sets_per_param["pooling_stride"] = [1, 2, 3]


        # convolutional layer
        self._value_sets_per_param["conv_stride"] = [1, 2, 3]
        self._value_sets_per_param["conv_filter_size"] = [2, 3, 4, 5, 6]
        self._value_sets_per_param["conv_filter_sizes"] = [[6],
                                                           [9],
                                                           [32],
                                                           [64],
                                                           [128],
                                                           [6, 6],
                                                           [9, 9],
                                                           [32, 32],
                                                           [64, 64],
                                                           [32, 64],
                                                           [64, 32],
                                                           [128, 128],
                                                           [6, 6, 6],
                                                           [32, 32, 32],
                                                           [3, 6, 9],
                                                           [9, 6, 3],
                                                           [9, 9, 9],
                                                           [6, 6, 6],
                                                           [12, 12, 12],
                                                           ]

        # online data augmentation
        self._value_sets_per_param["data_augmentation_online"] = [True, False]  # use this with caution, as it will make tuning of the following options senseless
        self._value_sets_per_param["dao_horizontal_flip"] = [True, False]
        self._value_sets_per_param["dao_vertical_flip"] = [True, False]
        # self._value_sets_per_param["dao_allow_vertical_flipping_of_foreground"] = [True, False]  # no upside-down faces
        self._value_sets_per_param["dao_max_rotation_angle"] = [0.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 45.0,
                                                                60.0, 90.0, 120.0, 180.0]
        self._value_sets_per_param["dao_max_foreground_rotation_angle"] = [0.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0,
                                                                           35.0, 45.0]
        self._value_sets_per_param["dao_crop_probability"] = [0.25, 0.5, 0.75, 0.9]
        self._value_sets_per_param["dao_crop_min_percent"] = [0.75, 0.8, 0.85, 0.9, 0.95]
        self._value_sets_per_param["dao_color_distortion"] = [True, False]
        self._value_sets_per_param["dao_color_distortion_fast_mode"] = [True, False]

    def tune(self, repeat_last_one=False):
        """Tune the current configuration by changing the actual parameters.

        Usually, this means selecting the next value for the current parameter.
        If all values for the current parameter have been used, the next parameter will be selected.
        :param repeat_last_one: If true, don't choose the next value or parameter. Repeat the very last configuration.
        :return:
        """

        # if this run is not only a repetition of the last one, choose the next config
        # (repetitions do not count as a separate iteration)
        if not repeat_last_one:
            self._get_next_changes()
            self._iter_total += 1

        # if we're not done yet
        if not self.finished:

            # begin logger output
            log.log('HYPER TUNING')
            log.log(' - iteration {}/{} in total'.format(
                self._iter_total,
                self.required_iterations()
            ))

            # actually override any values based on the method defined in a subclass
            self._apply_current_settings()

    def _override_configuration_entry(self, cf_key, value):
        """Override a single value of the configuration dictionary.

        This is more or less the same as just modifying the dictionary directly, but allows handling of some special
        cases.
        """
        # actually change the current parameter to the current value
        cf.set(cf_key, value)

        # evaluating momentum does only make sense, if the correct optimizer has been chosen
        if cf_key == "momentum":
            log.log("Automatically overriding the optimizer to Momentum (2).")
            cf.set("optimizer", 2)

        # enable online data augmentation if any related option should be tuned
        elif cf_key.startswith("dao_") and not cf.get("data_augmentation_online"):
            log.log("Enabling data_augmentation_online to allow tuning some subconfigs.")
            cf.set("data_augmentation_online", True)

        elif cf_key == "dao_color_distortion_fast_mode" and not cf.get("dao_color_distortion"):
            log.log("Enabling color distortions to tune the associated fast mode.")
            cf.set("data_augmentation_online", True)

        elif cf_key == "dao_crop_min_percent" and cf.get("dao_crop_probability") <= 0:
            log.log("WARNING: can not tune dao_crop_min_percent, if augmented cropping is disabled")

        elif cf_key == "dao_max_foreground_rotation_angle" and cf.get("dao_max_rotation_angle") <= 0:
            log.log("WARNING: can not tune dao_max_foreground_rotation_angle, if augmented rotations are disabled")

    @property
    def finished(self):
        """True, if everything is done."""
        return self._iter_total > self.required_iterations()

    def required_iterations(self):
        return self._required_iterations

    def finalize(self):
        """Finalize all current work.

        Regularly, this will be called after the very last iteration. If called before, this will cancel all undone
        iterations.
        TODO does this need to be called once again for the random part?
        """
        # if not already done, jump to the end
        if not self.finished:
            self._iter_total = self.required_iterations() + 1

        log.log("Hypertuning disabled")
