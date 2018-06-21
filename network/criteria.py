"""This module collects common configurations for all available evaluation criterias."""
# all available criterias will be saved in the following dictionary
_all_criterias = dict()


class Criteria:
    """An object of this class represents one criteria that can be used to evaluate net performance."""

    def __init__(self, key: str, format_func, acc_mean: bool):
        """

        :param key: A unique string that identifies this criteria.
        :param format_func: A python function (of this module) that formats the raw value of such a criteria to a human readable string.
        :param acc_mean: If true, multiple values of this criteria will be accumulated by using the mean. Otherwise, we use the sum.
        """
        self._key = key
        self._format_func = format_func
        self._acc_mean = acc_mean

        global _all_criterias
        if self._key in _all_criterias:
            raise ValueError("The criteria '{}' does already exist.".format(self._key))
        else:
            _all_criterias[self._key] = self

    @property
    def acc_mean(self):
        return self._acc_mean

    def format(self, value):
        return self._format_func(value)


# format functions

def format_perc_3(value):
    """Format a number in [0, 1] to a percentage number with 3 digits after the dot."""
    return "{:.3f}%".format(value * 100)


def format_3(value):
    """Round a number to 3 digits after the dot."""
    return "{:.3f}".format(value)


def format_int(value):
    """Cast the given value to an integer (and then to a string)."""
    return "{}".format(int(value))


def get(key: str) -> Criteria:
    """Get the global criteria with the given key."""
    global _all_criterias

    # usually, the criteria with the given key must habe been created before
    if key not in _all_criterias:
        # however, the f-beta scores are created dynamically, but use the same configuration as the f1_score.
        # so we create these criterias dynamically on the first access
        if (key.endswith("_score") or key.endswith("_score_diffable")) and key.startswith("f_"):
            f1_score = _all_criterias.get("f1_score")
            Criteria(key, f1_score._format_func, f1_score.acc_mean)

    # now, the criteria must really exist
    if key not in _all_criterias:
        raise ValueError("The criteria {} has not been configured yet.".format(key))

    return _all_criterias.get(key)


# create those configurations once
Criteria("accuracy", format_perc_3, True)
Criteria("f1_score", format_3, True)
Criteria("false_negatives", format_int, False)
Criteria("false_negatives_diffable", format_int, False)
Criteria("false_positives", format_int, False)
Criteria("false_positives_diffable", format_int, False)
Criteria("precision", format_perc_3, True)
Criteria("precision_diffable", format_perc_3, True)
Criteria("recall", format_perc_3, True)
Criteria("recall_diffable", format_perc_3, True)
Criteria("samples_negative", format_int, False)
Criteria("samples_positive", format_int, False)
Criteria("true_negatives", format_int, False)
Criteria("true_positives", format_int, False)
Criteria("true_positives_diffable", format_int, False)
Criteria("true_negative_rate", format_perc_3, True)
