import numpy as np
from utils import log


class Preprocessor:
    """A Preprocessor is meant to run standardization on a given set of raw data."""

    def __init__(self, data: np.ndarray, standardization=True):
        """Create a new Preprocessor.

        :param data: The data which should be pre-processed.
        :param standardization: Whether to actually run standardization.
        """
        # a preprocessor may be completely inactive. this allows to use the same code whether standardization is active
        # or not
        self.active = standardization

        if self.active:
            # preprocessing (will change original data, too!)
            log.log(".. initialize preprocessing")  # train only on previous images
            self.preprocess_init(standardization,
                                 data)  # call this independently of the value of standardization!

            log.log(".. preprocessing initialized")

    def preprocess_init(self, standardization, X):
        """This method needs to be called exactly ONCE before any calls of preprocess_data()

        X must be the training data, no validation data allowed!
        The calculated values will be reused to preprocess any following data.

        X will be left unchanged.
        """
        log.log('.. apply standardization (mean + std): {}'.format(standardization))

        # standardization
        if standardization:
            # generate mean image
            # TODO we should be able to take the mean from _online_variance, too
            self._mean_image = np.mean(X, axis=0, dtype=np.float32)

            # normalize by dividing by the standard deviation => unit std
            # note, that X does not need to be a float. However, if X is an int, the result might differ slightly from
            # the exact value
            self._std = self._std_memory_efficient(X)
            self._std[self._std == 0] = 0.001  # prevent division by 0

        else:  # ensure that preprocess_data() will not apply standardization
            self._mean_image = 0
            self._std = 1.0

    def _online_variance(self, data):
        """Calculate the variance of data.

        Required for _std_memory_efficient().
        See http://stackoverflow.com/questions/15638612/calculating-mean-and-standard-deviation-of-the-data-which-does-not-fit-in-memory
        """
        n = 0
        mean = 0.0
        M2 = 0.0

        for x in data:
            n += 1
            delta = x - mean
            mean += delta / n
            delta2 = x - mean
            M2 += delta * delta2

        if n < 2:
            return float('nan')
        else:
            return M2 / (n - 1)

    def _std_memory_efficient(self, data):
        """Calculate the standard deviation of data by using less memory than np.std."""
        var = self._online_variance(data)
        return np.sqrt(var)

    def preprocess_data(self, Xarr):
        """Apply pre-processing on Xarr.

        preprocess_init needs to be called before

        will only use params of previously processed training data. so X can be whatever you want, validation data, too.

        The returned data will be in [-1, 1]. Technically, there is a small chance that a very few outliers won't
        be in that interval though.

        :param Xarr:
        :return:
        """
        if self.active:
            for X in Xarr:
                # standardization:
                # subtract mean image from all data => zero mean
                # (this will change the original dataset, too)
                X -= self._mean_image

                # normalize by dividing by the standard deviation => unit std
                X /= self._std
