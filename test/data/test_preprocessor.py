from unittest import TestCase

import numpy as np
from data.preprocessor import Preprocessor


class TestPreprocessor(TestCase):
    def test__std_memory_efficient(self):

        random_int_example = self._get_random_input()
        self._test_std_memory_efficient_helper(random_int_example, 0)

        random_float_example = self._get_random_input(False)
        self._test_std_memory_efficient_helper(random_float_example, 2)


    def _test_std_memory_efficient_helper(self, sample, decimal):
        preprocessor = Preprocessor(sample)
        first = preprocessor._std_memory_efficient(sample)
        second = np.std(sample, axis=0)
        np.testing.assert_almost_equal(first, second, decimal=decimal)

    def _get_random_input(self, int_only=True):
        batch_size = 100
        img_height = 250
        img_width = 250
        channels = 3
        shape = (batch_size, img_height, img_width, channels)

        if int_only:
            max_int_value = 255
            result = np.random.randint(max_int_value, size=shape)
        else:
            result = np.random.rand(*shape)

        return result
