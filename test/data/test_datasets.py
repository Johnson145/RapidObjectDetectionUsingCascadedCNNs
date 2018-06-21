from unittest import TestCase

import numpy as np

from data.datasets import Batch


class TestBatch(TestCase):

    def test__constructor_and_getters(self):
        # create test data
        n_samples = 1000
        n_classes = 20
        labels = np.random.randint(n_classes, size=n_samples)
        img_width = 64
        img_height = img_width
        images = np.empty(shape=[n_samples, img_width, img_height], dtype=np.float32)
        bottleneck_size = 128
        bottlenecks = np.empty(shape=[n_samples, bottleneck_size], dtype=np.float32)

        # create test object
        batch = Batch(images, labels, bottlenecks)

        # compare input with output
        np.testing.assert_equal(images, batch.images)
        np.testing.assert_equal(labels, batch.labels)
        np.testing.assert_equal(bottlenecks, batch.bottlenecks)
