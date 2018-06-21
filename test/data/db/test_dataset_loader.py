from unittest import TestCase

import numpy as np
import copy
import config as cf
from data.db.dataset_loader import DatasetLoader


class TestDatasetLoader(TestCase):

    def test_shuffle_in_place(self):
        cf.set("shuffle_datasets_inplace", True)
        self._shuffle_helper()

    def test_shuffle_fast(self):
        cf.set("shuffle_datasets_inplace", False)
        self._shuffle_helper()

    def _shuffle_helper(self):
        # get Singleton dataset loader
        ds_loader = DatasetLoader(shuffle_new_data=True)

        # configure test data
        n_samples = 1000
        n_classes = 20
        img_width = 12
        img_height = 12

        # create test data
        # -> data_original and labels_original must have the same length
        # -> they must not have the same shape though, as some shuffling algorithms depend on these attributes
        #    (and the real data does not have the same shape as the real labels)
        labels_original = np.random.randint(n_classes, size=n_samples)

        # each data sample is a float image with one channel. all pixels of such an image are constantly the same
        # as the sample index.
        # -> using data points in this order allows us to simplify testing as they can be used as indices, too
        data_original = np.empty(shape=[n_samples, img_width, img_height], dtype=np.float32)
        for sample_index_and_value in range(n_samples):
            # set all (img_width x img_height) pixels of the current image to the same value as the image index
            data_original[sample_index_and_value] = sample_index_and_value

        # create a deep copy of that data
        data_copy = copy.deepcopy(data_original)
        labels_copy = copy.deepcopy(labels_original)

        # shuffle
        data_shuffled, labels_shuffled = ds_loader._shuffle(data_copy, labels_copy)

        if cf.get("shuffle_datasets_inplace"):
            # in-place changes imply that the provided parameters must have been changed by their reference and so they
            # must equal the returned values
            # (this does not guarantee in-place operations yet, but if this fails, in-place is impossible)
            np.testing.assert_equal(data_shuffled, data_copy, err_msg="Data array hasn't been changed by reference.")
            np.testing.assert_equal(labels_shuffled, labels_copy, err_msg="Label array hasn't been changed by reference.")

        # TODO can we test that data_shuffled is identical(not only equal!) to data_copy? (same for the labels)

        # first test: the shuffled data must differ from the original one
        np.testing.assert_equal(np.any(np.not_equal(data_shuffled, data_original)), True,
                                err_msg="Data array has not been changed.")
        np.testing.assert_equal(np.any(np.not_equal(labels_shuffled, labels_original)), True,
                                err_msg="Label array has not been changed.")

        # second test: ensure that the individual data points do still have the correct label
        for new_index in range(n_samples):
            new_data = data_shuffled[new_index]
            new_label = labels_shuffled[new_index]

            # the original data was build by using an index set, so we can use the data point itself as an index
            old_index = int(new_data[0][0])  # we can use any pixel to get the original index
            old_data = data_original[old_index]
            old_label = labels_original[old_index]

            # ensure that the testing itself works
            np.testing.assert_equal(new_data, old_data)

            # ensure that the shuffling didn't messed up the associated labels
            np.testing.assert_equal(new_label, old_label)

        # finally, we need to ensure that calling the shuffle method again using the same input, results in the same
        # output. this is important to ensure compatibility between datasets of different image size (cascade)
        data_copy2 = copy.deepcopy(data_original)
        labels_copy2 = copy.deepcopy(labels_original)
        data_shuffled2, labels_shuffled2 = ds_loader._shuffle(data_copy2, labels_copy2)
        np.testing.assert_equal(data_shuffled, data_shuffled2,
                                err_msg="Repeating the data shuffling changes the result.")
        np.testing.assert_equal(labels_shuffled, labels_shuffled2,
                                err_msg="Repeating the label shuffling changes the result.")
