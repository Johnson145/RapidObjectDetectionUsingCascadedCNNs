"""
This module bundles everything which is related to the structural organisation of any required input data.
It also provides iterators to allow accessing that data quite easily.

Note that some type hints used in this module are given by strings rather than the actually-required associated classes.
This is in deed correct as you can see e.g. here: https://stackoverflow.com/a/36286947/1665966
"""

import math
import random
from statistics import median, stdev
from typing import Dict, Optional, List

import numpy as np

import config as cf
from data.db.label import IID_FOREGROUND, IID_BACKGROUND
from data.preprocessor import Preprocessor
from utils import log

from abc import ABCMeta, abstractmethod

SPLIT_KEY_VAL = "valid"
SPLIT_KEY_TRAIN = "train"
SPLIT_KEY_TEST = "test"


class DataBundle(metaclass=ABCMeta):
    """A DataBundle is the most abstract way to organize data in a group.

    In everyday language, you may consider this a "dataset" already. However, we'll distinguish this further by
    introducing more specialized classes later on.

    Furthermore, this class should be kept as lightweight as possible. Why? Pay special attention to the Batch subclass
    which will be instantiated very often.
    """

    def __init__(self, images: np.ndarray, labels: np.ndarray, bottlenecks=None):
        """Create a new DataBundle.

        :param images: The images which will be used as input data for the used neural net.
                numpy array of type: cf.get("img_dtype"). Shape: [n_images, img_height, img_width, img_n_channels]
        :param labels: The labels represent the ground truth data. One label per image, so this is not(!) in one-hot.
                numpy array of type cf.get("label_dtype"). Shape: [n_images]
        :param bottlenecks: Optionally, you can provide the bottleneck value of each image. These values may be required
                along with the actual image data as an additional net input. The origin of the bottleneck values isn't
                specified in this class, but it should of course be the same for all images. This parameter must either
                be None or a numpy array of type np.float32 having the shape [n_images, bottleneck_size].
        """
        # TODO outsource bottleneck dtype to the config as we already have done for the image (cf.get("img_dtype"))

        # save given parameters
        self._images = images
        self._labels = labels
        self._bottlenecks = bottlenecks

        # derive further attributes
        self._n_samples = len(self.images)  # == len(self.labels)

    @property
    def images(self) -> np.ndarray:
        """Get the images."""
        return self._images

    @property
    def labels(self) -> np.ndarray:
        """Get the labels."""
        return self._labels

    @property
    def bottlenecks(self) -> Optional[np.ndarray]:
        """Get the labels."""
        return self._bottlenecks

    @property
    def n_samples(self) -> int:
        """Get the number of samples which are included in this DataBundle."""
        return self._n_samples


class DataBundleAdvanced(DataBundle, metaclass=ABCMeta):
    """DataBundleAdvanced extends DataBundle by features which are too heavy to be used in any kind of bundled data.

    So this can be used as the base class whenever objects of the new subclass won't be instantiated very often.
    """

    def __init__(self, images: np.ndarray, labels: np.ndarray, bottlenecks=None):
        """Create a new DataBundleAdvanced.

        :param images: See parent class.
        :param labels: See parent class.
        :param bottlenecks: See parent class.
        """
        # inform the user about potential memory improvements
        # (these refer to the casting done right after this check)
        if images.dtype != cf.get("img_dtype"):
            log.log("WARNING: copying image array, because it has the wrong dtype: {}".format(images.dtype))
        if labels.dtype != cf.get("label_dtype"):
            log.log("WARNING: copying label array, because it has the wrong dtype: {}".format(labels.dtype))

        # if the given parameters do not have the correct data type yet, we will convert them now
        images = np.asarray(images, dtype=cf.get("img_dtype"))
        labels = np.asarray(labels, dtype=cf.get("label_dtype"))
        # TODO why aren't we doing this for the bottlenecks? may they ever have varied?

        # now redirect the (maybe modified) parameters to the actual constructor of the parent class
        super(DataBundleAdvanced, self).__init__(images, labels, bottlenecks)

        # calculate and store the total number of foreground samples
        # (assuming that there are only two classes and the foreground class is described by "1")
        self._n_positive_samples = self.labels.sum()

    @property
    def n_positive_samples(self) -> int:
        """Get the number of positive samples given in this dataset."""
        return self._n_positive_samples


class Dataset(DataBundleAdvanced):
    """A Dataset object does not only bundle data, but also provide splits and pre-processing information.

    TODO outsource all of the following static label stuff. Or can we maybe already get rid of it because of the
            new label module?
    """

    # this index will be used as the next internal label id
    _next_new_iid = 0

    # only the bare ids.
    # ideally, this is identical with the numbers 0, .. , _next_new_iid-1
    # however, removing initialized iids again will result in a non-continuous list
    _label_internal_ids = []

    # label names as strings
    _label_name_by_internal_id = dict()

    # label ids
    _label_internal_id_by_name = dict()

    # _label_internal_id_by_db_id[db_id] = internal_id
    _label_internal_id_by_db_id = dict()

    # _label_db_id_by_internal_id[internal_id] = db_id
    _label_db_id_by_internal_id = dict()

    def __init__(self, images: np.ndarray, labels: np.ndarray, split_weights: List[float], preprocessor: Preprocessor,
                 name=None):
        """Create a new Dataset.

        Data won't be shuffled here to allow reconstructing results. If shuffling is required, do it earlier or use an
        iterator!

        :param images: See the parent class.
        :param labels: See the parent class.
        :param split_weights: A list containing the relative size of the individual splits. The first value describes
                the size of the train split, the second one refers to the validation split and the last one to the test
                set.  The sum of all values must be 1.
        :param preprocessor: The preprocessor which must be used before using any data of this dataset for inference.
        :param name: Optionally, this parameter may be a string describing this dataset object.
        """
        super(Dataset, self).__init__(images, labels)

        # save given parameters
        self._split_weights = split_weights
        self._name = name

        # The very first element added will define the shape of all future elements.
        # TODO fix incompatibility with the base class DataBundle. That base class already describes these shapes
        # explicitly.
        self._init_calculate_shapes()

        # Get split ids
        # (DON'T save all indices to do something like self.all_inputs[self.train_ids, ...], because it would duplicate
        #  all data)
        self._train_id_start = 0
        self._train_id_end = int(round(split_weights[0] * self.n_samples))
        self._valid_id_start = self._train_id_end
        self._valid_id_end = self._valid_id_start + int(round(split_weights[1] * self.n_samples))
        self._test_id_start = self._valid_id_end
        self._test_id_end = self._test_id_start + int(round(split_weights[2] * self.n_samples))

        # create actual splits
        self._train = DatasetSplit(self.images[self._train_id_start:self._train_id_end],
                                   self.labels[self._train_id_start:self._train_id_end])
        self._valid = DatasetSplit(self.images[self._valid_id_start:self._valid_id_end],
                                   self.labels[self._valid_id_start:self._valid_id_end])
        self._test = DatasetSplit(self.images[self._test_id_start:self._test_id_end],
                                  self.labels[self._test_id_start:self._test_id_end])

        # save preprocessor
        self._preprocessor = preprocessor

    def _init_calculate_shapes(self):
        """Calculate input and output shapes.

        TODO can we somehow get rid of this?
        """

        # shape of labels is already fixed
        self._shape_labels_single = 1  # 2 for one-hot

        if self._shape_labels_single > 1:
            self._shape_label_batch = [None, self._shape_labels_single]
        else:
            # if we use only one single value for the label, the batch size (None) is already the full shape
            self._shape_label_batch = [None]

        # The very first element added will define the shapes for all future elements.
        first_element = self.images[0]  # already needs to be a numpy array
        self._shape_data_single = first_element.shape
        # TODO 2-dimensional data should probably be extended by one dimension to allow convolution

        # use flexible batch size:
        self._shape_image_batch = Dataset.create_dynamic_batch_size_by_single_sample(first_element)

    @property
    def train(self) -> 'DatasetSplit':
        """Get the training split."""
        return self._train

    @property
    def valid(self) -> 'DatasetSplit':
        """Get the validation split."""
        return self._valid

    @property
    def test(self) -> 'DatasetSplit':
        """Get the test split."""
        return self._test

    def split(self, split_key: str) -> 'DatasetSplit':
        """Dynamically get the split described by the given key."""
        if split_key == SPLIT_KEY_TRAIN:
            return self.train
        elif split_key == SPLIT_KEY_TEST:
            return self.test
        elif split_key == SPLIT_KEY_VAL:
            return self.valid
        else:
            raise ValueError("Received invalid split key: {}".format(split_key))

    @property
    def splits(self) -> Dict[str, 'DatasetSplit']:
        """Get all splits matched by their keys."""
        splits = {
            SPLIT_KEY_TRAIN: self.train,
            SPLIT_KEY_VAL: self.valid,
            SPLIT_KEY_TEST: self.test
        }
        return splits

    @property
    def shape_image_batch(self) -> List:
        """Get the shape of a single batch containing images only."""
        return self._shape_image_batch

    @property
    def shape_label_batch(self) -> List:
        """Get the shape of a single batch containing labels only."""
        return self._shape_label_batch

    @property
    def name(self) -> Optional[str]:
        """Get the name of this dataset."""
        return self._name

    @staticmethod
    def create_dynamic_batch_size_by_single_sample(sample: np.ndarray) -> List:
        """Use the given sample to create the shape of a batch containing an arbitrary number of samples having the
        same shape as the given sample."""
        shape_single = list(sample.shape)
        shape_batch = shape_single
        shape_batch.insert(0, None)
        return shape_batch

    def log_stats(self):
        """Calculate and log stats about the used classes."""
        log.log("Dataset stats:")

        # we want the same stats for the complete dataset as well as for the single splits
        stat_groups = [
            ["complete dataset", self.labels],
            ["validation split", self.valid.labels],
            ["training split", self.train.labels],
            ["test split", self.test.labels],
        ]

        for stat_group in stat_groups:
            log_line = stat_group[0]
            labels = stat_group[1]

            log.log("- {}".format(
                log_line
            ))

            # calculate the absolute amount of each used label
            n_samples_per_iid_label = dict()
            for label in labels:
                if label not in n_samples_per_iid_label:
                    n_samples_per_iid_label[label] = 0
                n_samples_per_iid_label[label] += 1

            values_only = n_samples_per_iid_label.values()

            n_classes = len(n_samples_per_iid_label)

            if n_classes < 2:
                raise ValueError("Detected a dataset or split ({}) which contains less than two classes ({}).".format(
                    log_line,
                    n_classes
                ))
            elif n_classes > 2:
                log.log(".. different classes: {}".format(
                    n_classes
                ))

                log.log(".. absolute minimum: {}".format(
                    min(values_only)
                ))
                log.log(".. absolute maximum: {}".format(
                    max(values_only)
                ))
            else:  # n_classes == 2
                log.log(".. binary classification")
                log.log(".. foreground samples: {}".format(
                    n_samples_per_iid_label[IID_FOREGROUND]
                ))
                log.log(".. background samples: {}".format(
                    n_samples_per_iid_label[IID_BACKGROUND]
                ))

            log.log(".. average: {}".format(
                int(sum(values_only) / len(values_only))
            ))
            log.log(".. median: {}".format(
                int(median(values_only))
            ))
            log.log(".. standard variation: {}".format(
                int(stdev(values_only))
            ))

    @property
    def preprocessor(self) -> Preprocessor:
        """Get the preprocessor of this dataset."""
        return self._preprocessor


class Batch(DataBundle):
    """A batch is a smaller part of any larger DataBundle.

    For now, this class does not add additional features, but exists for semantic reasons.
    """


class DataBundleIterator(metaclass=ABCMeta):
    """This is the common base class of all iterators that can be used to iterate over a DataBundle.

    Iteration is always done using batches. If the dataset is smaller than the specified batch size, there will be only
    one batch though.

    An iterator may be used to process the associated DataBundle infinity times. If you want to process it only once
    (or almost once), you need to check the epoch counter accordingly.
    """

    def __init__(self, data_bundle: DataBundle, batch_size=None):
        """Create a new iterator for the given data bundle.

        :param data_bundle: The DataBundle which should be iterated over.
        :param batch_size: The number of samples per batch. If None or if greater than data_bundle.n_samples, all data
                will be used at once.
        """
        # save given parameters
        self._data_bundle = data_bundle
        self._batch_size_internal = batch_size

        # the number of batches that have already been provided by calling next_batch().
        self._n_provided_batches = 0

        # this is the epoch of the next(!) batch.
        # the very first epoch has the index 0 (and not 1).
        self._epoch = 0

    @property
    def batch_size(self) -> int:
        """Get the number of samples that are included in each batch.

        Note that, in contrast to the private attribute self._batch_size_internal, this will never be None.
        """
        if self._batch_size_internal is None or self._batch_size_internal > self._data_bundle.n_samples:
            # Special case: batch is no "real" batch: use all data at once
            return self._data_bundle.n_samples
        else:
            return self._batch_size_internal

    @property
    def epoch(self) -> int:
        """Get the epoch index of the epoch of the next(!) batch.

        The very first epoch has the index 0 (and not 1).
        """
        return self._epoch

    @property
    def in_first_epoch(self) -> bool:
        """Whether this iterator is still in the very first epoch.

        If you want to use the iterator only for a single epoch, validating this is a bit faster than evaluating
        next_batch_is_first_of_epoch().
        """
        return self.epoch == 0

    @property
    def n_batches_per_epoch(self) -> int:
        """Get the number of batches that will be processed in one epoch.

        As this object is an iterator, this method provides also the number of iterations that will be done per epoch.
        """
        return math.ceil(self._data_bundle.n_samples / self.batch_size)

    @property
    def n_provided_batches(self) -> int:
        """Get the number of batches that have already been provided by this iterator.

        This is the same as the index of the current/next iteration (sometimes also called "step").
        """
        return self._n_provided_batches

    @property
    def next_batch(self) -> Batch:
        """Get the next batch.

        Subclasses usually won't touch this method directly, but override _calculate_next_batch() instead.
        """
        # the subclass will handle the actual batch calculation
        result = self._calculate_next_batch()

        # we need to check this before changing any counters
        # (so self.next_batch_is_end_of_epoch still refers to the batch we've just calculated)
        if self.next_batch_is_last_of_epoch:
            self._epoch += 1

        # remember that we were here
        self._n_provided_batches += 1

        return result

    @property
    def next_batch_is_last_of_epoch(self) -> bool:
        """Whether the next batch is the very last batch of the current epoch.

        This can be used to implement epoch-wise iteration.
        """
        return (self._n_provided_batches + 1) % self.n_batches_per_epoch == 0

    @property
    def next_batch_is_first_of_epoch(self) -> bool:
        """Whether the next batch is the very first batch of the current epoch.

        This is True, iff the previous batch has been the very last one of the previous epoch.

        This can be used to implement epoch-wise iteration.
        """
        return self._n_provided_batches % self.n_batches_per_epoch == 0

    @abstractmethod
    def _calculate_next_batch(self) -> Batch:
        """This is the abstract internal method which will actually provide the next batch, but does neither touch the
        self._provided_batches counter nor the self._epoch attribute.

        :return:
        """
        return


class DeterministicIterator(DataBundleIterator):
    """A DeterministicIterator ensures that all data of the associated DataBundle is used exactly once during each
    epoch."""

    def __init__(self, data_bundle: DataBundle, batch_size=None, shuffle_every_epoch=True):
        """Create a new DeterministicIterator.

        :param data_bundle: See parent class.
        :param batch_size: See parent class.
        :param shuffle_every_epoch: If true, all images will be shuffled when a new epoch starts. This is the default
                behavior and it should be used for all training purposes. You can disable it to increase performance if
                the batch data is only used for prediction (e.g. accuracy calculation) and not for training purposes at
                all.
        """
        super(DeterministicIterator, self).__init__(data_bundle, batch_size)

        # Optionally, we may need to prepare a shuffled access
        # (in each case we will save the shuffle_every_epoch as an derived property here)
        if shuffle_every_epoch:
            # self._index_permutation will always contain a permutation of the available sample indices.
            # instead of iterating the original samples directly, we will iterate over that index permutation
            # (the very first permutation is just the original order. so the very first iteration may be a bit faster.)
            self._index_permutation = np.arange(self._data_bundle.n_samples)

            # alternative code snippet
            # self._index_permutation = np.random.permutation(list(range(self.data_bundle.n_samples)))
        else:
            self._index_permutation = None

        # prepare the very first batch
        self._next_batch_start = 0

    @property
    def shuffle_every_epoch(self) -> bool:
        """Whether data will be shuffled after each epoch.

        This is true, iff the shuffle_every_epoch parameter was set to True in the constructor. We don't save that
        attribute explicitly though.

        :return:
        """
        return self._index_permutation is not None

    def _calculate_next_batch(self) -> Batch:
        # update boundaries
        next_batch_end = min(self._next_batch_start + self.batch_size, self._data_bundle.n_samples)

        if self.shuffle_every_epoch:
            # get batch by using a slice of the current index permutation
            result = Batch(
                images=self._data_bundle.images[self._index_permutation[self._next_batch_start:next_batch_end]],
                labels=self._data_bundle.labels[self._index_permutation[self._next_batch_start:next_batch_end]],
                bottlenecks=self._data_bundle.bottlenecks[self._index_permutation[
                                                          self._next_batch_start:next_batch_end]] if self._data_bundle.bottlenecks is not None else None,
            )
        else:
            # get batch by directly accessing a slice of the stored data
            result = Batch(
                images=self._data_bundle.images[self._next_batch_start:next_batch_end],
                labels=self._data_bundle.labels[self._next_batch_start:next_batch_end],
                bottlenecks=self._data_bundle.bottlenecks[
                            self._next_batch_start:next_batch_end] if self._data_bundle.bottlenecks is not None else None,
            )

        # prepare the next batch calculation
        if self.next_batch_is_last_of_epoch:
            self._next_batch_start = 0

            # maybe we need to prepare the next epoch by shuffling
            if self.shuffle_every_epoch:
                # shuffling is done in-place
                random.shuffle(self._index_permutation)
        else:
            self._next_batch_start = next_batch_end

        return result


class RandomizedIterator(DataBundleIterator):
    """A RandomizedIterator creates batches based on a user-defined probability distribution.

    Note that, using this iterator never guarantees the coverage of all data. So the term "epoch" doesn't actually
    describe the single usage of each and every sample, but the usage of n samples where n is the total number of
    samples included in the associated DataBundle.
    """

    def __init__(self, data_bundle: DataBundle, probability_distribution: np.ndarray, batch_size=None):
        """Create a new RandomizedIterator.

        Note the changed parameter order which contrasts the parent class.

        :param data_bundle: See parent class.
        :param probability_distribution: The probability distribution used for generating batches. Must have exactly
                one entry per sample as given in the data_bundle. Values must be positive floats which sum up to 1.
        :param batch_size: See parent class.
        """
        super(RandomizedIterator, self).__init__(data_bundle, batch_size)
        self._probability_distribution = probability_distribution

        # create once a pool containing all available indices. we will pick a random choice out of these.
        # (we don't need(!) to persist these indices as we could just re-create them during each batch creation, but
        #  the latter is probably wasting a lot of runtime as we are creating lots of batches.)
        self._sample_indices = np.arange(self._data_bundle.n_samples)

    def _calculate_next_batch(self) -> Batch:
        # pick random choice of sample indices based on the given probability distribution
        # (we pick without replacement to force the batches to a certain amount of variety. as the batch size is usually
        #  much smaller than the total number of available samples, it's still almost certain that more likely samples
        #  will be chosen multiple times accross multiple batches)
        chosen_indices = np.random.choice(self._sample_indices, self.batch_size, replace=False,
                                          p=self._probability_distribution)
        # create and return a batch using the chosen samples
        return Batch(
            images=self._data_bundle.images[chosen_indices],
            labels=self._data_bundle.labels[chosen_indices],
            bottlenecks=self._data_bundle.bottlenecks[chosen_indices] if self._data_bundle.bottlenecks is not None else None,
        )


class DatasetSplit(DataBundleAdvanced):
    """A DatasetSplit is a smaller portion of a larger Dataset object."""

    def __init__(self, images: np.ndarray, labels: np.ndarray, bottlenecks=None, probability_distribution=None):
        """Create a new DatasetSplit.

        :param images: See parent class.
        :param labels: See parent class.
        :param bottlenecks: See parent class.
        :param probability_distribution: An optional probability distribution. If None is given, all samples will be
                distributed uniformly. This is used by the provided default iterator, but it may also be used to derive
                class weights.
        """
        super(DatasetSplit, self).__init__(images, labels, bottlenecks)
        self.set_probability_distribution(probability_distribution)

    def set_bottlenecks(self, bottlenecks: np.ndarray):
        """Replace the current bottlenecks with the new ones.

        Usually, we don't want to replace any values of a DataBundle. That's why this setter has been added to the
        DatasetSplit class only.

        :param bottlenecks: The new bottlenecks. Must include as many bottlenecks as self.n_samples.
        :return:
        """
        log.log("Replacing the split's current bottlenecks (old shape: {}, new shape: {}).".format(
            self.bottlenecks.shape if self.bottlenecks is not None else None,
            bottlenecks.shape if bottlenecks is not None else None,
        ))
        self._bottlenecks = bottlenecks

    def set_probability_distribution(self, probability_distribution: np.ndarray):
        """Set a new probability distribution.

        :param probability_distribution: See DatasetSplit.__init.
        :return:
        """
        self._probability_distribution = probability_distribution

        if self._probability_distribution is None:
            # the deterministic iterator will provide exactly as many positive samples as are given in the  dataset
            self._positive_proportion = float(self.n_positive_samples) / float(self.n_samples)
        else:
            # in case of a randomized access, the proportion is given by the probability distribution of the individual
            # samples
            self._positive_proportion = 0
            for i in range(self.n_samples):
                if self.labels[i] == IID_FOREGROUND:
                    self._positive_proportion += self._probability_distribution[i]

    def new_default_iterator(self, batch_size=None) -> DataBundleIterator:
        """Get a new default iterator which can be used to iterate over this DatasetSplit.

        Note that this isn't a saved attribute. So each time you use this method, you will receive a new iterator which
        hasn't provided any batches yet.

        If this DatasetSplit has a saved probability distribution, a RandomizedIterator will be returned. Otherwise,
        a DeterministicIterator will be used.

        :param batch_size: See DataBundleIterator.__init__().
        :return:
        """
        if self._probability_distribution is None:
            # TODO shuffle?
            return DeterministicIterator(self, batch_size)
        else:
            return RandomizedIterator(self, self._probability_distribution, batch_size)

    @property
    def positive_proportion(self) -> float:
        """Get the (mean) proportion of positive samples along all samples returned by a default iterator during one
        epoch.

        The result must be a float value in [0,1].

        This value can e.g. be used to calculate class weights.
        """
        return self._positive_proportion
