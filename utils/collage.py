"""This module provides features to visualize a bunch of small image samples by creating a collage."""
import abc
from typing import List, Dict

import cv2
import numpy as np

from data.datasets import SPLIT_KEY_VAL, SPLIT_KEY_TRAIN, DatasetSplit, DataBundle
from data.db.label import IID_FOREGROUND
import os
import config as cf
from utils import log


class Collage(metaclass=abc.ABCMeta):
    """Abstract base class to bundle some methods that can be used to create image collages."""

    def __init__(self):
        self._img = None

    @abc.abstractmethod
    def _create_img(self):
        """Create the actual collage image."""
        return

    @abc.abstractmethod
    def _sample_img_length(self):
        """The image length of a single sample integrated in _group_images()."""
        return

    @property
    def img(self):
        """Get the finished collage image.
        Available only after self._create_img() was called at least once.
        """
        if self._img is None:
            raise AttributeError("collage.img() was called before collage._create_img() could create it.")
        return self._img

    def save_img_file(self, collage_key: str) -> str:
        """Save self.img to disk by using an automatically-derived file path which is based on collage_key.

        :param collage_key: A short key that describes the purpose/content of the associated collage.
        :return: The file path which was used to save the image.
        """
        file_path = Collage._file_path_unique(collage_key)
        cv2.imwrite(file_path, self.img)
        return file_path

    @staticmethod
    def _file_path_unique(collage_key: str) -> str:
        """Get a full file path which is based on the given collage_key, but prevents overriding existing files.

        :param collage_key: A short key that describes the purpose/content of the associated collage.
        :return:
        """
        nr = 0
        while True:  # ensure unique file name (useful especially for cascades using the same session_key)
            # prepare file name without dir
            file_name = "{}_{}_{}.jpg".format(
                cf.get("session_key"),  # we will always start with the session key
                collage_key,  # this is user-defined
                nr  # this will ensure an unique file name
            )

            # prepend the dir: all collages are saved in cf.get("collages_dir")
            collage_file_path = os.path.join(cf.get("collages_dir"), file_name)

            if os.path.exists(collage_file_path):
                nr += 1
            else:
                break
        return collage_file_path

    def show_img(self):
        cv2.imshow('collage', self.img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def _group_images(self, samples, width, height):
        """Create a new image that contains images from the given samples organized in a grid of the given size.

        :param samples: samples to draw. TODO should be sorted such that the most important images are at the beginning.
        :param width: The width of the target canvas.
        :param height: The height of the target canvas.
        :return:
        """
        # create canvas
        canvas = np.zeros(shape=[height, width, 3], dtype=np.uint8)

        # iterate over samples as long as there are enough and we don't leave the canvas
        row = 0  # top left corner pixel of the next sample (not grid cell)
        col = 0  # top left corner pixel of the next sample (not grid cell)
        sample_i = 0
        while sample_i < len(samples) and ((row + self._sample_img_length()) < height) and ((col + self._sample_img_length()) < width):
            # fetch sample
            sample_raw = samples[sample_i]

            # convert sample into cv2 color space
            sample_raw = cv2.cvtColor(sample_raw, cv2.COLOR_RGB2BGR)

            # scale the sample
            sample_raw = cv2.resize(sample_raw, (self._sample_img_length(), self._sample_img_length()))

            # draw sample onto the canvas
            canvas[row:(row + self._sample_img_length()), col:(col + self._sample_img_length())] = sample_raw

            # prepare next iteration
            sample_i += 1  # always use the next sample
            col += self._sample_img_length()  # first of all, we stay in the same row, but move to the right
            if (col + self._sample_img_length()) >= width:  # leaving the end of the current row..
                col = 0  # jump to the beginning of the next row
                row += self._sample_img_length()

        return canvas

    def _draw_text(self, img, text, row, col):
        """Daw text on img at (col, row).

        :param img: The image to draw on.
        :param text: The text to draw.
        :param row: The y-coordinate of the text's top left corner.
        :param col: The x-coordinate of the text's top left corner.
        :return:
        """
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img, text, (col, row), font, 1, (255, 255, 255), 2, cv2.LINE_AA, bottomLeftOrigin=False)


class CollageRemovedSamples(Collage):
    """An object of this class visualizes a set of removed samples."""

    def __init__(self, removed_samples):
        Collage.__init__(self)
        self._removed_samples = removed_samples

        # configure the canvas
        self._IMG_TOTAL_WIDTH = 3840
        self._IMG_TOTAL_HEIGHT = 2160

        # samples are quite small for the human eye, so enlarge them a bit
        self._SAMPLE_IMG_LENGTH = 64

        self._img = self._create_img()

    def _create_img(self):
        """Create the actual collage image."""
        img = self._group_images(self._removed_samples, self._IMG_TOTAL_WIDTH, self._IMG_TOTAL_HEIGHT)
        return img

    def _sample_img_length(self):
        return self._SAMPLE_IMG_LENGTH


class CollageResampledSplits(Collage):
    """An object of this class visualizes some (not yet evaluated, but re-sampled) splits."""

    @staticmethod
    def visualize_train_valid(train_split, val_split):
        """Static helper method to save an image of a CollageResampledSplits containing training and validation data.

        :param train_split:
        :param val_split:
        :return:
        """
        collage = CollageResampledSplits(splits={
            SPLIT_KEY_TRAIN: train_split,
            SPLIT_KEY_VAL: val_split,
            # SPLIT_KEY_TEST: self._ds.test,  # usually we don't need this here
        })

        # save image file
        collage_key = "train_valid_resampled"
        collage_file_path = collage.save_img_file(collage_key)
        log.log("Saved image collage to visualize re-sampling of training and validation data to {}".format(
            collage_file_path))

    def __init__(self, splits: Dict[str, DatasetSplit]):
        """Initialize a new image collage based on the given samples.

        :param splits: Splits that should be drawn. Keys must use the DatasetSplit.SPLIT_KEY_xy constants.
        """
        Collage.__init__(self)

        # save params
        self._splits = splits

        # these splits will be printed onto the main canvas
        self._split_keys = sorted(splits.keys())

        # configure the canvas
        self._IMG_TOTAL_WIDTH = 3840
        self._IMG_TOTAL_HEIGHT = 2160
        self._HEADER_HEIGHT = 75
        self._PADDING_LEFT = 10

        # samples are quite small for the human eye, so enlarge them a bit
        self._SAMPLE_IMG_LENGTH = 64

        # automatically create the collage image
        self._img = self._create_img()

    def _sample_img_length(self):
        return self._SAMPLE_IMG_LENGTH

    def _create_img(self):
        """Create the actual collage image."""

        # create main canvas
        canvas_main = np.zeros(shape=[self._IMG_TOTAL_HEIGHT, self._IMG_TOTAL_WIDTH, 3], dtype=np.uint8)

        # remaining width for the inner table content
        INNER_WIDTH = self._IMG_TOTAL_WIDTH - self._PADDING_LEFT
        COL_INNER_WIDTH = int(INNER_WIDTH / len(self._split_keys))
        COL_INNER_HEIGHT = self._IMG_TOTAL_HEIGHT - self._HEADER_HEIGHT

        # draw split by split
        for split_i in range(len(self._split_keys)):
            split_key = self._split_keys[split_i]

            # draw column name = split key
            txt_col_px = int(self._PADDING_LEFT + (split_i + 0.5) * COL_INNER_WIDTH)
            self._draw_text(canvas_main, split_key, int(self._HEADER_HEIGHT / 2), txt_col_px)

            # draw split samples
            imgs_col = self._PADDING_LEFT + split_i * COL_INNER_WIDTH
            imgs_row = self._HEADER_HEIGHT
            batch = self._splits[split_key].new_default_iterator(500).next_batch
            canvas_main[imgs_row:(imgs_row + COL_INNER_HEIGHT), imgs_col:(imgs_col + COL_INNER_WIDTH)] = self._group_images(
                batch.images, COL_INNER_WIDTH, COL_INNER_HEIGHT)

        return canvas_main


class CollagePartitionedDataBundles(Collage):
    """An object of this class can visualize multiple DataBundles arranged in columns.

    Furthermore, each DataBundle is partitioned so that the partitions can be arranged in rows.
    """
    def __init__(self, data_bundles: Dict[str, DataBundle]):
        """Initialize a new image collage based on the given data_bundles.

        :param data_bundles: DataBundles that should be drawn matched by unique keys describing the bundles appropriately.
        """
        Collage.__init__(self)

        # save params
        self._data_bundles = data_bundles

        # these keys will be printed onto the main canvas
        self._data_bundles_keys = sorted(data_bundles.keys())

        # configure the canvas
        self._IMG_TOTAL_WIDTH = 3840
        self._IMG_TOTAL_HEIGHT = 2160
        self._HEADER_HEIGHT = 75
        self._PADDING_LEFT = 10
        self._COL_LEFT_WIDTH = 75

        # limit the maximum samples which are stored in one partition
        # (this is not meant to reduce the amount of shown samples, but the amount of samples which are stored, but
        #  never shown)
        self._MAX_N_SAMPLES_PER_PARTITION = 4000

        # samples are quite small for the human eye, so enlarge them a bit
        self._SAMPLE_IMG_LENGTH = 64

        # Divide all given data_bundles into subsets (e.g. TP, FP, FN, TN).
        self._partitions = self._partition_data_bundles()

        # ensure that none of the partitions exceed the max samples limit
        # (subclasses should already pay attention to it, but who knows)
        for data_bundle_key, partitions in self._partitions.items():
            for partition_key, partition in partitions.items():
                if len(partition) > self._MAX_N_SAMPLES_PER_PARTITION:
                    log.log("Warning: Found a collage partition which exceeds the internal max samples limit."
                            " Please check the subclass implementation.")
                    self._partitions[data_bundle_key][partition_key] = partition[0:self._MAX_N_SAMPLES_PER_PARTITION]

        # automatically create the collage image
        self._img = self._create_img()

    def _sample_img_length(self):
        return self._SAMPLE_IMG_LENGTH

    def _create_img(self):
        """Create the actual collage image."""

        # create main canvas
        canvas_main = np.zeros(shape=[self._IMG_TOTAL_HEIGHT, self._IMG_TOTAL_WIDTH, 3], dtype=np.uint8)

        # automatically defer the available row names = partition names by using the first data_bundle
        # => this assumes that all data_bundles are using the same partitions
        # row_names = [CollageEvaluation.KEY_TP, CollageEvaluation.KEY_FP, CollageEvaluation.KEY_FN,
        #              CollageEvaluation.KEY_TN]
        row_names = sorted(self._partitions[self._data_bundles_keys[0]].keys())
        n_partitions = len(row_names)

        # draw the very left column containing row labels
        remaining_height = self._IMG_TOTAL_HEIGHT - self._HEADER_HEIGHT
        ROW_HEIGHT = int((self._IMG_TOTAL_HEIGHT - self._HEADER_HEIGHT) / n_partitions)
        row_row_px = dict()
        for i in range(len(row_names)):
            name = row_names[i]
            row_px = int(self._HEADER_HEIGHT + (i + 0.5) * ROW_HEIGHT)
            self._draw_text(canvas_main, name, row_px, self._PADDING_LEFT)

            # remember which information begins at which y coordinate
            row_row_px[name] = self._HEADER_HEIGHT + i * ROW_HEIGHT

        # remaining width for the inner table content
        INNER_WIDTH = self._IMG_TOTAL_WIDTH - self._COL_LEFT_WIDTH - self._PADDING_LEFT
        COL_INNER_WIDTH = int(INNER_WIDTH / len(self._data_bundles_keys))

        # define the size of an inner cell
        CELL_WIDTH = int(INNER_WIDTH / len(self._data_bundles_keys))
        CELL_HEIGHT = ROW_HEIGHT

        # draw data_bundle by data_bundle
        for data_bundle_i in range(len(self._data_bundles_keys)):
            data_bundle_key = self._data_bundles_keys[data_bundle_i]

            # draw column name = data_bundle key
            txt_col_px = int(self._COL_LEFT_WIDTH + self._PADDING_LEFT + (data_bundle_i + 0.5) * COL_INNER_WIDTH)
            self._draw_text(canvas_main, data_bundle_key, int(self._HEADER_HEIGHT / 2), txt_col_px)

            # draw data_bundle samples
            imgs_col = self._COL_LEFT_WIDTH + self._PADDING_LEFT + data_bundle_i * COL_INNER_WIDTH
            for row_name in row_names:  # each row contains one element of the data_bundle's partition
                imgs_row = row_row_px[row_name]
                images = self._data_bundles[data_bundle_key].images[self._partitions[data_bundle_key][row_name]]
                canvas_main[imgs_row:(imgs_row + CELL_HEIGHT), imgs_col:(imgs_col + CELL_WIDTH)] = self._group_images(
                    images, CELL_WIDTH, CELL_HEIGHT)

        return canvas_main


    @abc.abstractmethod
    def _partition_data_bundles(self) -> Dict[str, List[int]]:
        """Divide all given data_bundles into subsets (e.g. TP, FP, FN, TN).

        Actually, the partitions won't contain the samples themselves, but their indices.

        The sublcass implementation should already consider self._MAX_N_SAMPLES_PER_PARTITION
        """
        return


class CollageEvaluation(CollagePartitionedDataBundles):
    """An object of this class visualizes some evaluated samples."""

    KEY_TP = "TP"
    KEY_FP = "FP"
    KEY_FN = "FN"
    KEY_TN = "TN"

    @staticmethod
    def visualize_train_valid(train_split, val_split, train_predictions, val_predictions):
        """Static helper method to save an image of a CollageEvaluation containing training and validation data.

        :param train_split:
        :param val_split:
        :param train_predictions:
        :param val_predictions:
        :return:
        """
        # create the collage
        collage = CollageEvaluation(splits={
            SPLIT_KEY_TRAIN: train_split,
            SPLIT_KEY_VAL: val_split,
            # SPLIT_KEY_TEST: self._ds.test,  # usually we don't need this here
        }, predictions={
            SPLIT_KEY_TRAIN: train_predictions,
            SPLIT_KEY_VAL: val_predictions,
            # SPLIT_KEY_TEST: self.predict(self._ds.test),  # usually we don't need this here
        })

        # save the image file
        collage_key = "split_evaluation"
        collage_file_path = collage.save_img_file(collage_key)
        log.log("Saved image collage to visualize evaluation to {}".format(collage_file_path))

    def __init__(self, splits: Dict[str, DatasetSplit], predictions: Dict[str, List[int]]):
        """Initialize a new image collage based on the given samples along with their predictions.

        :param data_bundles: Splits that should be drawn. Keys must use the DatasetSplit.SPLIT_KEY_xy constants.
        :param predictions: Same keys as splits. Values are the predicted class labels of the associated split.
        """
        # save subclass params
        self._predictions = predictions

        # call parent constructor (splits=data_bundles)
        super(CollageEvaluation, self).__init__(splits)

    def _partition_data_bundles(self) -> Dict[str, List[int]]:
        """Divide all given splits into subsets (TP, FP, FN, TN).

        Actually, the partitions won't contain the samples themselves, but their indices.
        """
        result = dict()

        for split_key in self._data_bundles_keys:
            split = self._data_bundles[split_key]
            predictions = self._predictions[split_key]

            split_result = {
                CollageEvaluation.KEY_TP: [],
                CollageEvaluation.KEY_FP: [],
                CollageEvaluation.KEY_FN: [],
                CollageEvaluation.KEY_TN: [],
            }

            for i in range(split.n_samples):
                ground_truth = split.labels[i]
                predicted = predictions[i]

                target_partition_key = None
                if ground_truth == predicted:  # correct prediction
                    if predicted == IID_FOREGROUND:
                        target_partition_key = CollageEvaluation.KEY_TP
                    else:
                        target_partition_key = CollageEvaluation.KEY_TN
                else:  # false prediction
                    if predicted == IID_FOREGROUND:
                        target_partition_key = CollageEvaluation.KEY_FP
                    else:
                        target_partition_key = CollageEvaluation.KEY_FN

                # only append data if this particular partition isn't empty yet
                if target_partition_key is not None \
                        and len(split_result[target_partition_key]) < self._MAX_N_SAMPLES_PER_PARTITION:
                    split_result[target_partition_key].append(i)
                    # TODO we can stop early, if all partitions are full

            result[split_key] = split_result

        return result


class CollageClassDistribution(CollagePartitionedDataBundles):
    """An object of this class visualizes multiple DataBundles partitioned by their class labels."""

    KEY_FOREGROUND = "+"
    KEY_BACKGROUND = "-"

    @staticmethod
    def visualize_class_distribution(data_bundles: Dict[str, DataBundle], collage_key_prefix: str):
        """Static helper method to save an image of a CollageClassDistribution containing arbitrary DataBundles.

        :param data_bundles:
        :param collage_key_prefix: A key which describes all of the given data_bundles.
        :return:
        """
        # ideally, the first parameter is already a dictionary
        # however, if we want to visualize only a single DataBundle, this would make things quite complicated
        # so we allow specifying a single DataBundle as well and just wrap it right here
        if isinstance(data_bundles, DataBundle):
            data_bundles = {
                collage_key_prefix: data_bundles
            }

        # create the collage
        collage = CollageClassDistribution(data_bundles)

        # save the image file
        collage_key = "{}_class_distribution".format(collage_key_prefix)
        collage_file_path = collage.save_img_file(collage_key)
        log.log("Saved image collage to visualize class distribution to {}".format(collage_file_path))

    def _partition_data_bundles(self) -> Dict[str, List[int]]:
        """Divide all given data_bundles based on their sample labels.

        Actually, the partitions won't contain the samples themselves, but their indices.
        """
        result = dict()

        for data_bundle_key in self._data_bundles_keys:
            data_bundle = self._data_bundles[data_bundle_key]

            data_bundle_result = {
                CollageClassDistribution.KEY_FOREGROUND: [],
                CollageClassDistribution.KEY_BACKGROUND: [],
            }

            for i in range(data_bundle.n_samples):
                ground_truth = data_bundle.labels[i]

                target_partition_key = None
                if ground_truth == IID_FOREGROUND:
                    target_partition_key = CollageClassDistribution.KEY_FOREGROUND
                else:
                    target_partition_key = CollageClassDistribution.KEY_BACKGROUND

                # only append data if this particular partition isn't empty yet
                if target_partition_key is not None \
                        and len(data_bundle_result[target_partition_key]) < self._MAX_N_SAMPLES_PER_PARTITION:
                    data_bundle_result[target_partition_key].append(i)
                    # TODO we can stop early, if all partitions are full

            result[data_bundle_key] = data_bundle_result

        return result
