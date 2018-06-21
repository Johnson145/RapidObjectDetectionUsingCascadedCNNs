import copy
import math
import os
import traceback

import numpy as np

import config as cf
import data.preprocessor as pp
from app.inference_app import InferenceApp
from data.cache import Cache
from data.datasets import Dataset, DataBundle
from data.db import label
from data.db.file_list_loader import FileListLoader
from utils import log
from utils.collage import CollageRemovedSamples, CollageClassDistribution


class DatasetLoader(FileListLoader):
    """An object of this class handles loading of the actually-used datasets.

    In contrast to the super class, this class is not handling any objects. Instead, the raw data along with the
    associated label is provided.
    """

    def __init__(self, shuffle_new_data=True):
        """Create new DatasetLoader.

        :param shuffle_new_data: If true, new (not cached) data will be shuffled once. Cached data will never be
            shuffled to allow reconstructing results.
        """
        self._shuffle_new_data = shuffle_new_data

        FileListLoader.__init__(self)

    def reset(self, reset_file_list=True):
        """Reset existing data.

        :param reset_file_list: whether the underlaying file list should be reloaded, too. usually it can be reused.
        :return:
        """
        if reset_file_list:
            FileListLoader.reset(self)

        # if a dataset hasn't been cached yet, it will be loaded file after file.
        # Those file samples will be collected in the following two vars before converted and merged to the numpy arrays
        # self._data and self._labels.
        self._temp_data_pool = None
        self._temp_label_pool = None

        # lazy load for actual data
        self._data = None
        self._labels = None
        self._dataset = None

    def dataset(self) -> Dataset:
        """Get a single object describing all loaded data."""
        # auto load data the first time it is requested
        if self._dataset is None:
            self._load_datasets()
        return self._dataset

    def _load_datasets(self):
        """Load all(!) requested datasets.

        Results will be merged in self._data and self._labels.
        Finally, everything will be saved in self._dataset.
        """
        # reset existing data
        self.reset()

        # load file lists, if not already done
        self.image_infos_per_dataset

        log.log("Load dataset(s): {}".format(
            cf.get("dataset_keys")
        ))

        # number of separated datasets
        n_datasets = len(cf.get("dataset_keys"))

        # check for combined cache result
        if n_datasets > 1:
            log.log("cache {}".format("enabled" if cf.get("cache_dataset") else "disabled"))
            self._data, self._labels, preprocessor = self._check_dataset_cache(self.dataset_key_combined)
            loaded_cached_data = self._data is not None and self._labels is not None
        else:
            loaded_cached_data = False

        if not loaded_cached_data:

            preprocessor = None
            if n_datasets > 1:
                log.log("Load each of the {} datasets separately and collect their data.".format(
                    n_datasets
                ))

                # load each dataset once to ensure that the individual ones are cached already
                # furthermore, we keep an eye on the required total memory without keeping the actual data in memory yet.
                # (this will mess up the runtime a bit, but we can save more valuable memory)
                log.log("Verifying that each dataset has been cached individually.")
                n_samples_per_dataset = []
                for dataset_key in cf.get("dataset_keys"):
                    _, dataset_labels, _ = self._load_dataset(dataset_key)
                    n_samples_per_dataset.append(len(dataset_labels))
                    dataset_labels = None  # manually release memory of the very last dataset
                n_samples_total = sum(n_samples_per_dataset)

                # pre-allocate numpy arrays to directly store all datasets in
                # (in contrast to using np.concatenate(data_per_set), this will prevent the need of doubling the required
                #  memory)
                log.log("Pre-allocating numpy arrays for the merged dataset.")
                self._data = np.empty(shape=[n_samples_total, cf.get("img_width"), cf.get("img_height"), 3],
                                                dtype=cf.get("img_dtype"))
                self._labels = np.empty(shape=[n_samples_total], dtype=cf.get("label_dtype"))

                # load each dataset separately and collect the data
                log.log("Actually starting to load the data")
                start_index = 0
                for ds_index, dataset_key in enumerate(cf.get("dataset_keys")):
                    end_index = start_index + n_samples_per_dataset[ds_index]
                    self._data[start_index:end_index], self._labels[start_index:end_index], _ = self._load_dataset(dataset_key)
                    start_index = end_index  # prepare next iteration

                # no need to filter anything here, because each single dataset was filtered already

                # shuffle all data
                # before(!) saving
                self._data, self._labels = self._shuffle(self._data, self._labels)

                # initialize the preprocessor (but do not preprocess anything yet)
                log.log("Create preprocessor once in order to cache it")
                preprocessor = pp.Preprocessor(self._data, cf.get("standardization"))

                # cache the merged data
                if cf.get("cache_dataset"):
                    log.log("Start caching merged dataset")
                    c = Cache()
                    c.save_dataset(self.dataset_key_combined, self._data, self._labels, preprocessor)
            else:
                # single dataset only
                # (already shuffled)
                log.log("Loading a single dataset only.")
                dataset_key = cf.get("dataset_keys")[0]
                self._data, self._labels, preprocessor = self._load_dataset(dataset_key)

        # filter all data again
        if cf.get("filter_dataset_after_caching"):
            self._data, self._labels = self._filter_data(self._data, self._labels,
                                                         dataset_key=self.dataset_key_combined)

        # build dataset object, no matter where the data is from
        log.log("Creating dataset object")
        self._dataset = Dataset(self._data, self._labels, cf.get("dataset_split"), preprocessor,
                                self.dataset_key_combined)

        # visualize the final dataset's class distribution
        CollageClassDistribution.visualize_class_distribution(self._dataset, "dataset_b_final_{}".format(self.dataset_key_combined))

        log.log("Finished dataset loading.")

    def _load_dataset(self, dataset_key: str):
        """Load a single dataset either from cache or from disk.

        While running, a non-cached dataset will temporarily add results to self._temp_data_pool and
        self._temp_label_pool.
        Finally, results will be returned as numpy arrays.
        """
        # check cache
        dataset_data, dataset_labels, preprocessor = self._check_dataset_cache(dataset_key)
        loaded_cached_data = dataset_data is not None and dataset_labels is not None

        # no cached data, load new data
        if not loaded_cached_data:

            # the following two vars need to be initialized as numpy arrays during each dataset loading
            self._temp_data_pool = None
            self._temp_label_pool = None

            log.log('Load dataset: {}'.format(dataset_key))

            if dataset_key not in cf.get("dataset_keys_available"):
                log.log("Error: Unknown dataset '" + dataset_key + "'")
                dataset_data = []
                dataset_labels = []
                preprocessor = None
            else:
                # actually process the data
                self._load_dataset_images(dataset_key)

                # convert to numpy arrays
                # (not needed if data is loaded from the cache)
                # log.log("Converting data into numpy arrays")
                # dataset_data = np.asarray(self._temp_data_pool, dtype=cf.get("img_dtype"))
                # dataset_labels = np.asarray(self._temp_label_pool, dtype=np.int32)
                dataset_data = self._temp_data_pool
                dataset_labels = self._temp_label_pool

                # ensure that we don't keep a global reference to this data, although we aren't using it anymore
                self._temp_data_pool = None
                self._temp_label_pool = None

                # visualize the dataset's class distribution before shuffling it
                temp_data_bundle = DataBundle(dataset_data, dataset_labels)
                CollageClassDistribution.visualize_class_distribution(temp_data_bundle,
                                                                      "dataset_a_temp_{}_a_new_ordered".format(dataset_key))
                # shuffle all data
                # before(!) saving
                dataset_data, dataset_labels = self._shuffle(dataset_data, dataset_labels)

                # visualize the dataset's class distribution after shuffling it
                temp_data_bundle = DataBundle(dataset_data, dataset_labels)
                CollageClassDistribution.visualize_class_distribution(temp_data_bundle,
                                                                      "dataset_a_temp_{}_b_new_shuffled".format(dataset_key))

                # intialize the preprocessor (but do not pre-process anything yet)
                log.log("Create preprocessor once in order to cache it")
                preprocessor = pp.Preprocessor(dataset_data, cf.get("standardization"))

                # if cache is enabled (but no data could be found there yet), save it before(!) converting data and labels
                # to numpy arrays, which might result in a memory error
                # UPDATE: this does not(!) help to prevent converting the data into numpy arrays. this would be done
                # anyway inside of the numpy save() method. try to cache here though, to try it before adding any(!) further
                # memory usage inside of the dataset constructor
                if cf.get("cache_dataset"):
                    log.log("Start caching of single dataset")
                    c = Cache()
                    c.save_dataset(dataset_key, dataset_data, dataset_labels, preprocessor)
        else:  # loaded dataset from cache
            # visualize the dataset's class distribution after loading it from the cache
            temp_data_bundle = DataBundle(dataset_data, dataset_labels)
            CollageClassDistribution.visualize_class_distribution(temp_data_bundle,
                                                                  "dataset_a_temp_{}_c_cached".format(dataset_key))

        return dataset_data, dataset_labels, preprocessor

    def _filter_data(self, data_input, labels_input, min_confidence=0.99999, dataset_key="unknown"):
        """Filters data_input and labels_input to remove all samples that seem to be foreground that was incorrectly
        labeled as background.

        :param data_input The data that should be filtered.
        :param labels_input The labels associated with data_input.
        :param min_confidence remove only samples which are predicted to be foreground with at least min_confidence confidence.
        :param dataset_key The dataset_key to which data_input and labels_input belong to. Used for debugging purposes only.
        :return:
        """
        log.log("Filtering input data to remove incorrectly-labeled background samples by using a pre-trained "
                "single cnn")

        # remember the currently requested image width
        target_img_width = cf.get("img_width")

        # load the default single net
        app_inference = InferenceApp()

        # loading the net may have changed the image width parameter in accordance to the persisted net
        # => we need to ensure that the two dimensions are compatible
        supported_img_width = app_inference.supported_img_width
        if supported_img_width != target_img_width:
            raise ValueError("Can not filter data, because the default single net has an input width of {}px, "
                             "but we need a width of {}px.".format(
                                    supported_img_width,
                                    target_img_width
                                ))

        # save filtered data indices in here
        indices_keep = []
        indices_drop = []

        # run in batches
        # TODO it would be sufficient to run only the positive samples
        start = 0
        while start < len(data_input):

            end = min(start + cf.get("max_batch_size"), len(data_input))

            input_batch = data_input[start:end]

            # run inference for all samples of the current batch
            class_probabilities_per_sample = app_inference.run_inference_on_raw_data(input_batch)

            # check the inference result of each single sample
            for batch_index in range(len(input_batch)):
                orig_index = start + batch_index
                ground_truth = labels_input[orig_index]
                best_guess_label_iid = class_probabilities_per_sample[batch_index].argmax()
                confidence = class_probabilities_per_sample[batch_index][best_guess_label_iid]

                # if we found labeled background data which looks like foreground..
                remove = ground_truth == label.IID_BACKGROUND and best_guess_label_iid == label.IID_FOREGROUND
                remove = remove and confidence >= min_confidence  # .. and we're confident enough

                # .. we'll remove it, otherwise we'll keep it in the filtered data
                if remove:
                    indices_drop.append(orig_index)
                else:
                    indices_keep.append(orig_index)

            # prepare next batch
            start += cf.get("max_batch_size")

        n_removed = len(data_input) - len(indices_keep)

        if n_removed > 0:
            log.log(
                "Removed {} {} samples which were labeled as background, but look like foreground with a confidence "
                "of at least {}".format(
                    n_removed,
                    dataset_key,
                    min_confidence
                ))

            # save a collage with the removed images
            dropped_images = data_input[indices_drop]
            collage = CollageRemovedSamples(dropped_images)
            collage_key = "removed_{}_{}".format(min_confidence, dataset_key)
            collage_file_path = collage.save_img_file(collage_key)
            log.log("collage of removed samples: {}".format(collage_file_path))

            # finally replace the loaded data with the filtered one
            data_filtered = data_input[indices_keep]
            labels_filtered = labels_input[indices_keep]
            return data_filtered, labels_filtered
        else:
            log.log("No samples were removed from {}".format(dataset_key))
            return data_input, labels_input

    def _shuffle(self, data, labels):
        """Shuffles the given data and labels, but only if shuffling has been activated in the first place.

        If cf.get("shuffle_datasets_inplace"), processing is done in-place and so the given references will be changed!
        This means that the returned arrays may actually not be required. However, we keep them to allow advanced unit
        testing and to be consistent with (cf.get("shuffle_datasets_inplace") == False).
        :param data:
        :param labels:
        """
        if self._shuffle_new_data:
            log.log("Shuffling dataset.")

            # shuffle in-place using the same seed for both elements
            # -> this does not work, because data and labels have different shapes
            # -> (their length is the same, but that's not enough)
            # -> these different shapes result in different permutations
            # -> in other words: it would work, iff data and labels had the same shape
            # random.Random(self.SEED).shuffle(data)
            # random.Random(self.SEED).shuffle(labels)

            # instead, we will shuffle using a pre-calculated permutation
            # this permutation uses a fixed seed, because we need to ensure that the same permutation is used whenever
            # the dataset contains the same number of samples. this is important especially for the cascade training,
            # because the datasets of different image sizes wouldn't be compatible otherwise)
            n_samples = len(data)  # == len(labels)
            ids_orig = list(range(n_samples))
            ids_shuffled = np.random.RandomState(seed=93452).permutation(ids_orig)

            if cf.get("shuffle_datasets_inplace"):
                log.log(".. forcing in-place shuffling. This may take a while.")
                # what we can do is using a modified version of the Bubblesort algorithm
                # (it isn't fast, but it works in-place and allows us to use the same permutation for both arrays)
                # TODO use a faster in-place(!) sorting algorithm which works with arrays (e.g. Comb sort)
                n = n_samples
                while n > 1:
                    self._progress(n_samples - n + 1, n_samples)
                    i = 0
                    while i < n-1:
                        if ids_shuffled[i] > ids_shuffled[i+1]:
                            # make the original swap action
                            ids_shuffled[i], ids_shuffled[i+1] = ids_shuffled[i+1], ids_shuffled[i]

                            # swap labels the same way
                            labels[i], labels[i+1] = labels[i+1], labels[i]

                            # swap data the same way, too
                            # (because of the multidimensional structure of data, we can't use the above code)
                            # (without making the deep copy, this doesn't work either!)
                            data_i_old = copy.deepcopy(data[i])
                            data[i] = data[i+1]
                            data[i+1] = data_i_old
                        i += 1
                    n -= 1
                # now: ids_shuffled == ids_orig
            else:
                # this is much faster, but it doesn't work in-place
                log.log(".. using faster shuffling without in-place restriction.")
                data = data[ids_shuffled, ...]
                labels = labels[ids_shuffled, ...]

        return data, labels

    def _check_dataset_cache(self, dataset_key):
        """Check the cache for the dataset with the key dataset_key.

        Got nothing to do with _check_filelist_cache().
        Will only load data from cache, when activated in the first place. If no data has been loaded, the returned
        values will be None.
        """
        dataset_data = None
        dataset_labels = None
        preprocessor = None

        if cf.get("cache_dataset"):
            c = Cache()
            cached_data = c.load_dataset(dataset_key)

            if cached_data is not None:
                dataset_data = cached_data[Cache.KEY_DATA_X]
                dataset_labels = cached_data[Cache.KEY_DATA_Y]
                preprocessor = cached_data[Cache.KEY_PREPROCESSOR]

        return dataset_data, dataset_labels, preprocessor

    def _load_dataset_images(self, dataset_key):
        """Load all images that have been collected in self._image_infos_per_dataset[dataset_key]."""
        n_files = len(self._image_infos_per_dataset[dataset_key])
        log.log(".. loading all {} images from the collected paths.".format(
            n_files
        ))

        # log the currently used image size
        log.log(".. input image dimension: {} x {} px".format(
            cf.get("img_width"),
            cf.get("img_height")
        ))

        # optionally, resize all files
        # (this code block could be removed completely, because img.raw_resized() will always resize the image on the
        #  first access. So removing this block would even speed up the total runtime. However, we keep it to get a
        #  separate notion about the time which is required exclusively for resizing)
        if cf.get("cache_resized_training_samples_individually"):
            # collect files that need to be resized
            log.log(".. -> collecting files that need to be resized")
            resize_files = []
            for img in self._image_infos_per_dataset[dataset_key]:
                if not os.path.isfile(img.path_resized) and os.path.isfile(img.path_original):
                    resize_files.append(img)
            log.log(".. -> found {} files that need to be resized".format(len(resize_files)))

            if len(resize_files) > 0:
                n_files_resized = 0
                for img in resize_files:
                    try:
                        _ = img.raw_resized()
                        n_files_resized += 1
                        self._progress(n_files_resized, len(resize_files))
                    except:
                        log.log(".. -> Error: Could not resize {}".format(
                            img.path_original
                        ))
                        log.log(traceback.format_exc())
                        # self._ignore_file(img.dataset_key, img.ignore_key)
                log.log(".. -> {} images were actually resized during this run.".format(n_files_resized))
        else:
            log.log(".. resizing is done online, so we don't need to check for any resized files here.")

        # actually load the files
        # use pre-known dimension to directly create numpy arrays
        log.log(".. Create empty numpy arrays of known dimensions to store images and labels in.")
        self._temp_data_pool = np.empty(shape=[n_files, cf.get("img_width"), cf.get("img_height"), 3],
                                        dtype=cf.get("img_dtype"))
        self._temp_label_pool = np.empty(shape=[n_files], dtype=np.int32)

        # in general, file_i and sample_i will be equal.
        # however, file_i is the i-th trial of transforming a file into a new sample. If the file is skipped, because
        # of an exception, sample_i will be less than file_i
        log.log(".. begin reading the actual image data")
        file_i = 0
        sample_i = 0
        for img in self._image_infos_per_dataset[dataset_key]:
            try:
                image_data = img.raw_resized()
                self._add_data_pair(image_data, img.label_internal_id, sample_i)
                sample_i += 1
            except FileNotFoundError:
                log.log(" .. Skipped {}, because the file could not be found".format(
                    img.path_resized
                ))
            except:
                self._files_skipped_unknown_error += 1
                log.log(" .. Skipped {}, because of an unexpected error:\n{}".format(
                    img.path_resized,
                    traceback.format_exc()
                ))

            # trigger progress, cause this file is finished
            file_i += 1
            if cf.get("max_samples") is None:  # no clue about the relative progress if we will break earlier
                self._progress(file_i, n_files)

            # check for maximum number of loaded samples
            if cf.get("max_samples") is not None and sample_i >= cf.get("max_samples"):
                log.log(
                    ".. Reached maximum number of {} samples. Cancelling further input processing.".format(
                        cf.get("max_samples")
                    ))
                break

        # shrink arrays, if some files have been skipped (because of any exception)
        if sample_i < len(self._temp_data_pool):
            log.log(".. Shrinking data again: {} samples have actually been used".format(
                sample_i
            ))
            self._temp_data_pool = self._temp_data_pool[0:sample_i]
            self._temp_label_pool = self._temp_label_pool[0:sample_i]

    def _add_data_pair(self, data_single, label, index):
        """Add a single sample pair to the internal temp pool.

        :param data_single: The raw data.
        :param label: The associated label.
        :param index: The target position. This is required as self._temp_data_pool are numpy arrays instead of
                        lists now.
        :return:
        """
        self._temp_data_pool[index] = data_single
        self._temp_label_pool[index] = label

    def _progress(self, file_i, num_total_files):
        """Print a log line that is similar to a progress bar indicating that file file_i of num_total_files was processed.
        You may skip intermediate steps of file_i and call this method only from time to time. However, you must either
        start with file_i == 1 or manually reset self._last_progress to 0 before.
        """
        if file_i < 1:
            # prevent missing initialization when starting with 0 instead of 1
            raise ValueError("You should not call the progess method with file_i < 1.")
        elif file_i == 1:
            # automatically initialize progress
            self._last_progress = 0

        file_progress = round(file_i / num_total_files * 100, 2)

        # show progress, whenever it increases of a new product pf min_per and itself
        min_per = 5

        if file_progress >= (self._last_progress + min_per):
            self._last_progress = math.floor(file_progress / min_per) * min_per
            log.log(" ### Progress: {0:.2f}% (file {1} of {2}) ###".format(file_progress, file_i, num_total_files))
