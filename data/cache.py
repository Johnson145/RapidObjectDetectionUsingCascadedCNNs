import os
import pickle
from typing import Optional, Dict

import config as cf
import numpy as np
from data.datasets import Dataset
from data.preprocessor import Preprocessor
from utils import log
from utils.singleton import Singleton


class Cache(metaclass=Singleton):
    """This singleton class handles persisting and loading of user-defined data in order to reduce the need of
    re-calculation.
    """

    # this version will be added to each created cached file. if the version number stored in a loaded file is
    # smaller than this, it will not be used
    # -> increase it whenever old cache data is incompatible
    _cache_version = 8

    # the following keys will be used to access the cached data in a dictionary
    # they will also be used as the name of the file they will be saved to

    # Dataset keys
    KEY_DATA_X = "x"
    KEY_DATA_Y = "y"
    KEY_CACHE_VERSION = "cache_version"
    KEY_CONFIG = "config"
    KEY_NEXT_NEW_IID = "next_new_iid"
    KEY_LABEL_IDS = "label_ids"
    KEY_LABEL_NAME_BY_ID = "label_name_by_id"
    KEY_LABEL_ID_BY_NAME = "label_id_by_name"
    KEY_PREPROCESSOR = "preprocessor"

    # file list loader keys
    CATEGORY_PREFIX_FILE_LIST_LOADER = "file_list_loader_"
    KEY_FLL_IMG_INFOS_PER_DS = "image_infos_per_dataset"
    KEY_FLL_IMG_INFOS = "image_infos"
    KEY_FLL_IMG_INFOS_PER_IID = "image_infos_per_iid_label"

    def __init__(self):
        """Create the singleton object."""
        # ensure that the root cache path does exist
        if not os.path.exists(self._ds_path("")):
            os.makedirs(self._ds_path(""))

        # inform the user about deprecated cache data
        deprecated_cache_num = self._count_old_cache_version_folders()
        if deprecated_cache_num > 0:
            log.log("Found {} deprecated cache folders. Go ahead and delete them manually.".format(deprecated_cache_num))

    def _ds_path(self, dataset_key: str, suffix=None, suffix_extension=".npy") -> str:
        """Get the file path to the specified dataset cache.

        Note, datasets are cached in a slightly-different structure than other objects.

        :param dataset_key: The key identifying the dataset which should be cached.
        :param suffix: An additional suffix which will be appended to the base file name.
        :param suffix_extension: The file extension. Either ".npy" or ".p".
        :return:
        """
        # each version gets its own subdirectory
        path = os.path.join(self._base_path("dataset"), "{}x{}".format(
            cf.get("img_width"),
            cf.get("img_height")
        ))

        # each dataset, too
        path = os.path.join(path, dataset_key)

        # suffix is used for the actual file name + .npy
        if suffix is not None:
            path = os.path.join(path, suffix + suffix_extension)

        return path

    def _base_path(self, category, suffix=None, suffix_extension=".npy") -> str:
        """Get the file path to the given non-dataset cache element.

        :param category: Cache elements are grouped in categories.
        :param suffix: This suffix should describe the individual element of the associated category.
        :param suffix_extension: The file extension. Either ".npy" or ".p".
        :return:
        """
        # each version gets its own subdirectory
        path = os.path.join(cf.get("cache_path_root"), "v{}".format(
            self._cache_version,
        ))

        # each dataset, too
        path = os.path.join(path, category)

        # suffix is used for the actual file name + .npy
        if suffix is not None:
            path = os.path.join(path, suffix + suffix_extension)

        return path

    def load_dataset(self, dataset_key: str) -> Optional[Dict]:
        """Return the requested dataset parts structured in a dictionary, or None if not available/valid.

        :param dataset_key: The key identifying the dataset which should be loaded from cache.
        :return:
        """
        # if a cached file does exist
        if os.path.isfile(self._ds_path(dataset_key, self.KEY_CACHE_VERSION)):

            log.log("Found cached data")
            log.log(".. loading cached dataset {}".format(dataset_key))
            loaded_data = dict()

            for file_name in os.listdir(self._ds_path(dataset_key)):
                key = os.path.splitext(file_name)[0]
                if file_name.endswith(".npy"):
                    loaded_data[key] = np.load(self._ds_path(dataset_key, key))
                elif file_name.endswith(".p"):
                    with open(self._ds_path(dataset_key, key, ".p"), "rb") as input_file:
                        loaded_data[key] = pickle.load(input_file)

            log.log(".. dataset has been loaded successfully from the cache")

            # restore class attributes
            # TODO do not use private vars, ensure that no conflict between different datasets can exist
            log.log(".. loading global meta information about available labels")
            Dataset._next_new_iid = loaded_data[self.KEY_NEXT_NEW_IID]
            Dataset._label_internal_ids = loaded_data[self.KEY_LABEL_IDS]
            Dataset._label_name_by_internal_id = loaded_data[self.KEY_LABEL_NAME_BY_ID]
            Dataset._label_internal_id_by_name = loaded_data[self.KEY_LABEL_ID_BY_NAME]

            return loaded_data

        else:
            log.log("Cache for dataset {} is empty".format(
                dataset_key
            ))
            return None

    def load(self, category, data_keys=None) -> Optional[Dict]:
        """Return the requested non-dataset data from cache, or None if not available/valid.

        :param category: The category which should be (partially) loaded.
        :param data_keys: If None, all found files of that category will be loaded. Otherwise, only the specified ones.
        :return:
        """
        # if a cached category folder does exist
        if not self.is_empty(category, data_keys):

            if data_keys is None:
                log.log("Loading everything from cached category {}".format(category))
            else:
                log.log("Loading {} from cached category {}".format(category, data_keys))

            loaded_data = dict()

            for file_name in os.listdir(self._base_path(category)):
                key = os.path.splitext(file_name)[0]
                if data_keys is None or key in data_keys:
                    if file_name.endswith(".npy"):
                        loaded_data[key] = np.load(self._base_path(category, key))
                    elif file_name.endswith(".p"):
                        with open(self._base_path(category, key, ".p"), "rb") as input_file:
                            loaded_data[key] = pickle.load(input_file)

            log.log(".. category {} has been loaded successfully from the cache".format(category))
            return loaded_data

        else:
            if data_keys is None:
                log.log("Cache for category {} is completely empty".format(
                    category
                ))
            else:
                log.log("Cache for category {} and data keys {} is empty".format(
                    category,
                    data_keys
                ))
            return None

    def load_single(self, category: str, data_key: str):
        """Load only a single data file from the cached category.

        :param category: The category which should be partially loaded.
        :param data_key: A key describing the specific element of the given category.
        :return: None, if no such element has been cached.
        """
        result_list = self.load(category, [data_key])
        if result_list is not None:
            result_list = result_list[data_key]
        return result_list

    def is_empty(self, category: str, data_keys=None) -> bool:
        """Check whether any (specific) data for category has been cached.

        :param category: The category which should be checked.
        :param data_keys: If None, just any data needs to exist. Otherwise, at least one file specified by the data_keys.
        :return:
        """
        category_dir_exists = os.path.isdir(self._base_path(category))

        if category_dir_exists and data_keys is not None:
            is_empty = True
            for file_name in os.listdir(self._base_path(category)):
                key = os.path.splitext(file_name)[0]
                if key in data_keys:
                    is_empty = False
                    break
        else:
            is_empty = not category_dir_exists

        return is_empty

    def save(self, category: str, data, suffix_extension=".npy"):
        """Save an arbitrary category to the cache.

        Currently, all elements of data must use the same suffix_extension.

        :param category: The category which should be saved.
        :param data: The actual data that should be cached.
        :param suffix_extension: The file extension. Either ".npy" or ".p".
        :return:
        """
        # create folder for this category
        if not os.path.exists(self._base_path(category)):
            os.makedirs(self._base_path(category))

        # each element of the data dictionary to a separate file
        for key, value in data.items():
            log.log(" .. saving {}.{}".format(
                category,
                key
            ))
            if suffix_extension == ".npy":
                np.save(self._base_path(category, key, suffix_extension), value)
            else:
                with open(self._base_path(category, key, ".p"), "wb") as output_file:
                    pickle.dump(value, output_file)

        # additional log message to signal the end of this category cache. but only, if there is more than one file
        if len(data) > 1:
            log.log(".. saved {}".format(category))

    def save_single(self, category, data_key, data_value, suffix_extension=".npy"):
        """Save only a single data file of a category."""
        self.save(category, {data_key: data_value}, suffix_extension)

    def save_dataset(self, dataset_key: str, x: np.ndarray, y: np.ndarray, preprocessor: Preprocessor):
        """Cache the specified dataset.

        Does not work directly with a Dataset object to allow saving python lists of x and y before they are
        converted to numpy arrays. the latter can not happen inplace and might cause a memory error. While first saving
        them to disk, will result in an automatic conversion to numpy arrays which do not need the memory.

        :param dataset_key: The key identifying the dataset which should be saved.
        :param x: The raw data of the associated dataset.
        :param y: The label data of the associated dataset.
        :param preprocessor: The preprocessor of the associated dataset.
        :return:
        """
        # create folder for this dataset
        if not os.path.exists(self._ds_path(dataset_key)):
            os.makedirs(self._ds_path(dataset_key))

        data_np = dict()
        data_np[self.KEY_CACHE_VERSION] = self._cache_version

        # do not save the complete dataset object, but X and Y.
        # this way the calculated data will be restored, but parameters for splitting etc. can be refreshed
        data_np[self.KEY_DATA_X] = x
        data_np[self.KEY_DATA_Y] = y

        # add the complete current configuration to ensure that no information about the loaded dataset are missing
        data_np[self.KEY_CONFIG] = cf._cf

        # save each element of the dictionary to a separate file
        for key, value in data_np.items():
            np.save(self._ds_path(dataset_key, key), value)

        # pickle instead of numpy
        data_pickle = dict()

        # store further dataset class attributes
        # TODO do not use private vars
        data_pickle[self.KEY_NEXT_NEW_IID] = Dataset._next_new_iid
        data_pickle[self.KEY_LABEL_IDS] = Dataset._label_internal_ids
        data_pickle[self.KEY_LABEL_NAME_BY_ID] = Dataset._label_name_by_internal_id
        data_pickle[self.KEY_LABEL_ID_BY_NAME] = Dataset._label_internal_id_by_name
        data_pickle[self.KEY_PREPROCESSOR] = preprocessor

        for key, value in data_pickle.items():
            with open(self._ds_path(dataset_key, key, ".p"), "wb") as output_file:
                pickle.dump(value, output_file)

        log.log("Cached dataset " + dataset_key)

        # save copy of current log file, but don't flush
        log.log_save(self._ds_path(dataset_key), flush=False)

    def _count_old_cache_version_folders(self) -> int:
        """Get the number of deprecated cache versions that still exist on disk."""
        # assuming that there is at least one folder for the current version. all others are old
        return len(os.listdir(cf.get("cache_path_root"))) - 1
