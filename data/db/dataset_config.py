from typing import List
import os
import config as cf
from data import annotation, imagenet_info
from data.db.label import KEY_BACKGROUND

################## Private Module State #######################
_all_configs = None
DATASET_KEY_IMAGENET = "imagenet"


class DatasetConfig:
    """An object of this class provides some meta data about an associated dataset."""

    def __init__(self, dataset_key: str):
        """Create new DatasetConfig.

        :param dataset_key: The key of the dataset which will be described by this object.
        """
        self._dataset_key = dataset_key
        self._has_annotations = annotation.has_annotations(dataset_key)

    @property
    def ignored_subfolder_names(self) -> List[str]:
        """
        Get a list containing the names of all subfolders that should be ignored.
        :return:
        """
        return []

    def label_key_from_folder_path(self, path: str) -> str:
        """Get the label key based on a full folder path.

        Usually, the folder name is identical to the label key.
        If None is returned, the given folder should be skipped.
        """
        return os.path.basename(os.path.normpath(path))

    @property
    def dataset_key(self) -> str:
        return self._dataset_key

    @property
    def has_annotations(self) -> str:
        return self._has_annotations

    @property
    def max_imgs_per_folder(self) -> int:
        """The maximum number of images taken from the same folder.
        None means no limit.
        """
        return None


class DatasetConfigImageNet(DatasetConfig):
    """An object of this class describes the ImageNet dataset.

    See
    http://www.image-net.org/
    """

    def __init__(self):
        """Create a new DatasetConfigImageNet."""
        DatasetConfig.__init__(self, DATASET_KEY_IMAGENET)

        # create imagenet index
        self._human_ids = imagenet_info.get_human_wordnet_ids()
        self._ignored_ids = imagenet_info.get_ignored_wordnet_ids()

    def label_key_from_folder_path(self, path: str) -> str:
        """Get the label key based on a full folder path.

        Here, we merge different imagenet ids to the desired labels.
        """
        folder_name = os.path.basename(os.path.normpath(path))
        if folder_name not in self._human_ids and folder_name not in self._ignored_ids:
            return KEY_BACKGROUND
        else:
            return None

    @property
    def max_imgs_per_folder(self) -> int:
        """The maximum number of images taken from the same folder.
        None means no limit.
        """
        if cf.get("dataset_path_root") == cf.get("dataset_native_path_root"):
            return cf.get("background_max_img_per_entity")
        else:
            # the pre-sampled dataset was already filtered when created
            return None


################## Public Module API ##########################


def get(ds_key: str) -> DatasetConfig:
    """Get the DatasetConfig which describes the dataset identified by ds_key.

    :param ds_key: The key of the dataset which should be described.
    :return:
    """
    global _all_configs
    if _all_configs is None:
        _all_configs = _create_all_configs()
    return _all_configs[ds_key]


def _create_all_configs():  # -> dict[str, DatasetConfig]
    """Create all dataset configuration objects for the specified dataset_keys."""
    result = {}
    for ds_key in cf.get("dataset_keys"):
        if ds_key == DATASET_KEY_IMAGENET:
            ds_config = DatasetConfigImageNet()
        else:
            ds_config = DatasetConfig(ds_key)
        result[ds_key] = ds_config
    return result
