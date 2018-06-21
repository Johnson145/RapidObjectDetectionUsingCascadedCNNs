import os
from typing import Optional, List

import PIL
import scipy
from PIL import Image

from data.annotation import Annotation
from data.db.label import Label
import config as cf
import numpy as np
from scipy import misc
from utils.img_manipulation import random_img_patch


class ImageInfo:
    """An object of this class collects some meta information about a specific image file.

    It may provide the image data (pixels) itself, too.
    """

    def __init__(self, path_original: str, label: Label, dataset_key: str):
        """Create a new ImageInfo.

        :param path_original: The file path pointing to the original (unscaled) image file.
        :param label: The ground truth label of this file.
        :param dataset_key: The key of the associated dataset.
        """

        # the file path of the original (unscaled etc.) image file
        self._path_original = path_original

        # an optional list of annotation objects including bounding box information
        self._annotations = None

        self._label = label

        # remember from which dataset this image comes
        self._dataset_key = dataset_key

        self._raw_img_cache = dict()

        # image attributes will be saved using lazy loading
        self._img_width_original = None
        self._img_height_original = None

    @property
    def path_original(self) -> str:
        """Get the file path pointing to the original (unscaled) image file."""
        return self._path_original

    @property
    def path_resized(self) -> str:
        """Get the file path of the resized image file (file does not need to exist yet)."""

        # always recalculate this path, because it will change as soon as the img_width, img_height have been changed.
        data_root_temp = os.path.join(cf.get("dataset_path_root"), self._dataset_key + '/images')
        data_root_resized = os.path.join(data_root_temp, "{}x{}".format(
            cf.get("img_width"),
            cf.get("img_height")
        ))
        file_path_resized = os.path.join(os.path.join(data_root_resized, self.label.key), self.basename)

        return file_path_resized

    @property
    def label_internal_id(self) -> int:
        """Get the iid of the associated image label."""
        return self.label.iid

    @property
    def annotations(self) -> Optional[List[Annotation]]:
        """Get a list containing all annotations of this image.

        Will be None, if no annotations are provided for the associated dataset.
        """
        # lazy loading annotations
        # TODO don't check for annotations more than once (happens for datasets that do not have annotations)
        from data import annotation
        if self._annotations is None and annotation.has_annotations(self.dataset_key):
            self._annotations = annotation.get_annotations(self)

        return self._annotations

    @property
    def dataset_key(self) -> str:
        """Get the key identifying the associated dataset."""
        return self._dataset_key

    @property
    def basename(self) -> str:
        """Get the (original) basename of this image file."""
        return os.path.basename(self.path_original)

    @property
    def original_label_folder_name(self) -> str:
        """Get the name of the very last folder in which this image file is included."""
        return os.path.basename(os.path.dirname(self._path_original))

    @property
    def ignore_key(self) -> str:
        """This string can be used to identify this file inside of a dataset's ignore list."""
        # inside of one dataset, the combination of the label folder name and the actual file name is unique
        return self.original_label_folder_name + "/" + self.basename

    @property
    def label(self) -> Label:
        """Get the ground truth label of this image."""
        return self._label

    @property
    def is_jpeg(self) -> bool:
        """Whether this a jpeg file."""
        filename = self._path_original.lower()
        return filename.endswith(".jpg") or filename.endswith(".jpeg")

    @property
    def is_png(self) -> bool:
        """Whether this a png file."""
        filename = self._path_original
        return ImageInfo._is_png(filename)

    @staticmethod
    def _is_png(filename) -> bool:
        """Static helper method to check whether a file is a png file."""
        filename = filename.lower()
        return filename.endswith(".png")

    @property
    def is_ignored(self) -> bool:
        """Whether this image file is listed in the ignore list."""
        from data.db.file_list_loader import FileListLoader
        return FileListLoader().file_is_ignored(self)

    @property
    def full_key(self) -> str:
        """This key should describe each file (no matter from which dataset) uniquely."""
        return self.dataset_key + "/" + self.ignore_key

    def raw_resized(self, cache=False) -> np.ndarray:
        """Provide this image resized to cf.get("img_width") x cf.get("img_height").

        If cf.get("cache_resized_training_samples_individually"), this will load the previously-resized image data from
        disk (If it doesn't exist yet, it will be created once on the first access). Otherwise, the image data will be
        scaled online.

        Don't confuse this method with raw_scaled().

        :param cache: Whether to cache the loaded data in RAM.
        :return:
        """
        # check whether we need to do the resizing right now
        need_to_resize_now = not cf.get("cache_resized_training_samples_individually") or not os.path.isfile(self.path_resized)  # and os.path.isfile(self.path_original)

        # maybe resize
        if need_to_resize_now:  # TODO this block doesn't use the cache parameter yet
            # open image file as a PIL object
            # (this is not the same as using raw_original(), because the latter will provide a numpy array)
            pil_img = Image.open(self.path_original).convert('RGB')

            # check annotations for given bounding box and crop image
            # (already done for the pre-sampled dataset)
            if cf.get("dataset_path_root") == cf.get("dataset_native_path_root"):
                annotation_used = False
                if self.annotations is not None:
                    # here, we can use only one bounding box per img_info
                    first_annotation = self.annotations[0]
                    if first_annotation.bbox_is_valid:
                        pil_img = pil_img.crop((first_annotation.xmin, first_annotation.ymin, first_annotation.xmax,
                                        first_annotation.ymax))
                        annotation_used = True

                # if no cropping was done yet and the original source isn't a cropped version already either,
                # crop a random image patch
                if not annotation_used:
                    pil_img = random_img_patch(pil_img)
                # else: use complete image
            # else: use complete image

            # actually resize the image "with brute force"
            pil_img = pil_img.resize((cf.get("img_width"), cf.get("img_height")), PIL.Image.ANTIALIAS)

            # optionally, save new image on disk
            if cf.get("cache_resized_training_samples_individually"):
                target_dir_without_file = os.path.dirname(self.path_resized)
                if not os.path.exists(target_dir_without_file):
                    os.makedirs(target_dir_without_file)
                pil_img.save(self.path_resized)

            # cast the pil image to a numpy array and return it
            return np.asarray(pil_img, dtype=cf.get("img_dtype"))

        else:  # cf.get("cache_resized_training_samples_individually") and os.path.isfile(self.path_resized)
            # requested image size should already exist cached on disk as an individual file, we just need to load it:
            return self.raw(self.path_resized, cache, "resized")

    def raw_original(self, cache=False) -> np.ndarray:
        """Get the original / unscaled raw image data."""
        return self.raw(self.path_original, cache, "original")

    def raw_scaled(self, cache=False, ratio=1.0) -> np.ndarray:
        """Get image data that is based on the original input, but rescaled online.
        This got nothing to do with raw_resized()!

        :param cache:
        :param ratio:
        :return:
        """
        if ratio != 1.0:
            return self.raw(self.path_original, cache, self._raw_scaled_cache_key(ratio), ratio)
        else:
            # do not create a second cache entry containing another unscaled version
            return self.raw_original(cache)

    def raw(self, file_path: str, cache=False, cache_key=None, ratio=1.0) -> np.ndarray:
        """Read the associated image file from disk and return its raw content as a nparray.

        :param file_path:
        :param cache: Whether the loaded data should be kept in memory for a later request (only relevant if not done before).
        :param cache_key: Must be provided, if cache is True
        :param ratio: Rescale the loaded image to this ratio. 1.0 will keep the original dimensions.
        :return: image data as a nparray with dtype=cf.get("img_dtype") and always(!) 3 color channels.
        """
        # validate params
        if cache and cache_key is None:
            raise ValueError("The provided cache_key must not be None, if the cache is enabled.")

        if self._raw_img_cache.get(cache_key) is None:
            # read image from file into an array
            # mode="RGB" is important here for two reasons:
            # - convert images with transparency information (e.g. PNG) from 4 channels to 3 channels
            # - convert greyscale images from 2 channels to 3 channels
            raw_image_data = misc.imread(file_path, mode="RGB")

            # ensure correct data type
            raw_image_data = np.asarray(raw_image_data, dtype=cf.get("img_dtype"))

            # apply online scaling
            if ratio != 1.0:
                raw_image_data = scipy.misc.imresize(raw_image_data, ratio)

            # cache data for next time?
            if cache:
                self._raw_img_cache[cache_key] = raw_image_data

            return raw_image_data

        else:
            # data is still cached and does not need to be loaded again
            return self._raw_img_cache.get(cache_key)

    def _raw_scaled_cache_key(self, ratio) -> str:
        """The cache_key that is used for a self.raw_scaled(.., cache=True, ratio)."""
        return "orig_scaled_{}".format(ratio)

    def is_raw_scaled_cached(self, ratio) -> bool:
        """Whether the result of self.raw_scaled(.., ratio) has already been cached or not."""
        return self._raw_scaled_cache_key(ratio) in self._raw_img_cache

    def clear_raw_img_cache(self):
        """Remove all cached raw image data."""
        self._raw_img_cache = dict()

    def _load_original_img_dimensions(self):
        """Save self._img_width_original and self._img_height_original."""
        # if the original image is already cached, we can get the dimensions by it's array shape
        if self._raw_img_cache.get("original") is not None:
            self._img_height_original = self.raw_original().shape[0]
            self._img_width_original = self.raw_original().shape[1]
        else:
            # load from file (without loading the actual image data)
            pil_img = Image.open(self.path_original)
            self._img_width_original, self._img_height_original = pil_img.size  # (width,height) tuple

    @property
    def img_width_original(self) -> int:
        """Get the original image width in pixels."""
        if self._img_width_original is None:
            self._load_original_img_dimensions()
        return self._img_width_original

    @property
    def img_height_original(self) -> int:
        """Get the original image height in pixels."""
        if self._img_height_original is None:
            self._load_original_img_dimensions()
        return self._img_height_original
