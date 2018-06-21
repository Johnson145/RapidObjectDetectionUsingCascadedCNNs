import operator
import os
import random
from glob import glob
from statistics import median, stdev
from typing import List, DefaultDict

from tensorflow.python.framework.errors_impl import InvalidArgumentError

import config as cf
import data.db.label as l
import data.cache
from data import cache
from data.datasets import Dataset
from data.db import dataset_config, label
from data.image_info import ImageInfo
from utils import log, file_handler
from utils.singleton import Singleton
from utils.time_watcher import TimeWatcher


class FileListLoader(metaclass=Singleton):
    """Base class for the DatasetLoader.

    This class will not load any actual data. Instead, it will only create lists containing all image file paths that
    are used by the DatasetLoader.
    """

    def __init__(self):
        """Create new FileListLoader."""
        # init vars
        self.reset()

    def reset(self):
        """Reset existing data.

        This equals an initial initialization in the constructor.
        """

        # collect an ImageInfo object for each image file that is loaded
        # those objects will be grouped by their dataset key
        self._image_infos_per_dataset = None

        # a merged list of all elements of self._image_infos_per_dataset
        # => one list containing all image infos
        self._image_infos = None

        # alternative split of self._image_infos
        self._image_infos_per_iid_label = None

        # save number of files that have been skipped because of..
        self._files_skipped_unknown_error = 0

        self._ignore_dicts = None
        self._whitelist_dicts = None
        self._image_info_by_path = None

    @property
    def image_infos_per_dataset(self) -> DefaultDict[str, List[ImageInfo]]:
        """Get a dictionary mapping each dataset_key to the associated image_infos."""
        # auto load data the first time it is requested
        if self._image_infos_per_dataset is None:
            self._load_file_lists()
        return self._image_infos_per_dataset

    @property
    def image_infos_per_iid_label(self) -> DefaultDict[int, List[ImageInfo]]:
        """Get a dictionary mapping each label iid to the associated image_infos."""
        # auto load data the first time it is requested
        if self._image_infos_per_iid_label is None:
            self._load_file_lists()
        return self._image_infos_per_iid_label

    @property
    def image_infos(self) -> List[ImageInfo]:
        """Get all(!) loaded image infos."""
        # auto load data the first time it is requested
        if self._image_infos is None:
            self._load_file_lists()
        return self._image_infos

    def _load_file_lists(self):
        """Load all file lists for all datasets."""

        tw = TimeWatcher("FileListLoading")

        log.log("Load file lists for dataset(s): {}".format(
            cf.get("dataset_keys")
        ))

        log.log(".. Required image dimension: {}x{}px".format(cf.get("img_width"),
                                                              cf.get("img_height")))

        self._image_infos_per_dataset, self._image_infos, self._image_infos_per_iid_label = self._check_filelist_cache_combined()

        if self._image_infos_per_dataset is None or self._image_infos is None or self._image_infos_per_iid_label is None:

            self._image_infos_per_dataset = dict()
            self._image_infos = []
            self._image_infos_per_iid_label = dict()

            # load each dataset separately
            for dataset_key in cf.get("dataset_keys"):
                self._load_file_list(dataset_key)

            self.log_stats()

            # ensure that each class meets the minimum and maximum requirements
            self._ensure_min_max()

            self._save_filelist_cache_combined()
        else:
            # if the file list was loaded from the cache, we need to initialize the labels manually
            for label_iid in self._image_infos_per_iid_label.keys():
                _ = label.get_by_iid(label_iid)

        log.log("Finished file list loading.")

        tw.stop()

    def _load_file_list(self, dataset_key):
        """Load a single file list for the dataset with the key dataset_key.

        Add all images of the dataset dataset_key to self._image_infos, if they are useful.

        Requires, that the folder structure of the dataset's files met the typical requirement:
        cf.get("dataset_path_root")/<dataset_key>/images/original/<subfolder_per_label>
        Inside of the <subfolder_per_label> may be further subfolders. Such subfolders will be merged to the same label
        without further distinguishing.
        Special cases can be implemented in the dataset_config module.
        """
        log.log('Load file list for dataset: {}'.format(dataset_key))
        self._image_infos_per_dataset[dataset_key] = []

        if dataset_key not in cf.get("dataset_keys_available"):
            log.log(".. Error: Unknown dataset '" + dataset_key + "'")
        else:
            # load dataset specific configuration
            ds_config = dataset_config.get(dataset_key)

            # check whether annotations are provided
            log.log(".. annotations provided: {}".format(
                ds_config.has_annotations
            ))

            # set img base paths and list all label folders
            data_root_temp = os.path.join(cf.get("dataset_path_root"), dataset_key + '/images')
            data_root_original_common = os.path.join(data_root_temp, "original")
            label_dirs_original = glob(data_root_original_common + "/*/")

            # log stats
            log.log(".. {} label folders in total".format(
                len(label_dirs_original)
            ))

            # search image files
            n_files = 0
            n_used_folders = 0

            for label_folder_path in label_dirs_original:
                # extract label_key
                label_key = ds_config.label_key_from_folder_path(label_folder_path)

                # ensure that this folder should not be ignored
                if label_key is not None:

                    # process all image files of that label folder
                    # attention: data_root_original_specific differs from data_root_original_common,
                    # because it contains the label folder as well as it may point to a subdirectory
                    for data_root_original_specific, _, filenames in os.walk(label_folder_path):
                        i_same_folder = 0

                        # check if this dir really contains any useful files
                        supported_files = []
                        for filename in filenames:
                            if self.is_supported_img_file(filename):
                                supported_files.append(filename)

                        # if this dir is really used..
                        if len(supported_files) > 0:
                            n_used_folders += 1
                            # .. add a new internal label (if it does not already exist)
                            label = l.get_by_key(label_key)
                        else:
                            label = None

                        for filename in supported_files:
                            # collect file path info
                            file_path_original = os.path.join(data_root_original_specific, filename)

                            # bundle all needed image information
                            img_info = ImageInfo(file_path_original, label, dataset_key)

                            # if this image file isn't on the ignore list, save it
                            if not self.file_is_ignored(img_info):
                                self._image_infos_per_dataset[dataset_key].append(img_info)
                                self._image_infos.append(img_info)
                                if label.iid not in self._image_infos_per_iid_label:
                                    self._image_infos_per_iid_label[label.iid] = []
                                self._image_infos_per_iid_label[label.iid].append(img_info)

                                # count
                                n_files += 1
                                i_same_folder += 1

                                # the background category got far too many images. keep only a small number per
                                # individual entity
                                if ds_config.max_imgs_per_folder is not None and \
                                                i_same_folder == ds_config.max_imgs_per_folder:
                                    break

            log.log(".. {} used label folders".format(
                n_used_folders
            ))

            log.log(".. {} used images".format(
                n_files
            ))

        # done
        log.log('.. finished file list for dataset: {}'.format(dataset_key))

    @staticmethod
    def is_supported_img_file(filename):
        """Check whether file_name describes a supported image file."""
        # TODO add support for .bmp
        filename = filename.lower()
        return filename.endswith(".jpg") or filename.endswith(".jpeg") or filename.endswith(".png")

    def log_stats(self):
        """Calculate and log stats about the used classes."""
        log.log("File list loader stats:")

        # calculate the absolute amount of each used label
        n_samples_per_iid_label = dict()
        for info in self.image_infos:
            if info.label_internal_id not in n_samples_per_iid_label:
                n_samples_per_iid_label[info.label_internal_id] = 0
            n_samples_per_iid_label[info.label_internal_id] += 1
        values_only = n_samples_per_iid_label.values()

        if len(values_only) > 0:
            log.log(".. absolute minimum: {}".format(
                min(values_only)
            ))
            log.log(".. absolute maximum: {}".format(
                max(values_only)
            ))
            log.log(".. average: {}".format(
                int(sum(values_only) / len(values_only))
            ))
            log.log(".. median: {}".format(
                int(median(values_only))
            ))
            if len(values_only) > 1:
                log.log(".. standard variation: {}".format(
                    int(stdev(values_only))
                ))

            # print labels with the lowest number of images (does not include the ones without any images)
            n_lowest = min(10, len(n_samples_per_iid_label))
            log.log(".. the following {} labels have the lowest number of images".format(
                n_lowest
            ))
            sorted_n_samples_per_iid_label = sorted(n_samples_per_iid_label.items(), key=operator.itemgetter(1))
            for i in range(n_lowest):
                iid_label = sorted_n_samples_per_iid_label[i][0]
                n_images = sorted_n_samples_per_iid_label[i][1]
                label = l.get_by_iid(iid_label)
                log.log(".. - {} ({} images)".format(
                    label.name,
                    n_images
                ))

    def remove_broken_images(self):
        """Try to open each image in the loaded list and add the failed ones to an ignore list."""
        log.log("Try to open each image in the list and add the failed ones to an ignore list.")

        import tensorflow as tf
        from tensorflow.python.platform import gfile

        # initialize TensorFlow to use the same open method as
        session = tf.Session()
        init = tf.global_variables_initializer()
        session.run(init)

        img_placeholder = tf.placeholder(name='img_placeholder', dtype=tf.string)
        # note that, using the decode_image tensor instead of decode_jpeg/png will imply more ignored files, because
        # some of the images can not be detected as either or, but would be loaded correctly, if directly fed into
        # one of the latter methods
        # UPDATE: most of those ignored files are caused by a bug which can be fixed manually.
        # see https://github.com/tensorflow/tensorflow/issues/7657
        decode_img = tf.image.decode_image(img_placeholder, channels=3)

        n_ignored = 0
        ignored_image_infos = []
        for info in self.image_infos:
            if not self._file_is_whitelisted(info):
                try:
                    contents = gfile.FastGFile(info.path_original, 'rb').read()

                    feed_dict = {img_placeholder: contents}
                    tf_result = session.run(decode_img, feed_dict)

                    # remove all images that can be read, but have a wrong shape
                    # most likely these are gifs. gifs always have len(tf_result.shape) == 4.
                    # it's still okay to remove those gifs as we only reach this point if the gif has a jpg extension
                    if len(tf_result.shape) != 3:
                        raise InvalidArgumentError(None, None, "Invalid image shape")

                    self._whitelist_file(info.dataset_key, info.ignore_key)
                    file_is_okay = True
                except InvalidArgumentError as e:
                    # catch errors like "Not a JPEG file: "
                    self._ignore_file(info.dataset_key, info.ignore_key)
                    n_ignored += 1
                    ignored_image_infos.append(info)

        # release TensorFlow resources
        session.close()
        tf.reset_default_graph()

        if n_ignored > 0:
            log.log("Added a total of {} new files to the ignore list.".format(
                n_ignored
            ))

            # remove all ignored files from the local file list
            # (do not just reload a completely new file list, because the new list will randomly contain other images
            # the we just checked)
            self._remove_loaded_files(ignored_image_infos)
        else:
            log.log("No further files need to be ignored.")

    def _remove_loaded_files(self, remove_list: List[ImageInfo]):
        """Remove all elements in the given list from the already loaded file lists."""

        # remove from global list
        self._image_infos = [x for x in self.image_infos if x not in remove_list]

        # recreate other index structures
        self._image_infos_per_dataset = dict()
        self._image_infos_per_iid_label = dict()
        for img_info in self.image_infos:

            if img_info.dataset_key not in self._image_infos_per_dataset:
                self._image_infos_per_dataset[img_info.dataset_key] = []
            self._image_infos_per_dataset[img_info.dataset_key].append(img_info)

            if img_info.label_internal_id not in self._image_infos_per_iid_label:
                self._image_infos_per_iid_label[img_info.label_internal_id] = []
            self._image_infos_per_iid_label[img_info.label_internal_id].append(img_info)

        # re-ensure that each class meets the minimum and maximum requirements
        # (some classes might have too few images now)
        self._ensure_min_max()

    def _get_ignore_file_path(self, dataset_key):
        """Get the file path of the ignore list for the dataset with dataset_key.

        File does not need to exist yet.
        """
        return os.path.join(cf.get("ignore_lists_dir"), "{}.txt".format(dataset_key))

    def _get_whitelist_file_path(self, dataset_key):
        return os.path.join(cf.get("whitelists_dir"), "{}.txt".format(dataset_key))

    def _read_ignore_dicts(self):
        """Read all ignore dicts from assets/ignore-lists."""
        self._ignore_dicts = dict()
        for dataset_key in cf.get("dataset_keys"):

            # use dictionaries instead of actual lists to speed up search
            self._ignore_dicts[dataset_key] = dict()

            ignore_file_path = self._get_ignore_file_path(dataset_key)
            if os.path.isfile(ignore_file_path):
                # each line is one ignored element
                lines = file_handler.read_txt_lines(ignore_file_path)

                for line in lines:
                    if line != "":  # there will be at least one empty line at the end of the file
                        if dataset_key not in self._ignore_dicts:
                            self._ignore_dicts[dataset_key] = dict()
                        self._ignore_dicts[dataset_key][line] = True

    def _read_whitelist_dicts(self):
        """Read all whitelist dicts from assets/white-lists."""
        self._whitelist_dicts = dict()
        for dataset_key in cf.get("dataset_keys"):

            # use dictionaries instead of actual lists to speed up search
            self._whitelist_dicts[dataset_key] = dict()

            whitelist_file_path = self._get_whitelist_file_path(dataset_key)
            if os.path.isfile(whitelist_file_path):
                # each line is one ignored element
                with open(whitelist_file_path) as f:
                    lines = f.readlines()
                lines = [x.strip() for x in lines]  # remove whitespaces

                for line in lines:
                    if line != "":  # there will be at least one empty line at the end of the file
                        if dataset_key not in self._whitelist_dicts:
                            self._whitelist_dicts[dataset_key] = dict()
                        self._whitelist_dicts[dataset_key][line] = True

    def ignore(self, image_info: ImageInfo):
        """Add image_info to the ignore list."""
        self._ignore_file(image_info.dataset_key, image_info.ignore_key)

    def _ignore_file(self, dataset_key: str, ignore_key: str):
        """Add the given file to the ignore list.
        
        Note, this will not yet remove any already loaded files from e.g. self._image_infos.
        """
        # if the ignore list hasn't been initialized yet, load existing info from files first
        if self._ignore_dicts is None:
            self._read_ignore_dicts()

        if ignore_key not in self._ignore_dicts[dataset_key]:
            log.log("Ignoring {} of dataset {}.".format(
                ignore_key,
                dataset_key
            ))

            # add file to the internal ignore list
            if dataset_key not in self._ignore_dicts:
                self._ignore_dicts[dataset_key] = dict()
            self._ignore_dicts[dataset_key][ignore_key] = True

            # persisting: add one line per image to the ignore list file
            ignore_file_path = self._get_ignore_file_path(dataset_key)
            with open(ignore_file_path, 'a') as file:
                file.write(ignore_key + "\n")

                # TODO check whether this file is used in the currently loaded image_infos
        else:
            log.log("Already ignored: {} of dataset {}.".format(
                ignore_key,
                dataset_key
            ))

    def _unignore_file(self, dataset_key: str, ignore_key: str):
        """Remove the given file from the ignore list.
        
        Note, this will not yet add any already added files from e.g. self._image_infos.
        """
        # if the ignore list hasn't been initialized yet, load existing info from files first
        if self._ignore_dicts is None:
            self._read_ignore_dicts()

        log.log("Unignoring {} of dataset {}.".format(
            ignore_key,
            dataset_key
        ))

        # remove file from the internal ignore list
        if dataset_key in self._ignore_dicts and ignore_key in self._ignore_dicts[dataset_key]:
            del self._ignore_dicts[dataset_key][ignore_key]

        # persisting: remove referencing file lines
        # each line is one ignored element
        ignore_file_path = self._get_ignore_file_path(dataset_key)
        file_str = ""
        lines = file_handler.read_txt_lines(ignore_file_path)
        for line in lines:
            if line != ignore_key:
                file_str += line + "\n"
        with open(ignore_file_path, "w") as file:
            file.write(file_str)

    def _whitelist_file(self, dataset_key: str, ignore_key: str):
        # if the ignore list hasn't been initialized yet, load existing info from files first
        if self._whitelist_dicts is None:
            self._read_whitelist_dicts()

        log.log("Whitelisting {} of dataset {}.".format(
            ignore_key,
            dataset_key
        ))

        # add file to the internal whitelist
        if dataset_key not in self._whitelist_dicts:
            self._whitelist_dicts[dataset_key] = dict()
        self._whitelist_dicts[dataset_key][ignore_key] = True

        # persisting: add one line per image to the whitelist file
        whitelist_file_path = self._get_whitelist_file_path(dataset_key)
        with open(whitelist_file_path, 'a') as file:
            file.write(ignore_key + "\n")

    def file_is_ignored(self, image_info: ImageInfo) -> bool:
        """Return true if image_info is listed in one of the ignore lists."""
        if self._ignore_dicts is None:
            self._read_ignore_dicts()
        return image_info.ignore_key in self._ignore_dicts[image_info.dataset_key]

    def _file_is_whitelisted(self, image_info: ImageInfo) -> bool:
        """Return true if image_info is listed in one of whitelists."""
        if self._whitelist_dicts is None:
            self._read_whitelist_dicts()
        return image_info.ignore_key in self._whitelist_dicts[image_info.dataset_key]

    def _ensure_min_max(self):
        """Ensure that each class has at least cf.get("class_min_images") and at most cf.get("class_max_images") images."""
        # check whether some files need to be dropped again (without applying any changes)
        change_dataset_max = False
        if cf.get("class_max_images") is not None:
            for iid_label, images in self._image_infos_per_iid_label.items():
                if len(images) > cf.get("class_max_images"):
                    change_dataset_max = True
                    break

        change_dataset_min = False
        if cf.get("class_min_images") is not None:
            for iid_label, images in self._image_infos_per_iid_label.items():
                if len(images) < cf.get("class_min_images"):
                    change_dataset_min = True
                    break

        if change_dataset_max or change_dataset_min:

            if change_dataset_max:
                # we can't iterate over the dictionary while changing, so we need a copy
                img_per_iid = self._image_infos_per_iid_label.copy()

                # ensure that all results can be reproduced
                random.seed(42)

                # actually change self._image_infos_per_iid_label
                self._image_infos_per_iid_label = dict()
                n_classes_reduced = 0
                for iid_label, images in img_per_iid.items():
                    if len(images) > cf.get("class_max_images"):
                        # uncomment the following lines to show which classes have been reduces
                        # log.log(".. reducing number of class images for label {} to an amount of {}".format(
                        #     iid_label,
                        #     cf.get("class_max_images")
                        # ))

                        self._image_infos_per_iid_label[iid_label] = random.sample(images, cf.get("class_max_images"))
                        n_classes_reduced += 1
                    else:
                        self._image_infos_per_iid_label[iid_label] = images
                log.log(".. reduced a total of {} classes".format(n_classes_reduced))

            if change_dataset_min:
                # we can't iterate over the dictionary while changing, so we need a copy
                img_per_iid = self._image_infos_per_iid_label.copy()

                # actually change self._image_infos_per_iid_label
                self._image_infos_per_iid_label = dict()
                n_classes_dropped = 0
                for iid_label, images in img_per_iid.items():
                    if len(images) >= cf.get("class_min_images"):
                        self._image_infos_per_iid_label[iid_label] = images
                    else:
                        # uncomment the following lines to show which classes have been dropped
                        log.log(".. dropping all images with label {}, cause they are too few ({} < {})".format(
                            iid_label,
                            len(images),
                            cf.get("class_min_images")
                        ))
                        # TODO this doesn't work anymore
                        Dataset.remove_label_type(iid_label)
                        n_classes_dropped += 1
                log.log(".. dropped a total of {} classes".format(
                    n_classes_dropped
                ))

            # reconstruct self._image_infos_per_dataset
            self._image_infos_per_dataset = dict()
            for iid_label, images in self._image_infos_per_iid_label.items():
                for image in images:
                    if image.dataset_key not in self._image_infos_per_dataset:
                        self._image_infos_per_dataset[image.dataset_key] = []
                    self._image_infos_per_dataset[image.dataset_key].append(image)

            # reconstruct self._image_infos
            # (by using the already reconstructed self._image_infos_per_dataset, the created order is more likely
            #  to be similar to the original one)
            self._image_infos = []
            for iid_label, images in self._image_infos_per_dataset.items():
                for image in images:
                    self._image_infos.append(image)

            log.log("New stats after applying min/max constraints:")
            self.log_stats()

    def image_info_by_path(self, path: str, refresh=False) -> ImageInfo:
        """Get an image info object of the currently loaded set by the full image file path.
        
        Note, that this is an (quite) expensive task!
        """
        if self._image_info_by_path is None or refresh:
            self._image_info_by_path = dict()
            for image_info in self.image_infos:
                self._image_info_by_path[image_info.path_original] = image_info

        if path in self._image_info_by_path:
            return self._image_info_by_path[path]
        else:
            return None

    def image_infos_by_label_key(self, label_key: str) -> List[ImageInfo]:
        """Get all image infos for the given label key."""
        result = []
        label = l.get_by_key(label_key)
        if label is not None:
            result = self.image_infos_by_label(label)
        return result

    def image_infos_by_label(self, label: l.Label) -> List[ImageInfo]:
        """Get all image infos for the given label."""
        result = []
        if label.iid is not None and label.iid in self.image_infos_per_iid_label:
            result = self.image_infos_per_iid_label[label.iid]
        return result

    @property
    def dataset_key_combined(self):
        """Get a combined key for all datasets requested."""
        sorted_keys = sorted(cf.get("dataset_keys"))
        separator = "-"
        combined_key = separator.join(sorted_keys)
        return combined_key

    def _cache_key(self):
        """Get the cache category key that is used for the loaded file list."""
        return cache.Cache.CATEGORY_PREFIX_FILE_LIST_LOADER + self.dataset_key_combined

    def _check_filelist_cache_combined(self):
        """Check the filelist cache for the combined dataset consisting of all datasets that should be loaded.

        File lists can only be cached in the combined fashion.
        Got nothing to do with _check_dataset_cache().
        Will only load data from cache, when activated in the first place. If no data has been loaded, the returned
        values will be None.
        """
        image_infos_per_dataset = None
        image_infos = None
        image_infos_per_iid_label = None

        if cf.get("cache_dataset"):

            c = data.cache.Cache()
            cached_data = c.load(self._cache_key())

            if cached_data is not None:
                image_infos_per_dataset = cached_data[cache.Cache.KEY_FLL_IMG_INFOS_PER_DS]
                image_infos = cached_data[cache.Cache.KEY_FLL_IMG_INFOS]
                image_infos_per_iid_label = cached_data[cache.Cache.KEY_FLL_IMG_INFOS_PER_IID]

        return image_infos_per_dataset, image_infos, image_infos_per_iid_label

    def _save_filelist_cache_combined(self):
        if cf.get("cache_dataset"):
            c = data.cache.Cache()
            c.save(self._cache_key(), {
                cache.Cache.KEY_FLL_IMG_INFOS_PER_DS: self._image_infos_per_dataset,
                cache.Cache.KEY_FLL_IMG_INFOS: self._image_infos,
                cache.Cache.KEY_FLL_IMG_INFOS_PER_IID: self._image_infos_per_iid_label,
            }, suffix_extension=".p")

    def sample_image_infos(self, max_positive_test_imgs: int, max_negative_test_imgs: int):
        """Get a subset of self.image_infos which has not more than the specified maximum of images per class.

        :param max_positive_test_imgs: The maximum number of foreground images.
        :param max_negative_test_imgs: The maximum number of background images.
        :return:
        """
        # iterate over all image infos until we either reached both maximums or we reached the end of all image infos
        sampled_img_infos = []
        n_positive_test_imgs = 0
        n_negative_test_imgs = 0
        for img in self.image_infos:
            if img.label.is_background:
                if n_negative_test_imgs < max_negative_test_imgs:
                    sampled_img_infos.append(img)
                    n_negative_test_imgs += 1
            else:
                if n_positive_test_imgs < max_positive_test_imgs:
                    sampled_img_infos.append(img)
                    n_positive_test_imgs += 1

            # end if we already got everything we need
            if n_negative_test_imgs >= max_negative_test_imgs and n_positive_test_imgs >= max_positive_test_imgs:
                break

        # log some stats about the sampled images
        log.log("Sampled image stats:")
        n_sampled_imgs = len(sampled_img_infos)
        avg_width_acc = 0
        avg_height_acc = 0
        avg_foreground_objects_acc = 0
        for img in sampled_img_infos:
            avg_width_acc += img.img_width_original
            avg_height_acc += img.img_height_original
            avg_foreground_objects_acc += len(img.annotations) if img.annotations is not None else 0
        avg_width = avg_width_acc / n_sampled_imgs
        avg_height = avg_height_acc / n_sampled_imgs
        avg_foreground_objects = avg_foreground_objects_acc / n_sampled_imgs
        log.log(" - total imgs: {}".format(n_sampled_imgs))
        log.log(" - positive imgs (showing at least one foreground object): {}".format(n_positive_test_imgs))
        log.log(" - negative imgs (no foreground object at all): {}".format(n_negative_test_imgs))
        log.log(" - avg image dimensions: {:.0f}x{:.0f}px".format(avg_width, avg_height))
        log.log(" - avg shown foreground objects: {:.2f}".format(avg_foreground_objects))

        return sampled_img_infos
