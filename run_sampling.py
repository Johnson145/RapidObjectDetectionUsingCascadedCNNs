"""
Use the native input in cf.get("dataset_native_path_root") to create a pre-sampled input in
cf.get("dataset_presampled_path_root").

ATTENTION, this script won't work, if dataset_path_root is set to dataset_presampled_path_root, because it will either
fail right in the config.py, because the folder does not exist yet, or it will fail in this script, because the folder
does exist.
=> manually set ensure: _cf["dataset_path_root"] = _cf["dataset_native_path_root"]
"""
import traceback

import cv2
from PIL import Image

import config as cf
import os

from data.db import label
from data.db.file_list_loader import FileListLoader
from data.db.dataset_loader import DatasetLoader
from data.db.label import Label
from data.cache import Cache
from utils import log

from data.rectangles import BoundingBox, RestrictedArea
import numpy as np

from utils.img_manipulation import random_img_patch, PotentialDeadlockError


class Sample:
    """Helper class to bundle information of new samples."""

    def __init__(self, label: Label, img_raw):
        self.label = label
        self.img_raw = img_raw


# sampling is supported only once for the complete input
if os.path.isdir(cf.get("dataset_presampled_path_root")):
    raise AttributeError("Can't create an augmented input, because there is already one on disk.")

# create missing base folder
os.makedirs(cf.get("dataset_presampled_path_root"))

# used base must be the native data
# (so this must be called before loading any image data
cf.set("dataset_path_root", cf.get("dataset_native_path_root"))

# cache must be disabled, otherwise we may still try to load an already pre-sampled dataset
cf.set("cache_dataset", False)

# load native input
FileListLoader().image_infos

# total number of saved samples
i_samples_total = 0

# this existing classifier will be used to identify potential faces (that were not annotated).
# the following settings for cv2_scale_factor and cf.get("nms_opencv_min_neighbors") will produce quite a lot false positives in favor
# of reducing false negatives. this would not be a useful configuration for a production environment, but we want to
# ensure that no faces make their way into the background sample pool.
if cf.get("foreground_equals_face"):
    log.log("background patches which look like human faces will be removed automatically")
    cv2_scale_factor = 1.1
    face_cascade = cv2.CascadeClassifier(
                os.path.join(cf.get("path_opencv_data"), 'haarcascade_frontalface_default.xml'))

# log some settings
log.log("number of additional background patches, which will be sampled from each original image: {}".format(
    cf.get("sampling_multiplier")
))
log.log("maximum allowed IoU between a new background sample and any known foreground region: {0:.2f}%".format(
    cf.get("sampling_background_max_iou_with_foreground") * 100
))

# process each native sample after another
i_imgs = 0
background_label = label.get_by_key(label.KEY_BACKGROUND)
log.log("begin processing one native image file after the other (this may take a while)")
for img_info in FileListLoader().image_infos:

    try:

        # collect new samples based on the current image
        img_new_samples = []
        restrictions = []  # ensure that no background patches intersect foreground information

        # load the original image only once
        # (there is no need to use the permanent internal cache though)
        img_raw = Image.open(img_info.path_original).convert('RGB')
        img_width, img_height = img_raw.size

        # first of all, we want to ensure that all annotated regions are used
        # (even if this implies that we get more samples than cf.get("sampling_multiplier"))
        if img_info.annotations is not None:
            for annotation in img_info.annotations:
                if annotation.bbox_is_valid:
                    # crop
                    annotation_img = img_raw.crop((annotation.xmin, annotation.ymin, annotation.xmax, annotation.ymax))
                    # annotated regions must always contain the same label as the complete image
                    annotation_sample = Sample(img_info.label, annotation_img)
                    img_new_samples.append(annotation_sample)

                    # remember annotated foreground regions
                    if img_info.label.is_foreground:
                        restricted_area = RestrictedArea(annotation.bbox, img_width=img_width, img_height=img_height)
                        restrictions.append(restricted_area)

        # some images do contain faces, although they are not annotated. this is true for background images as well
        # as for foreground images coming from AFLW or ImageNet.
        # so we will not only restrict known annotations, but potential faces detected by OpenCV, too
        # (they won't be used as foreground samples though)
        if cf.get("foreground_equals_face"):
            img_raw_np = np.array(img_raw)  # convert pil image to np array, which can be used by OpenCV
            img_raw_gray = cv2.cvtColor(img_raw_np, cv2.COLOR_RGB2GRAY)
            faces = face_cascade.detectMultiScale(img_raw_gray, cv2_scale_factor, cf.get("nms_opencv_min_neighbors"))
            for (x, y, w, h) in faces:
                # pay attention to the order of x and y!
                face_bbox = BoundingBox(x, y, x+w, y+h)
                restricted_area = RestrictedArea(face_bbox, img_width=img_width, img_height=img_height)
                restrictions.append(restricted_area)

        # produce the remaining new samples by using random background patches in different sizes
        # for background images, we can use everything. foreground images can only be used, if they contain partially
        # background
        if img_info.label.is_background or len(restrictions) > 0:
            while len(img_new_samples) < cf.get("sampling_multiplier"):
                try:
                    background_raw = random_img_patch(img_raw, restrictions,
                                                      cf.get("sampling_background_max_iou_with_foreground"))
                    background_sample = Sample(background_label, background_raw)
                    img_new_samples.append(background_sample)
                except PotentialDeadlockError as e:
                    # stop adding background patches when it fails once
                    break
                    log.log("{}".format(e))

        # save new samples on disk
        i_samples_img = 0  # number of saved samples belonging to the current native sample
        for sample in img_new_samples:
            # build the new file name
            dst_file_name = "aug_{}_{}_{}".format(
                i_samples_total,
                i_samples_img,
                img_info.basename
            )

            # build the new file path
            # (do not just replace parts of the original one, as at least the label folder can be different now, too)
            dst_folder = cf.get("dataset_presampled_path_root")
            dst_folder = os.path.join(dst_folder, img_info.dataset_key)
            dst_folder = os.path.join(dst_folder, "images")
            dst_folder = os.path.join(dst_folder, "original")  # unscaled original windows will be saved. this allows dynamic resizing in the "main" code
            dst_folder = os.path.join(dst_folder, sample.label.key)

            # create dirs
            if not os.path.exists(dst_folder):
                os.makedirs(dst_folder)

            # concat folder and file name
            dst = os.path.join(dst_folder, dst_file_name)

            # the new file path must not exist yet
            if os.path.exists(dst):
                raise ValueError("The destination path {} points to an existing file.".format(
                    dst
                ))

            # save augmented image sample on disk
            sample.img_raw.save(dst)

            i_samples_img += 1
            i_samples_total += 1
    except:
        log.log("WARNING: Skipped {}, because of an unexpected error:\n{}".format(
            img_info.full_key,
            traceback.format_exc()
        ))

    i_imgs += 1
    if i_imgs % 100 == 0:
        log.log("Processed {}/{} native files".format(
            i_imgs,
            len(FileListLoader().image_infos)
        ))

log.log("All augmented original files have been created.")

log.log("In order to use the new files, we need to recalculate the cached datasets")

# backup existing cache by renaming the folder
c = Cache()
old_ds_cache_path = c._base_path("dataset")
if os.path.exists(old_ds_cache_path):
    new_path_suffix = 0
    while True:
        new_path = "{}_pre_sampling_backup_{}".format(old_ds_cache_path, new_path_suffix)
        if os.path.exists(new_path):
            new_path_suffix += 1  # this backup already exists, try again
        else:
            log.log("Moving existing dataset cache to {}".format(new_path))
            os.rename(old_ds_cache_path, new_path)
            break

# new data should be loaded based on the just created pre-sampled data
cf.set("dataset_path_root", cf.get("dataset_presampled_path_root"))

# furthermore, we need to reset the already loaded file lists
FileListLoader().reset()

# now, we can try to load the dataset again
# this will start resizing of the pre-sampled data as well as caching afterwards
loader = DatasetLoader()
loader.dataset()

log.log("Done. Don't forget to set _cf[\"dataset_path_root\"] = _cf[\"dataset_presampled_path_root\"]")
