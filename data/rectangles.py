"""This module provides classes representing rectangle areas (of an image)."""
from typing import List

import numpy as np

from data.db.label import Label
import config as cf
from data.image_info import ImageInfo
from utils import log


class Rectangle:
    """An abstract base class for everything that is formed like a rectangle.

    The origin (of an image) is assumed to be in the top left corner:

    (xmin, ymin) - - - - - - - - (xmax, ymin)
    |                                       |

    |                                       |

    |                                       |

    |                                       |
    (xmin, ymax) - - - - - - - - (xmax, ymax)
    """

    def __init__(self, xmin, ymin, xmax, ymax):
        """Create a new Rectangle."""
        self._xmin = xmin
        self._ymin = ymin
        self._xmax = xmax
        self._ymax = ymax

    @property
    def xmin(self):
        return self._xmin

    @property
    def ymin(self):
        return self._ymin

    @property
    def xmax(self):
        return self._xmax

    @property
    def ymax(self):
        return self._ymax

    @property
    def width(self):
        return self.xmax - self.xmin

    @property
    def height(self):
        return self.ymax - self.ymin

    @property
    def is_valid(self):
        return None not in [self.xmin, self.ymin, self.xmax, self.ymax]

    def intersects(self, other: "Rectangle") -> bool:
        """Whether this rectangle intersects other.
        See https://stackoverflow.com/a/32088787/1665966

        :param other: Rectangle
        :return:
        """
        x1 = self.xmin
        y1 = self.ymin
        y2 = self.ymax
        x2 = self.xmax

        x3 = other.xmin
        y3 = other.ymin
        y4 = other.ymax
        x4 = other.xmax

        return not (x3 > x2 or y3 > y2 or x1 > x4 or y1 > y4)

    def intersection_over_union(self, other: "Rectangle") -> float:
        """Get the Intersection over Union (IoU) value describing the overlap of this rectangle and the other one.

        Code has been adapted based on the one found here:
        https://www.pyimagesearch.com/2016/11/07/intersection-over-union-iou-for-object-detection/

        :param other: Another rectangle to be checked for overlap.
        :return: The IoU of this rectangle and the other one.
        """
        # determine the (x, y)-coordinates of the intersection rectangle
        x_a = max(self.xmin, other.xmin)
        y_a = max(self.ymin, other.ymin)
        x_b = min(self.xmax, other.xmax)
        y_b = min(self.ymax, other.ymax)

        # compute the area of intersection rectangle
        inter_area = (x_b - x_a + 1) * (y_b - y_a + 1)

        # compute the area of both the prediction and ground-truth
        # rectangles
        own_area = (self.xmax - self.xmin + 1) * (self.ymax - self.ymin + 1)
        other_area = (other.xmax - other.xmin + 1) * (other.ymax - other.ymin + 1)

        # compute the intersection over union by taking the intersection
        # area and dividing it by the sum of prediction + ground-truth
        # areas - the intersection area
        iou = inter_area / float(own_area + other_area - inter_area)

        # return the intersection over union value
        return iou


class BoundingBox(Rectangle):
    """A bounding box describes a single rectangle surrounding the bounds of an entity.

    Right now, the difference to a more general Rectangle object is only semantic.
    """


class RestrictedArea(BoundingBox):
    """A restricted area enlarges a given bounding box by some padding to prevent any overlap with the bounded object
    even though it's bounding box may not be exact.
    """

    def __init__(self, bbox: BoundingBox, enlarge_by=0.05, img_width=None, img_height=None):
        """

        :param bbox: The bounding box that should be enlarged.
        :param enlarge_by: Enlarge bbox relative to this value.
        :param img_width: Optionally, the width of the surrounding image. This will prevent enlarging further than
                            the image border in x-dimension.
        :param img_height: Optionally, the height of the surrounding image. This will prevent enlarging further than
                            the image border in y-dimension.
        """
        # enlarge bbox by the factor enlarge_by into all directions
        old_width = bbox.xmax - bbox.xmin
        old_height = bbox.ymax - bbox.ymin
        width_extension = int(enlarge_by * old_width)
        height_extension = int(enlarge_by * old_height)

        xmin = bbox.xmin - width_extension
        ymin = bbox.ymin - width_extension
        xmax = bbox.xmax + height_extension
        ymax = bbox.ymax + height_extension

        # the lower border is already known
        xmin = max(0, xmin)
        ymin = max(0, ymin)

        # the upper borders may be given, too
        if img_width is not None:
            xmax = min(xmax, img_width)
        if img_height is not None:
            ymax = min(ymax, img_height)

        # finally call the super constructor
        BoundingBox.__init__(self, xmin, ymin, xmax, ymax)


class LabeledBoundingBox(BoundingBox):
    """An object of this class associated a given label to a bounding box."""

    def __init__(self, xmin, ymin, xmax, ymax, label: Label, confidence=0.0, image=None):
        self._label = label
        self._confidence = confidence
        self._image = image
        BoundingBox.__init__(self, xmin, ymin, xmax, ymax)

    @property
    def label(self) -> Label:
        """Get the label of this bounding box."""
        return self._label

    @property
    def confidence(self) -> float:
        """float in [0,1]. How sure are we that this bounding box has the correct label?"""
        return self._confidence

    @property
    def image(self) -> ImageInfo:
        """Optionally, a LabeledBoundingBox can refer to a specific image object."""
        return self._image

    @staticmethod
    def vertically_enlarge_bboxes(original_bboxes: List["LabeledBoundingBox"], enlarge_top=0.2, enlarge_bottom=None) -> List["LabeledBoundingBox"]:
        """As Li et al. suggested, we may want to enlarge all bounding boxes vertically to better approach the
        actually elliptic ground truth of the FDDB dataset.

        It's not clear whether Li et al. extended their bounding boxes by 20% into both directions or only into one.
        As our own bounding boxes seem to be aligned below the chin already, our detector performs best by just
        extending into the top direction.

        :param original_bboxes: A list containing all the bounding boxes that should be enlarged.
        :param enlarge_top: How much do we want to extend it at the top? Value must be in (0,1]
                                If None, no extension will be done at the top.
        :param enlarge_bottom: How much do we want to extend it at the bottom? Value must be in (0,1]
                                If None, no extension will be done at the bottom.
        :return:
        """
        log.log("Vertically extending {} bounding boxes (top: {}, bottom: {})".format(
            len(original_bboxes),
            enlarge_top,
            enlarge_bottom
        ))
        enlarged_bboxes = []
        for original_bbox in original_bboxes:
            if enlarge_top is not None:
                ymin_new = max(original_bbox.ymin - (enlarge_top * original_bbox.height), 0)
            else:
                ymin_new = original_bbox.ymin
            if enlarge_bottom is not None:
                ymax_new = min(original_bbox.ymax + (enlarge_bottom * original_bbox.height),
                               original_bbox.image.img_height_original - 1)
            else:
                ymax_new = original_bbox.ymax
            enlarged_bbox = LabeledBoundingBox(original_bbox.xmin, ymin_new, original_bbox.xmax, ymax_new,
                                               original_bbox.label, original_bbox.confidence, original_bbox.image)
            enlarged_bboxes.append(enlarged_bbox)
        return enlarged_bboxes


class Window(Rectangle):
    """A (sliding) window refers to a specific image patch.

    The inherited attributes refer to a scaled version of the used input image. The attributes with the "_norm" suffix
    provide data that refers to the original resolution of the used input image.
    """

    def __init__(self, xmin, ymin, xmax, ymax, image: ImageInfo, scale=1.0):
        """Create a new (sliding) window.

        :param xmin:
        :param ymin:
        :param xmax:
        :param ymax:
        :param image:
        :param scale: The image scale this window is referring to.
        """
        self._image = image
        self._scale = scale
        Rectangle.__init__(self, xmin, ymin, xmax, ymax)

    @property
    def scale(self):
        return self._scale

    @property
    def xmin_norm(self):
        return int(self.xmin / self.scale)

    @property
    def ymin_norm(self):
        return int(self.ymin / self.scale)

    @property
    def xmax_norm(self):
        return int(self.xmax / self.scale)

    @property
    def ymax_norm(self):
        return int(self.ymax / self.scale)

    @property
    def width_norm(self):
        return int(self.xmax_norm - self.xmin_norm)

    @property
    def height_norm(self):
        return int(self.ymax_norm - self.ymin_norm)

    @property
    def image(self) -> ImageInfo:
        """Get the image from which this window was taken."""
        return self._image

    @property
    def raw(self):
        """Get the raw image pixels that are covered by this window."""

        # get the full image data and cache it when loading for the first time
        # (assuming that we want to extract multiple windows from the same image)
        full_image_data = self.image.raw_scaled(True, self.scale)

        # validation
        if self.ymax > full_image_data.shape[0]:
            raise ValueError("ymax must not leave the image boundaries")
        if self.xmax > full_image_data.shape[1]:
            raise ValueError("xmax must not leave the image boundaries")

        return full_image_data[self.ymin:self.ymax, self.xmin:self.xmax]

    @property
    def raw_norm(self):
        """Get the unscaled(!) raw image pixels that are covered by this window in the original input image."""

        # get the full image data and cache it when loading for the first time
        # (assuming that we want to extract multiple windows from the same image)
        full_image_data = self.image.raw_original(True)

        # validation
        if self.ymax_norm > full_image_data.shape[0]:
            raise ValueError("ymax_norm must not leave the image boundaries")
        if self.xmax_norm > full_image_data.shape[1]:
            raise ValueError("xmax_norm must not leave the image boundaries")

        return full_image_data[self.ymin_norm:self.ymax_norm, self.xmin_norm:self.xmax_norm]

    @classmethod
    def extract_windows(cls, img: ImageInfo, convert_raw_to_np=True):
        """Extract all sliding windows from the given image.

        All returned windows will have the same width and height (according to the current config settings). However,
        they represent receptive fields of the original input image. These receptive fields differ in their size to
        allow the representation of different areas of the input image without (almost) any restriction caused by
        the finally-used window resolution.

        This method does not provide any other window scales which may be required in later stages of a cascade.
        Instead, those scaled windows will be created during the cascade's inference. They are based on the meta
        information provided by this method, but this method isn't evaluated a second time. So the windows will be
        scaled individually at a later stage. Otherwise, it would be difficult to identify windows which use different
        resolutions, but refer to the same receptive field. Furthermore, it doesn't make sense to pre-calculate more
        scales of the complete original image, if those scales are used in later stages of the cascade only. Those
        stages are already assuming a lot of background and so a lot of the re-scaled original images wouldn't be
        needed in the first place. That's why re-scaling the kept foreground windows manually is faster.

        :param convert_raw_to_np If True, the returned raw windows will be an numpy array. Otherwise, a list.
        :param img
        """

        windows_raw = []
        windows_info = []

        # begin with the full image
        # (other image scales of the original image (=scale pyramid) won't be pre-calculated explicitly here yet.
        #  instead, they will be created and cached on demand when using associated windows for the first time.
        #  so pre-calculating them here wouldn't make a difference.)
        scale = 1.0

        # get the full image data and cache it when loading for the first time
        # (assuming that we want to extract multiple windows from the same image)
        full_image_data = img.raw_original(True)

        img_height = full_image_data.shape[0]
        img_width = full_image_data.shape[1]

        # the windows must have the same dimensions as the network input
        window_width = cf.get("img_width")
        window_height = cf.get("img_height")

        # image will be loaded in different scales
        # it should be possible to scale the image such that the complete image could describe a foreground object, but
        # the scanned image can not get smaller than the window.
        # TODO currently, there is no guarantee that the full image is taken as one window, too
        min_img_width = window_width
        min_img_height = window_height

        # prevent too small windows
        max_windows_per_row = 1.0 / cf.get("min_window_length")  # maximum number of non-overlapping windows that could be placed right next to each other
        max_img_length = max_windows_per_row * window_width

        # iterate over different image scales
        while True:
            # if the image resolution is still too big, we won't extract windows, but go to the next scale
            # (without actually scaling the image yet)
            if img_width < max_img_length and img_height < max_img_length:

                if cf.get("log_window_extraction_details"):
                    log.log("  .. Using image scale {:.3f} = {:.0f}x{:.0f}.".format(
                        scale,
                        img_width,
                        img_height
                    ))

                # extract windows from the current scale

                # step == window_width => non-overlapping
                # step > window_width => non-overlapping + empty padding between consecutive windows
                # step < window_width and step > 0 => overlapping windows
                # this seems to have a large
                # effect on the final result
                # ..img_width.. =>  on small image scales, windows are quite big. so a window_width-based decision implies
                #                   quite big steps on the original image scale. There should be a minimum amount of windows
                #                   along each dimension
                # max(.., 1) => we need to move at least one pixel to prevent endless loop of doing nothing
                step_x = max(min(int(0.4 * window_width), int(0.1 * img_width)), 1)
                step_y = max(min(int(0.4 * window_height), int(0.1 * img_height)), 1)

                # iterate over all windows of one scale
                xmin = -1 * step_x
                while True:  # row
                    xmin += step_x
                    xmax = xmin + window_width

                    if xmax >= img_width:
                        break

                    ymin = -1 * step_y
                    while True:  # column
                        ymin += step_y
                        ymax = ymin + window_height

                        if ymax >= img_height:
                            break

                        # create new window and add it to the result
                        window = Window(xmin, ymin, xmax, ymax, img, scale)
                        windows_info.append(window)
                        windows_raw.append(window.raw)
            else:
                if cf.get("log_window_extraction_details"):
                    log.log("  .. Skipping image scale {:.3f} = {:.0f}x{:.0f}, because it's too big.".format(
                        scale,
                        img_width,
                        img_height
                    ))

            # prepare next scale
            scale /= cf.get("window_scale_factor")
            img_height /= cf.get("window_scale_factor")
            img_width /= cf.get("window_scale_factor")

            # stop criteria
            if img_height < min_img_height or img_width < min_img_width:
                break

        # convert raw window data into numpy array such that it can be used as a network input
        # TODO is there a way to pre-calculate the size of the following array so that we do not need to convert it?
        if convert_raw_to_np:
            windows_raw = np.asarray(windows_raw, dtype=cf.get("img_dtype"))

        if cf.get("log_window_extraction_details"):
            log.log("  .. Extracted {} windows.".format(len(windows_raw)))

        return windows_raw, windows_info
