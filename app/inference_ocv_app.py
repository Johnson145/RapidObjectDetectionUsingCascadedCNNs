import os
from typing import List

import cv2

import config as cf
from app.abstract_inference_app import AbstractInferenceApp
from data.db import label
from data.image_info import ImageInfo
from data.rectangles import LabeledBoundingBox, Window


class InferenceOCVApp(AbstractInferenceApp):
    """This app uses the OpenCV implementation of the Viola Jones method to run face detection."""

    def run_inference_on_windows(self, windows_info: List[Window], windows_raw) -> List[LabeledBoundingBox]:
        raise NotImplementedError("The OpenCV implementation of Viola Jones can't be applied to custom windows.")

    def _init_tf(self):
        # nothing to do here
        pass

    def __init__(self):
        """Create new InferenceOCVApp."""
        # windows will be extracted internally by OpenCV, so we can't merge them
        cf.set("inference_merge", False)

        # base constructor
        AbstractInferenceApp.__init__(self)

    def _main(self):
        if not cf.get("foreground_equals_face"):
            raise AttributeError("This app does not make sense, if you're not looking for a face detector")

        self._detector = cv2.CascadeClassifier(
            os.path.join(cf.get("path_opencv_data"), 'haarcascade_frontalface_default.xml'))

    def run_inference_on_image(self, image: ImageInfo) -> List[LabeledBoundingBox]:
        # collect all found foreground bounding boxes of this image in the following list
        img_results = []

        # the OpenCV implementation will only deliver foreground windows
        label_object = label.get_by_iid(label.IID_FOREGROUND)

        # confidence is static, too
        confidence = 1.0

        # for OpenCV, we need the greyscale image
        img_raw_color = image.raw_original()
        img_raw_gray = cv2.cvtColor(img_raw_color, cv2.COLOR_RGB2GRAY)

        # run inference for the complete image
        faces = self._detector.detectMultiScale(img_raw_gray, cf.get("window_scale_factor"),
                                                cf.get("nms_opencv_min_neighbors"))

        # convert OpenCV bbox to the custom format
        for (x, y, w, h) in faces:
            bbox = LabeledBoundingBox(x, y, x + w, y + h, label_object, confidence)
            img_results.append(bbox)

        return img_results
