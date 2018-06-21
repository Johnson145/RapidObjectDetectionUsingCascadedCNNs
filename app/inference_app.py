import os

import cv2
from typing import List

import numpy as np
import tensorflow as tf

import config as cf
from app.abstract_inference_app import AbstractInferenceApp
from data.db import label
from data.image_info import ImageInfo
from data.rectangles import Window, LabeledBoundingBox
from utils import log
from utils.time_watcher import TimeWatcher


class InferenceApp(AbstractInferenceApp):
    """An object of this class can be used to run inference using a pre-trained single net.

    Note, although this class isn't using any cascade yet, the InferenceCascadeApp is inheriting from this class.
    """

    def __init__(self, model_session_key=None):
        """Create a new inference app that is using a single custom net (no cascade yet).

        :param model_session_key: The session key of the model which will be used for inference.
                                    If None, cf.get("default_evaluation_model_single") will be used.
        :return:
        """
        self._tf_initialized = False
        self._session = None  # TensorFlow session

        # default value for model_session_key
        if model_session_key is None:
            model_session_key = cf.get("default_evaluation_model_single")

        # use properties to access the following attributes!
        self._model_session_key = model_session_key
        self._model_full_path = os.path.join(cf.get("output_graph_dir"), "graph_{}.pb".format(
            self._model_session_key
        ))

        # log settings which influence the inference result and speed
        log.log("Initializing inference app with the following settings:")
        log.log(" - pre-trained model: {}".format(model_session_key))
        log.log(" - min_window_length: {}".format(cf.get("min_window_length")))
        log.log(" - window_scale_factor: {}".format(cf.get("window_scale_factor")))
        log.log(" - nms: {}".format(cf.get("nms")))
        if cf.get("nms") == cf.NMS_OPENCV:
            log.log("   -> nms_opencv_min_neighbors: {}".format(cf.get("nms_opencv_min_neighbors")))

        AbstractInferenceApp.__init__(self)

    def _main(self):
        pass

    @property
    def model_full_path(self):
        """Get the full file path to a serialized TensorFlow graph file.

        Use this property to allow overriding in subclasses.
        """
        return self._model_full_path

    @property
    def _graph_name(self):
        """This name will be prepended to all tensor names in the loaded graph."""
        return "graph"

    @property
    def _input_img_name(self):
        """The (new) name of the input tensor."""
        return '{}/{}:0'.format(self._graph_name, cf.get("graph_input_inference_layer_name"))

    @property
    def _input_img_tensor(self):
        """The input tensor."""
        self._init_tf()
        return self._session.graph.get_tensor_by_name(self._input_img_name)

    @property
    def _softmax_name(self):
        """The (new) name of the softmax tensor."""
        return '{}/{}:0'.format(self._graph_name, cf.get("graph_final_inference_layer_name"))

    @property
    def _softmax_tensor(self):
        """The softmax tensor."""
        self._init_tf()
        return self._session.graph.get_tensor_by_name(self._softmax_name)

    def _init_tf(self):
        """Initialize TensorFlow and load required resources.

        If TensorFlow is already initialized, nothing will happen.
        """
        if not self._tf_initialized:
            self._tf_initialized = True
            self._session = tf.Session()
            self._parse_graph_file()

    def _parse_graph_file(self):
        """Creates a graph from saved GraphDef file."""
        # TensorFlow won't check file existence, so let's do this first
        if not os.path.exists(self.model_full_path):
            raise FileNotFoundError("Could not find the required graph file: {}".format(self.model_full_path))
        elif os.path.isdir(self.model_full_path):
            raise ValueError("The graph file path does not point to a file, but to a folder: {}".format(self.model_full_path))

        # parse the graph file
        with tf.gfile.FastGFile(self.model_full_path, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            _ = tf.import_graph_def(graph_def, name=self._graph_name)

    def run_inference_on_windows(self, windows_info: List[Window], windows_raw) -> List[LabeledBoundingBox]:
        # validate params
        if len(windows_info) < 1:
            raise ValueError("The given window list must not be empty.")
        if len(windows_info) != len(windows_raw):
            raise ValueError("The given window lists (raw vs. info) must have the same length.")

        results = []

        # run inference for all given windows
        # run in batches
        n_windows = len(windows_raw)
        class_probabilities_per_window = np.empty(shape=[n_windows, self._softmax_tensor.shape[1]],
                                                  dtype=np.float32)
        start = 0
        while start < n_windows:
            end = min(start + cf.get("max_batch_size"), n_windows)
            class_probabilities_per_window[start:end] = self.run_inference_on_raw_data(windows_raw[start:end])

            # prepare next batch
            start += cf.get("max_batch_size")

        # create bounding box objects for all foreground results
        for window_i in range(len(windows_info)):
            window_info = windows_info[window_i]
            best_guess_label_iid = class_probabilities_per_window[window_i].argmax()
            confidence = class_probabilities_per_window[window_i][best_guess_label_iid]

            if best_guess_label_iid != label.IID_BACKGROUND:
                label_object = label.get_by_iid(best_guess_label_iid)
                bbox = LabeledBoundingBox(window_info.xmin_norm, window_info.ymin_norm, window_info.xmax_norm,
                                          window_info.ymax_norm, label_object, confidence, window_info.image)
                results.append(bbox)

        # NMS / vertical extension
        results = self._postprocess_bboxes(results)

        return results

    def run_inference_on_raw_data(self, raw_data: np.ndarray) -> np.ndarray:
        """Just run inference on the given data and return the result.

        No batch splitting, no meta data..

        :param raw_data:
        :return:
        """
        return self._session.run(self._softmax_tensor, {
            self._input_img_tensor: raw_data
        })

    @staticmethod
    def _run_nms(candidates_bboxes: List[LabeledBoundingBox]) -> List[LabeledBoundingBox]:
        """Run Non-Maximum Suppression on the given bounding boxes.

        :param candidates_bboxes:
        :return:
        """
        log.log("Non-Maximum Suppression: {} ({} candidates)".format(cf.get("nms"), len(candidates_bboxes)))

        if cf.get("nms") == cf.NMS_DISABLED:
            return candidates_bboxes

        tw_nms = TimeWatcher("NMS")
        results = []

        if cf.get("nms") == cf.NMS_OPENCV:

            # prepare NMS by converting all bbox objects into the format required by OpenCV
            log.log("  -> prepare NMS by converting all bbox objects into the format required by OpenCV")
            candidates_opencv_infos_per_img = dict()  # each element is (x, y, w, h)
            image_infos_per_key = dict()
            for bbox in candidates_bboxes:
                if bbox.image.full_key not in candidates_opencv_infos_per_img:
                    candidates_opencv_infos_per_img[bbox.image.full_key] = []
                    image_infos_per_key[bbox.image.full_key] = bbox.image
                # TODO this does not take the original confidence into account! (that's why the different calc methods don't make any difference, too.)
                opencv_info = (bbox.xmin, bbox.ymin, bbox.width, bbox.height)
                candidates_opencv_infos_per_img[bbox.image.full_key].append(opencv_info)

            # actually run NMS
            min_neighbors = cf.get("nms_opencv_min_neighbors")
            log.log("  -> actually run NMS with a threshold of {}".format(min_neighbors))
            for img_key, candidates_opencv_infos in candidates_opencv_infos_per_img.items():
                results_opencv_infos, weights = cv2.groupRectangles(candidates_opencv_infos, min_neighbors)

                # convert the OpenCV information to bbox objects again
                # log.log("  -> convert the OpenCV information to bbox objects again")
                label_object = label.get_by_iid(label.IID_FOREGROUND)
                for i in range(len(results_opencv_infos)):
                    (x, y, w, h) = results_opencv_infos[i]
                    # note that, the new confidence score isn't normalized
                    confidence = float(weights[i])
                    bbox = LabeledBoundingBox(x, y, x + w, y + h, label_object, confidence,
                                              image_infos_per_key[img_key])
                    results.append(bbox)

        tw_nms.stop()
        log.log("  -> kept {}/{} windows".format(len(results), len(candidates_bboxes)))

        return results

    def _postprocess_bboxes(self, results: List[LabeledBoundingBox]) -> List[LabeledBoundingBox]:
        """Call this before returning any calculated bounding boxes to apply further post-processing to the complete set.

        :return:
        """
        # NMS
        results = self._run_nms(results)

        # maybe extend vertically
        if cf.get("vertically_enlarge_bboxes"):
            results = LabeledBoundingBox.vertically_enlarge_bboxes(results)

        return results

    def run_inference_on_image(self, image: ImageInfo) -> List[LabeledBoundingBox]:
        """Run inference on a single image.

        :param image: The image to process. Not yet split into windows.
        :return: A list containing all bounding boxes belonging to the foreground classes.
        """
        # ensure that TensorFlow has been initialized
        self._init_tf()

        log.log(" ")
        log.log("#################################################################")
        log.log("Run inference on {}".format(image.basename))
        tw_img_total = TimeWatcher("Run inference on {}".format(image.basename))

        # split the current image into sliding windows
        # windows_raw contains only the image patches and is ready to be fed into the network
        # windows_info contains further meta data about the windows
        log.log(" -> extracting windows")
        windows_raw, windows_info = self._extract_windows(image)

        # collect all found foreground bounding boxes of this image in the following list
        img_results = self.run_inference_on_windows(windows_info, windows_raw)

        tw_img_total.stop()

        # log
        log.log("-> final result: {}/{} ({:.2f}%) positive windows".format(
            len(img_results),
            len(windows_info),
            len(img_results) / len(windows_info) * 100
        ))
        log.log("#################################################################")
        log.log(" ")

        return img_results

    def clean(self):
        """Release TensorFlow resources."""
        self._tf_initialized = False
        self._session.close()
        self._session = None
        tf.reset_default_graph()

    def _update_input_dims(self):
        """Automatically configure the global settings required to load input having the correct dimensions."""
        # instead of using the latest settings from the current config file, we should use the settings applied while
        # the given net was trained.
        img_height_max = self.supported_img_height
        img_width_max = self.supported_img_width
        cf.set("img_height_max", img_height_max)
        cf.set("img_width_max", img_width_max)
        cf.set("img_height", img_height_max)
        cf.set("img_width", img_width_max)

    @property
    def supported_img_width(self):
        """Get the supported input image width."""
        return self._input_img_tensor.shape[2].value

    @property
    def supported_img_height(self):
        """Get the supported input image height."""
        return self._input_img_tensor.shape[1].value

    def _extract_windows(self, img: ImageInfo, convert_raw_to_np=True):
        # before extracting any windows, we need to ensure that the correct input dimensions are being used
        self._update_input_dims()
        return super()._extract_windows(img, convert_raw_to_np)
