import os
from concurrent.futures import ThreadPoolExecutor
from typing import List

import cv2
import numpy as np

import config as cf
from app.inference_app import InferenceApp
from app.train_cascade_app import TrainCascadeApp
from data.db import label
from data.rectangles import Window, LabeledBoundingBox
from utils import log, numbers


class InferenceCascadeApp(InferenceApp):
    """An object of this class can be used to run inference using a pre-trained cascade."""

    def __init__(self, model_session_key=None):
        """Create a new inference app that is using a cascade of nets.

        :param model_session_key: The session key of the model which will be used for inference.
                                    If None, cf.get("default_evaluation_model_cascade") will be used.
        :return:
        """

        # the currently-running net index (starting at 0)
        self._curr_net_index = None

        # replace default value for the parent's class default value of model_session_key
        if model_session_key is None:
            model_session_key = cf.get("default_evaluation_model_cascade")

        InferenceApp.__init__(self, model_session_key)

        # check how many nets we want to run, by checking file existence
        file_found = True
        self._curr_net_index = 0
        while file_found:
            file_found = os.path.exists(self.model_full_path)
            if file_found:
                self._curr_net_index += 1
        self._n_nets = self._curr_net_index  # the total number of nets equals the first non-existing file index
        if self._n_nets < 1:
            raise FileNotFoundError("Could not find any graph files belonging to the specified cascade: {}".format(
                self.model_full_path))
        elif self._n_nets == 1:
            raise FileNotFoundError("Found a single graph file, but a cascade must consist of at least two: {}".format(
                self.model_full_path))
        log.log("the loaded cascade contains {} nets".format(self._n_nets))
        self._curr_net_index = None  # reset

    def _parse_graph_file(self):
        # load multiple graph files instead of a single one
        old_curr_net_index = self._curr_net_index
        self._curr_net_index = 0
        while self._curr_net_index < self._n_nets:
            InferenceApp._parse_graph_file(self)
            self._curr_net_index += 1
        self._curr_net_index = old_curr_net_index  # reset

    @property
    def _graph_name(self):
        """This name will be prepended to all tensor names in the loaded graph."""
        return "{}_{}".format(super()._graph_name, self._curr_net_index)

    @property
    def _input_bottleneck_name(self):
        """The (new) name of the input tensor that accepts bottlenecks."""
        return '{}/{}:0'.format(self._graph_name, cf.get("graph_input_bottleneck_layer_name"))

    @property
    def _input_bottleneck_tensor(self):
        """The input tensor that accepts bottlenecks."""
        self._init_tf()
        return self._session.graph.get_tensor_by_name(self._input_bottleneck_name)

    @property
    def _output_bottleneck_name(self):
        """The (new) name of the output tensor that provides bottlenecks."""
        return '{}/{}/Relu:0'.format(self._graph_name, cf.get("graph_output_bottleneck_layer_name"))

    @property
    def _output_bottleneck_tensor(self):
        """The output tensor that provides bottlenecks."""
        self._init_tf()
        return self._session.graph.get_tensor_by_name(self._output_bottleneck_name)

    @property
    def model_full_path(self):
        """Get the full file path to a serialized TensorFlow graph file.

        Use this property to allow overriding in subclasses.
        """
        base = self._model_full_path

        # file extension
        ext = ".pb"

        # remove old file extension 1
        if base.endswith(ext):
            base = base[0:len(base) - len(ext)]

        # remove old file extension 2
        # (we start with the very first index, so we need to remove it, too)
        ext2 = "_0"
        if base.endswith(ext2):
            base = base[0:len(base) - len(ext2)]

        # append current net index along with the file extension
        result = "{}_{}{}".format(
            base,
            self._curr_net_index,
            ext
        )

        return result

    def run_inference_on_windows(self, windows_info: List[Window], windows_raw) -> List[LabeledBoundingBox]:

        results = []

        # The first net does not use any bottlenecks as an additional input
        windows_bottlenecks_in = None

        # we may use this to merge the confidence scores of the individual nets
        # (in accordance to cf.get("final_confidence_calculation"))
        windows_confidences_acc = None

        # remember how many windows we used in the very first iteration
        orig_n_windows = len(windows_raw)

        if orig_n_windows < 1:
            raise ValueError("Could not extract any windows from the given image")

        # run inference multiple times
        self._curr_net_index = 0
        while self._curr_net_index < self._n_nets:

            if len(windows_raw) < 1:
                log.log(" -> Stopping cascade inference, because no window candidates are remaining")
                break

            log.log(" -> Inference on net {}/{}".format(
                self._curr_net_index + 1,
                self._n_nets
            ))

            # maybe change the maximum batch size in accordance to the complexity of the currently-used cascade step
            if cf.get("cascade_increasing_input_dimensions"):
                # we double each side of a window within each step of the cascade, so we increase the total input by
                # a factor of four each time.
                # new_batch_size = cf.get("max_batch_size_original") * 4^exponent
                new_batch_size = cf.get("max_batch_size_original")
                exponent = self._n_nets - self._curr_net_index - 1
                for i in range(exponent):
                    new_batch_size = int(new_batch_size * 4)
                log.log("  .. new inference batch size for net {}/{}: {}".format(
                    self._curr_net_index + 1,
                    self._n_nets,
                    new_batch_size
                ))
                cf.set("max_batch_size", new_batch_size)

            # init threshold for foreground predictions
            if numbers.is_number(cf.get("foreground_confidence_threshold")) == 1:
                # same threshold for all stages:
                foreground_confidence_threshold = cf.get("foreground_confidence_threshold")
            elif len(cf.get("foreground_confidence_threshold")) == self._n_nets:
                foreground_confidence_threshold = cf.get("foreground_confidence_threshold")[self._curr_net_index]
            else:
                raise ValueError("Invalid foreground_confidence_threshold.")
            log.log("  .. minimum confidence threshold for foreground predictions: {:.2f}%".format(
                foreground_confidence_threshold * 100
            ))

            # the following vars will be used to store new data about predicted foreground windows
            # ("new data" may also refer to keeping the old data here.)
            # note, the required RAM for this action heavily depends on your operation system. it's efficient on Linux,
            # because Linux uses lazy memory allocations. so the created numpy arrays actually do not grow larger as
            # the number of windows we finally keep. if your OS doesn't support this, you may want to replace the numpy
            # array allocations with simple python lists (at least on Linux those lists will behave much worse though).
            # TODO at the time we processed some batch data, we can actually already get rid of it again. currently, we
            # are keeping it until all batches of one net have been processed
            n_windows_in = len(windows_raw)
            windows_bottlenecks_in_new = np.empty(shape=[n_windows_in, self._output_bottleneck_tensor.shape[1]],
                                                   dtype=np.float32)
            windows_confidences_new = np.empty(shape=[n_windows_in], dtype=np.float32)  # current confidence score per kept window for belonging to the foreground class
            windows_info_new = []
            windows_raw_new = []  #  if cf.get("cascade_increasing_input_dimensions"), this will simply be replaced

            # run in batches
            start = 0
            n_windows_kept = 0  # number of windows that will be kept for sure (so far)
            log.log("  .. actual inference")
            while start < n_windows_in:
                end = min(start + cf.get("max_batch_size"), n_windows_in)

                # collect required net input
                feed_dict = {
                    self._input_img_tensor: windows_raw[start:end]
                }
                if self._curr_net_index > 0 and cf.get("reuse_bottlenecks"):
                    feed_dict[self._input_bottleneck_tensor] = windows_bottlenecks_in[start:end]

                # run inference for all windows of this image
                if self.last_net or not cf.get("reuse_bottlenecks"):
                    # the last net does not require the outgoing bottlenecks, so we do not extract them
                    batch_out_class_probabilities_per_window = self._session.run(self._softmax_tensor, feed_dict)
                else:
                    # all nets before the last one require the outgoing bottlenecks
                    batch_out_bottlenecks_per_window, batch_out_class_probabilities_per_window = self._session.run(
                        [self._output_bottleneck_tensor, self._softmax_tensor],
                        feed_dict
                    )

                # directly filter the batch result for foreground predictions to prevent a huge RAM overhead caused by
                # all the background data
                for window_index_batch in range(len(batch_out_class_probabilities_per_window)):
                    foreground_probability = batch_out_class_probabilities_per_window[window_index_batch][label.IID_FOREGROUND]
                    # keep only predictions which satisfy the pre-defined foreground threshold
                    if foreground_probability > foreground_confidence_threshold:
                        window_index_original = start + window_index_batch
                        windows_info_new.append(windows_info[window_index_original])

                        # accumulate confidence scores
                        if windows_confidences_acc is None:  # => cf.get("final_confidence_calculation") == cf.FINAL_CONFIDENCE_CALCULATION_LAST_STEP or first net
                            windows_confidences_new[n_windows_kept] = foreground_probability
                        elif cf.get("final_confidence_calculation") == cf.FINAL_CONFIDENCE_CALCULATION_AVG:
                            windows_confidences_new[n_windows_kept] = windows_confidences_acc[window_index_original] + foreground_probability
                        else:  # =>  cf.get("final_confidence_calculation") == cf.FINAL_CONFIDENCE_CALCULATION_MULT
                            windows_confidences_new[n_windows_kept] = windows_confidences_acc[window_index_original] * foreground_probability

                        if not self.last_net and cf.get("reuse_bottlenecks"):
                            windows_bottlenecks_in_new[n_windows_kept] = batch_out_bottlenecks_per_window[window_index_batch]
                            if not cf.get("cascade_increasing_input_dimensions"):
                                windows_raw_new.append(windows_raw[window_index_original])

                        # remember next index for the numpy arrays
                        n_windows_kept += 1

                # prepare next batch
                start += cf.get("max_batch_size")

            # shrink numpy arrays to the actually-used parts
            windows_bottlenecks_in_new = windows_bottlenecks_in_new[0:n_windows_kept]
            windows_confidences_new = windows_confidences_new[0:n_windows_kept]

            # we don't need the old data anymore
            # (we may use the vars again as soon as we finished preparing the new data)
            windows_info = None
            windows_raw = None
            windows_bottlenecks_in = None
            windows_confidences_acc = None

            # log
            log.log("    - used {:.2f}% of the original window set".format(
                n_windows_in / orig_n_windows * 100
            ))
            log.log("    - dropped {:.2f}% of the used windows ({}/{}) as background".format(
                (n_windows_in - n_windows_kept) / n_windows_in * 100,
                n_windows_in - n_windows_kept,
                n_windows_in
            ))
            log.log("    - kept {:.2f}% of the used windows ({}/{}) as foreground".format(
                n_windows_kept / n_windows_in * 100,
                n_windows_kept,
                n_windows_in
            ))
            if cf.get("log_cascade_confidence_details"):
                log.log("    - confidence stats of the kept foreground predictions:")
                log.log("      ~ min: {:.3f}%".format(
                    np.min(windows_confidences_new) * 100
                ))
                log.log("      ~ max: {:.3f}%".format(
                    np.max(windows_confidences_new) * 100
                ))
                log.log("      ~ mean: {:.3f}%".format(
                    np.mean(windows_confidences_new) * 100
                ))

            # the following actions are only required if we're not yet in the very last net
            # (we won't see time-consuming stats here to ensure that they do not extend the total runtime)
            if not self.last_net:
                log.log("  .. preparing the next step ({})".format(
                    "multi-threaded" if cf.get("multi_threaded_step_preparation") else "single thread"
                ))

                # if the nets have different input sizes, we need to configure the next size now and immediately start
                # the replacing
                if cf.get("cascade_increasing_input_dimensions"):
                    # update dimensions before changing anything
                    TrainCascadeApp.update_img_dimensions(self._n_nets, self._curr_net_index + 1)

                    if cf.get("multi_threaded_step_preparation"):
                        # use multiple threads to prepare the new raw data
                        with ThreadPoolExecutor() as executor:
                            windows_raw_new = list(
                                executor.map(self._get_increased_raw_window_for_next_net, windows_info_new))
                    else:
                        windows_raw_new = np.empty([n_windows_kept, cf.get("img_height"), cf.get("img_width"), 3],
                                                   dtype=cf.get("img_dtype"))
                        for window_index_new, window_info in enumerate(windows_info_new):
                            windows_raw_new[window_index_new] = self._get_increased_raw_window_for_next_net(window_info)

                # we need to cast the data to an numpy array
                # (we either just created a new list or kept the one we created while iterating in batches)
                if not cf.get("cascade_increasing_input_dimensions") or cf.get("multi_threaded_step_preparation"):
                    log.log("  .. casting raw window list to numpy array")
                    windows_raw_new = np.asarray(windows_raw_new, dtype=cf.get("img_dtype"))

                # use the new data for the next net
                windows_info = windows_info_new
                windows_raw = windows_raw_new
                windows_bottlenecks_in = windows_bottlenecks_in_new

                if cf.get("final_confidence_calculation") != cf.FINAL_CONFIDENCE_CALCULATION_LAST_STEP:
                    windows_confidences_acc = windows_confidences_new
            else:
                # all windows that are kept until the very end, are part of the final result
                foreground_label_object = label.get_by_iid(label.IID_FOREGROUND)

                # we may need to finalize the accumulated confidence scores
                if cf.get("final_confidence_calculation") == cf.FINAL_CONFIDENCE_CALCULATION_AVG:
                    windows_confidences_new /= self._n_nets
                elif cf.get("final_confidence_calculation") == cf.FINAL_CONFIDENCE_CALCULATION_MULT:
                    windows_confidences_new[windows_confidences_new < cf.MIN_SCORE_FOR_FINAL_CONFIDENCE_CALCULATION_MULT] = cf.MIN_SCORE_FOR_FINAL_CONFIDENCE_CALCULATION_MULT

                for window_index_new, window_info in enumerate(windows_info_new):
                    confidence = windows_confidences_new[window_index_new]
                    bbox = LabeledBoundingBox(window_info.xmin_norm, window_info.ymin_norm, window_info.xmax_norm,
                                              window_info.ymax_norm, foreground_label_object, confidence,
                                              window_info.image)
                    results.append(bbox)

            # prepare a new inference iteration
            self._curr_net_index += 1

        # NMS / vertical extension
        results = self._postprocess_bboxes(results)

        return results

    # def _get_increased_raw_windows_for_next_net(self):

    @staticmethod
    def _get_increased_raw_window_for_next_net(window_info: Window) -> np.ndarray:
        """Get the next raw image data for the given window_info.

        This method assumes that (not self.last_net and cf.get("cascade_increasing_input_dimensions")). Furthermore,
        TrainCascadeApp.update_img_dimensions(self._n_nets, self._curr_net_index + 1) should have just been called.
        So this method is meant to prepare the "increased raw window for the next net".

        :param window_info:
        :return:
        """

        next_scale_multiplicator = cf.get("img_width") / window_info.width
        next_scale = window_info.scale * next_scale_multiplicator
        # next_scale = round(next_scale, 5)

        # decide whether we want to crop before resizing or resize before cropping
        if cf.get("cascade_scale_patches_individually") and \
                (not cf.get("cascade_scale_patches_individually_iff_not_cached")
                 or not window_info.image.is_raw_scaled_cached(next_scale)):
            # crop patch from unscaled original image and resize each patch individually
            # - quite a lot of resizing operations
            # + each resizing operation is done on a quite small patch only
            # + no need to access more than one version of the original image
            # see #rectangles.Window.extract_windows(..) method for further explanation
            window_raw = window_info.raw_norm
            # (the default value for cv2.resize(.., interpolation=xy) already applies the fastest resizing algorithm available)
            window_raw = cv2.resize(window_raw, (cf.get("img_width"), cf.get("img_height")))
        else:
            # scale the complete image and crop patches
            # + need to resize only once per scale
            # + the scaled image may already exist, because it was used for a larger receptive field
            #   (iff window_info.image.is_raw_scaled_cached(next_scale)) <= this is of course not only true,
            #                           if resizing was done outside of this loop, but also if it was done once here
            # - need to resize a large image
            # - need to access multiple versions of the larger image
            # => in most cases this solution will be slower than the
            next_xmin = int(window_info.xmin_norm * next_scale)
            next_xmax = next_xmin + cf.get("img_width")  # window_info.xmax_norm * next_scale,
            next_ymin = int(window_info.ymin_norm * next_scale)
            next_ymax = next_ymin + cf.get("img_height")  # window_info.ymax_norm * next_scale,
            next_scale_window_info = Window(next_xmin,
                                            next_ymin,
                                            next_xmax,
                                            next_ymax,
                                            window_info.image,
                                            next_scale
                                            )
            window_raw = next_scale_window_info.raw

            # it doesn't make a difference whether we safe the new or the old window info object,
            # because the new scales are already calculated such that they use the current window
            # scale (no matter which one) and fix it to fit the target data
            # window_info = next_scale_window_info

        return window_raw

    def _update_input_dims(self):
        backup_curr_net_index = self._curr_net_index

        # we need the dims of the very first net to compare them with the dims of the very last net
        self._curr_net_index = 0
        super()._update_input_dims()
        width_very_first = cf.get("img_width")

        # we need the dimensions of the very last net to initialize the maximum dimensions correctly
        self._curr_net_index = self._n_nets - 1

        # this is primarily important to set the maximum dimensions
        # the current dimensions may be overwritten below
        super()._update_input_dims()

        # check whether this cascade uses increasing input dimensions
        increasing_dims = width_very_first < cf.get("img_width")
        cf.set("cascade_increasing_input_dimensions", increasing_dims)

        # if the nets have different input sizes, we need to configure the very first input size right here, otherwise
        # we would extract a wrong window size
        if cf.get("cascade_increasing_input_dimensions"):
            TrainCascadeApp.update_img_dimensions(self._n_nets, 0)

        # restore original state
        self._curr_net_index = backup_curr_net_index

    @property
    def last_net(self):
        """Whether we are currently handling the very last net of the cascade.

        :return:
        """
        return self._curr_net_index == (self._n_nets - 1)

    def run_inference_on_raw_data(self, raw_data):
        raise NotImplementedError("The cascade does not support running raw data directly.")
