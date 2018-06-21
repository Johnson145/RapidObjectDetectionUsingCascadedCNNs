import abc
import traceback
from typing import List

import numpy as np
from concurrent.futures.thread import ThreadPoolExecutor

from app.base_app import BaseApp
from data.image_info import ImageInfo
from data.rectangles import LabeledBoundingBox, Window
from utils import log
from utils.time_watcher import TimeWatcher
import config as cf


class AbstractInferenceApp(BaseApp):
    """This class is the common root of all custom and external inference apps."""

    @abc.abstractmethod
    def run_inference_on_windows(self, windows_info: List[Window], windows_raw) -> List[LabeledBoundingBox]:
        """Run inference on a set of sliding windows.

        :param windows_info: The sliding windows to process. May belong to different images.
        :param windows_raw: The sliding windows to process. May belong to different images.
        :return: A list containing all bounding boxes belonging to the foreground classes.
        """
        return

    @abc.abstractmethod
    def run_inference_on_image(self, image: ImageInfo) -> List[LabeledBoundingBox]:
        """Run inference on a single image.

        :param image: The image to process. Not yet split into windows.
        :return: A list containing all bounding boxes belonging to the foreground classes.
        """
        return

    def run_inference_on_images(self, images: List[ImageInfo], merge=True) -> List[List[LabeledBoundingBox]]:
        """Run inference on the given image list.

        :param images: The images to process. Not yet split into windows.
        :param merge: If True, performance will be optimized by processing all images at once. Otherwise,
                    performance may be worse, but you can get additional evaluations referring to a single image only.
        :return: The outer list contains one inner list for each provided input image. Each of such inner lists contains
                    bounding boxes for all found foreground classes.
        """
        # init TensorFlow before starting any timers
        self._init_tf()

        # The outer result list contains one inner list for each provided input image.
        all_results = []

        timer_multiple = TimeWatcher("inference_img_multiple: {} imgs".format(len(images)))

        if merge:
            # extract windows from all images first and merge them
            timer_extracting = TimeWatcher("extract windows from all images and merge them")

            # extract using multiple threads
            # TODO most of the code's runtime is currently required for the "thread.acquire" method
            log.log(" -> extract")
            with ThreadPoolExecutor() as executor:
                results_per_img = list(executor.map(lambda img: self._extract_windows(img, convert_raw_to_np=False),
                                                    images))

            # merge
            log.log(" -> merge")
            windows_merged_info = []  # merge infos first
            for _, window_infos_of_one_image in results_per_img:
                windows_merged_info += window_infos_of_one_image

            # merge raw data by directly creating a common numpy array
            # (so no combined list in between)
            windows_merged_raw = np.empty(
                shape=[len(windows_merged_info), cf.get("img_width"), cf.get("img_height"), 3],
                dtype=cf.get("img_dtype"))
            raw_window_index = 0
            for raw_windows_of_one_image, _ in results_per_img:
                for raw_window in raw_windows_of_one_image:
                    windows_merged_raw[raw_window_index] = raw_window
                    raw_window_index += 1

            # release memory
            results_per_img = None

            timer_extracting.stop()

            # run inference using the merged windows
            log.log("run inference using the merged windows (total: {}, avg per img: {:.0f})".format(
                len(windows_merged_info),
                len(windows_merged_info) / len(images)
            ))
            merged_bboxes = self.run_inference_on_windows(windows_merged_info, windows_merged_raw)

            # separate results: group them by the input images
            all_results_dict = dict()
            for img in images:
                all_results_dict[img.path_original] = []

            for bbox in merged_bboxes:
                all_results_dict[bbox.image.path_original].append(bbox)

            # transform dict to final result list
            all_results = []
            for img in images:
                all_results.append(all_results_dict[img.path_original])
        else:
            # process image after image
            # TODO implement multi-threading for the non-merging mode, too
            for img in images:
                img_results = []
                try:
                    timer_single = TimeWatcher("inference_img_single")
                    img_results = self.run_inference_on_image(img)
                    timer_single.stop()
                except FileNotFoundError:
                    log.log(" .. Skipped {}, because the file could not be found".format(
                        img.path_resized
                    ))
                except:
                    log.log(" .. Skipped {}, because of an unexpected error:\n{}".format(
                        img.path_resized,
                        traceback.format_exc()
                    ))

                all_results.append(img_results)

        timer_multiple.stop()

        if merge:
            # runtime stats for inference only are available in merge mode only
            runtime_total = timer_extracting.elapsed_seconds
            runtime_avg = runtime_total / float(len(images))
            log.log("Runtime window extraction: {} images in {} (avg: {}).".format(
                len(images),
                TimeWatcher.seconds_to_str(runtime_total),
                TimeWatcher.seconds_to_str(runtime_avg)
            ))
            runtime_total = timer_multiple.elapsed_seconds - timer_extracting.elapsed_seconds
            runtime_avg = runtime_total / float(len(images))
            log.log("Runtime inference only: {} images in {} (avg: {}).".format(
                len(images),
                TimeWatcher.seconds_to_str(runtime_total),
                TimeWatcher.seconds_to_str(runtime_avg)
            ))

        # log runtime stats: inference including extracting
        runtime_total = timer_multiple.elapsed_seconds
        runtime_avg = runtime_total / float(len(images))
        log.log("Runtime inference including window extraction: {} images in {} (avg: {}).".format(
            len(images),
            TimeWatcher.seconds_to_str(runtime_total),
            TimeWatcher.seconds_to_str(runtime_avg)
        ))

        return all_results

    def _extract_windows(self, img: ImageInfo, convert_raw_to_np=True):
        """Extract all sliding windows from the given img.

        Essentially, this is a wrapper for Window.extract_windows(img) to allow additional steps required by subclasses.
        Exceptions will be caught and replaced by an empty list along with an error message, because we don't want the
        complete inference process to get stopped because of single images.
        """
        try:
            windows_raw, windows_info = Window.extract_windows(img, convert_raw_to_np)

            if len(windows_raw) < 1:
                raise ValueError("Could not extract any windows from the given image")

            return windows_raw, windows_info

        except FileNotFoundError:
            log.log(" .. Skipped {}, because the file could not be found".format(
                img.path_resized
            ))
            return [], []
        except:
            log.log(" .. Skipped {}, because of an unexpected error:\n{}".format(
                img.path_resized,
                traceback.format_exc()
            ))
            return [], []

    @abc.abstractmethod
    def _init_tf(self):
        """Initialize TensorFlow and load required resources.

        If TensorFlow is already initialized, nothing will happen.
        """
        return
