import config as cf
from app.base_app import BaseApp
from app.inference_app import InferenceApp
from app.inference_cascade_app import InferenceCascadeApp
from data.db.file_list_loader import FileListLoader
from utils import log
from utils.cpu_gpu_switcher import CpuGpuSwitcher


class EvaluateRuntimeApp(BaseApp):
    """An object of this class will run inference on all combinations of cascade/single gpu/cpu configurations.

    Final runtimes can be taken from the saved log file.

    This app is not extending but using an InferenceApp.
    """

    def __init__(self, cascade_session_key: str, single_session_key: str, max_positive_test_imgs: int,
                 max_negative_test_imgs: int):
        """Create a new EvaluateRuntimeApp.

        :param cascade_session_key: The session key of the serialized cascade model which should be evaluated.
                    If None, cf.get("default_evaluation_model_cascade") will be used.
        :param single_session_key: The session key of the serialized single-net model which should be evaluated.
                    If None, cf.get("default_evaluation_model_single") will be used.
        :param max_positive_test_imgs: The maximum number of foreground images which should be evaluated.
        :param max_negative_test_imgs: The maximum number of background images which should be evaluated.
        """
        self._cascade_session_key = cascade_session_key
        self._single_session_key = single_session_key
        self._max_positive_test_imgs = max_positive_test_imgs
        self._max_negative_test_imgs = max_negative_test_imgs

        # prevent using image patches instead of the original images
        cf.set("cache_dataset", False)

        # sample images only once to ensure that all apps use the exact same files
        self._img_infos = FileListLoader().sample_image_infos(max_positive_test_imgs, max_negative_test_imgs)

        BaseApp.__init__(self)

    def _main(self):
        log.log("Evaluating the runtime of all cascade/single cpu/gpu combinations.")
        log.log(" .. cascade_session_key: {}".format(self._cascade_session_key))
        log.log(" .. single_session_key: {}".format(self._single_session_key))
        log.log(" .. max_positive_test_imgs: {}".format(self._max_positive_test_imgs))
        log.log(" .. max_negative_test_imgs: {}".format(self._max_negative_test_imgs))
        log.log(" .. total number of actually used images: {}".format(len(self._img_infos)))

        self._run_inference(True, True)
        self._run_inference(True, False)
        # TODO currently it is impossible to switch between the processing unit after(!) TensorFlow has been used once.
        # => so the next two lines will still use the GPU
        # self._run_inference(False, True)
        # self._run_inference(False, False)

    def _run_inference(self, enable_gpu: bool, use_cascade: bool):
        """Run a single combination of enable_gpu and use_cascade.

        :param enable_gpu: Whether the GPU should be used. If not, the CPU will be used.
        :param use_cascade: Whether the cascade should be used. If not, the single net will be used.
        :return:
        """

        log.log("")
        log.log("")
        log.log("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        log.log("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        log.log("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        log.log("~~~~~~~~~~~~~~~~~~~~~~~~~~  {} / {}  ~~~~~~~~~~~~~~~~~~".format(
            "Cascade" if use_cascade else "Single Net",
            "GPU" if enable_gpu else "CPU",
        ))
        log.log("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        log.log("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        log.log("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")

        # the GPU should already be enabled, but let's get sure
        if enable_gpu:
            CpuGpuSwitcher().enable_gpu()
        else:
            CpuGpuSwitcher().disable_gpu()

        # create an inference app without running it yet
        if use_cascade:
            app_inference = InferenceCascadeApp(self._cascade_session_key)
        else:
            app_inference = InferenceApp(self._single_session_key)

        # run inference
        _ = app_inference.run_inference_on_images(self._img_infos, merge=cf.get("inference_merge"))

        # tidy up to ensure that a later run does not benefit from anything done in a previous run
        app_inference.clean()
        app_inference = None
        for img_info in self._img_infos:
            img_info.clear_raw_img_cache()
