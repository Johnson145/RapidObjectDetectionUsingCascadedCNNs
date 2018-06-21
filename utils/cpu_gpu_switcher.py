import os
import config as cf
from utils import log
from utils.singleton import Singleton


class CpuGpuSwitcher(metaclass=Singleton):
    """This singleton provides helper methods to choose which processing units are available to TensorFlow.

    The implementation is based on the environment variable "CUDA_VISIBLE_DEVICES". If no GPU is available, it will
    be an empty string. Otherwise, it will be a comma-separated list of GPU ids that are available to TensorFlow. GPU
    ids start by 0.

    If "CUDA_VISIBLE_DEVICES" points to more than one GPU, TensorFlow will allocate all their memory, even though it
    may actually not be used. Because this project does not support the usage of multiple GPUs at once yet, it will
    prevent waisting resources by explicitly limiting the GPU access to a single user-defined GPU id.

    TODO currently it is impossible to switch between the processing unit after(!) TensorFlow has been used once.
    """

    def __init__(self):
        """Create the singleton object."""

        # if the environment variable isn't set yet, we need to set it right here in order to prevent further errors.
        # (if it wasn't set yet, it means that all available GPUs are enabled, but we don't know whether any GPU is
        #  available in the first place and if it is, we don't know how many)
        if 'CUDA_VISIBLE_DEVICES' not in os.environ:
            os.environ['CUDA_VISIBLE_DEVICES'] = cf.get("preferred_gpu_id")

        # ensure that either the CPU is used or the configured preferred GPU id is
        if self.is_gpu_enabled:
            pu_string = "GPU:{}".format(cf.get("preferred_gpu_id"))
            if os.environ['CUDA_VISIBLE_DEVICES'] != cf.get("preferred_gpu_id"):
                pu_string += " (changed from {})".format(os.environ['CUDA_VISIBLE_DEVICES'])
                os.environ['CUDA_VISIBLE_DEVICES'] = cf.get("preferred_gpu_id")
        else:
            pu_string = "CPU"

        # log the initially-used processing unit
        log.log("processing unit: {}".format(pu_string))

    @property
    def is_gpu_enabled(self) -> bool:
        """Whether the GPU is enabled.

        :return:
        """
        return os.environ['CUDA_VISIBLE_DEVICES'] != ''

    def enable_gpu(self, gpu_id=None):
        """Enable GPU usage.

        :param gpu_id: The id of the GPU which should be used. If None, cf.get("preferred_gpu_id") will be used.
                        If not None, this must be a string containing only an int >= 0.
        :return:
        """
        # default parameter value
        if gpu_id is None:
            gpu_id = cf.get("preferred_gpu_id")

        if not self.is_gpu_enabled or os.environ['CUDA_VISIBLE_DEVICES'] != gpu_id:
            if not self.is_gpu_enabled:
                log.log("Enabling GPU:{} usage.".format(gpu_id))
            else:
                log.log("Switching GPU from {} to {}.".format(os.environ['CUDA_VISIBLE_DEVICES'], gpu_id))
            os.environ['CUDA_VISIBLE_DEVICES'] = gpu_id

    def disable_gpu(self):
        """Disable GPU usage such that only the CPU will be used."""
        if self.is_gpu_enabled:
            os.environ['CUDA_VISIBLE_DEVICES'] = ''
            log.log("Disabling GPU usage.")
