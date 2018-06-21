"""
Run inference using the single net.
Results won't be visualized, but some stats will be logged.
"""
from app.inference_app import InferenceApp
import config as cf
from data.db.file_list_loader import FileListLoader
from utils.cpu_gpu_switcher import CpuGpuSwitcher

# visualizing makes much more sense on the original native data
cf.set("dataset_path_root", cf.get("dataset_native_path_root"))

# prevent using image patches instead of the original images
cf.set("cache_dataset", False)

# we don't need too many images here
cf.set("class_min_images", 1000)

# USE CPU ONLY (so we can execute this while training is active, too)
CpuGpuSwitcher().disable_gpu()

# create inference app without actually running anything yet
app = InferenceApp()

# run inference on some random samples
app.run_inference_on_images(FileListLoader().sample_image_infos(max_positive_test_imgs=80,
                                                                max_negative_test_imgs=20))
