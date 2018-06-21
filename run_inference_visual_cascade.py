"""
Run inference using the cascade and visualize the results.
"""
from app.inference_cascade_app import InferenceCascadeApp
from app.inference_visualizer_app import InferenceVisualizerApp
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

# run the actual inference along with the visualization
app_visual = InferenceVisualizerApp(inference_app=InferenceCascadeApp(),
                                    images=FileListLoader().sample_image_infos(max_positive_test_imgs=80,
                                                                               max_negative_test_imgs=20))
