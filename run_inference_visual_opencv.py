"""
Run inference using the OpenCV implementation of the Viola Jones face detector and visualize the results.
"""
from app.inference_ocv_app import InferenceOCVApp
from app.inference_visualizer_app import InferenceVisualizerApp
import config as cf
from data.db.file_list_loader import FileListLoader

# visualizing makes much more sense on the original native data
cf.set("dataset_path_root", cf.get("dataset_native_path_root"))

# prevent using image patches instead of the original images
cf.set("cache_dataset", False)

# we don't need too many images here
cf.set("class_min_images", 1000)

# create an inference app without running it
app_inference = InferenceOCVApp()

# run the actual inference along with the visualization
app_visual = InferenceVisualizerApp(inference_app=app_inference,
                                    images=FileListLoader().sample_image_infos(max_positive_test_imgs=80,
                                                                               max_negative_test_imgs=20))
