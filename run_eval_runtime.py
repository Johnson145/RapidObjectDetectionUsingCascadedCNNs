"""
Evaluate the runtime of the cascade as well as the single net's runtime.
CPU and GPU will be used separately.
"""
from app.evaluate_runtime_app import EvaluateRuntimeApp
import config as cf

# this makes much more sense on the original native data
cf.set("dataset_path_root", cf.get("dataset_native_path_root"))

# configure and run the evaluation app
evaluation_app = EvaluateRuntimeApp(
    cascade_session_key=None,
    single_session_key=None,
    max_positive_test_imgs=80,
    max_negative_test_imgs=20,
)
