"""
Tune the hyper parameters to get the best single net performance.
"""
from app.tune_single_app import TuneSingleApp

tune_params = [
    "fc1_size",
    "learning_rate_init",
    "L2_regularization_strength",
    "L1_regularization_strength",
    "dropout_rate",
    "learning_rate_decay",
    "conv_filter_sizes",
    "conv_filter_size",
    "conv_stride",
    "pooling_size",
    "pooling_stride",
    "batch_size",
    "optimizer",

    "data_augmentation_online",
    "dao_horizontal_flip",
    "dao_vertical_flip",
    "dao_max_rotation_angle",
    "dao_max_foreground_rotation_angle",
    "dao_crop_min_percent",  # tune this before dao_crop_probability
    # "dao_crop_probability",
    "dao_color_distortion_fast_mode",  # tune this before dao_color_distortion
    # "dao_color_distortion",
]

app = TuneSingleApp(tune_params, random=True)
