"""
Tune the hyper parameters to get the best cascade performance.
"""
from app.tune_cascade_app import TuneCascadeApp

tune_params = [
    # general
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

    # relevant for cascades only
    "cascade_n_nets",
    "min_beta",
    "max_beta",
    "f_beta_cascade_loss_very_last",

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

app = TuneCascadeApp(tune_params, random=True)
