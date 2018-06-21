"""
This module handles all global settings.
You may change the default settings right here, but it's suggested to copy the config_local_sample.py file instead.
"""
import os
import random
import string
import time
import numpy as np
from importlib import util

from utils import log
from utils.cpu_gpu_switcher import CpuGpuSwitcher

###############################################################
########### Initialize configuration module ###################
############# (Don't change anything here) ####################
###############################################################

# all global settings will be saved as key-value pairs
# (do not access this dictionary directly from outside of this module)
_cf = dict()

# create session key which is (very likely to be) unique. this key will be used as an suffix for file names etc.
# => timestamp, underscore, 3 random letters
_cf["session_key"] = "{}_{}{}{}".format(
    time.strftime("%Y-%m-%d_%H-%M-%S"),
    random.choice(string.ascii_letters),
    random.choice(string.ascii_letters),
    random.choice(string.ascii_letters)
)
log.log("Session key: {}".format(_cf["session_key"]))


def _load_dataset_keys_available():
    """Automatically set the value of _cf["dataset_keys_available"]."""
    global _cf

    # ensure that the dataset_path_root does exist in the first place
    if not os.path.exists(_cf["dataset_path_root"]):
        raise ValueError(
            "The configured dataset_path_root does not exist: {}".format(_cf["dataset_path_root"]))

    # ensure that it does contain some subdirectories as well
    try:
        _cf["dataset_keys_available"] = next(os.walk(_cf["dataset_path_root"]))[1]
    except StopIteration:
        raise ValueError(
            "The configured dataset_path_root does not contain any datasets: {}".format(_cf["dataset_path_root"]))

###############################################################
########### Initialize default configuration ##################
###############################################################
#### (you may change the values here, but it's preferred    ###
####  to create a separate config_local.py file)            ###
###############################################################

# DEBUG mode: True/False
# Enabling this will affect some other config values (see below) in order to allow faster code execution without
# changing the actual default values permanently.
_cf["debug"] = False


################# Inference configuration #####################
###############################################################

# confidence threshold that is required during inference in order to keep foreground predictions
# => confidence must be greater (not equal) to this threshold
# this can either be
#   - a single number which will be applied after each stage of the cascade
#   - or a list containing _cf["cascade_n_nets"] entries, specifying separate thresholds for each stage of the cascade
# value(s) must be in [0.0, 1.0]. usually, they should be in [0.5, 1.0).
_cf["foreground_confidence_threshold"] = 0.5

# whether multiple images should be processed at once or sequentially
# currently, multi-threading is supported for the merge mode only.
_cf["inference_merge"] = True

# minimum window length in [0, 1]
# the value describes the relative proportion of the shorter input image side.
# e.g. 0.1 means that a single window must cover at least 10% of the input width/height.
# this prevents very very small windows on high quality images. objects that are smaller than the specified size, can
# not be detected anymore.
_cf["min_window_length"] = 0.075

# sliding windows are extracted at different image scales
# the following setting defines how much the image size is reduced at each scale
# e.g.: a value of 2 will half the image dimensions again and again
# OpenCV uses a default value of 1.1 for its Viola Jones implementation. Many examples increase it to 1.3 though.
# the lower the value the more likely it gets to detect all objects. however, a smaller value will also increase the
# required runtime dramatically.
_cf["window_scale_factor"] = 1.1

# configure the non-maximum suppression
NMS_DISABLED = "NMS_DISABLED"
NMS_OPENCV = "NMS_OPENCV"
_cf["nms"] = NMS_OPENCV

# minimum number of neighbors required to keep a bounding box when using the OpenCV NMS implementation
# only relevant if _cf["nms"] == NMS_OPENCV
_cf["nms_opencv_min_neighbors"] = 1  # 0 => keep all.

# if nothing else has been specified, the following already-trained models will be used for any kind of evaluation
# (each value must be the session key of an locally-existing model)
_cf["default_evaluation_model_cascade"] = "cnn_cascade_for_face_detection_Mkd"
_cf["default_evaluation_model_single"] = "single_cnn_for_face_detection_SfN"

# whether we want to enlarge the square bounding boxes vertically
_cf["vertically_enlarge_bboxes"] = False

# whether the preparation of a new cascade step should be done using multiple threads
# (because of heavy overhead, this will probably not help yet)
_cf["multi_threaded_step_preparation"] = False

# choose how to calculate the cascade's final confidence score
# this refers to the anyway kept foreground predictions only
# NOTE: currently, this won't affect anything if _cf["nms"] == NMS_OPENCV
FINAL_CONFIDENCE_CALCULATION_LAST_STEP = "FINAL_CONFIDENCE_CALCULATION_LAST_STEP"  # just use the score of the very last net
FINAL_CONFIDENCE_CALCULATION_AVG = "FINAL_CONFIDENCE_CALCULATION_AVG"  # use the mean of all net's confidence
FINAL_CONFIDENCE_CALCULATION_MULT = "FINAL_CONFIDENCE_CALCULATION_MULT"  # multiply all net's confidence scores (but ensure the following min: MIN_SCORE_FOR_FINAL_CONFIDENCE_CALCULATION_MULT)
MIN_SCORE_FOR_FINAL_CONFIDENCE_CALCULATION_MULT = 0.5001  # see FINAL_CONFIDENCE_CALCULATION_MULT
_cf["final_confidence_calculation"] = FINAL_CONFIDENCE_CALCULATION_LAST_STEP

################# Training configuration ######################
###############################################################

# batch size used in the training phase
_cf["batch_size"] = 1200  # this should be as big as your hardware supports (especially because of the unbalanced data)

# _cf["max_batch_size"] is independent from the actual batch size used during training (_cf["batch_size"])
# instead, this may be a greater number which is used during inference only.
# some(!) code parts do support setting this to None to use the complete dataset at once
_cf["max_batch_size"] = _cf["batch_size"]

# (maximum) number of epochs for one training process
_cf["epochs_total"] = 50

# How often should the same training (=> same config etc.) be repeated? 1 => no repetition
# You may want to increase this while tuning hyper parameters to verify the effect of individual config changes.
_cf["n_repeat_same_session"] = 1

# start with given weights for initial values
# empty string to not load any weights
_cf["snapshot_full_path"] = ""

# if this number of iterations has been exceeded without making any progress => restore best snapshot found so far
# None => never restore during training
_cf["restore_after"] = None

# make your life convenient
_cf["timeout_minutes"] = 0  # maximum number of minutes used for training. 0=unlimited
_cf["log_auto_save"] = True  # if True, the log file will be saved automatically as soon as all calculations have been finished correctly
_cf["auto_save_on_abort"] = True  # Automatically save the latest results on user abort. Otherwise the user will be asked first.

# the maximum number of validation evaluations that are allowed to produce the same constant prediction function
# after this limit was reached, the training will be stopped
# (e.g. n_max_constant_evals=3 => at the 4th time recognizing the foreground-only or background-only prediction, we will cancel)
_cf["n_max_constant_evals"] = 3  # 0 => cancel on first occurrence, None => no cancelling will be caused by this

# initial learning rate
# (must be a value in (0, 1) to prevent errors in the exponential decay)
_cf["learning_rate_init"] = 0.01

# learning rate decay
# (1 means no decay)
_cf["learning_rate_decay"] = 0.9

# optimizer
OPTIMIZER_SGD = 0
OPTIMIZER_ADAM = 1
OPTIMIZER_MOMENTUM = 2  # (see also _cf["momentum))
_cf["optimizer"] = OPTIMIZER_MOMENTUM

# momentum update parameter
# (only relevant, iff _cf["optimizer"] == OPTIMIZER_MOMENTUM)
_cf["momentum"] = 0.9  # 0 deactivates the momentum update.

# regularization
# L2 or L1 can be used for the weights of both fully-connected layers.
# dropout is applied between those two layers.
_cf["dropout_rate"] = 0.5  # 1.0 => no dropout
_cf["L2_regularization_strength"] = 0  # 0 => no L2 regularization
_cf["L1_regularization_strength"] = 0  # 0 => no L1 regularization

# default beta value for the f-measure
# (relevant for training a single net only)
_cf["f_beta_default"] = None  # None to use (weighted) cross entropy instead

# general loss function for training a cascade
# True to use the f-measure
# False to use (weighted) cross entropy instead
_cf["f_beta_cascade_loss"] = True

# whether the very last net of a cascade should be trained with the f-measure, as well.
# if True => the very last net will be trained with the f-measure using _cf["min_beta"]
# if False => the very last net will be trained with the (weighted) cross entropy, although the previous nets may have
#               used the f-measure.
# (only relevant for f_beta_cascade_loss==True)
_cf["f_beta_cascade_loss_very_last"] = False

# the minimum beta value that is used while training a cascade.
# so this is the beta value that will be used for the very last net of the cascade.
# we may choose a value of 1 to weigh equally in the end
# alternatively, we may choose a smaller value such as 0.5 to pay more attention to reducing the False Positives, which
# have been tolerated in earlier nets of the cascade.
# there is one special case: this value will not be used by the very last net, if _cf["f_beta_cascade_loss_very_last"]
# is set to False. _cf["min_beta"] is still relevant for all preceding nets, because they "keep assuming that the final
# value will be _cf["min_beta"]". So the influence to preceding beta values remains the same.
# (only relevant for f_beta_cascade_loss==True)
_cf["min_beta"] = 1

# the maximum beta value that is used while training a cascade
# so this is the beta value that will be used for the very first net of the cascade
# (only relevant for f_beta_cascade_loss==True)
_cf["max_beta"] = 24

# training of an individual net may fail during cascade creation, because n_max_constant_evals was reached.
# usually this is due to a bad choice of the f-measure's beta parameter. however, if the choice is not too bad, it's
# likely that we will succeed in another try with the same beta value. we will try it cascade_max_same_beta times.
_cf["cascade_max_same_beta"] = 3

# whether to use a weighted or unweighted cross entropy
# (this setting will be ignored, if the cross entropy isn't the loss function in the first place)
_cf["weighted_cross_entropy"] = True

# whether the cross entropy class weights should be normalized (all positive, sum=1)
# enabling this may lead to the problem of Vanishing Gradients
# (technically, disabling it may lead to the problem of Exploding Gradients. this has not yet happened though)
_cf["weighted_cross_entropy_normalize"] = False

# the main validation criteria that is used while tuning hyper parameters as well as while training a net using the
# (weighted) cross entropy
_cf["tuning_main_criteria"] = "f1_score"  # f1_score, accuracy, ..

# whether to validate all images before starting the training
# (this can be helpful, if some broken files crash the training)
_cf["remove_broken_images_before_training"] = False

################## Network architecture #######################
###############################################################

# number of neurons used in the "main" fully-connected layer
_cf["fc1_size"] = 512

# configure convolutional layers
# value must be a list:
#   - number of entries equals number of used convolutional layers
#   - each entry describes the number of filter maps used in one convolutional layer
_cf["conv_filter_sizes"] = [32]

# the stride between two equal filters belonging to the same filter map.
# stride is used uniformly in both dimensions (horizontal and vertical).
# set to 1 in order to place filters at each possible location.
_cf["conv_stride"] = 1

# the width and height of a single filter used in the convolutional layer.
# TensorFlow default is 5
_cf["conv_filter_size"] = 3

# configure the max pooling layer
# (options similar to the ones of the convolutional layer)
_cf["pooling_size"] = 3
_cf["pooling_stride"] = 1

# the number of custom neural nets that are trained in the cascade
# (must be an int greater than 1)
# if _cf["append_inception"], the actual number of used nets will be _cf["cascade_n_nets"] + 1
_cf["cascade_n_nets"] = 3

# whether the individual nets of a cascade should use different input dimensions
# if set to False, all nets will use _cf["img_width"] x _cf["img_height"]
# if set to True, _cf["img_width"] x _cf["img_height"] will be used for the very last net only. all preceding nets will
# use a smaller image size, i.e. the i-th net will use 1/2^(_cf["cascade_n_nets"] - i) * (_cf["img_width"] x _cf["img_height"])
_cf["cascade_increasing_input_dimensions"] = True

# if set to True, each net of a cascade will use the bottleneck of the previous net as an additional input
_cf["reuse_bottlenecks"] = True

# whether to append the inception net
# if set to True
# -> the single net will only use the inception net rather than any custom architecture
# -> the cascade ..
#   - will still use the custom architecture in general, but a very last net is appended and uses inception
#   - the total number of nets in the cascade will be _cf["cascade_n_nets"] + 1
#   - _cf["img_width"] will still be used for the last custom(!) net
#   - the very last inception net will always use an input of 299x299
_cf["append_inception"] = False

# layer names
_cf["graph_final_inference_layer_name"] = "final_softmax"
_cf["graph_input_training_layer_name"] = "X_train"
_cf["graph_input_inference_layer_name"] = "X"
_cf["graph_input_bottleneck_layer_name"] = "bottleneck_in"
_cf["graph_output_bottleneck_layer_name"] = "bottleneck_out"
_cf["inception_bottleneck_tensor_name"] = 'pool_3:0'  # name of the last inception tensor that will be extracted from the given model base



######################### Data(sets) ##########################
###############################################################

# Percentages used to split the complete dataset into training, validation and testing data.
# Must be a list containing three positive floats which sum up to one.
_cf["dataset_split"] = [0.8, 0.1, 0.1]

# pre-processing
_cf["standardization"] = True

# datasets which provide annotations
_cf["dataset_keys_annotated"] = ["aflw"]

# enable/disable using a cache for the datasets
_cf["cache_dataset"] = True

# resize images to the following size
# if using a cascade..
# -> _cf["img_width"] will be used as the size of the last custom net
# -> previous nets will always use half of the input size of their proceeding net
# -> e.g. (_cf["img_width"] = 48 and _cf["cascade_n_nets"] = 3) => 12x12, 24x24, 48x48
# -> if _cf["append_inception"], still describes the size of the last custom(!) net, but the appended inception net will use 299x299
# NOTE: currently, only squared dimensions are supported completely
_cf["img_width"] = 48
_cf["img_height"] = _cf["img_width"]

# the following data type will be used to represent images before(!) fed into the network
_cf["img_dtype"] = np.uint8

# this data type will be used to store labels inside of numpy arrays
_cf["label_dtype"] = np.int32

# Minimum and maximum number of samples per(!) class
# - None means no Limit
# - values lower than 50 can cause trouble by providing too few data for the individual splits
# - if _cf["cache_dataset"], changing these values won't have an effect without manually clearing the dataset cache
#   (of course you don't need to clear it, if the dataset hasn't been cached yet in the first place)
_cf["class_min_images"] = 20000
_cf["class_max_images"] = None

# maximum number of samples used in total. None => unlimited
# (if _cf["cache_dataset"] is enabled, you will need to clean the cache first)
_cf["max_samples"] = None

# when using additional background images grouped into further sub-categories, we do not want too many images of the
# same sub-category.
# the background category got far too many images. keep only a small number per different entity
# (currently, this applies only to images from the ImageNet dataset)
_cf["background_max_img_per_entity"] = 25

# whether image patches should be scaled individually or as a part of the larger original image
# only relevant if _cf["cascade_increasing_input_dimensions"] == True
_cf["cascade_scale_patches_individually"] = True

# assuming we do resize patches individually in the first place, we can limit this to the cases when the original image
# hasn't been cached yet in the relevant scale
# True => use cache if available, other rescale individually
# False => always rescale individually
# only relevant if _cf["cascade_increasing_input_dimensions"] == True and _cf["cascade_scale_patches_individually"] == True
_cf["cascade_scale_patches_individually_iff_not_cached"] = False

# determine how splits are resampled in between of consecutive cascade stages
RESAMPLING_ADABOOST_LIKE = "RESAMPLING_ADABOOST_LIKE"  # decrease weights of all background samples equally
RESAMPLING_CONFIDENCE = "RESAMPLING_CONFIDENCE"  # decrease weights of background samples in accordance to their confidence scores
RESAMPLING_DEACTIVATED = "RESAMPLING_DEACTIVATED"  # do not re-sample. use the complete original dataset for all nets.
_cf["cascade_resampling_method"] = RESAMPLING_ADABOOST_LIKE

# (maximum) number of background samples that will be extracted from each native image to augment the dataset
# (this will only be used when running the run_sampling.py file manually)
_cf["sampling_multiplier"] = 30

# The maximum allowed IoU between any sampled background patch and any known foreground region.
# Must be a float in [0, 1). Set to 0 to disallow any intersection.
_cf["sampling_background_max_iou_with_foreground"] = 0.05

# Whether the dataset should be filtered before each run, but after caching
# This will remove samples that are labeled as background, but look like they contain foreground
# this uses a pre-trained single cnn (specified by _cf["default_evaluation_model_single"])
_cf["filter_dataset_after_caching"] = False

# This option refers to the training process only and has no influence to the separate inference.
# If this is True, each training sample will be saved in every required resolution. So when using a 12x24x48 cascade,
# we would create three files for each sample.
# If this option is False, rescaling will be done online while reading the original samples.
# This step is independently of the later _cf["cache_dataset"] option, which will cache all samples of the same size as
# a single file.
# advantages of enabling / use cases:
# - when using (quite) high resolutions, loading the cached files will be faster than online scaling
# - when keeping the cascade architecture constant, but changing the number of used samples quite often
# disadvantages:
# - loading small files (small resolutions) can take much longer than online scaling
# - when changing the cascade architecture quite often, there will be a lot of additional small files on disk
#   -> these take space
#   -> and even may slow done the complete file system
_cf["cache_resized_training_samples_individually"] = False

# whether the dataset loader will be forced to shuffle new data in-place
# doing so will reduce memory requirements, but also increase the required runtime
# (this is only relevant for the dataset creation. once the dataset has been cached, this doesn't matter)
_cf["shuffle_datasets_inplace"] = False

###############################################################
################# Online Augmentation #########################
###############################################################

# whether training samples should be modified online in the first place
# disabling this will make all settings with the prefix "dao_" irrelevant
# changing this setting after the neural net has been created may cause trouble
_cf["data_augmentation_online"] = True

# whether samples should be flipped horizontally (applies to foreground as well as to background)
_cf["dao_horizontal_flip"] = True

# whether samples should be flipped vertically
# don't get confused here: vertical(!) flipping will mirror the image at a horizontal(!) line such that it gets upside
# down.
# (generally, this applies to foreground as well as to background. however, it can be individually disabled for the
#  foreground class. see _cf["dao_allow_vertical_flipping_of_foreground"])
_cf["dao_vertical_flip"] = False

# this option allows to disable vertical flipping during online augmentation for foreground images only.
# so background images may still be flipped.
# only relevant iff (_cf["data_augmentation_online"] == True and _cf["dao_vertical_flip"] == True)
_cf["dao_allow_vertical_flipping_of_foreground"] = False

# rotations
DAO_ROTATION_MODE_DISABLED = "DAO_ROTATION_MODE_DISABLED"  # no rotation
DAO_ROTATION_MODE_CONTINUOUS = "DAO_ROTATION_MODE_CONTINUOUS"  # continuous rotation based on the following settings
DAO_ROTATION_MODE_90 = "DAO_ROTATION_MODE_90"  # 90° and 270° rotations only. background will be rotated by 180°, too.
_cf["dao_rotation_mode"] = DAO_ROTATION_MODE_CONTINUOUS

# whether samples should be rotated up to the given angle
# must be a value in [0.0, 180.0]
# set to 0 to disable any rotations
# (generally, this applies to foreground as well as to background. however, it can be individually changed for the
#  foreground class. see _cf["dao_dao_max_foreground_rotation_angle"])
# only relevant if _cf["dao_rotation_mode"] == DAO_ROTATION_MODE_CONTINUOUS
_cf["dao_max_rotation_angle"] = 0.0

# the maximum angle in degrees which is used to augment foreground samples during online augmentation.
# set to None to use the same value as used for the background
# set to 0 to disable rotation for foreground images only
# a value of 20 will result in a rotation of [-20.0°, 20.0°]
# only relevant iff (_cf["data_augmentation_online"] == True and _cf["dao_max_rotation_angle"] > 0)
# only relevant if _cf["dao_rotation_mode"] == DAO_ROTATION_MODE_CONTINUOUS
_cf["dao_max_foreground_rotation_angle"] = 0.0

# probability to apply additional cropping
# set to 0 to disable cropping completely
_cf["dao_crop_probability"] = 0.5

# minimum length of the cropped path
# must be a value in (0, 1] describing the new length in proportion to the original one
# only relevant iff (_cf["data_augmentation_online"] == True and _cf["dao_crop_min_percent"] > 0)
_cf["dao_crop_min_percent"] = 0.9

# whether to apply color distortions
_cf["dao_color_distortion"] = True

# if True, the color distortions will not use any (very) slow operations.
# only relevant, iff (_cf["data_augmentation_online"] == True and _cf["dao_color_distortion"] == True)
_cf["dao_color_distortion_fast_mode"] = False

################## Other configuration ########################
###############################################################

# the id of the GPU which will be used (if any is available in the first place)
_cf["preferred_gpu_id"] = "0"  # this must be a string containing only an int >= 0

# if you want to overclock your gpu automatically each time you use it, just create a shell script and specify its
# name right here. None means that no shell script will be called.
_cf["overclock_gpu_shell"] = None

# whether the window extraction should log some details
_cf["log_window_extraction_details"] = False

# whether to log some additional information about the cascade's confidence score on the kept foreground samples
_cf["log_cascade_confidence_details"] = False

# whether the foreground class describes human faces
# => this will enable some features which work only for face detection
_cf["foreground_equals_face"] = True

###############################################################
######################### Paths ###############################
###############################################################

# ensure existence of local config file and load it, if available
temp_spec = util.find_spec("config_local")
module_can_be_imported = temp_spec is not None
if not module_can_be_imported:
    raise EnvironmentError("The file 'config_local.py' does not seem to exist. Please copy config_local_sample.py"
                           " and adjust it to your needs.")
import config_local
for key, value in config_local._cf.items():
    _cf[key] = value

# paths
_cf["output_root_dir"] = os.path.join(_cf["project_extension_root"], "output")
_cf["log_dir"] = _cf["log_dir_init"] = os.path.join(_cf["output_root_dir"], "logs")
_cf["snapshot_dir"] = os.path.join(_cf["output_root_dir"], "snapshots")
_cf["summary_dir"] = os.path.join(_cf["output_root_dir"], "summaries")
_cf["collages_dir"] = os.path.join(_cf["output_root_dir"], "collages")
_cf["cache_path_root"] = os.path.join(_cf["project_extension_root_fast"], "cache")

# the native input may already be the final input, but can also be used as the template for
# a pre-sampled augmented input
_cf["dataset_native_path_root"] = os.path.join(_cf["project_extension_root"], "input")

# this will be used to create the augmented pre-sampled version of the "native input"
_cf["dataset_presampled_path_root"] = os.path.join(_cf["output_root_dir"], "input_augmented")

# the (actual) common root of (all) datasets
# this is either _cf["dataset_native_path_root"] or _cf["dataset_presampled_path_root"]
_cf["dataset_path_root"] = _cf["dataset_presampled_path_root"]
# _cf["dataset_path_root"] = _cf["dataset_native_path_root"]

_cf["assets_root"] = os.path.join(_cf["output_root_dir"], "assets")
_cf["ignore_lists_dir"] = os.path.join(_cf["assets_root"], "ignore-lists")
_cf["whitelists_dir"] = os.path.join(_cf["assets_root"], "whitelists")
_cf["path_opencv_data"] = "opencv_data"  # path to the pretrained models of OpenCV. original source: https://github.com/opencv/opencv/tree/master/data/haarcascades
_cf["bbox_visualization_dir"] = os.path.join(_cf["output_root_dir"], "bbox_visualization")  # ATTENTION! this folder will be deleted automatically

# Where to save the trained graph.
_cf["output_graph_dir"] = os.path.join(_cf["output_root_dir"], "graph")
_cf["output_graph_file"] = os.path.join(_cf["output_graph_dir"], "graph_{}.pb".format(
    _cf["session_key"]
))

# this folder may be used to download the inception model, if _cf["append_inception"] is True
_cf["inception_model_base"] = os.path.join(_cf["output_root_dir"], "inception_model_base")

# FDDB paths
_cf["fddb_root_dir"] = os.path.join(_cf["dataset_native_path_root"], "fddb")
_cf["fddb_folds_dir"] = os.path.join(_cf["fddb_root_dir"], "FDDB-folds")
_cf["fddb_img_base_dir"] = os.path.join(_cf["fddb_root_dir"], "images/original/foreground")
_cf["fddb_detection_output_dir"] = os.path.join(_cf["output_root_dir"], "fddb_detection_output")
_cf["fddb_latest_detection_output_dir"] = os.path.join(_cf["fddb_detection_output_dir"], "latest")  # symlink
_cf["fddb_per_evaluation_script_path"] = os.path.join(_cf["fddb_root_dir"], "evaluation_code/runEvaluate.pl")
_cf["fddb_gnuplot_compare_dir"] = os.path.join(_cf["fddb_root_dir"], "rocs")

###############################################################
################ Anything after Paths ########################
###############################################################

# automatically set and validate _cf["dataset_keys_available"]
_load_dataset_keys_available()

# these datasets will actually be loaded and used
# _cf["dataset_keys"] = _cf["dataset_keys_available"]
_cf["dataset_keys"] = ["aflw", "imagenet"]

###############################################################
############## DEBUG MODE Config Changes ######################
###############################################################

if _cf["debug"]:
    # _cf["dataset_keys"] = ["aflw"]
    _cf["class_min_images"] = 100
    _cf["class_max_images"] = 100
    _cf["epochs_total"] = 3
    _cf["cascade_n_nets"] = 3


###############################################################
########### Parameters derived by others ######################
###############################################################

# minimum value for learning rate (after decay)
# (this does not necessarily need to be a derived value)
def refresh_learning_rate_min():
    _cf["learning_rate_min"] = 0.1 * _cf["learning_rate_init"]
refresh_learning_rate_min()

_cf["timeout_seconds"] = _cf["timeout_minutes"] * 60

# only relevant for cf.get("cascade_increasing_input_dimensions") == True:
# the original configuration for the image dimension will be used for the very last net of the cascade
_cf["img_width_max"] = _cf["img_width"]
_cf["img_height_max"] = _cf["img_height"]

# remember the initial value for the max_batch_size option.
# we will need this to calculate a dynamic batch size in accordance to the currently used cascade step.
_cf["max_batch_size_original"] = _cf["max_batch_size"]

###############################################################
############## Don't change anything here #####################
###############################################################

# copy default params to be able to restore them later
_cf_default = _cf.copy()

# create folders, if they do not exist yet
_folders = ["output_root_dir", "log_dir", "snapshot_dir", "summary_dir", "ignore_lists_dir", "inception_model_base",
            "whitelists_dir", "bbox_visualization_dir", "collages_dir", "output_graph_dir",
            "fddb_detection_output_dir"]
for folder_key in _folders:
    if folder_key in _cf and not os.path.exists(_cf[folder_key]):
        os.makedirs(_cf[folder_key])

###############################################################
################### Public Interface ##########################
###############################################################

def get(param_name):
    return _cf[param_name]

def set(param_name, param_value):
    global _cf
    _cf[param_name] = param_value

    if param_name == "learning_rate_init":
        refresh_learning_rate_min()

    elif param_name == "dataset_path_root":
        # changing the dataset root, changes the availability of datasets, too
        log.log("Resetting dataset_keys_available, because dataset_path_root was changed")
        _load_dataset_keys_available()

        if len(_cf["dataset_keys"]) == 0:
            log.log("Resetting dataset_keys to all datasets, because the old list was empty")
            _cf["dataset_keys"] = _cf["dataset_keys_available"]

def reset():
    """Reset current configuration to its default values."""
    global _cf  # keyword global is only necessary for write(!) access
    _cf = _cf_default.copy()


###############################################################
######### Execute stuff which requires a completely ###########
######### initialized config module already.  #################
###############################################################

# initialize GPU/CPU state
# (must be done after _cf['preferred_gpu_id'] has been set and the public interface has been defined)
CpuGpuSwitcher()
