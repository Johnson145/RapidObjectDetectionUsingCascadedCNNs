"""
Try to open each image in the loaded file list and add the failed ones to an ignore list.
"""
from data.db.file_list_loader import FileListLoader
from utils import log
import config as cf

# overwrite settings such that not only a random subset of files will be checked
cf.set("class_min_images", None)
cf.set("class_max_images", None)

# define which datasets you want to validate
# cf.set("dataset_keys", cf.get("dataset_keys_available"))

# load file lists
loader = FileListLoader()
loader.image_infos

# check for broken images and blacklist them
loader.remove_broken_images()

# save log files
log.log_set_name("log_broken_images")
log.log_save(cf.get("log_dir"))
