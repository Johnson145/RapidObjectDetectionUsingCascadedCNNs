"""
Just load the file list once to see the associated stats.
"""
from data.db.file_list_loader import FileListLoader
from utils import log
import config as cf

# load file lists
loader = FileListLoader()
loader.image_infos

# save log files
log.log_set_name("log_file_list")
log.log_save(cf.get("log_dir"))
