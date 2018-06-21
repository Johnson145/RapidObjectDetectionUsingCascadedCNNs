"""This will load the database once such that all pre-processing and caching is executed."""
import config as cf
from data.db.dataset_loader import DatasetLoader
from utils import log

# load file lists
loader = DatasetLoader()
loader.dataset()

# save log files
log.log_set_name("log_dataset_loading")
log.log_save(cf.get("log_dir"))
