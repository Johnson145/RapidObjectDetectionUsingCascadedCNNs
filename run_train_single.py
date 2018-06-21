"""
Train a new single net.
"""
import config as cf
from app.train_app import TrainApp

# for some reason, the single net isn't able to generalize very well while the dao is enabled
cf.set("data_augmentation_online", False)

# run the basic training once
app = TrainApp()
