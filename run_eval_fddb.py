"""
Run inference/detection on the FDDB dataset using the cascade and export the detected bounding boxes such that they
can (and will) be evaluated by the FDDB evaluation script.
"""
from app.evaluate_fddb_app import EvaluateFDDBApp


EvaluateFDDBApp()
