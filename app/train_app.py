from typing import Dict

import os
from tensorflow.python.platform import gfile

import config as cf
from app.base_app import BaseApp
from data.datasets import Dataset, SPLIT_KEY_VAL, SPLIT_KEY_TRAIN, SPLIT_KEY_TEST
from data.db import dataset_loader
from data.db.file_list_loader import FileListLoader
from network import net_trainable as nt, inception_builder
from network.inference_optimizer import InferenceOptimizer
from network.net_trainable import ConstantPredictionException
from utils import log
from utils.time_watcher import TimeWatcher
import tensorflow as tf


class TrainApp(BaseApp):
    """This app can be used to train a single net.

    Training of a cascade is handled in the subclass: TrainCascadeApp.
    """

    def __init__(self, run_now=True):
        """Create new TrainApp.

        :param run_now: Whether to start the training right now.
        """
        # introduce an additional attribute pointing to the currently used network
        self._net = None

        # the beta parameter of the f-measure that will be used as the loss function
        self._f_beta = cf.get("f_beta_default")

        # some more default values
        self._tw_training_complete = None
        self._files_checked = False
        self._final_results = None

        # call the super constructor
        BaseApp.__init__(self, run_now)

    def _main(self):
        self._check_files()
        try:
            self._run_training()
        except ConstantPredictionException:
            log.log("Cancelling because of an ConstantPredictionException exception")
            self._on_cancel()

    def _check_files(self):
        """Test all loaded input images for TensorFlow compatibility."""
        if not self._files_checked and cf.get("remove_broken_images_before_training"):
            tw_broken_images = TimeWatcher("RemoveBrokenImages")
            FileListLoader().remove_broken_images()
            tw_broken_images.stop()
            self._files_checked = True

    def _run_training(self, bottleneck_in_size=None):
        """Run the training with current settings."""

        self._tw_training_complete = TimeWatcher("SingleNetTrainingComplete")

        # load dataset
        ds = self._load_dataset()
        ds.log_stats()

        # create the neural network
        log.log('Creating the network')
        self._net = nt.NetTrainable(dataset=ds,
                                    snapshot_full_path=cf.get("snapshot_full_path"),
                                    f_beta=self._f_beta,
                                    bottleneck_in_size=bottleneck_in_size,
                                    nr=self.net_nr,
                                    nr_max=self.net_nr_max,
                                    use_inception_architecture=self.use_inception_architecture)

        # log some relevant configuration settings before starting the training
        self._log_current_config()

        # actually train the net
        self._net.train()

        # training has been completed correctly, finalize this session
        self._finalize_latest_session()

    def _log_current_config(self):
        """Log some relevant configuration settings (before starting the training)."""
        # Training
        log.log('Start Training..')
        if cf.get("timeout_minutes") > 0:
            log.log('.. timeout after {} minutes'.format(cf.get("timeout_minutes")))
        log.log('.. total number of epochs: {}'.format(cf.get("epochs_total")))
        log.log('.. batch size in each iteration: {}'.format(cf.get("batch_size")))
        log.log('.. learning rate init: {}'.format(cf.get("learning_rate_init")))
        log.log('.. learning rate decay: {}'.format(cf.get("learning_rate_decay")))
        log.log('.. learning rate minimum: {}'.format(cf.get("learning_rate_min")))
        log.log('.. L2 regularization active: {}'.format(cf.get("L2_regularization_strength") > 0))
        if cf.get("L2_regularization_strength") > 0:
            log.log('.. L2 regularization strength: {}'.format(cf.get("L2_regularization_strength")))
        log.log('.. L1 regularization active: {}'.format(cf.get("L1_regularization_strength") > 0))
        if cf.get("L1_regularization_strength") > 0:
            log.log('.. L1 regularization strength: {}'.format(cf.get("L1_regularization_strength")))
        log.log('.. drop out active: {}'.format(cf.get("dropout_rate") > 0 and cf.get("dropout_rate") < 1))
        if cf.get("dropout_rate") > 0 and cf.get("dropout_rate") < 1:
            log.log('.. drop out rate: {}'.format(cf.get("dropout_rate")))
        log.log(".. filter_dataset_after_caching: {}".format(cf.get("filter_dataset_after_caching")))
        log.log(".. data augmentation online: {}".format(cf.get("data_augmentation_online")))

        # log the used loss function
        if self._f_beta is None:
            if cf.get("weighted_cross_entropy"):
                if cf.get("weighted_cross_entropy_normalize"):
                    loss_text = "normalized"
                else:
                    loss_text = "UNnormalized"
                loss_text += " weighted"
            else:
                loss_text = "(unweighted)"
            loss_text += " cross entropy"
        else:
            loss_text = "f_{:.2f}".format(self._f_beta)
        log.log(".. loss function: {}".format(loss_text))

        # print optimizer
        optimizer_name = "unknown"
        if cf.get("optimizer") == 0:
            optimizer_name = "GradientDescentOptimizer"
        elif cf.get("optimizer") == 1:
            optimizer_name = "AdamOptimizer"
        elif cf.get("optimizer") == 2:
            optimizer_name = "MomentumOptimizer"
        log.log('.. optimizer: {}'.format(optimizer_name))

        if cf.get("optimizer") == 2:
            log.log('.. momentum update: {}'.format(cf.get("momentum")))

    def _finalize_latest_session(self):
        """Finalize the training session that was started at last."""
        # done
        self._tw_training_complete.stop()

        # the final evaluation may change the current net one last time, so we need to do this before exporting
        val_eval, test_eval, train_eval = self._net.final_evaluation()

        # save final results
        self._final_results = {
            SPLIT_KEY_TRAIN: train_eval,
            SPLIT_KEY_VAL: val_eval,
            SPLIT_KEY_TEST: test_eval
        }

        # exporting
        self._export_graph()

    def _on_cancel(self):
        """This method will be called when the user interrupted the main method."""
        if self._net is not None:
            # cancel running training session
            self._net.stop_training()

            # the following line allows the user to enter just the letter y instead of "y".
            # (in PyCharm)
            y = "y"
            n = "n"

            # ask the user whether the latest results should be saved
            finalize_and_save = cf.get("auto_save_on_abort") or eval(input("Do you want to save the latest data? [y/n]"))
            if finalize_and_save != "n":
                log.log("Saving latest results.")
                # finalize as usual
                self._finalize_latest_session()
            else:
                log.log("Results deleted.")

    def _export_graph(self):
        """Persist the trained graph including the weights stored as constants."""
        log.log("Exporting..")

        # define several file paths for the exported graph file
        graph_file_path_final = self._output_graph_file_path()
        graph_file_path_frozen = graph_file_path_final.replace(".pb", "_training_frozen.pb")
        graph_file_path_optimized = graph_file_path_final.replace(".pb", "_inference_optimized.pb")

        # freeze and serialize the current training graph
        log.log("  .. frozen version of the original training graph")
        frozen_training_graph_def = self._net.output_graph_def
        with gfile.FastGFile(graph_file_path_frozen, 'wb') as f:
            f.write(frozen_training_graph_def.SerializeToString())

        # export a version optimized for inference
        # (actually, this is even mandatory to remove at least all data augmentation nodes)
        log.log("  .. inference-optimized graph")
        _ = InferenceOptimizer(input=graph_file_path_frozen,
                               output=graph_file_path_optimized,
                               frozen_graph=True,
                               input_names=cf.get("graph_input_training_layer_name"),
                               output_names=cf.get("graph_final_inference_layer_name"))

        # the inference optimization removes the shape information of the input node. so we need to add a new
        # placeholder providing explicit shape information.
        # (this information is necessary to allow dynamic cascade evaluation)
        # -> re-import the serialized graph, but replace the image placeholder
        tf.reset_default_graph()
        x_new = tf.placeholder(name=cf.get("graph_input_inference_layer_name"), shape=self._net.shape_data_batch,
                               dtype=tf.float32)
        with tf.gfile.FastGFile(graph_file_path_optimized, 'rb') as f:
            unmodified_reimported_graph_def = tf.GraphDef()
            unmodified_reimported_graph_def.ParseFromString(f.read())
            _ = tf.import_graph_def(unmodified_reimported_graph_def,
                                    name="",  # this is important to prevent a default prefix
                                    input_map={cf.get("graph_input_training_layer_name"): x_new})
            unmodified_reimported_graph_def = None  # prevent using the wrong graph def
            # modified_reimported_graph_def = self._net.output_graph_def  # don't use this one, as it won't provide the new graph yet
            modified_reimported_graph_def = tf.get_default_graph().as_graph_def()

        # and export it once again
        log.log("  .. finally a modified graph version using another placeholder to:\n{}".format(
            graph_file_path_final
        ))
        with gfile.FastGFile(graph_file_path_final, 'wb') as f:
            f.write(modified_reimported_graph_def.SerializeToString())

        # delete temp graph files
        os.remove(graph_file_path_frozen)
        os.remove(graph_file_path_optimized)

    def _output_graph_file_path(self) -> str:
        """Get the file path that will be used to save the final net graph."""
        return cf.get("output_graph_file")

    def delete_graph_file(self):
        """Delete the graph file that was created by calling self._export_graph()."""
        if os.path.exists(self._output_graph_file_path()):
            log.log("Deleting graph file {}".format(
                self._output_graph_file_path()
            ))
            os.remove(self._output_graph_file_path())

    def _load_dataset(self) -> Dataset:
        """Load and provide the dataset used for training, validation and testing."""
        # when using the inception net, we need to ignore the dimension settings made by the user
        if cf.get("append_inception"):
            cf.set("img_width", inception_builder.MODEL_INPUT_WIDTH)
            cf.set("img_height", inception_builder.MODEL_INPUT_HEIGHT)

        # use all available data
        ds_loader = dataset_loader.DatasetLoader()
        ds = ds_loader.dataset()
        return ds

    def _on_finished(self):
        """This method will be called when the main method is done (either finished regularly or cancelled)."""
        BaseApp._on_finished(self)
        # net can't be used after this point
        self._net.close_session()

    @property
    def net_nr(self):
        """The number (=index+1) of the current net."""
        return 1

    @property
    def net_nr_max(self):
        """The maximum net number (=index+1) of the current cascade."""
        return 1

    @property
    def final_results(self) -> Dict[str, Dict[str, float]]:
        """Get the final evaluation results.

        Results will be available after calling _finalize_latest_session, otherwise None will be returned.
        """
        return self._final_results

    @property
    def use_inception_architecture(self):
        """If True, the inception architecture will be used to build the net.
        Otherwise, a custom architecture will be chosen.

        :return:
        """
        return cf.get("append_inception")
