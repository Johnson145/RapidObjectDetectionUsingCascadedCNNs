import math
import os
import time

import numpy as np
import tensorflow as tf

import config as cf
from data import data_augmentation_online
from data.db import label
from data.db.label import IID_BACKGROUND, IID_FOREGROUND
from network import criteria
from utils import log
from utils.collage import CollageEvaluation
from .net import Net


class NetTrainable(Net):
    """This class extends the parent class with everything that is required to train the artificial neural net."""

    def __init__(self, dataset, snapshot_full_path=None, f_beta=None, bottleneck_in_size=None, nr=1, nr_max=1,
                 use_inception_architecture=False):
        """Create a new NetTrainable.

        :param dataset:
        :param snapshot_full_path:
        :param f_beta: The beta parameter of the f-measure that will be used as the loss function.
                        If None, the cross entropy will be used instead.
        :param bottleneck_in_size
        :param nr the number of this net. Usually this remains 1. If a cascade is used, this will be the nr-th net.
        :param nr_max the maximum number of any net in the current cascade. For single nets this remains 1.
        :param use_inception_architecture If set to True, this net will use the inception architecture.
                                            Otherwise, a custom architecture will be chosen.
        """
        # save given parameter values
        self._ds = dataset

        # derive information based on the given dataset
        self._iterations_per_epoch = math.ceil(self._ds.train.n_samples / cf.get("batch_size"))
        self.iterations_total = int(cf.get("epochs_total") * self._iterations_per_epoch)

        # call parent constructor
        Net.__init__(self, self._ds.shape_image_batch, self._ds.shape_label_batch, self._ds.preprocessor,
                     snapshot_full_path, f_beta, bottleneck_in_size, nr, nr_max, use_inception_architecture)

        if self._load_given_snapshot:
            # validate once before training to compare old and new results
            self._full_evaluation(self._ds.valid, "Initial validation set results:")

    def _set_up_architecture(self):
        # parent
        Net._set_up_architecture(self)

        # extension
        self._set_up_architecture_training()

    def _set_up_architecture_training(self):
        """Extends the basic _set_up_architecture() method by training-specific stuff.

        _set_up_architecture() must have been called before.
        :return:
        """
        # loss function
        with tf.name_scope("loss"):

            if self._f_beta is not None and self._ds.train.positive_proportion > 0.5:
                log.log("Warning: Disabling the usage of F-Beta, because there are more positive samples "
                        "than negative ones. Weighted cross entropy will be used instead.")
                self._f_beta = None

            if self._f_beta is None:

                if cf.get("weighted_cross_entropy"):
                    # increase loss value of foreground samples to the level of the unbalanced data ratio
                    if cf.get("weighted_cross_entropy_normalize"):
                        # the foreground weight is 1 - (probability for a foreground sample)
                        # => normalized weights in [0,1] with (foreground_loss_multiplier + background_loss_multiplier = 1)
                        # (calculate the weight based on the training data!!)
                        foreground_loss_multiplier = 1 - self._ds.train.positive_proportion
                        background_loss_multiplier = self._ds.train.positive_proportion
                    else:
                        # there are foreground_loss_multiplier times as much background samples as foreground samples
                        background_loss_multiplier = 1.0
                        foreground_loss_multiplier = (1 - self._ds.train.positive_proportion) / self._ds.train.positive_proportion

                    log.log("increase loss value of foreground samples to the level of the unbalanced data ratio: {:.3f}".format(
                        foreground_loss_multiplier))

                    weights_per_batch = self._Y_float * tf.constant((foreground_loss_multiplier - background_loss_multiplier), tf.float32)
                    weights_per_batch += background_loss_multiplier

                    self._loss = tf.losses.sparse_softmax_cross_entropy(labels=self._Y,
                                                                        logits=self._scores,
                                                                        weights=weights_per_batch)
                else:
                    # use the (unweighted) cross entropy
                    loss_input = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self._scores,
                                                                                labels=self._Y,
                                                                                name="xentropy")
                    self._loss = tf.reduce_mean(loss_input, name="loss_mean")

            else:
                # use the f-measure based on the given beta
                # the f-measure itself is optimal at 1 and bad at 0
                # as we want to minimize instead of maximize, we change this as follows
                loss_input = 1 - self._f_beta_score_diffable
                self._loss = tf.reduce_mean(loss_input, name="loss_mean")

            # L2 regularization for the fully-connected layers
            if cf.get("L2_regularization_strength") > 0:
                regularizers = tf.nn.l2_loss(self._W["fc1"]) + tf.nn.l2_loss(self._b["fc1"]) + \
                               tf.nn.l2_loss(self._W["fc2"]) + tf.nn.l2_loss(self._b["fc2"])
                reg = tf.constant(cf.get("L2_regularization_strength"), dtype=tf.float32)
                self._loss += reg * regularizers

            # L1 regularization for the fully-connected layers
            if cf.get("L1_regularization_strength") > 0:

                l1_regularizer = tf.contrib.layers.l1_regularizer(
                    scale=cf.get("L1_regularization_strength"), scope=None
                )
                l1_weights = [self._W["fc1"], self._W["fc2"], self._b["fc1"], self._b["fc2"]]
                regularizers = tf.contrib.layers.apply_regularization(l1_regularizer, l1_weights)
                self._loss += regularizers

        # Create a variable to track the global step (should be equal to the index var "step" in the training loop)
        self._global_step = tf.Variable(0, name='global_step', trainable=False)

        # automatically decay learning rate
        self.learning_rate_calc = tf.train.exponential_decay(cf.get("learning_rate_init"), self._global_step,
                                                             self.iterations_total / 20, cf.get("learning_rate_decay"),
                                                             staircase=True)
        self.learning_rate = tf.maximum(self.learning_rate_calc, cf.get("learning_rate_min"))

        # create optimizer
        if cf.get("optimizer") == 2 and cf.get("momentum") != 0:
            optimizer = tf.train.MomentumOptimizer(self.learning_rate, cf.get("momentum"))
        elif cf.get("optimizer") == 1:
            optimizer = tf.train.AdamOptimizer(self.learning_rate)
        else:
            optimizer = tf.train.GradientDescentOptimizer(self.learning_rate)

        self.train_op = optimizer.minimize(self._loss, global_step=self._global_step)

        # keep track of loss values
        tf.summary.scalar(self._loss.op.name, self._loss)

        # and maybe of the learning rate
        tf.summary.scalar(self.learning_rate.op.name, self.learning_rate)

        self.summary_op = tf.summary.merge_all()

    def start_tf_session_and_initialize_vars(self):
        Net.start_tf_session_and_initialize_vars(self)

        if self._load_given_snapshot:
            self._session.run(self._global_step.assign(0))

    def train(self):
        """Actually start the training process."""
        # runtime evaluation
        self.runtime_training_start = time.time()

        # allow saving results to file
        summary_writer = tf.summary.FileWriter(os.path.join(cf.get("summary_dir"), cf.get("session_key")),
                                               self._session.graph)

        # set interruption points (interrupt every x iterations)
        # (interrupt_seldom should be a multiple of interrupt_sometimes)
        interrupt_often = 100
        interrupt_sometimes = min(10000, math.floor(cf.get("epochs_total") * self._iterations_per_epoch / 4))
        interrupt_seldom = interrupt_sometimes * 3

        # we should not start evaluation in the last x percent of the training process, because result is irrelevant
        # when compared to the coming final evaluation. this can speed up training a bit.
        no_eval_last_x_percent = 0.15  # in [0, 1]
        max_eval_step = self.iterations_total * (1 - no_eval_last_x_percent)

        # we'll try to detect constant predictions (always the same class label)
        # the following var will count the number of validation evaluations in which a specific class index was the
        # only one predicted
        # e.g.: constant background prediction => n_const_predict_iid[IID_BACKGROUND] += 1
        # as soon as this happened too often, we will cancel the training
        n_const_predict_iid = np.zeros(shape=[label.n_labels()], dtype=np.uint8)

        # Now iterate over our dataset n_epoch times
        # One epoch = use all training data once
        cancel = False  # set to True in order to cancel current training

        ds_i = self._ds.train.new_default_iterator(cf.get("batch_size"))
        while ds_i.epoch < cf.get("epochs_total"):
            log.log('Epoch {}/{}'.format(ds_i.epoch + 1, cf.get("epochs_total")))

            if cancel:
                break

            # mini batches with training data until all training data has been used once:
            curr_epoch = ds_i.epoch
            while ds_i.epoch == curr_epoch:
                batch_i = ds_i.next_batch

                # normalize
                preprocessed_images_i = self._prepare_input(batch_i.images)

                # collect common net params
                feed_dict = {
                    self.dropout_prob: cf.get("dropout_rate")
                }

                # optionally, augment the data (once again)
                if cf.get("data_augmentation_online"):
                    feed_dict[self._X_augmentation_input] = preprocessed_images_i
                    feed_dict[self._Y_augmentation_input] = batch_i.labels
                else:
                    feed_dict[self._X] = preprocessed_images_i
                    feed_dict[self._Y] = batch_i.labels

                if self._bottleneck_in_size is not None:
                    feed_dict[self._bottleneck_in] = batch_i.bottlenecks

                _, loss_value = self._session.run([self.train_op, self._loss], feed_dict=feed_dict)

                if math.isnan(loss_value):
                    log.log("ERROR: loss value is nan. Cancelling training.")
                    cancel = True  # break outer loop
                    break  # break inner loop

                # this step is over
                # (after this point comes only validation based on the number of already done steps)

                # write the summaries and print an overview quite often
                if ds_i.n_provided_batches % interrupt_often == 1 or ds_i.n_provided_batches == self.iterations_total:  # =1 to show after the first iteration already
                    # Print status
                    log.log(
                        'Iteration {0}/{1}: loss = {2:.2f}, learning rate = {3:.4f}'.format(ds_i.n_provided_batches, self.iterations_total,
                                                                                            loss_value,
                                                                                            self._session.run(
                                                                                                self.learning_rate)))
                    # Update the events file.
                    summary_str = self._session.run(self.summary_op, feed_dict=feed_dict)
                    summary_writer.add_summary(summary_str, ds_i.n_provided_batches)
                    summary_writer.flush()

                # never evaluate the net in the very last iteration, because it will be done anyway in stop_training()
                # as well as in final_evaluation() [ensured by ds_iterator.n_provided_batches != self.iterations_total, although ds_iterator.n_provided_batches < max_eval_step
                # does it as well].
                if ds_i.n_provided_batches % interrupt_sometimes == 0 and ds_i.n_provided_batches < max_eval_step and ds_i.n_provided_batches != self.iterations_total:  # ==0 to skip first iteration

                    log.log("Updated evaluation after {}/{} iterations:".format(ds_i.n_provided_batches, int(self.iterations_total)))

                    # 1/2: validation data (evaluation only, without logging so far)
                    res_val = self._full_evaluation(self._ds.valid, " -> validation:")

                    # check for constant prediction (all labels are equal)
                    # TODO generalize from two classes to n classes (requires the calculated predictions to be given)
                    if (res_val["true_positives"] + res_val["false_positives"]) == 0:
                        # no positive predictions => only negative ones
                        n_const_predict_iid[IID_BACKGROUND] += 1

                        log.log("WARNING: validation evaluation suggests constant background prediction ({} times)".format(
                            n_const_predict_iid[IID_BACKGROUND]
                        ))

                        if cf.get("n_max_constant_evals") is not None and n_const_predict_iid[IID_BACKGROUND] > cf.get("n_max_constant_evals"):
                            raise ConstantPredictionException("ERROR: validation evaluation suggests constant background prediction too often. Cancelling training.")

                    elif (res_val["true_negatives"] + res_val["false_negatives"]) == 0:
                        # no negative predictions => only positive ones
                        n_const_predict_iid[IID_FOREGROUND] += 1

                        log.log("WARNING: validation evaluation suggests constant foreground prediction ({} times)".format(
                            n_const_predict_iid[IID_FOREGROUND]
                        ))

                        if cf.get("n_max_constant_evals") is not None and n_const_predict_iid[IID_FOREGROUND] > cf.get("n_max_constant_evals"):
                            raise ConstantPredictionException("ERROR: validation evaluation suggests constant foreground prediction too often. Cancelling training.")

                    # training data shouldn't be evaluated as often as the validation set.
                    if ds_i.n_provided_batches % interrupt_seldom == 0:

                        # 2/2: training data
                        self._full_evaluation(self._ds.train, " -> training:")

                    # check whether the new snapshot is better than everything else seen so far
                    self._update_best_val_results(res_val, ds_i.n_provided_batches)

                # auto restore best model if cf.get("restore_after") iterations have passed without any improvement
                # (doesn't make sense in the very last iteration)
                if cf.get("restore_after") is not None and self.iterations_since_best_found > cf.get("restore_after") \
                        and ds_i.n_provided_batches != self.iterations_total \
                        and self.best_snapshot_path is not None:  # and at least one snapshot has been saved before
                    self.iterations_since_best_found = 0
                    log.log("Step back: load best snapshot found so far, because we haven't made any progress with the"
                            " current one for more than {} iterations.".format(cf.get("restore_after")))
                    self._saver.restore(self._session, self.best_snapshot_path)

                # always increase after(!) all possible resets of this var
                self.iterations_since_best_found += 1

                # check timeout
                if cf.get("timeout_minutes") > 0:
                    self.runtime_training_end = time.time() - self.runtime_training_start

                    if self.runtime_training_end > cf.get("timeout_seconds"):
                        log.log("TIMEOUT: stopping earlier. saving current work.")
                        break

        # everything done
        self.stop_training()

    def _update_best_val_results(self, res_val, step):
        """Check whether res_val contains better results than the best seen so far and remember the answer.

        :param res_val: Validation results as returned by self._full_evaluation(self._ds.valid)
        :param step: The iteration number in which res_val were obtained.
        :return:
        """
        if self.best_val_results is None \
                or res_val[self._main_criteria] > self.best_val_results[self._main_criteria]:
            # snapshots
            log.log("Saving snapshot..")
            snapshot_path_prefix = os.path.join(self._snapshot_dir_session,
                                                "val_{}_{:.3f}".format(
                                                    self._main_criteria,
                                                    res_val[self._main_criteria]
                                                ))
            self.best_snapshot_path = self._saver.save(self._session, snapshot_path_prefix, global_step=step)
            self.best_val_results = res_val
            self.iterations_since_best_found = 0

            # log the new high score
            best_val_txt = criteria.get(self._main_criteria).format(self.best_val_results[self._main_criteria])
            log.log("Updated best model with validation {} of {}".format(
                self._main_criteria,
                best_val_txt)
            )

    def stop_training(self):
        """Stop the currently running training process.

        This method must be called whenever the training has been stopped. No matter whether the training has been
        ended successfully or aborted earlier.
        """
        if self.iterations_since_best_found > 1:
            # after finishing all training iterations, we must evaluate the current validation data one last time to
            # prevent restoring a worse version in a proceeding self.final_evaluation() call.
            # (doing this here instead of inside of the train() method, because we need to do this after cancelling
            #  training, too.)
            # TODO if the training has been cancelled, we could replace self.iterations_total with the actual step value
            log.log("Ensure that the last known best snapshot is still better than the current state.")
            res_val = self._full_evaluation(self._ds.valid)
            self._update_best_val_results(res_val, self.iterations_total)

        # save final runtime one last time
        # (intermediate updates might have already calculated this value, but only if timeout_minutes > 0)
        self.runtime_training_end = time.time() - self.runtime_training_start
        log.log('.. training finished.')


    def final_evaluation(self):
        """Run a final evaluation of the current net state.

        This method might get called after the training has been finished.

        :return:
        """
        log.log("starting final evaluation")

        # Reset weights to best model so far (if the best one isn't the current one)
        if self.iterations_since_best_found > 1:
            if self.best_snapshot_path is None:
                log.log("Error: tried to restore best snapshot, but couldn't find any one at all.")
            else:
                self._saver.restore(self._session, self.best_snapshot_path)
                log.log("Restoring best snapshot of this run ({})".format(self.best_snapshot_path))
        else:
            log.log("Best snapshot of this run was created in the very last iteration: {}".format(
                self.best_snapshot_path
            ))

        # The (best) validation set evaluation probably already exists, so do not evaluate it again
        val_first_log_line = "FINAL validation set evaluation:"
        if self.best_val_results is not None:
            val_eval = self.best_val_results
            self.log_results(self.best_val_results, val_first_log_line)
        else:
            val_eval = self._full_evaluation(self._ds.valid, val_first_log_line)

        train_eval = self._full_evaluation(self._ds.train, "FINAL training set evaluation:")

        test_eval = self._full_evaluation(self._ds.test, "FINAL test set evaluation:")

        # TODO do not re-calculate the predictions, but use the one from the above evaluation
        log.log("Preparing result visualization")
        train_predictions, _ = self.predict(self._ds.train)
        valid_predictions, _ = self.predict(self._ds.valid)
        CollageEvaluation.visualize_train_valid(self._ds.train, self._ds.valid, train_predictions, valid_predictions)

        log.log("final evaluation is done.")

        return val_eval, test_eval, train_eval

    def create_input_placeholder(self):
        """Extend the default image input by optional augmentation operations.

        self._X will be set in the parent class.
        :return:
        """
        if cf.get("data_augmentation_online"):
            log.log("Extending the input placeholders with augmentation operations")
            # this can replace the default input node (self._X) with additional augmentation operations done in TensorFlow
            self._X_augmentation_input = tf.placeholder(name=cf.get("graph_input_training_layer_name") + "_augmented",
                                                        shape=self._shape_data_batch,
                                                        dtype=tf.float32)
            self._Y_augmentation_input = tf.placeholder(name='Y_augmented', shape=self._shape_labels_batch,
                                                        dtype=tf.int32)
            self._X_augmented, self._Y_augmented = data_augmentation_online.add_augmentation_operations(
                self._X_augmentation_input, self._Y_augmentation_input)

            # network input
            # (dtype=tf.uint8 is not allowed)
            _X = tf.placeholder_with_default(name=cf.get("graph_input_training_layer_name"),
                                                  shape=self._shape_data_batch,
                                                  input=self._X_augmented)

            _Y = tf.placeholder_with_default(name="Y",
                                             shape=self._shape_labels_batch,
                                             input=self._Y_augmented)
            return _X, _Y
        else:
            return super().create_input_placeholder()

    @property
    def shape_data_batch(self):
        return self._shape_data_batch


class ConstantPredictionException(Exception):
    """An object of this type will be raised if the current net learned a constant prediction function."""
    pass
