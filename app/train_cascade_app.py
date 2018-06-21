import numpy as np

from app.train_app import TrainApp
import config as cf
from data.datasets import SPLIT_KEY_VAL, SPLIT_KEY_TRAIN, SPLIT_KEY_TEST, Dataset
from data.db import dataset_loader, label
from data.db.label import IID_FOREGROUND
from network import inception_builder
from network.net_trainable import ConstantPredictionException
from utils import log
from utils.collage import CollageEvaluation, CollageResampledSplits


class TrainCascadeApp(TrainApp):
    """Instead of training a single net, this app trains multiple nets to create a cascade."""

    def __init__(self, run_now=True):
        """Create a new TrainCascadeApp.

        :param run_now: Whether to start the training process right now.
        """
        # the total number of nets that will be trained
        self._n_nets = cf.get("cascade_n_nets")

        # check whether we want to append an extra inception net
        if cf.get("append_inception"):
            self._n_nets += 1

        # the currently-trained net index (starting at 0)
        self._curr_net_index = None

        # first dimension: one dictionary containing a numpy array for each split
        # second dimension: a weight for each sample of the associated split
        # will be initialized when calling _load_dataset for the first time
        # TODO we may get rid of _sample_weights_normalized and just use the split's attribute
        self._sample_weights_normalized = None  # sum per split will be 1
        self._sample_weights_acc = None  # accumulator for the unnormalized weights. each single value will be in [0,1], but the sum per split may be bigger

        TrainApp.__init__(self, run_now)

    def _run_training(self):
        # each iteration triggers the training process of a single net
        # (do not use "for i in range(self._n_nets):", because we need to manipulate self._curr_net_index)
        self._curr_net_index = 0
        while self._curr_net_index < self._n_nets:

            log.log("************************************************************")
            log.log("******** Training net {}/{} to create a cascade ***********".format(
                self._curr_net_index + 1,
                self._n_nets
            ))

            # configure the loss function
            # in the beginning, we want to pay special attention to reducing the false negatives
            # in the very end, we want to weigh everything equally
            if cf.get("f_beta_cascade_loss"):
                self._f_beta = cf.get("max_beta") - ((self._curr_net_index / (self._n_nets - 1)) * (cf.get("max_beta") - cf.get("min_beta")))

                # the very last net's loss may be replaced by the (weighted) cross entropy, even though the preceding
                # nets were using the f-measure
                if self.last_net and not cf.get("f_beta_cascade_loss_very_last"):
                    self._f_beta = None

            # configure the re-usage of the previous net's bottleneck
            if self._curr_net_index == 0 or not cf.get("reuse_bottlenecks"):
                # the very first net does not use any other bottleneck input
                bottleneck_in_size = None
            else:
                bottleneck_in_size = self._net.bottleneck_out_size

            # this try block prevents reaching the try block in the upper's class _main() method.
            trial_nr = 1
            while trial_nr <= cf.get("cascade_max_same_beta"):
                try:
                    # actually start the training
                    # (this must use the super class implementation!)
                    TrainApp._run_training(self, bottleneck_in_size)
                    break
                except ConstantPredictionException as e:
                    # if the net is learning a constant function, we are probably using a bad beta value.
                    # we will try a few times. if it still fails, we cancel the training
                    if trial_nr >= cf.get("cascade_max_same_beta"):
                        raise e
                    else:
                        log.log("WARNING: Retrying with same beta value: {}/{}".format(trial_nr, cf.get("cascade_max_same_beta")))
                        trial_nr += 1

            # the following steps can be skipped, if we do not have any proceeding net
            if not self.last_net:

                ds_loader = dataset_loader.DatasetLoader()
                ds_all = ds_loader.dataset()

                for split_key, split in ds_all.splits.items():

                    # updating bottlenecks
                    # (this can't be done during the previous training process evaluation anymore, because the splits
                    #  used during the training process aren't the same as the filtered ones
                    #  -> keep in mind that no other prediction with the same net can be done after the evaluation in that case though
                    # -> for the enlarged(!) splits, this can be combined with the prediction we need anyway

                    log.log("Updating {} sample weights{}".format(
                        split_key,
                        " and bottlenecks" if cf.get("reuse_bottlenecks") else ""
                    ))

                    # run inference on the final net using all(!) original split data as the input
                    # (this may be more data than actually used during training process)
                    # don't add a log_line="Inference evaluation for enlarged training set split", because this will cause
                    # the predictions to take twice as much time. we don't need the evaluation here anyway
                    predicted_label_iids, probabilities = \
                        self._net.predict(split, update_bottlenecks=cf.get("reuse_bottlenecks"),
                                          return_probabilities=True)  # this must update the bottlenecks!

                    if cf.get("cascade_resampling_method") == cf.RESAMPLING_CONFIDENCE:

                        for j in range(len(predicted_label_iids)):
                            predicted_iid = predicted_label_iids[j]
                            actual_iid = split.labels[j]

                            # real foreground, must be recognized by all(!) nets. so its sample weight will remain at
                            # the maximum value of 1
                            if actual_iid == IID_FOREGROUND:
                                weight_change = 1

                            # for all others: the probability of being seen in the current net (during production) is the
                            # product of all keep-probabilities (=prob to be foreground) provided by preceding nets
                            else:
                                weight_change = probabilities[j][label.IID_FOREGROUND]

                            self._sample_weights_normalized[split_key][j] *= weight_change

                        # normalize weights
                        self._sample_weights_normalized[split_key] = self._sample_weights_acc[split_key] / \
                                                                     self._sample_weights_acc[split_key].sum()

                    elif cf.get("cascade_resampling_method") == cf.RESAMPLING_ADABOOST_LIKE:
                        # this resampling method is similar to the AdaBoost.M1 algorithm of Freund and Schapire.
                        # it mainly differs in the fact, that instead of decreasing the weights of correct predictions,
                        # we decrease the weights of background samples.

                        # calculate a weighted error
                        error = 0
                        for j in range(len(predicted_label_iids)):
                            predicted_iid = predicted_label_iids[j]
                            actual_iid = split.labels[j]
                            # only incorrect predictions account to the weighted error
                            if predicted_iid != actual_iid:
                                error += self._sample_weights_normalized[split_key][j]

                        # update weights on the given training results
                        if error == 0 or error >= 0.5:
                            # the AdaBoost.M1 algorithm assumes error < 0.5. Usually, this assumption should be
                            # satisfied, but in case it doesn't, we will reset the weight distribution
                            log.log("resetting weight distribution, because of an unsupported error rate.")
                            self._sample_weights_normalized[split_key] = np.full([split.n_samples], 1 / split.n_samples)
                        else:
                            # update by decreasing the weights of some samples
                            update_factor = error / (1 - error)
                            for j in range(len(predicted_label_iids)):
                                predicted_iid = predicted_label_iids[j]
                                actual_iid = split.labels[j]
                                #
                                # original: only correct predictions need to be updated here
                                # if predicted_iid == actual_iid:
                                #
                                # only the ones predicted to be background
                                # if predicted_iid == label.IID_BACKGROUND
                                if predicted_iid == label.IID_BACKGROUND:
                                    self._sample_weights_normalized[split_key][j] = self._sample_weights_normalized[split_key][j] * update_factor

                            # normalize weights
                            self._sample_weights_normalized[split_key] = self._sample_weights_normalized[split_key] / self._sample_weights_normalized[split_key].sum()
                    elif cf.get("cascade_resampling_method") == cf.RESAMPLING_DEACTIVATED:
                        do_nothing_to_keep_dataset = True

            # now is the time to release all resources of the current net, as we need them for the next one
            self._net.close_session()

            # prepare iteration for the next net
            self._curr_net_index += 1

    def _output_graph_file_path(self) -> str:
        # as we create multiple nets, we do have multiple graph files, too
        base = TrainApp._output_graph_file_path(self)

        # file extension
        ext = ".pb"

        # remove old file extension
        if base.endswith(ext):
            base = base[0:len(base) - len(ext)]

        # append current net index along with the file extension
        result = "{}_{}{}".format(
            base,
            self._curr_net_index,
            ext
        )

        return result

    @staticmethod
    def update_img_dimensions(n_nets: int, curr_net_index: int):
        """Update the configuration to use the next pair of width and height settings."""
        if cf.get("append_inception") and curr_net_index == (n_nets -1):  # static version of self.use_inception_architecture
            # special case for the inception architecture
            width_curr_net = inception_builder.MODEL_INPUT_WIDTH
            height_curr_net = inception_builder.MODEL_INPUT_HEIGHT
        else:  # default behavior
            # begin calculation with the maximum value
            width_curr_net = cf.get("img_width_max")
            height_curr_net = cf.get("img_height_max")

            # and decrease it this many times
            exponent = n_nets - curr_net_index - 1

            # if an extra inception net is appended, we need to ignore it here explicitly
            if cf.get("append_inception"):
                exponent -= 1

            # width_curr_net = cf.get("img_width_max") * 1/2^exponent
            for i in range(exponent):
                width_curr_net = int(width_curr_net / 2)
                height_curr_net = int(height_curr_net / 2)

        cf.set("img_width", width_curr_net)
        cf.set("img_height", height_curr_net)

    def _load_dataset(self) -> Dataset:
        """We may manipulate the probability distribution used to generate batches for each new net of the cascade."""

        # load the current dataset first
        ds_loader = dataset_loader.DatasetLoader()

        # if we want to increase the image dimensions..
        if cf.get("cascade_increasing_input_dimensions"):
            log.log("need to reload the dataset using the new image dimensions")

            # calculate the new width x height
            TrainCascadeApp.update_img_dimensions(self._n_nets, self._curr_net_index)

            # if this is not the very first net of the cascade, ..
            if self._curr_net_index > 0:
                # .. we need to ensure that the new data is compatible with the old one
                old_dataset = ds_loader.dataset()

                # reload the dataset with the new dimension settings
                # (but keep the current file list)
                ds_loader.reset(reset_file_list=False)
                ds = ds_loader.dataset()

                new_labels = ds.labels
                if len(new_labels) != len(old_dataset.labels):
                    raise ValueError("The previous net's dataset length ({}) is incompatible with the current"
                                     " one ({}).".format(
                                        len(old_dataset.labels),
                                        len(new_labels)
                                    ))
                elif not np.array_equal(old_dataset.labels, new_labels):
                    raise ValueError(
                        "The previous net's dataset is incompatible with the current one. Although the length "
                        "is the same, the label values do not match.")

                # if required, copy bottlenecks
                if cf.get("reuse_bottlenecks"):
                    ds.train.set_bottlenecks(old_dataset.train.bottlenecks)
                    ds.valid.set_bottlenecks(old_dataset.valid.bottlenecks)
                    ds.test.set_bottlenecks(old_dataset.test.bottlenecks)

                # not needed anymore (allow releasing the memory)
                old_dataset = None
            else:
                # load dataset for the very first time
                ds = ds_loader.dataset()
        else:
            ds = ds_loader.dataset()

        # if we want to update the probability distribution..
        if cf.get("cascade_resampling_method") != cf.RESAMPLING_DEACTIVATED:

            # if this is the very first net..
            if self._curr_net_index == 0:
                log.log("initializing sample probability distribution for usage in later nets")

                # if not done yet, we need to initialize the sample weights
                # (they'll be updated in self._main())
                self._sample_weights_normalized = {
                    SPLIT_KEY_TRAIN: np.full([ds.train.n_samples], 1 / ds.train.n_samples),
                    SPLIT_KEY_VAL: np.full([ds.valid.n_samples], 1 / ds.valid.n_samples),
                    SPLIT_KEY_TEST: np.full([ds.test.n_samples], 1 / ds.test.n_samples)
                }
                if cf.get("cascade_resampling_method") == cf.RESAMPLING_CONFIDENCE:
                    self._sample_weights_acc = {
                        SPLIT_KEY_TRAIN: np.ones([ds.train.n_samples]),
                        SPLIT_KEY_VAL: np.ones([ds.valid.n_samples]),
                        SPLIT_KEY_TEST: np.ones([ds.test.n_samples])
                    }

                # we'll use the complete training set, because the very first net of the cascade will always have to
                # process all inputs during production, too.
                # -> so we don't use any probability distribution here to ensure that all data is used for sure
                # -> as of now, .set_probability_distribution(None) isn't necessary, because the associated splits
                #    have never owned a probability distribution yet. let's be sure though:
                ds.train.set_probability_distribution(None)
                ds.valid.set_probability_distribution(None)
                ds.test.set_probability_distribution(None)

            else:  # consecutive nets use a RandomizedIterator based on the new probability distribution
                log.log("using a new sample probability distribution")
                ds.train.set_probability_distribution(self._sample_weights_normalized[SPLIT_KEY_TRAIN])
                ds.valid.set_probability_distribution(self._sample_weights_normalized[SPLIT_KEY_VAL])
                ds.test.set_probability_distribution(self._sample_weights_normalized[SPLIT_KEY_TEST])

        # visualize the used training/validation samples
        CollageResampledSplits.visualize_train_valid(ds.train, ds.valid)

        return ds

    def _finalize_latest_session(self):
        """Evaluate the created cascade by merging the results of the single nets.

        This method will be called separately for each single net. Results will be saved independently first. When
        calling this method for the very last net, all previously-saved results will be merged.
        """
        # call super method first!
        TrainApp._finalize_latest_session(self)

        # load the complete dataset for the following evaluation
        ds_loader = dataset_loader.DatasetLoader()
        ds_all = ds_loader.dataset()

        # make the different splits iterable
        splits = ds_all.splits

        # before evaluating the very first net, initialize required accumulators
        if self._curr_net_index == 0:
            self._predictions = dict()
            for key, split in splits.items():
                self._predictions[key] = np.full([split.n_samples], label.IID_FOREGROUND, dtype=np.int8)

        # run current net and merge with older results
        new_predicted_label_iids = dict()
        for key, split in splits.items():
            new_predicted_label_iids[key], _ = self._net.predict(split)

        # merge the single-net result with the previous ones
        for key, split in splits.items():
            self._predictions[key] *= new_predicted_label_iids[key]

        # print ground truth distribution of the still included samples
        # (may be a bit confusing, because the surrounding code does actually evaluate all(!) samples of the splits.
        #  however, the following calculation is meant to simulate the behavior of the deployed inference. in that
        #  case, later stages do only use the samples which have been predicted to be foreground by all previous
        #  stages)
        ##
        log.log("Class distribution of samples (according to the ground truth: foreground and background), "
                "which are still predicted(!) to be foreground after net {}/{}".format(
            self._curr_net_index + 1,
            self._n_nets
        ))
        for key, split in splits.items():
            n_still_in_and_actually_positive = (self._predictions[key] * split.labels).sum()  # same as true_positives
            n_still_in_but_actually_negative = (self._predictions[key] * (split.labels - 1)).sum() * -1  # same as false_positives
            n_still_in = n_still_in_and_actually_positive + n_still_in_but_actually_negative
            log.log("-> {} split".format(key))
            log.log(" - n_positive_samples: {}".format(
                n_still_in_and_actually_positive
            ))
            log.log(" - n_negative_samples: {}".format(
                n_still_in_but_actually_negative
            ))
            log.log(" - n_total_samples: {}".format(
                n_still_in
            ))

        # if this is the very last net, calculate the final evaluation criteria
        if self.last_net:

            # globally save the combined results as the final one
            self._final_results = dict()

            for key, split in splits.items():
                # we need to calculate some values by ourselves
                results = dict()
                results["true_positives"] = (self._predictions[key] * split.labels).sum()
                results["true_negatives"] = ((self._predictions[key] - 1) * (split.labels - 1)).sum()
                results["false_negatives"] = ((self._predictions[key] - 1) * split.labels).sum() * -1
                results["false_positives"] = (self._predictions[key] * (split.labels - 1)).sum() * -1

                # extend results by further evaluation criterias and log it
                self._final_results[key] = self._net.process_results(results, "Combined cascade evaluation for the {}"
                                                                              " split".format(key))

            # visualize the combined cascade evaluation in a separate collage
            log.log("Visualize the combined cascade evaluation in a separate collage")
            CollageEvaluation.visualize_train_valid(ds_all.train, ds_all.valid, self._predictions[SPLIT_KEY_TRAIN],
                                            self._predictions[SPLIT_KEY_VAL])
        else:
            # TrainApp._finalize_latest_session(self) has set the self._final_results attribute, but we don't want to
            # provide the single net results
            self._final_results = None

    def delete_graph_file(self):
        index_cache = self._curr_net_index
        for i in range(index_cache + 1):
            self._curr_net_index = i
            TrainApp.delete_graph_file(self)

    def _log_current_config(self):
        # begin with information already provided by the super class
        TrainApp._log_current_config(self)

        # append additional information that is only relevant for cascade training
        log.log(".. resampling method: {}".format(cf.get("cascade_resampling_method")))
        log.log(".. reuse_bottlenecks: {}".format(cf.get("reuse_bottlenecks")))
        log.log(".. max_beta: {}".format(cf.get("max_beta")))
        log.log(".. min_beta: {}".format(cf.get("min_beta")))

    @property
    def net_nr(self):
        """The number (=index+1) of the current net."""
        return self._curr_net_index + 1

    @property
    def net_nr_max(self):
        """The maximum net number (=index+1) of the current cascade."""
        return self._n_nets

    @property
    def last_net(self):
        """Whether we are currently handling the very last net of the cascade.

        :return:
        """
        return self._curr_net_index == (self._n_nets - 1)

    @property
    def use_inception_architecture(self):
        return super().use_inception_architecture and self.last_net
