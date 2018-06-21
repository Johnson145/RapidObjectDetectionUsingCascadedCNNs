import os

import config as cf
import numpy as np
import tensorflow as tf
import collections
from data.datasets import DatasetSplit, DeterministicIterator
from data.db import label
from network import net_builder, criteria, inception_builder
from utils import log


class Net:
    """An object of this class represents an artificial neural network.

    It already provides everything you need to create and run the network. All training stuff is outsourced to a
    separate subclass though.
    """

    def __init__(self, shape_data_batch, shape_labels_batch, preprocessor, snapshot_full_path=None, f_beta=None,
                 bottleneck_in_size=None, nr=1, nr_max=1, use_inception_architecture=False):
        """Create a new artificial neural network.

        :param shape_data_batch: The shape of a single input batch (referring to the "main image input").
        :param shape_labels_batch: The shape of a single label batch.
        :param preprocessor: The preprocessor which is required to standardize the input data.
        :param snapshot_full_path: Optionally, a file path pointing to a snapshot of a previous training.
        :param f_beta: The beta parameter of the f-measure that will calculated during the evaluation phase.
                        If None, no additional F-measure will be calculated.
        :param bottleneck_in_size: The size of the last fully connected layer (=bottleneck) of the previous net that
                is part of the same cascade as this net. If set to None, no connection to any other net is established.
                Otherwise, the bottleneck of the previous net is expected to be part of this net's input.
        :param nr the number of this net. Usually this remains 1. If a cascade is used, this will be the nr-th net.
        :param nr_max the maximum number of any net in the current cascade. For single nets this remains 1.
        :param use_inception_architecture If set to True, this net will use the inception architecture.
                                            Otherwise, a custom architecture will be chosen.
        """
        self._use_inception_architecture = use_inception_architecture
        self._preprocessor = preprocessor
        self._bottleneck_in_size = bottleneck_in_size
        self._f_beta = f_beta
        self._nr = nr
        self._nr_max = nr_max

        # this key will be used to identify the value of the f-measure using self._f_beta as the beta value
        if self._f_beta is None:
            self._f_beta_key = None
        else:
            self._f_beta_key = "f_{:.2f}_score".format(self._f_beta)

        # snapshot to init weights with
        self._load_given_snapshot = snapshot_full_path is not None and len(snapshot_full_path) > 0
        self._snapshot_full_path = snapshot_full_path

        # save given parameter values
        self._shape_data_batch = shape_data_batch
        self._shape_labels_batch = shape_labels_batch

        # dictionaries containing references to the current weights
        # (key is the name of the weighted layer) e.g. self._W["fc1"]
        self._W = dict()
        self._b = dict()

        # remember the best snapshot so far
        self.best_val_results = None
        self.best_snapshot_path = None
        self.iterations_since_best_found = 0

        # this is the evaluation criteria that will be used to determine the above defined "best snapshot"
        if self._f_beta is None:
            self._main_criteria = cf.get("tuning_main_criteria")
        else:
            self._main_criteria = self._f_beta_key

        # group snapshots in separate session dir
        self._snapshot_dir_session = os.path.join(cf.get("snapshot_dir"), cf.get("session_key"))
        if not os.path.exists(self._snapshot_dir_session):
            os.makedirs(self._snapshot_dir_session)

        log.log('Creating network..')
        self._set_up_architecture()

        self.start_tf_session_and_initialize_vars()

    def create_input_placeholder(self):
        """Create the main image input placeholder self._X.

        This method may be overridden in subclasses.
        :return:
        """
        # (dtype=tf.uint8 is not allowed)
        _X = tf.placeholder(name=cf.get("graph_input_training_layer_name"),
                            shape=self._shape_data_batch,
                            dtype=tf.float32)

        # labels / network output. not(!) in one-hot. so just one-dimensional
        _Y = tf.placeholder(name='Y', shape=self._shape_labels_batch, dtype=tf.int32)

        return _X, _Y

    def _set_up_architecture(self):
        """Create the net's architecture as a TensorFlow graph.

        The base architecture is as follows:
        X => conv1 => relu1 => pool1 => conv2 => .. convn => relun => pooln => fc1 => fc1_relu => fc2 => softmax

        However, this base architecture may be modified e.g. to include the inception architecture or to handle
        additional bottleneck inputs.

        :return:
        """
        tf.reset_default_graph()

        # basic network input
        self._X, self._Y = self.create_input_placeholder()

        log.log("x shape: {}".format(self._shape_data_batch))
        log.log("y shape: {}".format(self._shape_labels_batch))

        # casted labels
        self._Y_one_hot = tf.one_hot(self._Y, 2)
        self._Y_float = tf.cast(self._Y, dtype=tf.float32)

        # bottleneck input
        if self._bottleneck_in_size is not None:
            bottleneck_in_shape = [None, self._bottleneck_in_size]
            self._bottleneck_in = tf.placeholder(name=cf.get("graph_input_bottleneck_layer_name"),
                                                 shape=bottleneck_in_shape, dtype=tf.float32)
            log.log("{} shape: {}".format(cf.get("graph_input_bottleneck_layer_name"), bottleneck_in_shape))
        else:
            self._bottleneck_in = None

        # get the fully-connected bottleneck layer (not yet merged with any old bottleneck values).
        if self._use_inception_architecture:
            self._fc1 = self._set_up_architecture_middle_inception()
        else:
            self._fc1 = self._set_up_architecture_middle_custom_net()

        # if requested, stack the "own bottleneck" with the "old bottleneck"
        if self._bottleneck_in_size is not None:
            # TODO which above fc should actually add a "/Relu" suffix here?? this doesn't make sense as the
            # stacked(!) new layer won't be surrounded by another ReLU. do we need this name though?
            name = "{}/Relu".format(cf.get("graph_output_bottleneck_layer_name"))  # the above fc adds a "/Relu" suffix
            self._fc1 = tf.concat([self._fc1, self._bottleneck_in], 1,
                                  name=name)
            log.log(name + " has the following shape: " + str(self._fc1.get_shape().as_list()))

        # dropout
        # if no dropout value is specified in the feed dict, dropout will be deactivated by using 1.0
        dropout_default_deactivated = tf.constant(1.0, dtype=tf.float32, shape=[])
        self.dropout_prob = tf.placeholder_with_default(input=dropout_default_deactivated, shape=[],
                                                        name='dropout_placeholder')
        self._fc1 = tf.nn.dropout(self._fc1, self.dropout_prob)

        # And another fully-connected layer, now with just 2 outputs: one for each label representation
        self._scores, self._W["fc2"], self._b["fc2"] = net_builder.fully_connected(self._fc1, label.n_labels(),
                                                                                   activation=None,
                                                                                   name='fc2')
        # shape: [<batch_size>, 1] the best guess class label per sample
        self._best_guess = tf.argmax(self._scores, 1, output_type=tf.int32)

        # Note, that no softmax was applied yet
        # the training softmax is applied inside of the loss definition separately
        # if needed for prediction after deployment, the following softmax can be used
        self._final_softmax = tf.nn.softmax(self._scores, name=cf.get("graph_final_inference_layer_name"))

        # accuracy etc.
        # (required here to allow further architecture extension in the sub classes)
        log.log("Initializing evaluation methods")
        self._init_evaluation()

    def _set_up_architecture_middle_custom_net(self):
        """Set up all variable architecture elements between the static input and output by using a custom net.

        :return: The fully-connected bottleneck layer (not yet merged with any old bottleneck values).
        """
        log.log("hidden architecture: custom")

        # start network with input X
        H = self._X

        required_input_dims = 4
        if len(self._shape_data_batch) != required_input_dims:
            raise ValueError("Can't create network architecture, because of a wrong input shape (should have {} "
                             "dimensions, but found {}).".format(
                                required_input_dims,
                                len(self._shape_data_batch)
                            ))

        # add convolutional layers along with ReLUs and max-pooling
        for layer_i, n_filters_i in enumerate(cf.get("conv_filter_sizes")):
            layer_name = "conv_layer_" + str(layer_i)

            # create single convolutional layer
            H, W = net_builder.conv2d(
                H, n_filters_i, k_h=cf.get("conv_filter_size"), k_w=cf.get("conv_filter_size"),
                stride_vertical=cf.get("conv_stride"), stride_horizontal=cf.get("conv_stride"),
                name=layer_name)

            # add ReLU
            H = tf.nn.relu(H)

            # just to check what's happening:
            log.log("{} (incl. activation) with {} filter maps of the size {}x{} with stride {} has the "
                    "following shape: {}".format(
                layer_name,
                n_filters_i,
                cf.get("conv_filter_size"),
                cf.get("conv_filter_size"),
                cf.get("conv_stride"),
                H.get_shape().as_list()
            ))

            # activation and/or pooling? In general, both is used at once.
            pool_name = "pool_layer_{}".format(layer_i)
            with tf.name_scope(pool_name):
                H = net_builder.max_pool(H, cf.get("pooling_size"), cf.get("pooling_stride"))
                log.log("{} with size {}x{} and stride {} has the following shape: {}".format(
                    pool_name,
                    cf.get("pooling_size"),
                    cf.get("pooling_size"),
                    cf.get("pooling_stride"),
                    str(H.get_shape().as_list())
                ))

        # Connect the last convolutional layer to a fully connected layer (relu inclusive)
        # (the name depends on the fact whether this is already the "final version of this tensor")
        if self._bottleneck_in_size is None:
            fc1_name = cf.get("graph_output_bottleneck_layer_name")
        else:
            fc1_name = "{}_part1".format(cf.get("graph_output_bottleneck_layer_name"))

        H, self._W["fc1"], self._b["fc1"] = net_builder.fully_connected(H,
                                                                        cf.get("fc1_size"),
                                                                        activation=tf.nn.relu,
                                                                        name=fc1_name)

        log.log(fc1_name + " (incl. activation) has the following shape: " + str(H.get_shape().as_list()))

        return H

    def _set_up_architecture_middle_inception(self):
        """Set up all variable architecture elements between the static input and output by using a the inception net.

        :return: The fully-connected bottleneck layer (not yet merged with any old bottleneck values).
        """
        log.log("hidden architecture: inception")
        return inception_builder.build(self._X)

    def start_tf_session_and_initialize_vars(self):
        """Start the TensorFlow session and initialize all weights (may use an existing snapshot)."""
        # create saver to be able saving current parameters at certain checkpoints
        # must be called after(!) initializing the optimizer
        self._saver = tf.train.Saver()

        # start a session
        log.log("Start TensorFlow session")
        self._session = tf.Session()

        if self._load_given_snapshot:
            log.log("Restoring given TensorFlow snapshot")
            self._saver.restore(self._session, self._snapshot_full_path)
        else:
            # initialize all vars
            log.log("Initializing TensorFlow with a new set of variables")
            init = tf.global_variables_initializer()
            self._session.run(init)

    def close_session(self):
        """Release any resources used by TensorFlow.

        This net can't be used anymore after this point.

        :return:
        """
        self._session.close()
        tf.reset_default_graph()
        log.log("session closed")

    ######################################## EVALUATION ##############################

    def _run_full_dataset_in_batches(self, eval_tensor, dataset_split: DatasetSplit):
        """Run eval_tensor on the complete dataset_split.

        To prevent memory overflow, dataset_split will be evaluated in batches of size cf.get("max_batch_size").
        if cf.get("max_batch_size") = None => do all at once
        The results of each batch are summed up and divided by the number of batches in the end (mean value).

        If eval_tensor is a dictionary with tensors rather than a single value one only, the returned value will be a dictionary
        with one element per input tensor and the same keys as in the input tensor dictionary.

        Note that, this method does not actually guarantee that each sample of the given dataset split will be used
        exactly once. If the given split stores a probability distribution, that one will be used. This should be the
        expected behavior in most use cases though.
        """
        i = 0
        accumulator = {}
        ds_iterator = dataset_split.new_default_iterator(cf.get("max_batch_size"))
        while ds_iterator.in_first_epoch:
            batch_i = ds_iterator.next_batch

            images_preprocessed_i = self._prepare_input(batch_i.images)

            feed_dict = {
                self._X: images_preprocessed_i,
                self._Y: batch_i.labels,
                self.dropout_prob: 1.0  # deactivate dropout for testing
            }

            if self._bottleneck_in_size is not None:
                feed_dict[self._bottleneck_in] = batch_i.bottlenecks

            temp = self._session.run(eval_tensor, feed_dict=feed_dict)

            # don't init accumulator outside of this loop, as the type is unknown=
            for key, value in temp.items():
                if i == 0:
                    accumulator[key] = value
                else:
                    accumulator[key] += value

            i += 1

        if i > 0:
            for key, value in accumulator.items():
                # some values need to be divided by the number of batches. the other ones will be kept as sums
                if criteria.get(key).acc_mean:
                    accumulator[key] = value / float(i)

        final_value = accumulator

        return final_value

    def _prepare_input(self, x):
        """Prepare x to be a valid input for the network.

        Basically, this converts x (from uint8) to float32 and applies preprocessing. A modified copy of x will be
        returned.
        This method needs to be called each and every time the network is evaluated.
        """
        x = np.array(x, dtype=np.float32, copy=True)  # don't use asarray here. we explicitly need to copy x, otherwise we would apply preprocessing multiple times to the same sample
        self._preprocessor.preprocess_data(x)  # changes x inplace
        return x

    def accuracy(self, dataset_split):
        """Get the accuracy for running the current network on the complete dataset_split."""
        return self._run_full_dataset_in_batches(
            {"accuracy": self.calc_accuracy},
            dataset_split)["accuracy"]

    def _init_evaluation(self):
        """Initialize accuracy and further evaluation methods."""

        # create a list with <batch_size> elements. each element is either 1 (prediction correct) or 0 (false)
        pred_correctness = tf.nn.in_top_k(self._scores, self._Y, 1)

        # using the f1 score as the loss function may result in learning a constant function that always returns the
        # same probability for all classes, because this makes "all classes being the best guess at once"
        # => prevent that all probabilities are equal
        different_probs = tf.not_equal(self._scores[:, 0], self._scores[:, 1])
        pred_correctness = tf.logical_and(pred_correctness, different_probs)

        # based on a list of "rights or wrongs", calculate the accuracy
        self.calc_accuracy = tf.reduce_mean(tf.cast(pred_correctness, tf.float32))

        # the following code is based on code from:
        # http: // stackoverflow.com / a / 38349755 / 1665966

        # Step 1:
        # Let's create 2 vectors that will contain boolean values, and will describe our labels

        is_label_one = tf.cast(self._Y, dtype=tf.bool)
        is_label_zero = tf.logical_not(is_label_one)
        # Imagine that labels = [0,1]
        # Then
        # is_label_one = [False,True]
        # is_label_zero = [True,False]

        # Step 2:
        # get the prediction and false prediction vectors. correct_prediction is something that you choose within your model.
        false_prediction = tf.logical_not(pred_correctness)

        # Step 3:
        # get the 4 metrics by comparing boolean vectors
        # TRUE POSITIVES
        self._true_positives = tf.reduce_sum(tf.to_int32(tf.logical_and(pred_correctness, is_label_one)))

        # FALSE POSITIVES
        self._false_positives = tf.reduce_sum(tf.to_int32(tf.logical_and(false_prediction, is_label_zero)))

        # TRUE NEGATIVES
        self._true_negatives = tf.reduce_sum(tf.to_int32(tf.logical_and(pred_correctness, is_label_zero)))

        # FALSE NEGATIVES
        self._false_negatives = tf.reduce_sum(tf.to_int32(tf.logical_and(false_prediction, is_label_one)))

        # PRECISION
        self._precision = tf.truediv(self._true_positives, self._true_positives + self._false_positives)

        # RECALL
        self._recall = tf.truediv(self._true_positives, self._true_positives + self._false_negatives)

        # or F1 score:
        self._f1_score = tf.truediv(2 * (self._precision * self._recall), self._precision + self._recall)

        # the more general F score, based on the given beta param
        if self._f_beta is not None:
            self._f_beta_sq = self._f_beta * self._f_beta  # calc this only once
            self._f_beta_score = (1 + self._f_beta_sq) * \
                                 tf.truediv(self._precision * self._recall,
                                            (self._f_beta_sq * self._precision) + self._recall)
            self._f_beta_score = tf.cast(self._f_beta_score, tf.float32)  # cast from 64 to 32

            ##### the implementations above are exact, but can't be used as a custom loss function, because their is no
            ##### way to determine the gradients of functions such as tf.cast() or tf.argmax()
            ##### => so we use a different version for the loss function

            # probabilities for each sample to belong to the (positive) foreground class
            # shape: [<batch_size>, 1]
            self._probs_foreground = self._final_softmax[:, 1]

            # see self._probs_foreground
            self._probs_background = self._final_softmax[:, 0]

            self._true_positives_diffable = tf.reduce_sum(self._probs_foreground * self._Y_float)
            self._false_positives_diffable = tf.reduce_sum(self._probs_foreground * (1 - self._Y_float))
            self._false_negatives_diffable = tf.reduce_sum(self._probs_background * self._Y_float)

            pd_divide_by = self._true_positives_diffable + self._false_positives_diffable
            self._precision_diffable = tf.cond(pd_divide_by > 0,
                                               lambda: tf.truediv(self._true_positives_diffable, pd_divide_by),
                                               lambda: 0.0)

            rd_divide_by = self._true_positives_diffable + self._false_negatives_diffable
            self._recall_diffable = tf.cond(rd_divide_by > 0,
                                               lambda: tf.truediv(self._true_positives_diffable, rd_divide_by),
                                               lambda: 0.0)

            fd_divide_by = (self._f_beta_sq * self._precision_diffable) + self._recall_diffable
            self._f_beta_score_diffable = tf.cond(fd_divide_by > 0,
                                               lambda: (1 + self._f_beta_sq) * tf.truediv(self._precision_diffable * self._recall_diffable, fd_divide_by),
                                               lambda: 0.0)


    def _full_evaluation(self, dataset_split, log_line=None):
        """Evaluate the current net on dataset_split by calculating all relevant measurements.

        :param dataset_split:
        :param log_line: if not None, all results will be logged. If log_line is furthermore a string, this will be the first line of the log.
        :return:
        """

        # combine all tensor values to single list, so it can be summed up etc. in other places
        tensors = dict()

        # accuracy is always required
        tensors["accuracy"] = self.calc_accuracy

        # the following measures do only make sense if we are creating a binary classifier
        if label.n_labels() == 2:
            tensors["true_positives"] = self._true_positives
            tensors["false_positives"] = self._false_positives
            tensors["true_negatives"] = self._true_negatives
            tensors["false_negatives"] = self._false_negatives
            # tensors["precision"] = self._precision
            # tensors["recall"] = self._recall
            # tensors["f1_score"] = self._f1_score
            if self._f_beta is not None:
                tensors["true_positives_diffable"] = self._true_positives_diffable
                tensors["false_positives_diffable"] = self._false_positives_diffable
                tensors["false_negatives_diffable"] = self._false_negatives_diffable
                tensors["precision_diffable"] = self._precision_diffable
                tensors["recall_diffable"] = self._recall_diffable
                tensors["{}_diffable".format(self._f_beta_key)] = self._f_beta_score_diffable

        results = self._run_full_dataset_in_batches(tensors, dataset_split)

        # all values could be calculated by the tensors above, but instead we will calculate some only once
        # (instead of running the tensor again for each batch and merge the results)
        # this is much more numerically-stable. (and probably a bit faster, too)
        results = self.process_results(results, log_line)

        return results

    def process_results(self, results, log_line=None):
        """Process the given results and extend them by even more criterias.
        Some of the added criterias may existed before, but will be replaced by numerically-stable values.

        :param results:
        :param log_line: if not None, all results will be logged. If log_line is furthermore a string, this will be the first line of the log.
        :return:
        """

        if label.n_labels() == 2:
            # validate params
            if not ("true_positives" in results and "true_negatives" in results and "false_negatives" in results
                    and "false_positives" in results):
                raise ValueError("Missing result values.")

            # accuracy
            n_total_samples = results["true_positives"] + results["true_negatives"] + results["false_negatives"] + results["false_positives"]
            results["accuracy"] = float(
                results["true_positives"] + results["true_negatives"]) / n_total_samples

            # precision
            results["precision"] = float(results["true_positives"])
            divide_by = results["true_positives"] + results["false_positives"]
            if divide_by > 0:
                results["precision"] /= float(divide_by)

            # recall / true positive rate
            results["recall"] = float(results["true_positives"])
            divide_by = float(results["true_positives"] + results["false_negatives"])
            if divide_by > 0:
                results["recall"] /= float(divide_by)

            # specificity / true negative rate
            results["true_negative_rate"] = float(results["true_negatives"])
            divide_by = float(results["true_negatives"] + results["false_positives"])
            if divide_by > 0:
                results["true_negative_rate"] /= float(divide_by)

            # f1 score
            results["f1_score"] = float(2 * results["precision"] * results["recall"])
            divide_by = float(results["precision"] + results["recall"])
            if divide_by > 0:
                results["f1_score"] /= float(divide_by)

            # the more general F score, based on the given beta param
            if self._f_beta is not None:
                results[self._f_beta_key] = float((1 + self._f_beta_sq) * results["precision"] * results["recall"])
                divide_by = float((self._f_beta_sq * results["precision"]) + results["recall"])
                if divide_by > 0:
                    results[self._f_beta_key] /= float(divide_by)

            # total number of positive samples include in the given set
            results["samples_positive"] = results["true_positives"] + results["false_negatives"]

            # total number of negative samples include in the given set
            results["samples_negative"] = results["true_negatives"] + results["false_positives"]

        # create a sorted dictionary
        results = collections.OrderedDict(sorted(results.items()))

        # log
        if log_line is not None:
            self.log_results(results, log_line)

        return results

    @staticmethod
    def log_results(results, first_line="results:"):
        """Log all values given in results."""
        log.log(first_line)
        for key, value in results.items():
            # format and log the value
            value_format = criteria.get(key).format(value)
            log.log("    - {}: {}".format(key, value_format))

    @property
    def output_graph_def(self):
        """Return a static version of the current graph which can be used to persist it."""
        session = self._session
        graph = session.graph
        return tf.graph_util.convert_variables_to_constants(session, graph.as_graph_def(),
                                                            [cf.get("graph_final_inference_layer_name")])
    @property
    def bottleneck_out_size(self) -> int:
        """Get the size of the fc1 layer (ignoring the batch dimension)."""
        return int(self._fc1.shape[1])

    def predict(self, ds_split: DatasetSplit, ds_split_is_already_preprocessed=False, log_line=None,
                update_bottlenecks=False, return_probabilities=False):
        """Predict the classes of all windows in ds_split.

        :param ds_split: The split which should be evaluated.
        :param ds_split_is_already_preprocessed: Whether we still need to preprocess the input data.
        :param log_line: If this is not None, ann additional evaluation will be done and logged.
        :param update_bottlenecks: Whether the bottlenecks of this net should be saved in the given ds_split.
        :param return_probabilities:
        :return:
        """

        # optional evaluation
        # (must be done before swapping the bottlenecks)
        # TODO this should not need an extra run (all samples are evaluated twice)
        if log_line is not None:
            self._full_evaluation(ds_split, log_line)

        # Optionally, initialize a common cache for all new bottlenecks
        if update_bottlenecks:
            if self._nr == self._nr_max:
                # we don't need anymore bottlenecks
                log.log("Not caching the new bottlenecks, because the last net of a cascade has been reached.")
                update_bottlenecks = False
            else:
                # create a new empty numpy array that can be used as a cache for all new bottlenecks of this split
                # TODO using np.empty instead of np.zeros is a bit faster, but provides a higher risk of failures, too
                new_bottlenecks_cache = np.zeros(shape=[ds_split.n_samples, self.bottleneck_out_size], dtype=np.float32)

        # we do not need to save all probs, if we will not return them
        if return_probabilities:
            probabilities_all = np.empty([ds_split.n_samples, label.n_labels()], dtype=np.float32)
        else:
            probabilities_all = None

        # run actual prediction
        label_predictions = np.empty([ds_split.n_samples], dtype=np.int)
        # do not shuffle! otherwise the returned results can not be compared to the original ground truth input
        # that's why we can't use ds_split.new_default_iterator either, because it may provide a randomized iterator
        ds_iterator = DeterministicIterator(ds_split, cf.get("max_batch_size"), shuffle_every_epoch=False)
        while ds_iterator.in_first_epoch:
            batch_i = ds_iterator.next_batch

            # keep track of the index range which has been used by this batch
            # => we need this for everything we want to save about the current prediction
            # => this requires a DeterministicIterator and shuffle_every_epoch=False
            batch_begin = (ds_iterator.n_provided_batches - 1) * ds_iterator.batch_size
            batch_end = batch_begin + batch_i.n_samples  # batch_i.n_samples may be less than ds_iterator.batch_size (!)

            if not ds_split_is_already_preprocessed:
                preprocessed_images_i = self._prepare_input(batch_i.images)
            else:
                preprocessed_images_i = batch_i.images

            feed_dict = {
                self._X: preprocessed_images_i,
                self.dropout_prob: 1.0  # deactivate dropout for prediction
            }

            if self._bottleneck_in_size is not None:
                feed_dict[self._bottleneck_in] = batch_i.bottlenecks

            if update_bottlenecks:
                probabilities_batch, new_bottlenecks_cache[batch_begin:batch_end] = \
                    self._session.run([self._scores, self._fc1], feed_dict=feed_dict)

            else:
                probabilities_batch = self._session.run(self._scores, feed_dict=feed_dict)

            # maybe keep the probabilities of the current batch
            if return_probabilities:
                probabilities_all[batch_begin:batch_end] = probabilities_batch

            # save best guesses of the current batch
            label_predictions[batch_begin:batch_end] = np.argmax(probabilities_batch, 1)

        # if requested, we can now replace all bottlenecks of the current split by the new cached ones
        if update_bottlenecks:
            ds_split.set_bottlenecks(new_bottlenecks_cache)

        return label_predictions, probabilities_all
