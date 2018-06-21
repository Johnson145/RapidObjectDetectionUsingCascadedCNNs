"""
This module provides methods to create (parts of) the Inception net.

Code is based on the Inception retrainer:
https://github.com/tensorflow/hub/blob/master/examples/image_retraining/retrain.py
"""
import os

import sys
import tarfile
import urllib

import config as cf
import tensorflow as tf
from utils import log

# some constants specific to the inception net:
DATA_URL = 'http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz'
# pylint: enable=line-too-long
BOTTLENECK_TENSOR_SIZE = 2048
MODEL_INPUT_WIDTH = 299
MODEL_INPUT_HEIGHT = 299
MODEL_INPUT_DEPTH = 3
RESIZED_INPUT_TENSOR_NAME = 'ResizeBilinear:0'
MAX_NUM_IMAGES_PER_CLASS = 2 ** 27 - 1  # ~134M


def build(new_resized_input_tensor):
    """This is the main public interface to this module.

    Create all required inception net parts, connect them with the given input and return the bottleneck layer.
    :return:
    """
    # ensure that the required model base exists
    _maybe_download_and_extract()
    return _create_inception_graph(new_resized_input_tensor)


def _maybe_download_and_extract():
    """Download and extract model tar file.

    If the pretrained model we're using doesn't already exist, this function
    downloads it from the TensorFlow.org website and unpacks it into a directory.
    """
    dest_directory = cf.get("inception_model_base")
    if not os.path.exists(dest_directory):
        os.makedirs(dest_directory)
    filename = DATA_URL.split('/')[-1]
    filepath = os.path.join(dest_directory, filename)
    if not os.path.exists(filepath):
        def _progress(count, block_size, total_size):
            sys.stdout.write('\r>> Downloading %s %.1f%%' %
                             (filename,
                              float(count * block_size) / float(total_size) * 100.0))
            sys.stdout.flush()

        filepath, _ = urllib.request.urlretrieve(DATA_URL,
                                                 filepath,
                                                 _progress)
        statinfo = os.stat(filepath)
        log.log('Successfully downloaded {} {}bytes.'.format(
            filename,
            statinfo.st_size
        ))
    tarfile.open(filepath, 'r:gz').extractall(dest_directory)


def _create_inception_graph(new_resized_input_tensor):
    """"Creates a graph from saved GraphDef file and returns a Graph object.

    Returns:
      Graph holding the trained Inception network, and various tensors we'll be
      manipulating.
    """
    with tf.Session() as sess:
        model_filename = os.path.join(cf.get("inception_model_base"), 'classify_image_graph_def.pb')
        with tf.gfile.FastGFile(model_filename, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            bottleneck_tensor_unshaped = (
                tf.import_graph_def(
                    graph_def,
                    name='',
                    input_map={RESIZED_INPUT_TENSOR_NAME: new_resized_input_tensor},
                    return_elements=[
                        cf.get("inception_bottleneck_tensor_name")
                    ]))

            # the original inception retrainer script extracts the "pool_3/_reshape:0" tensor as the bottleneck_tensor.
            # however, that tensor does work with a single image only, as it has a shape of [1, 2048]
            # we use the tensor just before that: "pool_3:0", which has a shape of [None, 1, 1, 2048].
            # then we append a new tensor to it that does the reshaping while keeping support for multiple images
            # final shape of the bottleneck_tensor should be [None, 2048]
            bottleneck_tensor = tf.reshape(bottleneck_tensor_unshaped, [-1, BOTTLENECK_TENSOR_SIZE])

    return bottleneck_tensor
