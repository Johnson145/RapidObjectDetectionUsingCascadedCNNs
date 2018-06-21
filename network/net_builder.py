"""This module provides some helper methods to create parts of an artificial neural network."""

import tensorflow as tf


def max_pool(x, size=2, stride=2):
    """Create a max-pooling layer.

    :param x: The input of the pooling layer.
    :param size: The width and height of the kernel.
    :param stride: The stride between consecutive kernels.
    :return:
    """
    return tf.nn.max_pool(x,
                          ksize=[1, size, size, 1],
                          strides=[1, stride, stride, 1],
                          padding="SAME")


def conv2d(x, n_output, k_h=5, k_w=5, stride_vertical=2, stride_horizontal=2, padding="SAME", name="conv2d"):
    """Create a 2D convolutional layer.

    :param x: The input of the convolutional layer.
    :param n_output: The number of filters / feature maps.
    :param k_h: The kernel height (height of the input neighborhood).
    :param k_w: The kernel width (width of the input neighborhood).
    :param stride_vertical: Vertical stride between consecutive kernel duplicates.
    :param stride_horizontal: Horizontal stride between consecutive kernel duplicates.
    :param padding: Padding method.
    :param name: The name of the convolutional layer.
    :return:
    """
    with tf.variable_scope(name, reuse=None):
        # create new weight matrix
        W = tf.get_variable(
            name='W',
            shape=[k_h, k_w, x.get_shape()[-1], n_output],
            initializer=tf.contrib.layers.xavier_initializer_conv2d())

        # create the actual convolutional layer using the given weights
        conv = tf.nn.conv2d(
            name='conv',
            input=x,
            filter=W,
            strides=[1, stride_vertical, stride_horizontal, 1],
            padding=padding)

        # create bias
        b = tf.get_variable(
            name='b',
            shape=[n_output],
            initializer=tf.constant_initializer(0.0))

        # add bias to the convolutional layer
        h = tf.nn.bias_add(
            name='h',
            value=conv,
            bias=b)

    return h, W


def fully_connected(x, n_output, name="fc", activation=None):
    """Create a fully-connected layer.

    :param x: The input of the fully-connected layer.
    :param n_output: The number of neurons used in the fully-connected layer.
    :param name: The name of the fully-connected layer.
    :param activation: An optional activation method which may be applied right after the fully-connected layer.
    :return:
    """
    if n_output < 1:
        raise ValueError("Can not create a fully-connected layer with {} neurons.".format(n_output))

    if len(x.get_shape()) != 2:
        x = flatten(x, reuse=None)

    with tf.variable_scope(name, reuse=None):
        # create weight matrix
        n_input = x.get_shape().as_list()[1]
        W = tf.get_variable(
            name='W',
            shape=[n_input, n_output],
            dtype=tf.float32,
            initializer=tf.contrib.layers.xavier_initializer())

        # create bias
        b = tf.get_variable(
            name='b',
            shape=[n_output],
            dtype=tf.float32,
            initializer=tf.constant_initializer(0.0))

        # create the actual fully-connected layer using the weights and bias
        h = tf.nn.bias_add(
            name='h',
            value=tf.matmul(x, W),
            bias=b)

        # maybe append the activation function
        if activation:
            h = activation(h)

        return h, W, b


def flatten(x):
    """Flatten the tensor x to two dimensions."""
    with tf.variable_scope("flatten"):
        dims = x.get_shape().as_list()
        if len(dims) == 4:
            flattened = tf.reshape(
                x,
                shape=[-1, dims[1] * dims[2] * dims[3]])
        elif len(dims) == 3:
            flattened = tf.reshape(
                x,
                shape=[-1, dims[1] * dims[2]])
        elif len(dims) == 2 or len(dims) == 1:
            flattened = x
        else:
            raise ValueError("Expected n dimensions of 1, 2 or 4, but found: {}".format(len(dims)))
        return flattened
