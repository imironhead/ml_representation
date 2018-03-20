"""
"""
import tensorflow as tf


def build_encoder(tensors, initializer):
    """
    figure 4
    """
    with tf.variable_scope('encoder', reuse=tf.AUTO_REUSE):
        # NOTE: 3 successive conv - relu - pool
        params = [(64, 10), (128, 7), (128, 4)]

        for filters, kernel_size in params:
            tensors = tf.layers.conv2d(
                tensors,
                filters=filters,
                kernel_size=kernel_size,
                padding='VALID',
                activation=tf.nn.relu,
                use_bias=True,
                kernel_initializer=initializer)

            tensors = tf.nn.max_pool(
                tensors,
                ksize=[1, 2, 2, 1],
                strides=[1, 2, 2, 1],
                padding='VALID')

        tensors = tf.layers.conv2d(
            tensors,
            filters=256,
            kernel_size=4,
            padding='VALID',
            activation=tf.nn.relu,
            use_bias=True,
            kernel_initializer=initializer)

        # NOTE: flatten for fully connection
        tensors = tf.layers.flatten(tensors)

        # NOTE: fully connect to make features
        tensors = tf.layers.dense(
            tensors,
            units=4096,
            activation=tf.nn.sigmoid,
            use_bias=True,
            kernel_initializer=initializer)

    return tensors


def build_siamese_network():
    """
    Siamese Neural Networks for One-shot Image Recognition

    figure 4
    """
    initializer = tf.truncated_normal_initializer(stddev=0.02)

    # NOTE: sources of siamese networks
    images_a = tf.placeholder(
        shape=[None, 105, 105, 1], dtype=tf.float32, name='images_a')

    images_b = tf.placeholder(
        shape=[None, 105, 105, 1], dtype=tf.float32, name='images_b')

    # NOTE: labels to mark same character pairs
    labels = tf.placeholder(
        shape=[None, 1], dtype=tf.float32, name='labels')

    # NOTE: encode images with the same weights
    features_a = build_encoder(images_a, initializer)
    features_b = build_encoder(images_b, initializer)

    # NOTE: siamese twin joins here, L1
    joins = tf.abs(features_a - features_b)

    # NOTE: the alpha_j are additional parameters that are learned by the model
    #       during training, weighting the importance of the component-wise
    #       distance
    logits = tf.layers.dense(
        joins,
        units=1,
        activation=None,
        use_bias=False,
        kernel_initializer=initializer,
        name='outputs')

    # NOTE: sigmoidal activation function
    predictions = tf.nn.sigmoid(logits)

    # NOTE: cross entropy without regularizer
    loss = tf.nn.sigmoid_cross_entropy_with_logits(
        labels=labels, logits=logits)
    loss = tf.reduce_mean(loss)

    step = tf.get_variable(
        'training_step',
        [],
        trainable=False,
        initializer=tf.constant_initializer(0, dtype=tf.int64),
        dtype=tf.int64)

    trainer = tf.train \
        .AdamOptimizer(learning_rate=0.0001) \
        .minimize(loss, global_step=step)

    return {
        'images_a': images_a,
        'images_b': images_b,
        'labels': labels,
        'loss': loss,
        'predictions': predictions,
        'trainer': trainer,
        'step': step,
    }
