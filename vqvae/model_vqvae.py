"""
build model base on the paper:
    arXiv:1711.00937v1
    Neural Discrete Representation Learning
"""
import tensorflow as tf


def quantize_vectors(tensors, embedding_space):
    """
    """
    # NOTE: book keeping the shape of input shape for the output tensors
    #       they must be in the same shape because the input is just quantized
    shape_in = tf.shape(tensors)

    shape_h, shape_w, shape_c = shape_in[1], shape_in[2], shape_in[3]

    # NOTE: flatten to h * w vectors for quantization
    # NOTE: shape_c is also the D on the paper
    tensors = tf.reshape(tensors, [-1, shape_h * shape_w, 1, shape_c])

    # NOTE: embedding_space is k * c
    shape_embedding = tf.shape(embedding_space)

    k = shape_embedding[0]

    # NOTE: tile each vector k times, then we can get l1 distance between one
    #       feature vector and each quantized candidates
    tensors = tf.tile(tensors, [1, 1, k, 1])

    # NOTE: l2 distance
    distances = tf.square(tensors - embedding_space)

    distances = tf.reduce_sum(distances, axis=3)

    # NOTE: find indices of nearest embeddings
    nearest_indices = tf.argmin(distances, axis=2)

    # NOTE: do quantization
    quantized = tf.gather(embedding_space, nearest_indices, axis=0)

    # NOTE: reshape back to the original shape
    tensors = tf.reshape(quantized, shape_in)

    return tensors


def resnet(tensors, num_filters):
    """
    """
    initializer = tf.truncated_normal_initializer(stddev=0.02)

    x_tensors = tf.layers.conv2d(
        tensors,
        filters=num_filters,
        kernel_size=3,
        strides=1,
        padding='same',
        activation=tf.nn.leaky_relu,
        kernel_initializer=initializer)

    x_tensors = tf.layers.conv2d(
        x_tensors,
        filters=num_filters,
        kernel_size=1,
        strides=1,
        padding='same',
        activation=None,
        kernel_initializer=initializer)

    tensors = tensors + x_tensors

    tensors = tf.nn.leaky_relu(tensors)

    return tensors


def build_encoder(tensors, use_resnet, embedding_d):
    """
    """
    initializer = tf.truncated_normal_initializer(stddev=0.02)

    for layer in range(3):
        filters = embedding_d // (2 ** (2 - layer))

        tensors = tf.layers.conv2d(
            tensors,
            filters=filters,
            kernel_size=3,
            strides=1 if layer == 0 else 2,
            padding='same',
            activation=None,
            kernel_initializer=initializer)

        tensors = tf.contrib.layers.instance_norm(tensors)

        tensors = tf.nn.leaky_relu(tensors)

        if not use_resnet:
            tensors = tf.layers.conv2d(
                tensors,
                filters=filters,
                kernel_size=3,
                strides=1,
                padding='same',
                activation=None,
                kernel_initializer=initializer)

            tensors = tf.contrib.layers.instance_norm(tensors)

            tensors = tf.nn.leaky_relu(tensors)

    if use_resnet:
        for _ in range(2):
            tensors = resnet(tensors, embedding_d)

    return tensors


def build_decoder(tensors, use_resnet, embedding_d, num_channels):
    """
    """
    initializer = tf.truncated_normal_initializer(stddev=0.02)

    if use_resnet:
        for _ in range(2):
            tensors = resnet(tensors, embedding_d)

    for layer in range(3):
        filters = embedding_d // (2 ** layer)

        tensors = tf.layers.conv2d_transpose(
            tensors,
            filters=filters,
            kernel_size=3,
            strides=1 if layer == 0 else 2,
            padding='same',
            activation=tf.nn.relu,
            kernel_initializer=initializer)

        if not use_resnet:
            tensors = tf.layers.conv2d(
                tensors,
                filters=filters,
                kernel_size=3,
                strides=1,
                padding='same',
                activation=tf.nn.relu,
                kernel_initializer=initializer)

    # NOTE: to match the number of channels to the target tensors
    tensors = tf.layers.conv2d(
        tensors,
        filters=num_channels,
        kernel_size=1,
        strides=1,
        padding='same',
        activation=tf.nn.tanh,
        kernel_initializer=initializer)

    return tensors


def build_model(
        use_resnet, source_images, num_channels, embedding_k, embedding_d):
    """
    source_images: placeholder or tensors
    """
    # NOTE: embedding space variables
    embedding_space = tf.get_variable(
        'embedding_space',
        [embedding_k, embedding_d],
        trainable=True,
        initializer=tf.truncated_normal_initializer(stddev=0.2),
        dtype=tf.float32)

    # NOTE:
    tensors_ze = build_encoder(source_images, use_resnet, embedding_d)

    # NOTE: vector quantisation
    tensors_zq = quantize_vectors(tensors_ze, embedding_space)

    # NOTE: arXiv:1711.00937v1
    #       equation 3
    #       embedding space vectors will not be optimized throough tensors_zq
    # NOTE: subtract ze before stop gradient, then add it back so that:
    #       1. the values of input tensors of decoder are still zq
    #       2. the gradients of tensors_ze are completely contribed by
    #          tensors_zq
    tensors = tf.stop_gradient(tensors_zq - tensors_ze) + tensors_ze

    # NOTE:
    result_images = build_decoder(
        tensors, use_resnet, embedding_d, num_channels)
    result_images = tf.identity(result_images, name='result_images')

    # NOTE: reconstruction loss
    loss_reconstruction = \
        tf.losses.mean_squared_error(source_images, result_images)

    # NOTE: vector quantisation loss
    loss_quantization = tf.losses.mean_squared_error(
        tf.stop_gradient(tensors_ze),
        tensors_zq)

    # NOTE: commitment loss
    loss_commitment = tf.losses.mean_squared_error(
        tf.stop_gradient(tensors_zq),
        tensors_ze)

    # NOTE: arXiv:1711.00937v1
    #       we found the resulting algorithm to be quite robust to beta, as the
    #       results did not vary for values of beta ranging from 0.1 to 2.0.
    #       we use beta = 0.25 in all our experiments.
    loss = loss_reconstruction + loss_quantization + 0.25 * loss_commitment

    # NOTE: trainer
    step = tf.get_variable(
        'step',
        [],
        trainable=False,
        initializer=tf.constant_initializer(0, dtype=tf.int64),
        dtype=tf.int64)

    learning_rate = tf.get_variable(
        'learning_rate',
        [],
        trainable=False,
        initializer=tf.constant_initializer(1e-5, dtype=tf.float32),
        dtype=tf.float32)

    trainer = tf.train \
        .AdamOptimizer(learning_rate=learning_rate) \
        .minimize(loss, global_step=step)

    return {
        'source_images': source_images,
        'result_images': result_images,
        'step': step,
        'loss': loss,
        'trainer': trainer,
        'learning_rate': learning_rate,
    }
