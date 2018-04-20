"""
"""
import numpy as np
import os
import tensorflow as tf

import datasets
import model_vqvae


def build_model(
        use_resnet,
        num_channels,
        embedding_k,
        embedding_d,
        batches_output_types,
        batches_output_shapes):
    """
    """
    # NOTE: a string handle as training/validation set switch
    dataset_handle = tf.placeholder(tf.string, shape=[])

    iterator = tf.data.Iterator.from_string_handle(
        dataset_handle, batches_output_types, batches_output_shapes)

    # NOTE: create an op to iterate the datasets
    next_batch = iterator.get_next()

    # NOTE: build the vqvae model with next_batch from tf.data.Dataset as input
    model = model_vqvae.build_model(
        use_resnet, next_batch, num_channels, embedding_k, embedding_d)

    # NOTE: we need dataset_handle to switch training and validation
    #       let's say it's part of the model
    model['dataset_handle'] = dataset_handle

    return model


def build_summaries(model):
    """
    """
    # NOTE: the shape of both source_images and result_images are (N,H,W,C)
    source_images = model['source_images']
    result_images = model['result_images']

    batch_shape = tf.shape(source_images)

    n, h, w, c = batch_shape[0], batch_shape[1], batch_shape[2], batch_shape[3]

    cmp_images = tf.concat([source_images, result_images], axis=2)
    cmp_images = tf.reshape(cmp_images, [1, n * h, 2 * w, c])
    cmp_images = 127.5 * cmp_images + 127.5
    cmp_images = tf.saturate_cast(cmp_images, tf.uint8)

    # NOTE: summary of training images
    summary_train_images = tf.summary.image(
        'train_images', cmp_images, max_outputs=1)

    # NOTE: summary of validation images
    summary_valid_images = tf.summary.image(
        'valid_images', cmp_images, max_outputs=1)

    summary_loss = tf.summary.scalar('loss', model['loss'])

    return {
        'loss': summary_loss,
        'train_images': summary_train_images,
        'valid_images': summary_valid_images,
        }


def train(session, model, train_handle, valid_handle, ckpt_path, log_path):
    """
    """
    source_ckpt_path = tf.train.latest_checkpoint(ckpt_path)
    target_ckpt_path = os.path.join(ckpt_path, 'model.ckpt')

    summaries = build_summaries(model)

    reporter = tf.summary.FileWriter(log_path)

    saver = tf.train.Saver()

    if source_ckpt_path is None:
        session.run(tf.global_variables_initializer())
    else:
        tf.train.Saver().restore(session, source_ckpt_path)

    # NOTE: initial or saved (from check point) step
    step = session.run(model['step'])

    # NOTE: exclude log which does not happend yet :)
    reporter.add_session_log(
        tf.SessionLog(status=tf.SessionLog.START), global_step=step)

    while True:
        feeds = {
            model['dataset_handle']: train_handle,
            model['learning_rate']: 5e-6
        }

        fetch = {
            'trainer': model['trainer'],
            'loss': model['loss'],
            'step': model['step'],
        }

        if step % 1000 == 0:
            fetch['summary_loss'] = summaries['loss']
            fetch['summary_images'] = summaries['train_images']

        fetched = session.run(fetch, feed_dict=feeds)

        step = fetched['step']

        if step % 100 == 0:
            print('loss[{}]: {}'.format(step, fetched['loss']))

        if 'summary_loss' in fetched:
            reporter.add_summary(fetched['summary_loss'], step)
        if 'summary_images' in fetched:
            reporter.add_summary(fetched['summary_images'], step)

        if step % 100000 == 0:
            feeds = {
                model['dataset_handle']: valid_handle,
                model['learning_rate']: 1e-4
            }

            summary_valid_images = \
                session.run(summaries['valid_images'], feed_dict=feeds)

            reporter.add_summary(summary_valid_images, step)

        if step % 100000 == 0:
            saver.save(session, target_ckpt_path, global_step=step)


def main(_):
    """
    """
    FLAGS = tf.app.flags.FLAGS

    if FLAGS.dataset == 'mnist':
        batch_size = 128
        embedding_k = 64
        embedding_d = 16
        num_channels = 1

        build_batch_iterator = datasets.build_mnist_batch_iterator
    elif FLAGS.dataset == 'images':
        batch_size = 16
        embedding_k = 512
        embedding_d = 128
        num_channels = 3

        build_batch_iterator = datasets.build_image_batch_iterator
    else:
        raise Exception('invalid dataset')

    train_batches = build_batch_iterator(FLAGS.train_data_path, batch_size)
    valid_batches = build_batch_iterator(FLAGS.valid_data_path, batch_size)

    model = build_model(
        FLAGS.use_resnet,
        num_channels,
        embedding_k,
        embedding_d,
        train_batches.output_types,
        train_batches.output_shapes)

    next_train_batch = train_batches.get_next()
    next_valid_batch = valid_batches.get_next()

    with tf.Session() as session:
        session.run(train_batches.initializer)
        session.run(valid_batches.initializer)

        # NOTE: generate handles for switching dataset
        train_handle = session.run(train_batches.string_handle())
        valid_handle = session.run(valid_batches.string_handle())

        train(
            session,
            model,
            train_handle,
            valid_handle,
            FLAGS.ckpt_path,
            FLAGS.log_path)


if __name__ == '__main__':
    tf.app.flags.DEFINE_string('train_data_path', None, '')
    tf.app.flags.DEFINE_string('valid_data_path', None, '')

    tf.app.flags.DEFINE_string('ckpt_path', None, '')
    tf.app.flags.DEFINE_string('log_path', None, '')

    tf.app.flags.DEFINE_string('dataset', 'mnist', '')

    tf.app.flags.DEFINE_boolean('use_resnet', True, '')

    tf.app.run()

