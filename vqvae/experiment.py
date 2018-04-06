"""
"""
import numpy as np
import os
import tensorflow as tf

import model_vqvae


def mnist_batches(images, batch_size):
    """
    """
    # NOTE: get dataset shape
    n, h, w, c = images.shape

    indices = np.arange(n)

    while True:
        # NOTE: shuffle index pool
        np.random.shuffle(indices)

        for i in range(n % batch_size, n, batch_size):
            sampled_images = images[indices[i:i+batch_size]]

            y, x = np.random.randint(5, size=2)

            padded_images =  np.pad(
                sampled_images,
                [(0, 0), (y, 4 - y), (x, 4 - x), (0, 0)],
                mode='constant',
                constant_values=0.0)

            yield padded_images


def load_mnist(path_mnist, batch_size=128):
    """
    """
    # NOTE: load all images of mnist
    import mnist

    path_train_eigens = os.path.join(path_mnist, 'train-images-idx3-ubyte.gz')
    path_issue_eigens = os.path.join(path_mnist, 't10k-images-idx3-ubyte.gz')

    dataset = mnist.load_mnist(path_train_eigens, '', path_issue_eigens, '')

    train_images = dataset['train_eigens']
    valid_images = dataset['issue_eigens']

    train_image_batches = mnist_batches(train_images, batch_size)
    valid_image_batches = mnist_batches(valid_images, batch_size)

    return train_image_batches, valid_image_batches


def build_summary(model):
    """
    """
    source_images = model['source_images']
    result_images = model['result_images']

    cmp_images = tf.concat([source_images, result_images], axis=2)

    cmp_images = tf.reshape(cmp_images, [1, -1, 64, 1])

    summary_images = tf.summary.image('images', cmp_images, max_outputs=1)

    summary_loss = tf.summary.scalar('loss', model['loss'])

    return tf.summary.merge([summary_loss, summary_images])


def train(model, train_batches, valid_batches, ckpt_path, log_path):
    """
    """
    source_ckpt_path = tf.train.latest_checkpoint(ckpt_path)
    target_ckpt_path = os.path.join(ckpt_path, 'model.ckpt')

    summary = build_summary(model)

    reporter = tf.summary.FileWriter(log_path)

    with tf.Session() as session:
        if source_ckpt_path is None:
            session.run(tf.global_variables_initializer())
        else:
            tf.train.Saver().restore(session, source_ckpt_path)

        # NOTE:
        step = session.run(model['step'])

        reporter.add_session_log(
            tf.SessionLog(status=tf.SessionLog.START), global_step=step)

        for images in train_batches:
            feeds = {
                model['source_images']: images,
                model['learning_rate']: 5e-6,
            }

            fetch = {
                'trainer': model['trainer'],
                'loss': model['loss'],
                'step': model['step'],
            }

            if step % 1000 == 0:
                fetch['summary'] = summary

            fetched = session.run(fetch, feed_dict=feeds)

            step = fetched['step']

            if 'summary' in fetched:
                reporter.add_summary(fetched['summary'], step)

            if step % 10000 == 0:
                tf.train.Saver().save(
                    session, target_ckpt_path, global_step=model['step'])


def main(_):
    """
    """
    FLAGS = tf.app.flags.FLAGS

    if FLAGS.dataset == 'mnist':
        model = model_vqvae.build_model()

        train_image_batches, valid_image_batches = load_mnist(
            FLAGS.data_path)
    else:
        raise Exception('invalid dataset')

    train(
        model,
        train_image_batches,
        valid_image_batches,
        FLAGS.ckpt_path,
        FLAGS.log_path)


if __name__ == '__main__':
    tf.app.flags.DEFINE_string('data_path', None, '')
    tf.app.flags.DEFINE_string('ckpt_path', None, '')
    tf.app.flags.DEFINE_string('log_path', None, '')

    tf.app.flags.DEFINE_string('dataset', 'mnist', '')

    tf.app.run()


