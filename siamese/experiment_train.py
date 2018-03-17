"""
"""
import itertools
import numpy as np
import os
import skimage.transform
import sklearn.metrics
import tensorflow as tf

import datasets
import model

FLAGS = tf.app.flags.FLAGS


def train_one_step(session, siamese_model, characters, batch_size):
    """
    """
    image_group_a, image_group_b, labels = \
        datasets.generate_mini_batch(characters, batch_size, distort=True)

    feeds = {
        siamese_model['images_a']: image_group_a,
        siamese_model['images_b']: image_group_b,
        siamese_model['labels']: labels,
    }

    fetch = {
        'loss': siamese_model['loss'],
        'step': siamese_model['step'],
        'trainer': siamese_model['trainer'],
    }

    fetched = session.run(fetch, feed_dict=feeds)

    return fetched['step'], fetched['loss']


def validate(session, siamese_model, characters, batch_size):
    """
    """
    image_group_a, image_group_b, labels = \
        datasets.generate_mini_batch(characters, batch_size, distort=False)

    feeds = {
        siamese_model['images_a']: image_group_a,
        siamese_model['images_b']: image_group_b,
    }

    fetch = {
        'predictions': siamese_model['predictions'],
    }

    fetched = session.run(fetch, feed_dict=feeds)

    predictions = fetched['predictions'].flatten()

    predictions[predictions >= 0.5] = 1.0
    predictions[predictions <= 0.5] = 0.0
    predictions = predictions.astype(np.int)

    labels = labels.flatten()
    labels = labels.astype(np.int)

    return sklearn.metrics.accuracy_score(labels, predictions)


def train():
    """
    """
    training_set, validation_set = datasets.load_dataset(FLAGS.data_path)

    siamese_model = model.build_siamese_network()

    with tf.Session() as session:
        session.run(tf.global_variables_initializer())

        while True:
            step, loss = \
                train_one_step(session, siamese_model, training_set, 128)

            if step % 200 == 0:
                print('loss[{}]: {}'.format(step, loss))

            if step == 20000:
                break

        score = validate(session, siamese_model, validation_set, 256)

        print('accu[{}]: {}'.format(step, score))

        if FLAGS.ckpt_path is not None:
            tf.train.Saver().save(
                session, FLAGS.ckpt_path, global_step=siamese_model['step'])

def main(_):
    """
    """
    train()


if __name__ == '__main__':
    tf.app.flags.DEFINE_string('data_path', None, '')
    tf.app.flags.DEFINE_string('ckpt_path', None, '')
    tf.app.flags.DEFINE_string('log_path', None, '')

    tf.app.run()
