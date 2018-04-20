"""
"""
import numpy as np
import os
import skimage.io
import tensorflow as tf


def build_mnist_generator(images):
    """
    """
    def mnist_generator():
        while True:
            # NOTE: shuffle on axis 0
            np.random.shuffle(images)

            for image in images:
                image =  np.pad(
                    image,
                    [(4, 4), (4, 4), (0, 0)],
                    mode='constant',
                    constant_values=-1.0)

                yield image

    return mnist_generator


def build_mnist_batch_iterator(path, batch_size=128):
    """
    """
    def preprocess_mnist(image):
        # NOTE: random crop to 28x28x1
        image = tf.random_crop(image, size=[28, 28, 1])

        # NOTE: random horizontal flip
        image = tf.image.random_flip_left_right(image)

        return image

    import mnist

    images = mnist.extract_images(path)
    images = images.astype(np.float32) / 127.5 - 1.0

    # NOTE: build path list dataset
    images = build_mnist_generator(images)

    data = tf.data.Dataset.from_generator(
        images, (tf.float32), (tf.TensorShape([36, 36, 1])))

    # NOTE: the image generator never ends

    # NOTE: the image generator shuffled images in each epoch

    # NOTE: preprocess images
    data = data.map(preprocess_mnist, 32)

    # NOTE: combine images to batch
    data = data.batch(batch_size=batch_size)

    # NOTE: create the final iterator
    iterator = data.make_initializable_iterator()

    return iterator


def build_path_generator(dir_path):
    """
    """
    # NOTE: check if the extension is of an image
    def is_image_name(name):
        name, ext = os.path.splitext(name)

        return ext.lower() in ['.png', '.jpg', '.jpeg']

    names = os.listdir(dir_path)
    names = [n for n in names if is_image_name(n)]

    def paths_generator():
        while True:
            # NOTE: shuffle names
            np.random.shuffle(names)

            for name in names:
                yield os.path.join(dir_path, name)

    return paths_generator


def build_image_batch_iterator(dir_path, batch_size=128):
    """
    """
    # NOTE: the path generator never ends
    # NOTE: the path generator shuffled path list in each epoch
    def preprocess_image(path):
        image = tf.read_file(path)
        image = tf.image.decode_image(image, channels=3)
        image = tf.random_crop(image, size=[128, 128, 3])
        image = tf.image.random_flip_left_right(image)
        image = tf.image.convert_image_dtype(image, tf.float32)
        image = tf.multiply(image, 2.0)
        image = tf.subtract(image, 1.0)

        return image

    # NOTE: build path list dataset
    image_paths = build_path_generator(dir_path)

    data = tf.data.Dataset.from_generator(image_paths, (tf.string))

    # NOTE: preprocess image concurrently
    data = data.map(preprocess_image, 16)

    # NOTE: combine images to batch
    data = data.batch(batch_size=batch_size)

    # NOTE: create the final iterator
    iterator = data.make_initializable_iterator()

    return iterator

