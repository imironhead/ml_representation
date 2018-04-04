"""
"""
import gzip
import numpy as np
import os


def read32(bytestream):
    """
    read a 32 bit unsigned int fron the bytestream.
    """
    dt = np.dtype(np.uint32).newbyteorder('>')

    return np.frombuffer(bytestream.read(4), dtype=dt)[0]


def extract_images(path_file):
    """
    extract the images into a numpy array in shape [number, y, x, depth].
    """
    print('extracting: {}'.format(path_file))

    with gzip.open(path_file) as bytestream:
        magic = read32(bytestream)

        if magic != 2051:
            raise Exception('invalid mnist data: {}'.format(path_file))

        size = read32(bytestream)
        rows = read32(bytestream)
        cols = read32(bytestream)

        buff = bytestream.read(size * rows * cols)
        data = np.frombuffer(buff, dtype=np.uint8)
        data = data.reshape(size, rows, cols, 1)

    return data


def extract_labels(path_file):
    """
    extract the labels into a numpy array with shape [index].
    """
    print('extracting: {}'.format(path_file))

    with gzip.open(path_file) as bytestream:
        magic = read32(bytestream)

        if magic != 2049:
            raise Exception('invalid mnist data: {}'.format(path_file))

        size = read32(bytestream)
        buff = bytestream.read(size)
        labels = np.frombuffer(buff, dtype=np.uint8)

    return labels.astype(np.int32)


def load_mnist(
        path_train_eigens, path_train_labels,
        path_issue_eigens, path_issue_labels):
    """
    """
    train_eigens, train_labels_onehot = None, None
    issue_eigens, issue_labels_onehot = None, None

    if os.path.isfile(path_train_eigens):
        train_eigens = extract_images(path_train_eigens)
        train_eigens = train_eigens.astype(np.float32) / 255.0

    if os.path.isfile(path_train_labels):
        train_labels = extract_labels(path_train_labels)
        train_labels_onehot = np.zeros((train_labels.size, 10))
        train_labels_onehot[np.arange(train_labels.size), train_labels] = 1.0

    if os.path.isfile(path_issue_eigens):
        issue_eigens = extract_images(path_issue_eigens)
        issue_eigens = issue_eigens.astype(np.float32) / 255.0

    if os.path.isfile(path_issue_labels):
        issue_labels = extract_labels(path_issue_labels)
        issue_labels_onehot = np.zeros((issue_labels.size, 10))
        issue_labels_onehot[np.arange(issue_labels.size), issue_labels] = 1.0

    return {
        'train_eigens': train_eigens,
        'train_labels': train_labels_onehot,
        'issue_eigens': issue_eigens,
        'issue_labels': issue_labels_onehot,
    }


if __name__ == '__main__':
    path_root = '/home/ironhead/datasets/mnist'

    path_train_eigens = os.path.join(path_root, 'train-images-idx3-ubyte.gz')
    path_train_labels = os.path.join(path_root, 'train-labels-idx1-ubyte.gz')
    path_issue_eigens = os.path.join(path_root, 't10k-images-idx3-ubyte.gz')
    path_issue_labels = os.path.join(path_root, 't10k-labels-idx1-ubyte.gz')

    dataset = load_mnist(
        path_train_eigens, path_train_labels,
        path_issue_eigens, path_issue_labels)

    print(dataset['train_eigens'].shape)
    print(dataset['issue_eigens'].shape)

    print(np.sum(dataset['train_eigens'][:, :2]))
    print(np.sum(dataset['train_eigens'][:, -2:]))

    print(np.sum(dataset['train_eigens'][:, :, :2]))
    print(np.sum(dataset['train_eigens'][:, :, -2:]))

    print(np.sum(dataset['issue_eigens'][:, :2]))
    print(np.sum(dataset['issue_eigens'][:, -2:]))

    print(np.sum(dataset['issue_eigens'][:, :, :2]))
    print(np.sum(dataset['issue_eigens'][:, :, -2:]))
