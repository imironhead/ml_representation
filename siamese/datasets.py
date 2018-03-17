"""
"""
import glob
import itertools
import numpy as np
import os
import skimage.io
import skimage.transform


def omniglot_path_to_keys(image_path):
    """
    """
    temp, basename = os.path.split(image_path)
    temp, character = os.path.split(temp)
    temp, alphabet = os.path.split(temp)
    temp, category = os.path.split(temp)

    character = int(character[-2:]) - 1
    drawer = int(basename[5:7]) - 1

    return category, alphabet, character, drawer


def omniglot_enum_image_paths(root_dir_path, category):
    """
    """
    image_paths = []

    category_path = os.path.join(root_dir_path, category)

    for dirname, dirnames, filenames in os.walk(category_path):
        names = filter(lambda x: x.endswith('.png'), filenames)
        paths = map(lambda x: os.path.join(dirname, x), names)

        image_paths.extend(paths)

    return image_paths


def omniglot_enum_images(root_dir_path, category):
    """
    """
    image_paths = omniglot_enum_image_paths(root_dir_path, category)

    alphabet_names = []
    character_number = {}

    # NOTE: give alphabets indices
    # NOTE: count number of characters of each alphabet
    for image_path in image_paths:
        _, alphabet_key, character_idx, drawer_idx = \
            omniglot_path_to_keys(image_path)

        if alphabet_key not in character_number:
            character_number[alphabet_key] = character_idx + 1

            alphabet_names.append(alphabet_key)

        if character_number[alphabet_key] <= character_idx:
            character_number[alphabet_key] = character_idx + 1

    # NOTE: sort names so that the structure of omniglot is consistant.
    alphabet_names = sorted(alphabet_names)

    alphabet_indices = {n: i for i, n in enumerate(alphabet_names)}

    # NOTE: we have len(alphabet_indices) alphabets
    omniglot = [[] for _ in range(len(alphabet_indices))]

    for image_path in image_paths:
        _, alphabet_key, character_idx, drawer_idx = \
            omniglot_path_to_keys(image_path)

        # NOTE: alphabet name to index
        alphabet_idx = alphabet_indices[alphabet_key]

        # NOTE: initail alphabet slot if neceeary
        # NOTE: each alphabet has character_number[alphabet_key] characters
        # NOTE: each character has 20 drawers
        if len(omniglot[alphabet_idx]) == 0:
            omniglot[alphabet_idx] = \
               [[None] * 20 for _ in range(character_number[alphabet_key])]

        image = skimage.io.imread(image_path)
        image = image.reshape(1, 105, 105, 1)
        image = (255.0 - image.astype(np.float)) / 255.0

        omniglot[alphabet_idx][character_idx][drawer_idx] = image

    return omniglot


def load_dataset(root_path, seed=0):
    """
    """
    # NOTE: load all available dataset
    omniglot_background = \
        omniglot_enum_images(root_path, 'images_background')
    omniglot_evaluation = \
        omniglot_enum_images(root_path, 'images_evaluation')

    # NOTE: merge in character level
    omniglot = \
        [ch for ch in itertools.chain.from_iterable(omniglot_background)] + \
        [ch for ch in itertools.chain.from_iterable(omniglot_evaluation)]

    # NOTE: split data randomly base on the seed (then the training and
    #       validation sets can be rebuild with the seed)
    # NOTE: 20 drawers drew all characters in omniglot
    num_characters = len(omniglot)
    num_drawers = 20

    np.random.seed(seed)

    indices_characters = np.random.choice(
        num_characters, size=num_characters, replace=False)
    indices_drawers = np.random.choice(
        num_drawers, size=num_drawers, replace=False)

    # NOTE: 964 characters and 16 drawers for training
    # NOTE: there are 964 characters in images_background
    num_training_characters = 964
    num_training_drawers = 16
    num_validation_characters = len(omniglot) - num_training_characters

    training_set = [[] for _ in range(num_training_characters)]
    validation_set = [[] for _ in range(num_validation_characters)]

    # NOTE: all alphabets & drawers are split into 4 parts:
    #                            drawers trsin    drawers validation
    #       alphabet train:      A,               B
    #       alphabet validation: C,               D
    #
    #       then use A to training & D for validation
    #       so the alphabets & drawers in both sets are not overlapped
    for i, ch in enumerate(indices_characters[:num_training_characters]):
        for dr in indices_drawers[:num_training_drawers]:
            training_set[i].append(omniglot[ch][dr])

    for i, ch in enumerate(indices_characters[num_training_characters:]):
        for dr in indices_drawers[num_training_drawers:]:
            validation_set[i].append(omniglot[ch][dr])

    return training_set, validation_set


def affine_distort(image):
    """
    siamese neural networks for one-shot image recognition
    3.2 learning - affine distortions
    """
    affine = skimage.transform.AffineTransform(
        rotation=np.random.uniform(low=-0.055, high=0.055) * np.pi,
        shear=np.random.uniform(low=-0.3, high=0.3),
        scale=np.random.uniform(low=0.8, high=1.2, size=[2]),
        translation=np.random.uniform(low=-2.0, high=2.0, size=[2]))

    h, w = image.shape[1:3]

    image = image.reshape(h, w)

    image = skimage.transform.warp(
        image, affine.inverse, mode='constant', cval=0.0)

    return image.reshape(1, h, w, 1)


def generate_mini_batch(characters, batch_size, distort=True):
    """
    """
    image_group_a = []
    image_group_b = []
    labels = []

    positive_size = batch_size // 2
    negative_size = batch_size - positive_size

    # NOTE: 50% for positive part
    for _ in range(positive_size):
        ch = np.random.choice(len(characters))
        dr1, dr2 = np.random.choice(len(characters[ch]), size=2)

        im_a, im_b = characters[ch][dr1], characters[ch][dr2]

        if distort:
            im_a, im_b = affine_distort(im_a), affine_distort(im_b)

        image_group_a.append(im_a)
        image_group_b.append(im_b)

        labels.append(1.0)

    # NOTE: 50% for negative part
    for _ in range(negative_size):
        ch1, ch2 = np.random.choice(len(characters), size=2, replace=False)
        dr1, dr2 = np.random.choice(len(characters[ch1]), size=2)

        im_a, im_b = characters[ch1][dr1], characters[ch2][dr2]

        if distort:
            im_a, im_b = affine_distort(im_a), affine_distort(im_b)

        image_group_a.append(im_a)
        image_group_b.append(im_b)

        labels.append(0.0)

    # NOTE: concat as numpy arrays
    image_group_a = np.concatenate(image_group_a, axis=0)
    image_group_b = np.concatenate(image_group_b, axis=0)

    labels = np.array(labels, dtype=np.float32).reshape([-1, 1])

    return image_group_a, image_group_b, labels

