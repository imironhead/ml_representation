"""
"""
import glob
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

    alphabet_indices = {}
    character_number = {}

    # NOTE: give alphabets indices
    # NOTE: count number of characters of each alphabet
    for image_path in image_paths:
        _, alphabet_key, character_idx, drawer_idx = \
            omniglot_path_to_keys(image_path)

        if alphabet_key not in alphabet_indices:
            alphabet_indices[alphabet_key] = len(alphabet_indices)
            character_number[alphabet_key] = character_idx + 1

        if character_number[alphabet_key] <= character_idx:
            character_number[alphabet_key] = character_idx + 1

    # NOTE: we have len(alphabet_indices) alphabets
    omniglot = [[]] * len(alphabet_indices)

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
                [[None] * 20] * character_number[alphabet_key]

        image = skimage.io.imread(image_path)
        image = image.reshape(1, 105, 105, 1)
        image = (255.0 - image.astype(np.float)) / 255.0

        omniglot[alphabet_idx][character_idx][drawer_idx] = image

    return omniglot

