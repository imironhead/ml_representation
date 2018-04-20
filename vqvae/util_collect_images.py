"""
"""
import argparse
import hashlib
import multiprocessing
import os
import skimage.io
import skimage.transform


def do_collect_images(params):
    """
    """
    task_index = params['index']
    extension = params['extension']
    min_source_image_size = params['min_source_image_size']
    min_target_image_size = params['min_target_image_size']
    target_dir_path = params['target_dir_path']
    source_image_paths = params['source_image_paths']

    for index, path in enumerate(source_image_paths):
        if index % 100 == 0:
            print('[{}] done {} / {}'.format(
                task_index, index, len(source_image_paths)))

        target_name = hashlib.md5(path.encode('utf-8')).hexdigest()
        target_name = target_name + '.' + extension
        target_path = os.path.join(target_dir_path, target_name)

        if os.path.isfile(target_path):
            continue

        image = skimage.io.imread(path)

        # NOTE: is not interested in mono images
        if len(image.shape) != 3:
            continue

        h, w, c = image.shape

        # NOTE: only interested in rgb images
        if c != 3:
            continue

        # NOTE: ignore small images
        if min(h, w) < min_source_image_size:
            continue

        # NOTE: scale up
        if min(h, w) < min_target_image_size:
            new_w = w * min_target_image_size // min(w, h)
            new_h = h * min_target_image_size // min(w, h)

            h = max(new_h, min_target_image_size)
            w = max(new_w, min_target_image_size)

            image = skimage.transform.resize(image, (h, w))

        skimage.io.imsave(target_path, image)


def collect_images(
        source_dir_path,
        target_dir_path,
        extension,
        min_source_image_size,
        min_target_image_size):
    """
    """
    # NOTE: collect all images under cource_dir_path
    def is_image_name(name):
        name, ext = os.path.splitext(name)

        return ext.lower() in ['.png', '.jpg', '.jpeg']

    names = os.listdir(source_dir_path)
    names = [n for n in names if is_image_name(n)]

    source_image_paths = [os.path.join(source_dir_path, n) for n in names]

    # NOTE: split all images into 8 groups (for 8 process)
    num_processes = 16

    tasks = [{
        'extension': extension,
        'min_source_image_size': min_source_image_size,
        'min_target_image_size': min_target_image_size,
        'target_dir_path': target_dir_path,
        'source_image_paths': []} for _ in range(num_processes)]

    num_paths = len(source_image_paths)
    num_paths_per_task = (num_paths + num_processes - 1) // num_processes

    for begin in range(0, num_paths, num_paths_per_task):
        index = begin // num_paths_per_task
        end = min(begin + num_paths_per_task, num_paths)

        tasks[index]['index'] = index
        tasks[index]['source_image_paths'] = source_image_paths[begin:end]

    # NOTE: dispatch the tasks
    with multiprocessing.Pool(num_processes) as pool:
        pool.map(do_collect_images, tasks)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--extension',
        default='png',
        type=str,
        help='format of the target images, can be "png" or "jpg"')

    parser.add_argument(
        '--source-dir-path',
        type=str,
        help='the dir which contains source image files')

    parser.add_argument(
        '--target-dir-path',
        type=str,
        help='the dir for keeping the processed images')

    parser.add_argument(
        '--min-source-image-size',
        default=100,
        type=int,
        help='drop the image if min(width, height) < min_source_image_size')

    parser.add_argument(
        '--min-target-image-size',
        default=128,
        type=int,
        help='if min(width, height) < min_target_image_size, up-scale it ' +
             'so that min(w, h) == min_target_image_size')

    args = parser.parse_args()

    # NOTE: sanity check, only support jpg and png
    if args.extension != 'jpg' and args.extension != 'png':
        raise Exception('non-supported extension: {}'.format(args.extension))

    if not os.path.isdir(args.source_dir_path):
        raise Exception('invalid source dir: {}'.format(args.source_dir_path))

    if not os.path.isdir(args.target_dir_path):
        raise Exception('invalid target dir: {}'.format(args.target_dir_path))

    collect_images(
        args.source_dir_path,
        args.target_dir_path,
        args.extension,
        args.min_source_image_size,
        args.min_target_image_size)
