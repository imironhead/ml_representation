# Replicated "Neural Discrete Representation Learning"

## Experiments

### MSCOCO without ResNet

![loss](/assets/vqvae_mscoco_nores_loss.png)
Training Loss.

![train](/assets/vqvae_mscoco_nores_train_430k.png)
Training results in step 430,000 (MSCOCO).

![test](/assets/vqvae_mscoco_nores_test_400k.png)
Validation results in step 400,000 (IMAGENET).

### MSCOCO with ResNet

![loss](/assets/vqvae_mscoco_res_loss.png)
Training Loss.

![train](/assets/vqvae_mscoco_res_train_405k.png)
Training results in step 405,000 (MSCOCO).

![test](/assets/vqvae_mscoco_res_test_400k.png)
Validation results in step 400,000 (IMAGENET).

### MNIST

### Fashion MNIST

![loss](/assets/vqvae_fashion_nores_loss.png)

![train](/assets/vqvae_fashion_nores_train_224k.png)

![test](/assets/vqvae_fashion_nores_test_200k.png)

## NOTES

### Gradients Descend between Encoder and Decoder in Tensorflow

``` python
tensors_ze = build_encoder(source_images, embedding_d)

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
result_images = build_decoder(tensors, embedding_d, num_channels)
```

### Parallel Training Data Loading in Tensorflow

[tf.data.Dataset map](https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map) can do things parallelly.

``` python
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
```
