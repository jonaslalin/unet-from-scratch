import tensorflow as tf

from unet.model import unet


def test_unet_with_valid_padding():
    _, _, autoencoder = unet(
        shape=(572, 572, 1),
        num_classes=2,
        padding="valid",
        filters=(64, 128, 256, 512, 1024),
    )
    with tf.device("/device:CPU:0"):  # tensorflow-metal fix
        g = tf.random.Generator.from_seed(1)
        inputs = g.uniform(shape=(2, 572, 572, 1))
    outputs = autoencoder(inputs)
    assert outputs.shape == (2, 388, 388, 2)


def test_unet_with_same_padding():
    _, _, autoencoder = unet(
        shape=(512, 512, 3),
        num_classes=3,
        padding="same",
        filters=(64, 128, 256, 512, 1024),
    )
    with tf.device("/device:CPU:0"):  # tensorflow-metal fix
        g = tf.random.Generator.from_seed(1)
        inputs = g.uniform(shape=(2, 512, 512, 3))
    outputs = autoencoder(inputs)
    assert outputs.shape == (2, 512, 512, 3)
