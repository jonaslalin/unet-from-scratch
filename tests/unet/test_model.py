import tensorflow as tf

from unet.model import unet


def test_ronneberger_unet():
    model = unet(
        input_shape=(572, 572, 1),
        num_classes=2,
        padding="valid",
        filters=(64, 128, 256, 512, 1024),
    )
    assert model.output_shape == (None, 388, 388, 2)

    with tf.device("/device:CPU:0"):  # tensorflow-metal fix
        g = tf.random.Generator.from_seed(1)
        inputs = g.uniform(shape=(2, 572, 572, 1))
    outputs = model(inputs)
    assert outputs.shape == (2, 388, 388, 2)
