import tensorflow as tf


def conv_batchnorm_relu(filters, padding):
    def call(inputs):
        x = tf.keras.layers.Conv2D(
            filters,
            kernel_size=3,
            padding=padding,
            use_bias=False,
            kernel_initializer="he_normal",
        )(inputs)
        x = tf.keras.layers.BatchNormalization()(x)
        outputs = tf.keras.layers.Activation("relu")(x)
        return outputs

    return call


def double_conv_batchnorm_relu(filters, padding):
    def call(inputs):
        x = conv_batchnorm_relu(filters, padding)(inputs)
        outputs = conv_batchnorm_relu(filters, padding)(x)
        return outputs

    return call


def crop(inputs):
    source, target = inputs
    height_diff = source.shape[1] - target.shape[1]
    width_diff = source.shape[2] - target.shape[2]
    top_crop = height_diff // 2
    bottom_crop = height_diff // 2 + height_diff % 2
    left_crop = width_diff // 2
    right_crop = width_diff // 2 + width_diff % 2
    outputs = tf.keras.layers.Cropping2D(
        cropping=((top_crop, bottom_crop), (left_crop, right_crop))
    )(source)
    return outputs


def encoder(filters, padding):
    def call(inputs):
        outputs = []
        x = inputs
        for i, _filters in enumerate(filters):
            x = double_conv_batchnorm_relu(_filters, padding)(x)
            outputs.insert(0, x)
            if i < len(filters) - 1:
                x = tf.keras.layers.MaxPool2D(strides=2)(x)
        return outputs

    return call


def decoder(num_classes, padding):
    def call(inputs):
        x = inputs[0]
        for shortcut in inputs[1:]:
            filters = shortcut.shape[-1]
            x = tf.keras.layers.Conv2DTranspose(
                filters,
                kernel_size=2,
                strides=2,
                kernel_initializer="he_normal",
            )(x)
            x = tf.keras.layers.Concatenate()([crop([shortcut, x]), x])
            x = double_conv_batchnorm_relu(filters, padding)(x)
        outputs = tf.keras.layers.Conv2D(
            filters=num_classes,
            kernel_size=1,
            kernel_initializer="glorot_uniform",
        )(x)
        return outputs

    return call


def unet(
    input_shape=(572, 572, 1),
    num_classes=2,
    padding="valid",
    filters=(64, 128, 256, 512, 1024),
):
    def call(inputs):
        x = encoder(filters, padding)(inputs)
        outputs = decoder(num_classes, padding)(x)
        return outputs

    inputs = tf.keras.Input(shape=input_shape)
    outputs = call(inputs)
    return tf.keras.Model(inputs, outputs)
