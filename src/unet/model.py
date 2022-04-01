import tensorflow as tf


def conv_batchnorm_relu(filters, padding):
    def forward(inputs):
        x = tf.keras.layers.Conv2D(
            filters,
            kernel_size=3,
            padding=padding,
            use_bias=False,
        )(inputs)
        x = tf.keras.layers.BatchNormalization()(x)
        outputs = tf.keras.layers.Activation("relu")(x)
        return outputs

    return forward


def double_conv_batchnorm_relu(filters, padding):
    def forward(inputs):
        x = conv_batchnorm_relu(filters, padding)(inputs)
        outputs = conv_batchnorm_relu(filters, padding)(x)
        return outputs

    return forward


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


def encoder_fn(filters, padding):
    def forward(inputs):
        outputs = []
        x = inputs
        for i, _filters in enumerate(filters):
            x = double_conv_batchnorm_relu(_filters, padding)(x)
            outputs.insert(0, x)
            if i < len(filters) - 1:
                x = tf.keras.layers.MaxPool2D(strides=2)(x)
        return outputs

    return forward


def decoder_fn(num_classes, padding):
    def forward(inputs):
        x = inputs[0]
        for shortcut in inputs[1:]:
            filters = shortcut.shape[-1]
            x = tf.keras.layers.Conv2DTranspose(
                filters,
                kernel_size=2,
                strides=2,
            )(x)
            x = tf.keras.layers.Concatenate()([crop([shortcut, x]), x])
            x = double_conv_batchnorm_relu(filters, padding)(x)
        outputs = tf.keras.layers.Conv2D(filters=num_classes, kernel_size=1)(x)
        return outputs

    return forward


def unet(
    shape=(572, 572, 1),
    num_classes=2,
    padding="valid",
    filters=(64, 128, 256, 512, 1024),
):
    inputs = tf.keras.Input(shape)
    x = encoder_fn(filters, padding)(inputs)
    outputs = decoder_fn(num_classes, padding)(x)

    encoder = tf.keras.Model(inputs, x, name="encoder")
    decoder = tf.keras.Model(x, outputs, name="decoder")
    autoencoder = tf.keras.Model(inputs, outputs, name="autoencoder")

    return encoder, decoder, autoencoder
