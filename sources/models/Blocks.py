import tensorflow as tf
from tensorflow.keras import layers

class DownResBlock(layers.Layer):
    def __init__(self, channels, kernel_size, w_norm_clip, initial_activation=None):
        super(DownResBlock, self).__init__()
        self.out_channels = channels
        self.in_act = initial_activation
        self.conv1 = layers.Conv1D(filters=channels, kernel_size=kernel_size, padding="same", activation=layers.LeakyReLU(),
                                   kernel_constraint=tf.keras.constraints.MaxNorm(max_value=w_norm_clip, axis=[0,1]))
        self.conv2 = layers.Conv1D(filters=channels, kernel_size=kernel_size, padding="same",
                                   kernel_constraint=tf.keras.constraints.MaxNorm(max_value=w_norm_clip, axis=[0,1]))

        self.pool = layers.AveragePooling1D(pool_size=2, strides=2, padding="same")

        self.shortcut_conv = layers.Conv1D(filters=channels, kernel_size=1, padding="same",
                                           kernel_constraint=tf.keras.constraints.MaxNorm(max_value=w_norm_clip, axis=[0,1]))
        self.shortcut_pool = layers.AveragePooling1D(pool_size=2, strides=2, padding="same")

    def call(self, inputs, **kwargs):
        x = inputs
        input_channels = inputs.shape.as_list()[-1]

        if self.in_act is not None:
            x = self.in_act(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.pool(x)

        x_0 = self.shortcut_pool(inputs)
        if self.out_channels != input_channels:
            x_0 = self.shortcut_conv(x_0)

        out = x + x_0

        return out


class DownResLayer(layers.Layer):
    def __init__(self, channels_out, dropout_rate=0.5, kernel_size=3, w_norm_clip=2, first_layer=False, use_dropout=False, **kwargs):
        super(DownResLayer, self).__init__(**kwargs)
        self.use_dropout = use_dropout
        if first_layer:
            self.down_resblock = DownResBlock(channels_out, kernel_size, w_norm_clip)
        else:
            self.down_resblock = DownResBlock(channels_out, kernel_size, w_norm_clip, initial_activation=layers.LeakyReLU())
        self.dropout = layers.Dropout(rate=dropout_rate)

    def call(self, inputs, **kwargs):
        x = self.down_resblock(inputs)

        if self.use_dropout:
            x = self.dropout(x, training=kwargs["training"])

        return x

class UpResBlock(layers.Layer):
    def __init__(self, channels, kernel_size=4, w_norm_clip=2, **kwargs):
        super(UpResBlock, self).__init__(**kwargs)
        self.out_channels = channels

        self.up = layers.UpSampling1D(size=2)
        self.conv0 = layers.Conv1D(filters=channels, kernel_size=kernel_size, padding="same", activation=layers.LeakyReLU(),
                                   kernel_constraint=tf.keras.constraints.MaxNorm(max_value=w_norm_clip, axis=[0,1]))
        self.conv1 = layers.Conv1D(filters=channels, kernel_size=kernel_size, padding="same",
                                   kernel_constraint=tf.keras.constraints.MaxNorm(max_value=w_norm_clip, axis=[0, 1]))

        self.up_s = layers.UpSampling1D(size=2)
        self.upconv_s = layers.Conv1D(filters=channels, kernel_size=1, padding="same")

    def call(self, inputs, **kwargs):
        x = inputs
        input_channels = inputs.shape.as_list()[-1]

        x = self.up(x)
        x = self.conv0(x)
        x = self.conv1(x)

        x_0 = self.up_s(inputs)
        if self.out_channels != input_channels:
            x_0 = self.upconv_s(x_0)

        out = x + x_0

        return out


class UpResLayer(layers.Layer):
    def __init__(self, channels_out, dropout_rate=0.4, kernel_size=4, w_norm_clip=2, use_actnormdrop=False, **kwargs):
        super(UpResLayer, self).__init__(**kwargs)
        self.use_actnormdrop = use_actnormdrop
        self.up_resblock = UpResBlock(channels_out, kernel_size, w_norm_clip)
        self.act = layers.LeakyReLU()
        self.batchnorm = layers.BatchNormalization()
        self.dropout = layers.Dropout(rate=dropout_rate)

    def call(self, inputs, **kwargs):
        x = self.up_resblock(inputs)

        if not self.use_actnormdrop:
            x = self.act(x)
            x = self.batchnorm(x, training=kwargs["training"])
            x = self.dropout(x, training=kwargs["training"])

        return x

class AttentionLayer(layers.Layer):
    def __init__(self, filters, use_input_as_value=False, use_actnormdrop=False, dropout_rate=0.4, **kwargs):
        super().__init__(**kwargs)
        self.use_input_as_value = use_input_as_value
        self.use_actnormdrop = use_actnormdrop

        self.act = layers.LeakyReLU()
        self.norm = layers.BatchNormalization()
        self.query_mat = layers.Conv1D(
            filters=filters,
            kernel_size=3,
            padding="same"
        )
        self.value_mat = layers.Conv1D(
            filters=filters,
            kernel_size=3,
            padding="same"
        )

        self.attention0 = layers.Attention(
            use_scale=True,
            causal=False
        )

        self.short_downres = layers.Conv1D(
            filters=filters,
            kernel_size=3,
            padding="same"
        )

        self.drop = layers.Dropout(rate=dropout_rate)

    def call(self, inputs, **kwargs):
        x_0 = self.short_downres(inputs)
        x = inputs

        q = self.query_mat(x)
        v = self.value_mat(x)

        x = self.attention0([q, v])

        x = layers.add([x, x_0])

        if self.use_actnormdrop:
            x = self.act(x)
            x = self.norm(x, training=kwargs["training"])
            x = self.drop(x, training=kwargs["training"])

        return x