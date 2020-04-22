import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

class DownResBlock(layers.Layer):
    def __init__(self, channels, kernel_size, w_norm_clip, initial_activation=None):
        super(DownResBlock, self).__init__()
        self.out_channels = channels
        self.in_act = initial_activation
        self.conv1 = layers.Conv1D(filters=channels, kernel_size=kernel_size, padding="same", activation=layers.LeakyReLU(),
                                   )#kernel_constraint=tf.keras.constraints.MaxNorm(max_value=w_norm_clip, axis=[0,1,2]))
        self.conv2 = layers.Conv1D(filters=channels, kernel_size=kernel_size, padding="same",
                                   )#kernel_constraint=tf.keras.constraints.MaxNorm(max_value=w_norm_clip, axis=[0,1,2]))

        self.pool = layers.AveragePooling1D(pool_size=2, strides=2, padding="same")

        self.shortcut_conv = layers.Conv1D(filters=channels, kernel_size=1, padding="same",
                                           )#kernel_constraint=tf.keras.constraints.MaxNorm(max_value=w_norm_clip, axis=[0,1,2]))
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
        x = inputs
        if self.use_dropout:
            x = self.dropout(x, training=kwargs["training"])

        x = self.down_resblock(inputs)

        return x

class UpResBlock(layers.Layer):
    def __init__(self, channels, kernel_size=4, w_norm_clip=2,
                 use_initial_activation=True, use_batchnorm=False, **kwargs):
        super(UpResBlock, self).__init__(**kwargs)
        self.out_channels = channels
        self.use_initial_activation = use_initial_activation
        self.use_batchnorm = use_batchnorm
        self.batchnorm = layers.BatchNormalization()
        self.initial_activation = layers.LeakyReLU(alpha=0.2)
        self.up = layers.UpSampling1D(size=2)
        self.conv0 = layers.Conv1D(filters=channels, kernel_size=kernel_size, padding="same", activation=layers.LeakyReLU(),
                                   )#kernel_constraint=tf.keras.constraints.MaxNorm(max_value=w_norm_clip, axis=[0,1,2]))
        self.conv1 = layers.Conv1D(filters=channels, kernel_size=kernel_size, padding="same",
                                   )#kernel_constraint=tf.keras.constraints.MaxNorm(max_value=w_norm_clip, axis=[0,1,2]))

        self.up_s = layers.UpSampling1D(size=2)
        self.upconv_s = layers.Conv1D(filters=channels, kernel_size=1, padding="same",
                                      )#kernel_constraint=tf.keras.constraints.MaxNorm(max_value=w_norm_clip, axis=[0,1,2]))

    def call(self, inputs, **kwargs):
        x_0 = inputs
        x = inputs
        input_channels = inputs.shape.as_list()[-1]

        if self.use_batchnorm:
            x = self.batchnorm(x, training=kwargs["training"])
        if self.use_initial_activation:
            x = self.initial_activation(x)
        x = self.up(x)
        x = self.conv0(x)
        x = self.conv1(x)

        if self.out_channels != input_channels:
            x_0 = self.upconv_s(x_0)
        x_0 = self.up_s(x_0)

        out = x + x_0

        return out


class UpResLayer(layers.Layer):
    def __init__(self, channels_out, dropout_rate=0.4, kernel_size=4, w_norm_clip=1,
                 use_initial_activation=True, use_batchnorm=False, **kwargs):
        super(UpResLayer, self).__init__(**kwargs)
        self.up_resblock = UpResBlock(channels_out, kernel_size, w_norm_clip, use_initial_activation, use_batchnorm)
        self.dropout = layers.Dropout(rate=dropout_rate)

    def call(self, inputs, **kwargs):
        x = self.dropout(inputs)
        x = self.up_resblock(x)
        return x


class AttentionLayer(layers.Layer):
    def __init__(self, channels_out, filters, kernel_size=3, use_input_as_value=False, use_positional_encoding=False, **kwargs):
        super().__init__(**kwargs)
        self.use_input_as_value = use_input_as_value
        self.use_positional_encoding = use_positional_encoding
        self.act = layers.LeakyReLU()
        self.norm = layers.BatchNormalization()
        self.positional_encoding = get_positional_encoding(500, channels_out)
        self.key_mat = layers.Conv1D(
            filters=filters,
            kernel_size=kernel_size,
            padding="same"
        )
        self.query_mat = layers.Conv1D(
            filters=filters,
            kernel_size=kernel_size,
            padding="same"
        )

        self.value_mat = layers.Conv1D(
            filters=channels_out,
            kernel_size=kernel_size,
            padding="same"
        )

        self.attention0 = layers.Attention(
            use_scale=False,
            causal=False
        )

        self.attention_conv = layers.Conv1D(
            filters=channels_out,
            kernel_size=1,
            padding="same"
        )

        self.gamma = tf.Variable(initial_value=0.05, trainable=True, name="gamma")

    def call(self, inputs, **kwargs):
        x = inputs
        seq_length = inputs.shape.as_list()[1]
        if self.use_positional_encoding:
            x = x + self.positional_encoding[:, :seq_length, :]

        q = self.query_mat(x)
        k = self.key_mat(x)
        v = self.value_mat(x)

        x = self.attention0([q, v, k])
        x = self.attention_conv(x)

        out = inputs + x

        return out

def get_positional_encoding(seq_length, seq_depth, with_batch_dim=True):
    sequence_ids = tf.expand_dims(tf.range(seq_length, dtype=tf.float64), axis=-1)
    depth_ids = tf.expand_dims(tf.range(seq_depth), axis=0)

    angle_rates = 1 / tf.pow(10000, (2 * (depth_ids // 2)) / seq_depth)
    angle_rads = tf.cast(sequence_ids * angle_rates, tf.float32)

    even_mask = tf.tile([1., 0.], [seq_depth // 2])
    odd_mask = tf.tile([0., 1.], [seq_depth // 2])

    angle_rads = tf.sin(angle_rads)*tf.expand_dims(even_mask, axis=0) + tf.cos(angle_rads)*tf.expand_dims(odd_mask, axis=0)

    if with_batch_dim:
        angle_rads = tf.expand_dims(angle_rads, axis=0)

    return angle_rads
