import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

class DownResBlock(layers.Layer):
    def __init__(self, channels, kernel_size, w_norm_clip, initial_activation=None, normalization=None, downsample_rate=2, regularization=None):
        super(DownResBlock, self).__init__()
        self.out_channels = channels
        self.in_act = initial_activation
        self.norm0 = None
        self.norm1 = None
        if normalization is "layer":
            self.norm0 = layers.LayerNormalization(axis=[2])
            self.norm1 = layers.LayerNormalization(axis=[2])
        elif normalization is "batch":
            self.norm0 = layers.BatchNormalization()
            self.norm1 = layers.BatchNormalization()

        self.conv1 = layers.Conv1D(filters=channels, kernel_size=kernel_size, padding="same", activation=layers.LeakyReLU(),
                                   kernel_regularizer=regularization)
        self.conv2 = layers.Conv1D(filters=channels, kernel_size=kernel_size, padding="same", strides=downsample_rate,
                                   kernel_regularizer=regularization)

        #self.pool = layers.AveragePooling1D(pool_size=downsample_rate, strides=downsample_rate, padding="same")

        self.shortcut_conv = layers.Conv1D(filters=channels, kernel_size=1, padding="same",
                                           kernel_regularizer=regularization)
        self.shortcut_pool = layers.AveragePooling1D(pool_size=downsample_rate, strides=downsample_rate, padding="same")

    def call(self, inputs, **kwargs):
        x = inputs
        input_channels = inputs.shape.as_list()[-1]

        if self.in_act is not None:
            x = self.in_act(x)
            if self.norm0 is not None:
                x = self.norm0(x, training=kwargs["training"])
        x = self.conv1(x)
        if self.norm1 is not None:
            x = self.norm1(x, training=kwargs["training"])
        x = self.conv2(x)
        #x = self.pool(x)

        x_0 = self.shortcut_pool(inputs)
        if self.out_channels != input_channels:
            x_0 = self.shortcut_conv(x_0)

        out = x + x_0

        return out


class DownResLayer(layers.Layer):
    def __init__(self, channels_out, dropout_rate=0.0, kernel_size=3, w_norm_clip=2, normalization=None, first_layer=False, downsample_rate=2, regularization=None, **kwargs):
        super(DownResLayer, self).__init__(**kwargs)
        if first_layer:
            self.down_resblock = DownResBlock(channels_out, kernel_size, w_norm_clip, normalization=normalization, downsample_rate=downsample_rate, regularization=regularization)
        else:
            self.down_resblock = DownResBlock(channels_out, kernel_size, w_norm_clip, initial_activation=layers.LeakyReLU(), normalization=normalization, downsample_rate=downsample_rate, regularization=regularization)
        self.dropout = layers.Dropout(rate=dropout_rate)

    def call(self, inputs, **kwargs):
        x = inputs
        x = self.dropout(x, training=kwargs["training"])

        x = self.down_resblock(x)

        return x

class UpResBlock(layers.Layer):
    def __init__(self, channels, kernel_size=4, w_norm_clip=2,
                 use_initial_activation=True, normalization=None, **kwargs):
        super(UpResBlock, self).__init__(**kwargs)
        self.out_channels = channels
        self.use_initial_activation = use_initial_activation
        self.norm0 = None
        self.norm1 = None
        if normalization is "layer":
            self.norm0 = layers.LayerNormalization(axis=2)
            self.norm1 = layers.LayerNormalization(axis=2)
        elif normalization is "batch":
            self.norm0 = layers.BatchNormalization()
            self.norm1 = layers.BatchNormalization()
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

        if self.use_initial_activation:
            x = self.initial_activation(x)
            if self.norm0 is not None:
                x = self.norm0(x, training=kwargs["training"])
        x = self.up(x)
        x = self.conv0(x)
        if self.norm1 is not None:
            x = self.norm1(x, training=kwargs["training"])
        x = self.conv1(x)

        if self.out_channels != input_channels:
            x_0 = self.upconv_s(x_0)
        x_0 = self.up_s(x_0)

        out = x + x_0

        return out


class UpResLayer(layers.Layer):
    def __init__(self, channels_out, dropout_rate=0.4, kernel_size=4, w_norm_clip=1,
                 use_initial_activation=True, normalization=None, **kwargs):
        super(UpResLayer, self).__init__(**kwargs)
        self.up_resblock = UpResBlock(channels_out, kernel_size, w_norm_clip, use_initial_activation, normalization)
        self.dropout = layers.Dropout(rate=dropout_rate)

    def call(self, inputs, **kwargs):
        x = self.dropout(inputs)
        x = self.up_resblock(x)
        return x


class AttentionLayer(layers.Layer):
    def __init__(self, channels_out, filters_per_head, num_attention_heads=1, kernel_size=3, use_positional_encoding=False, regularization=None, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)
        self.use_positional_encoding = use_positional_encoding
        self.num_heads = num_attention_heads
        self.filters_per_head = filters_per_head

        self.conv_filters = num_attention_heads * filters_per_head
        self.positional_encoding = get_positional_encoding(500, channels_out)

        self.act = layers.LeakyReLU()

        self.key_mat = layers.Conv1D(
            filters=self.conv_filters,
            kernel_size=kernel_size,
            padding="same",
            kernel_regularizer=regularization
        )
        self.query_mat = layers.Conv1D(
            filters=self.conv_filters,
            kernel_size=kernel_size,
            padding="same",
            kernel_regularizer=regularization
        )

        self.value_mat = layers.Conv1D(
            filters=self.conv_filters,
            kernel_size=kernel_size,
            padding="same",
            kernel_regularizer=regularization
        )

        self.attention_heads = [
            layers.Attention(use_scale=True, causal=False)
            for h in range(self.num_heads)
        ]

        self.attention_conv = layers.Conv1D(
            filters=channels_out,
            kernel_size=kernel_size,
            padding="same",
            kernel_regularizer=regularization
        )

        self.gamma = tf.Variable(initial_value=0.05, trainable=True, name="gamma")

    def split_heads(self, x, batch_size):
        x = tf.reshape(x, [batch_size, -1, self.num_heads, self.filters_per_head])
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def scaled_dot_product_attention(self, q, k, v):
        matmul_qk = tf.matmul(q, k, transpose_b=True)

        dk = tf.cast(tf.shape(k)[-1], tf.float32)
        scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

        attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)
        output = tf.matmul(attention_weights, v)

        return output, attention_weights

    def call(self, inputs, **kwargs):
        x = inputs
        batch_size = tf.shape(inputs)[0]
        seq_length = tf.shape(inputs)[1]
        if self.use_positional_encoding:
            x = x + self.positional_encoding[:, :seq_length, :]

        q = self.query_mat(x)
        k = self.key_mat(x)
        v = self.value_mat(x)

        q = self.split_heads(q, batch_size)
        k = self.split_heads(k, batch_size)
        v = self.split_heads(v, batch_size)

        scaled_attention, attention_weights = self.scaled_dot_product_attention(q, k, v)
        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])  #(batch_size, seq_len_q, num_heads, depth)
        concat_attention = tf.reshape(scaled_attention, (batch_size, -1, self.conv_filters))
        x = self.attention_conv(concat_attention)

        out = inputs + x

        return out


def get_positional_encoding(seq_length, seq_depth, with_batch_dim=True):
    sequence_ids = tf.expand_dims(tf.range(seq_length, dtype=tf.float64), axis=-1)
    depth_ids = tf.expand_dims(tf.range(2, seq_depth + 2), axis=0)

    angle_rates = 1 / tf.pow(500, (2 * (depth_ids // 2)) / seq_depth)
    angle_rads = tf.cast(sequence_ids * angle_rates, tf.float32)

    even_mask = tf.tile([1., 0.], [seq_depth // 2])
    odd_mask = tf.tile([0., 1.], [seq_depth // 2])

    angle_rads = tf.sin(angle_rads)*tf.expand_dims(even_mask, axis=0) + tf.cos(angle_rads)*tf.expand_dims(odd_mask, axis=0)

    if with_batch_dim:
        angle_rads = tf.expand_dims(angle_rads, axis=0)

    return angle_rads
