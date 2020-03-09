import tensorflow as tf
import tensorflow.keras.layers as layers

import config
from models.Blocks import *


class AttentionLayer(layers.Layer):
    def __init__(self, filters, downsample=False, **kwargs):
        super().__init__(**kwargs)
        strides = 2 if downsample else 1
        self.query_mat = layers.Conv1D(
            filters=filters,
            kernel_size=4,
            strides=strides,
            padding="same"
        )
        self.value_mat = layers.Conv1D(
            filters=filters,
            kernel_size=4,
            strides=strides,
            padding="same"
        )

        self.attention0 = layers.Attention(
            use_scale=True,
            causal=False
        )

    def call(self, inputs, **kwargs):
        q = self.query_mat(inputs)
        v = self.value_mat(inputs)

        x = self.attention0([q, v])

        return x


class AttentionNET(tf.keras.Model):

    def __init__(self, hparams):
        super(AttentionNET, self).__init__()
        self.use_last_layer = hparams[config.HP_ATT_EXTRA_LAYER]
        self.upchannnel_attention = 2 if hparams[config.HP_ATT_UPCHANNEL] else 1
        self.downres0 = DownResLayer(
            channels_out=hparams[config.HP_ATT_FILTERS],
            first_layer=True,
            use_dropout=True
        )

        self.attention_layer = AttentionLayer(
            filters=hparams[config.HP_ATT_FILTERS] * self.upchannnel_attention,
            downsample=hparams[config.HP_ATT_DOWNRESATT]
        )

        self.downres1 = DownResLayer(
            channels_out=hparams[config.HP_ATT_FILTERS] * 2 * self.upchannnel_attention
        )

        self.downres_f = DownResLayer(
            channels_out=hparams[config.HP_ATT_FILTERS] * 4 * self.upchannnel_attention
        )

        self.avg = layers.GlobalAveragePooling1D()

        self.dense_output = layers.Dense(1, activation="sigmoid")

    def call(self, inputs, training=None, mask=None):
        x = self.downres0(inputs)
        x = self.attention_layer(x)
        x = self.downres1(x)
        if self.use_last_layer:
            x = self.downres_f(x)
        x = self.avg(x)
        return self.dense_output(x)

    def model(self):
        x = layers.Input(shape=(500, 2))
        return tf.keras.Model(inputs=[x], outputs=self.call(x))