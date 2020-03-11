import tensorflow as tf
import tensorflow.keras.layers as layers

import config
from models.Blocks import *


class AttentionLayer(layers.Layer):
    def __init__(self, filters, use_input_as_value=False, initial_layer=False, **kwargs):
        super().__init__(**kwargs)
        self.use_input_as_value = use_input_as_value
        self.initial_layer = initial_layer

        self.act = layers.LeakyReLU()
        self.norm = layers.BatchNormalization()
        self.query_mat = layers.Conv1D(
            filters=filters,
            kernel_size=4,
            strides=2,
            padding="same"
        )
        self.value_mat = layers.Conv1D(
            filters=filters,
            kernel_size=4,
            strides=2,
            padding="same"
        )

        self.attention0 = layers.Attention(
            use_scale=True,
            causal=False
        )

        self.short_downres = layers.Conv1D(
            filters=filters,
            kernel_size=4,
            strides=2,
            padding="same"
        )

        self.drop = layers.Dropout(rate=0.4)

    def call(self, inputs, **kwargs):
        x_0 = self.short_downres(inputs)
        x = inputs

        if not self.initial_layer:
            x = self.act(inputs)
            x = self.norm(x, training=kwargs["training"])

        q = self.query_mat(x)
        v = self.value_mat(x)

        x = self.attention0([q, v])

        x = layers.add([x, x_0])

        if self.initial_layer:
            x = self.drop(x, training=kwargs["training"])

        return x


class AttentionNET(tf.keras.Model):

    def __init__(self, hparams):
        super(AttentionNET, self).__init__()
        self.num_layers = hparams[config.HP_ATT2_LAYERS]

        self.att_layers = []
        for l in range(self.num_layers):
            self.att_layers.append(AttentionLayer(
                filters=hparams[config.HP_ATT2_FILTERS],
                initial_layer=l == 0
            ))

        self.avg = layers.GlobalAveragePooling1D()

        self.dense_output = layers.Dense(1, activation="sigmoid")

    def call(self, inputs, training=None, mask=None):
        x = inputs
        for layer in range(self.num_layers):
            x = self.att_layers[layer](x, training=training)

        x = self.avg(x)
        return self.dense_output(x)

    def model(self):
        x = layers.Input(shape=(500, 2))
        return tf.keras.Model(inputs=[x], outputs=self.call(x))