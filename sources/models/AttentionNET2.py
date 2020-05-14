import tensorflow as tf
import tensorflow.keras.layers as layers

import config
from models.Blocks import *


class AttentionNET(tf.keras.Model):

    def __init__(self, hparams):
        super(AttentionNET, self).__init__()
        self.num_layers = hparams[config.HP_ATT2_LAYERS]

        self.att_layers = []
        for l in range(self.num_layers):
            self.att_layers.append(AttentionLayer(
                filters=hparams[config.HP_ATT2_FILTERS] if l <= 1 else hparams[config.HP_ATT2_OTHERFILTERS],
                initial_layer=l <= 1
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
