import tensorflow as tf
import tensorflow.keras.layers as layers

import config
from models.Blocks import *


class AttentionNETDual(tf.keras.Model):

    def __init__(self, hparams):
        super(AttentionNETDual, self).__init__()
        self.num_layers = hparams[config.HP_ATTD_LAYERS]
        self.use_common_layer = hparams[config.HP_ATTD_COMMON_INIT]

        self.common_downconv = layers.Conv1D(
            filters=hparams[config.HP_ATTD_FILTERS],
            kernel_size=3,
            padding="same"
        )

        self.arr_att_layers = []
        for l in range(self.num_layers):
            self.arr_att_layers.append(AttentionLayer(
                filters=hparams[config.HP_ATTD_FILTERS],
                initial_layer=l == 0
            ))

        self.val_att_layers = []
        for l in range(self.num_layers):
            self.val_att_layers.append(AttentionLayer(
                filters=hparams[config.HP_ATTD_FILTERS],
                initial_layer=l == 0
            ))

        self.arr_avg = layers.GlobalAveragePooling1D()
        self.val_avg = layers.GlobalAveragePooling1D()

        self.arr_dense_output = layers.Dense(1, activation="sigmoid", name="Arousal")
        self.val_dense_output = layers.Dense(1, activation="sigmoid", name="Valence")

    def call(self, inputs, training=None, mask=None):
        x = inputs
        if self.use_common_layer:
            x = self.common_downconv(x)

        x_arr = x
        x_val = x

        for layer in range(self.num_layers):
            x_arr = self.arr_att_layers[layer](x_arr, training=training)

        for layer in range(self.num_layers):
            x_val = self.val_att_layers[layer](x_val, training=training)

        x_arr = self.arr_avg(x_arr)
        x_val = self.val_avg(x_val)

        return self.arr_dense_output(x_arr), self.val_dense_output(x_val)

    def model(self):
        x = layers.Input(shape=(500, 5))
        return tf.keras.Model(inputs=[x], outputs=self.call(x))