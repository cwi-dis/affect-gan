import config

import tensorflow as tf
import tensorflow.keras.layers as layers
from models.Blocks import *


class DeepCNN(tf.keras.Model):

    def __init__(self, hparams, *args, **kwargs):
        super(DeepCNN, self).__init__(*args, **kwargs)
        self.layers_count = hparams[config.HP_DEEP_LAYERS]
        self.dual_output = hparams[config.HP_LOSS_TYPE] == "DUAL_BCE"
        self.input_len = 500

        self.down_res_layers = [DownResLayer
            (
                hparams[config.HP_DEEP_CHANNELS] * 2**l,
                kernel_size=hparams[config.HP_DEEP_KERNEL_SIZE],
                first_layer=(l == 0)
            ) for l in range(self.layers_count - 1)]

        self.down_res_layer_final_a = DownResLayer(
                hparams[config.HP_DEEP_CHANNELS] * 2**(self.layers_count-1),
                kernel_size=hparams[config.HP_DEEP_KERNEL_SIZE],
                first_layer=False,
                last_layer=True
        )
        self.down_res_layer_final_v = DownResLayer(
                hparams[config.HP_DEEP_CHANNELS] * 2**(self.layers_count-1),
                kernel_size=hparams[config.HP_DEEP_KERNEL_SIZE],
                first_layer=False,
                last_layer=True
        )

        self.feature_pool_a = layers.GlobalAveragePooling1D()
        self.feature_pool_v = layers.GlobalAveragePooling1D()
        self.lrelu_out_a = layers.LeakyReLU()
        self.lrelu_out_v = layers.LeakyReLU()

        if hparams[config.HP_LOSS_TYPE] == "MSE":
            activation = None
        else:
            activation = 'sigmoid'

        self.dense_out_a = layers.Dense(units=1, activation=activation, name="arousal_class")
        self.dense_out_v = layers.Dense(units=1, activation=activation, name="valence_class")

    def call(self, inputs, training=None, mask=None):
        x = inputs

        for i in range(self.layers_count - 1):
            x = self.down_res_layers[i](x, training=training)

        x_a = self.down_res_layer_final_a(x, training=training)
        x_a = self.lrelu_out_a(x_a)
        x_a = self.feature_pool_a(x_a)

        if self.dual_output:
            x_v = self.down_res_layer_final_v(x, training=training)
            x_v = self.lrelu_out_v(x_v)
            x_v = self.feature_pool_v(x_v)
            return self.dense_out_a(x_a), self.dense_out_v(x_v)
        else:
            return self.dense_out_a(x_a)

    def model(self):
        x = layers.Input(shape=(500, 5))
        return tf.keras.Model(inputs=[x], outputs=self.call(x))
