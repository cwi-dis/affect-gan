import config

import tensorflow as tf
import tensorflow.keras.layers as layers
from models.Blocks import *

class ChannelDownResLayer(layers.Layer):
    def __init__(self, channels_out, dropout_rate=0.4, kernel_size=3, w_norm_clip=2, first_layer=False, last_layer=False, **kwargs):
        super(ChannelDownResLayer, self).__init__(**kwargs)
        self.last_layer = last_layer
        if first_layer:
            self.down_resblock = DownResBlock(channels_out, kernel_size, w_norm_clip)
        else:
            self.down_resblock = DownResBlock(channels_out, kernel_size, w_norm_clip, initial_activation=layers.LeakyReLU())
        self.lrelu = layers.LeakyReLU()
        self.dropout = layers.Dropout(rate=dropout_rate)

    def call(self, inputs, **kwargs):
        x = self.down_resblock(inputs)

        if not self.last_layer:
            x = self.dropout(self.lrelu(x), training=kwargs["training"])

        return x


class DeepCNN(tf.keras.Model):

    def __init__(self, hparams, *args, **kwargs):
        super(DeepCNN, self).__init__(*args, **kwargs)
        self.layers_count = hparams[config.HP_DEEP_LAYERS]
        self.dual_output = hparams[config.HP_LOSS_TYPE] == "DUAL_BCE"
        self.input_len = 500
        self.dropout_rate = 0.5 / (self.layers_count - 1)

        self.down_res_layers = [ChannelDownResLayer
            (
                hparams[config.HP_DEEP_CHANNELS] * 2**l,
                kernel_size=hparams[config.HP_DEEP_KERNEL_SIZE],
                first_layer=(l == 0),
                dropout_rate=self.dropout_rate
            ) for l in range(self.layers_count - 1)]

        self.down_res_layer_final_a = ChannelDownResLayer(
                hparams[config.HP_DEEP_CHANNELS] * 2**(self.layers_count-1),
                kernel_size=hparams[config.HP_DEEP_KERNEL_SIZE],
                first_layer=False,
                last_layer=True,
                dropout_rate=self.dropout_rate
        )
        self.down_res_layer_final_v = ChannelDownResLayer(
                hparams[config.HP_DEEP_CHANNELS] * 2**(self.layers_count-1),
                kernel_size=hparams[config.HP_DEEP_KERNEL_SIZE],
                first_layer=False,
                last_layer=True,
                dropout_rate=self.dropout_rate
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
