import config

import tensorflow as tf
import tensorflow.keras.layers as layers

from models.DeepCNN import ChannelDownResLayer

class LateFuseCNN(tf.keras.Model):

    def __init__(self, hparams, views):
        super(LateFuseCNN, self).__init__()
        self.views = views
        self.view_layers_count = hparams[config.HP_LDEEP_V_LAYERS]
        self.fuse_layers_count = hparams[config.HP_LDEEP_F_LAYERS]

        self.channel_down_res_layers = [[
            ChannelDownResLayer(
                channels_out=hparams[config.HP_LDEEP_V_CHANNELS] * (2 ** l),
                dropout_rate=hparams[config.HP_LDEEP_DROPOUT],
                kernel_size=hparams[config.HP_LDEEP_KSIZE],
                w_norm_clip=hparams[config.HP_LDEEP_WEIGHTNORM]
            ) for c in range(self.views)
        ] for l in range(self.view_layers_count - 1)]

        self.channel_down_res_final_layer = [
            ChannelDownResLayer(
                channels_out=hparams[config.HP_LDEEP_V_CHANNELS] * (2 ** (self.view_layers_count-1)),
                dropout_rate=hparams[config.HP_LDEEP_DROPOUT],
                kernel_size=hparams[config.HP_LDEEP_KSIZE],
                w_norm_clip=hparams[config.HP_LDEEP_WEIGHTNORM],
                last_layer=True
            ) for c in range(self.views)
        ]

        self.merged_channel_n = hparams[config.HP_LDEEP_V_CHANNELS] * (2 ** (self.view_layers_count-1)) * self.views

        self.down_res_layers = [
            ChannelDownResLayer(
                self.merged_channel_n // (2 ** l),
                dropout_rate=hparams[config.HP_LDEEP_DROPOUT],
                kernel_size=hparams[config.HP_LDEEP_KSIZE],
                w_norm_clip = hparams[config.HP_LDEEP_WEIGHTNORM]
            ) for l in range(self.fuse_layers_count - 1)
        ]
        self.down_res_layer_final = ChannelDownResLayer(
            self.merged_channel_n // (2 ** (self.fuse_layers_count - 1)),
            dropout_rate=hparams[config.HP_LDEEP_DROPOUT],
            kernel_size=hparams[config.HP_LDEEP_KSIZE],
            w_norm_clip=hparams[config.HP_LDEEP_WEIGHTNORM],
            last_layer=True
        )

        self.feature_pool = layers.GlobalAveragePooling1D()
        self.lrelu_out = layers.LeakyReLU()
        self.dense_out = layers.Dense(units=1, activation='sigmoid')


    def call(self, inputs, training=None, mask=None):
        x_c = tf.split(inputs, self.views, -1)
        for layer in range(self.view_layers_count - 1):
            x_c = [self.channel_down_res_layers[layer][c](x_c[c], training=training) for c in range(self.views)]
        x_c = [self.channel_down_res_final_layer[c](x_c[c], training=training) for c in range(self.views)]

        x = layers.concatenate(x_c)

        for i in range(self.fuse_layers_count - 1):
            x = self.down_res_layers[i](x, training=training)
        x = self.down_res_layer_final(x, training=training)

        x = self.feature_pool(x)
        x = self.lrelu_out(x)
        return self.dense_out(x)


    def model(self):
        x = layers.Input(shape=(500, 5))
        return tf.keras.Model(inputs=[x], outputs=self.call(x))
