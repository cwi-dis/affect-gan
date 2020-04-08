import tensorflow as tf
import tensorflow.keras.layers as layers

import config
from models.Blocks import *

class Generator(tf.keras.Model):

    def __init__(self, *args, **kwargs):
        super(Generator, self).__init__(*args, **kwargs)

        self.up_0 = UpResLayer(channels_out=16, kernel_size=3, use_actnormdrop=True, dropout_rate=0.5)
        self.up_1 = UpResLayer(channels_out=16, kernel_size=5)
        self.non_local = AttentionLayer(filters=16, use_actnormdrop=True)
        self.final_conv = layers.Conv1D(filters=5, kernel_size=3, padding="same",
                                        kernel_constraint=tf.keras.constraints.MaxNorm(max_value=1, axis=[0, 1]))

    def call(self, inputs, training=None, mask=None):
        x = tf.expand_dims(inputs, axis=-1)
        x = self.up_0(x)
        x = self.up_1(x)
        x = self.non_local(x)
        x = self.final_conv(x)
        x = tf.keras.activations.tanh(x)
        return x

    def model(self):
        x = layers.Input(shape=(125))
        return tf.keras.Model(inputs=[x], outputs=self.call(x))


class Discriminator(tf.keras.Model):
    def __init__(self, hparams, *args, **kwargs):
        super(Discriminator, self).__init__(*args, **kwargs)

        self.downres0 = DownResLayer(
            channels_out=4,
            dropout_rate=0.3,
            first_layer=True,
            use_dropout=True
        )
        self.non_local = AttentionLayer(
            filters=4,
            use_actnormdrop=True,
            dropout_rate=0.0
        )
        self.downres1 = DownResLayer(
            channels_out=2,
            use_dropout=True
        )
        self.downres2 = DownResLayer(
            channels_out=2,
            use_dropout=False
        )

        self.avg = layers.GlobalAveragePooling1D()

        self.dense_output = layers.Dense(1, activation="sigmoid")

    def call(self, inputs, training=None, mask=None):
        x = self.downres0(inputs, training=training)
        x = self.non_local(x, training=training)
        x = self.downres1(x, training=training)
        x = self.downres2(x, training=training)

        x = self.avg(x)
        return self.dense_output(x)

    def model(self):
        x = layers.Input(shape=(500, 5))
        return tf.keras.Model(inputs=[x], outputs=self.call(x))
